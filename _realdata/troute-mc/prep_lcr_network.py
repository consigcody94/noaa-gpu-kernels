"""
Prep script: build a flat binary input for the t-route MC GPU harnesses from
REAL NOAA data (Lower Colorado River, TX test domain shipped in NOAA-OWP/t-route).

Sources (downloaded 2026-06-10 from NOAA-OWP/t-route @ master
12a8eae0cdfed437143c590659fa7077605a5e70):
  - test/LowerColorado_TX/domain/RouteLink.nc
      NWM RouteLink subset: real NHDPlus-derived channel geometry
      (Length, BtmWdth, TopWdth, TopWdthCC, n, nCC, ChSlp, So) + topology (link, to)
  - test/LowerColorado_TX/channel_forcing/202108231300.CHRTOUT_DOMAIN1
      Real NWM Analysis-and-Assimilation channel output, valid 2021-08-23 13:00 UTC:
      streamflow, qBucket, qSfcLatRunoff per feature_id.

Mapping to harness inputs (per segment, ordered upstream->downstream within reach):
  dx=Length, bw=BtmWdth, tw=TopWdth, twcc=TopWdthCC, n_ch=n, ncc=nCC, cs=ChSlp, s0=So
  ql  = qBucket + qSfcLatRunoff   (t-route's preferred qlat composition, nhd_io.py)
  qd  = streamflow at the segment (previous-timestep flow at segment)
  qu  = sum of streamflow over upstream neighbors (links whose 'to' == this link); 0 for headwaters
  dp  = 0.0  (cold start -- matches test_AnA.yaml, which leaves the WRF restart commented out;
              the shipped HYDRO_RST has 11141 links vs RouteLink's 11248 so a positional
              join per t-route's own convention would misalign -- NOT used)

Waterbody-internal links (341; all have NHDWaterbodyComID set) carry only
_FillValue streamflow in CHRTOUT (WRF-Hydro/NWM does not route them) and are
excluded -- the same exclusion t-route's test performs via
break_network_at_waterbodies: True. Children of excluded links become reach heads;
their qu sums only the surviving parents (lake outflow boundary set to 0, documented).

Reach construction: maximal linear chains of the 'to' topology; a link starts a new
reach iff its in-degree (within the routed domain) != 1. Matches t-route's reach
concept (network broken at junctions).

Output binary (little-endian), lcr_network.bin:
  int32 n_seg, int32 n_reach
  float64 params[n_seg*8]  (dx,bw,tw,twcc,n_ch,ncc,cs,s0 interleaved per segment)
  float64 ql[n_seg], qu[n_seg], qd[n_seg], dp[n_seg]
  int32 rs[n_reach] (reach start index), int32 rl[n_reach] (reach length)
Plus lcr_network_ids.csv (link id per ordered segment, for traceability).

NO synthetic/fabricated values enter the input path.
"""
import struct
import sys
import numpy as np
import netCDF4 as nc

DATA = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\troute-mc\data"
OUT_BIN = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\troute-mc\lcr_network.bin"
OUT_CSV = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\troute-mc\lcr_network_ids.csv"

# ---- RouteLink: geometry + topology ----
rl = nc.Dataset(DATA + r"\RouteLink.nc")
link = np.asarray(rl["link"][:], dtype=np.int64)
to = np.asarray(rl["to"][:], dtype=np.int64)
geom = {
    "dx": np.asarray(rl["Length"][:], dtype=np.float64),
    "bw": np.asarray(rl["BtmWdth"][:], dtype=np.float64),
    "tw": np.asarray(rl["TopWdth"][:], dtype=np.float64),
    "twcc": np.asarray(rl["TopWdthCC"][:], dtype=np.float64),
    "n_ch": np.asarray(rl["n"][:], dtype=np.float64),
    "ncc": np.asarray(rl["nCC"][:], dtype=np.float64),
    "cs": np.asarray(rl["ChSlp"][:], dtype=np.float64),
    "s0": np.asarray(rl["So"][:], dtype=np.float64),
}
rl.close()
n = len(link)
print(f"RouteLink: {n} segments")

# ---- CHRTOUT: real flows (auto scale_factor applied by netCDF4) ----
ch = nc.Dataset(DATA + r"\202108231300.CHRTOUT_DOMAIN1.nc")
fid = np.asarray(ch["feature_id"][:], dtype=np.int64)
streamflow = ch["streamflow"][:]  # masked array: waterbody-internal links are _FillValue
qbucket = ch["qBucket"][:]
qsfc = ch["qSfcLatRunoff"][:]
t_valid = nc.num2date(ch["time"][0], ch["time"].units)
ch.close()
print(f"CHRTOUT: {len(fid)} features, valid {t_valid}")

# Align CHRTOUT to RouteLink by id
fmap = {f: i for i, f in enumerate(fid)}
idx = np.array([fmap[l] for l in link])  # KeyError if any link missing => hard fail
sf_mask = np.ma.getmaskarray(streamflow)[idx] | np.ma.getmaskarray(qbucket)[idx] \
    | np.ma.getmaskarray(qsfc)[idx]
print(f"links with masked CHRTOUT flow (waterbody-internal, excluded): {sf_mask.sum()}")

# Exclude waterbody-internal links (no real flow state exists for them; t-route's
# own test config also removes them via break_network_at_waterbodies: True).
keep = ~sf_mask
link = link[keep]
to = to[keep]
for k in geom:
    geom[k] = geom[k][keep]
idx = idx[keep]
n = len(link)
print(f"Routed domain after waterbody exclusion: {n} segments")

qd_all = np.ma.filled(streamflow, np.nan)[idx].astype(np.float64)
ql_all = (np.ma.filled(qbucket, np.nan)[idx] + np.ma.filled(qsfc, np.nan)[idx]).astype(np.float64)
assert np.isfinite(qd_all).all() and np.isfinite(ql_all).all()

# ---- upstream previous-timestep flow: sum of parents' streamflow ----
lmap = {l: i for i, l in enumerate(link)}
qu_all = np.zeros(n)
indeg = np.zeros(n, dtype=np.int64)
for i in range(n):
    t = to[i]
    j = lmap.get(t)
    if j is not None:
        qu_all[j] += qd_all[i]
        indeg[j] += 1

# ---- reach construction: maximal linear chains ----
heads = [i for i in range(n) if indeg[i] != 1]
order = []
rs, rlen = [], []
visited = np.zeros(n, dtype=bool)
for h in heads:
    rs.append(len(order))
    cnt = 0
    cur = h
    while True:
        assert not visited[cur], f"cycle at link {link[cur]}"
        visited[cur] = True
        order.append(cur)
        cnt += 1
        j = lmap.get(to[cur])
        if j is None or indeg[j] != 1 or visited[j]:
            break
        cur = j
    rlen.append(cnt)
assert visited.all(), f"unvisited segments: {(~visited).sum()}"
order = np.array(order)
nreach = len(rs)
print(f"Reaches: {nreach}, segments: {len(order)}, "
      f"max reach len {max(rlen)}, mean {np.mean(rlen):.2f}")

# ---- stats / sanity (report only, no modification of real values) ----
def stats(name, a):
    print(f"  {name:5s} min {np.min(a):.6g}  max {np.max(a):.6g}  mean {np.mean(a):.6g}")
print("Real-data ranges (full domain):")
for k, v in geom.items():
    stats(k, v)
stats("ql", ql_all)
stats("qu", qu_all)
stats("qd", qd_all)
print(f"  segments with s0<=0: {(geom['s0']<=0).sum()}, n<=0: {(geom['n_ch']<=0).sum()}, "
      f"bw<=0: {(geom['bw']<=0).sum()}, tw<bw: {(geom['tw']<geom['bw']).sum()}, "
      f"all-zero-inflow: {((ql_all<=0)&(qu_all<=0)&(qd_all<=0)).sum()}")

# ---- write binary in reach order ----
P = np.empty((len(order), 8))
for c, k in enumerate(["dx", "bw", "tw", "twcc", "n_ch", "ncc", "cs", "s0"]):
    P[:, c] = geom[k][order]
ql_o, qu_o, qd_o = ql_all[order], qu_all[order], qd_all[order]
dp_o = np.zeros(len(order))  # cold start per test_AnA.yaml

with open(OUT_BIN, "wb") as f:
    f.write(struct.pack("<ii", len(order), nreach))
    f.write(P.astype("<f8").tobytes())
    f.write(ql_o.astype("<f8").tobytes())
    f.write(qu_o.astype("<f8").tobytes())
    f.write(qd_o.astype("<f8").tobytes())
    f.write(dp_o.astype("<f8").tobytes())
    f.write(np.array(rs, dtype="<i4").tobytes())
    f.write(np.array(rlen, dtype="<i4").tobytes())
print(f"Wrote {OUT_BIN}")

with open(OUT_CSV, "w") as f:
    f.write("seg_index,link,dx,bw,tw,twcc,n_ch,ncc,cs,s0,ql,qu,qd\n")
    for s, i in enumerate(order):
        f.write(f"{s},{link[i]},{geom['dx'][i]},{geom['bw'][i]},{geom['tw'][i]},"
                f"{geom['twcc'][i]},{geom['n_ch'][i]},{geom['ncc'][i]},{geom['cs'][i]},"
                f"{geom['s0'][i]},{ql_all[i]},{qu_all[i]},{qd_all[i]}\n")
print(f"Wrote {OUT_CSV}")
