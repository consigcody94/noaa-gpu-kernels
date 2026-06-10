"""
Extract real Argo (T, S, P) data for the MOM6 EOS + triDiagTS GPU kernel harnesses.

Data source (public, authoritative):
  Argo GDAC (Ifremer mirror), daily geo profile files for 2026-05-15:
    https://data-argo.ifremer.fr/geo/atlantic_ocean/2026/05/20260515_prof.nc
    https://data-argo.ifremer.fr/geo/pacific_ocean/2026/05/20260515_prof.nc
    https://data-argo.ifremer.fr/geo/indian_ocean/2026/05/20260515_prof.nc
  Accessed 2026-06-10.

QC policy (standard Argo usage):
  - Per-profile DATA_MODE: if 'A' (adjusted-RT) or 'D' (delayed mode), use
    <PARAM>_ADJUSTED with <PARAM>_ADJUSTED_QC; if 'R', use raw <PARAM> with
    <PARAM>_QC.
  - Keep a level only if PRES, TEMP and PSAL are all unmasked AND all three
    QC flags are in {1, 2} (good / probably good).
  - Additionally require physically plausible bounds (paranoia only; values
    are NOT altered): -3 <= T <= 40 degC, 0 < S <= 42 PSU, 0 <= P <= 7000 dbar.

Outputs (in data/):
  eos_tsp.bin     : int32 n, then float32 T[n], S[n], P[n]   (P in dbar)
  eos_region.u8   : uint8 region[n]; 0=open ocean, 1=polar (|lat|>=60),
                    2=Mediterranean box (30..46N, -6..36.5E)
  columns.bin     : int32 ncol, then per column:
                    int32 nz, float32 h[nz], T[nz], S[nz]
                    h = layer thickness (m) from pressure spacing (1 dbar ~ 1 m),
                    h[k] = midpoint-to-midpoint interface spacing.
                    Columns capped at 75 levels (kernel MAX_OC_LEV) by even
                    subsampling of the good levels; require >= 20 good levels
                    with strictly increasing pressure.
  summary.json    : provenance + stats
"""
import json
import numpy as np
from netCDF4 import Dataset

DATA = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\mom6_eos\data"
FILES = [
    ("atlantic_ocean", DATA + r"\atlantic_ocean_20260515_prof.nc"),
    ("pacific_ocean",  DATA + r"\pacific_ocean_20260515_prof.nc"),
    ("indian_ocean",   DATA + r"\indian_ocean_20260515_prof.nc"),
]
MAX_OC_LEV = 75
MIN_LEVELS = 20

all_T, all_S, all_P, all_R = [], [], [], []
columns = []  # list of (h, T, S) arrays
stats = {"files": {}, "n_profiles_used": 0, "n_profiles_skipped": 0}

def qc_ok(qc_arr):
    """QC flags come as masked char array; return boolean good (1 or 2)."""
    q = np.ma.filled(qc_arr, b" ")
    if q.dtype.kind in ("S", "U"):
        q = q.astype("S1")
        return (q == b"1") | (q == b"2")
    return (q == 1) | (q == 2)

for basin, path in FILES:
    ds = Dataset(path)
    nprof = ds.dimensions["N_PROF"].size
    dm = np.ma.filled(ds.variables["DATA_MODE"][:], b"R").astype("S1")
    lat = np.ma.filled(ds.variables["LATITUDE"][:], np.nan)
    lon = np.ma.filled(ds.variables["LONGITUDE"][:], np.nan)

    raw = {v: ds.variables[v][:] for v in ("PRES", "TEMP", "PSAL")}
    rawq = {v: ds.variables[v + "_QC"][:] for v in ("PRES", "TEMP", "PSAL")}
    adj = {v: ds.variables[v + "_ADJUSTED"][:] for v in ("PRES", "TEMP", "PSAL")}
    adjq = {v: ds.variables[v + "_ADJUSTED_QC"][:] for v in ("PRES", "TEMP", "PSAL")}

    used = skipped = 0
    npts_file = 0
    for p in range(nprof):
        adjusted = dm[p] in (b"A", b"D")
        vals, good = {}, None
        ok_profile = True
        for v in ("PRES", "TEMP", "PSAL"):
            arr = adj[v][p] if adjusted else raw[v][p]
            qc = adjq[v][p] if adjusted else rawq[v][p]
            g = qc_ok(qc) & ~np.ma.getmaskarray(arr)
            vals[v] = np.ma.filled(arr, np.nan).astype(np.float64)
            good = g if good is None else (good & g)
        if good is None or not ok_profile:
            skipped += 1
            continue
        T, S, P = vals["TEMP"], vals["PSAL"], vals["PRES"]
        # plausibility bounds (filter only, never modify)
        good &= np.isfinite(T) & np.isfinite(S) & np.isfinite(P)
        good &= (T >= -3.0) & (T <= 40.0) & (S > 0.0) & (S <= 42.0)
        good &= (P >= 0.0) & (P <= 7000.0)
        idx = np.where(good)[0]
        if idx.size == 0:
            skipped += 1
            continue
        T, S, P = T[idx], S[idx], P[idx]
        # sort by pressure, drop non-strictly-increasing duplicates
        order = np.argsort(P, kind="stable")
        T, S, P = T[order], S[order], P[order]
        keep = np.concatenate(([True], np.diff(P) > 0))
        T, S, P = T[keep], S[keep], P[keep]

        # ---- EOS triplets ----
        la, lo = lat[p], lon[p]
        if np.isfinite(la) and abs(la) >= 60.0:
            reg = 1
        elif np.isfinite(la) and np.isfinite(lo) and 30.0 <= la <= 46.0 and -6.0 <= lo <= 36.5:
            reg = 2
        else:
            reg = 0
        all_T.append(T); all_S.append(S); all_P.append(P)
        all_R.append(np.full(T.size, reg, dtype=np.uint8))
        npts_file += T.size

        # ---- triDiagTS column ----
        if T.size >= MIN_LEVELS:
            if T.size > MAX_OC_LEV:
                sel = np.unique(np.round(np.linspace(0, T.size - 1, MAX_OC_LEV)).astype(int))
            else:
                sel = np.arange(T.size)
            Pc, Tc, Sc = P[sel], T[sel], S[sel]
            nz = Pc.size
            # interface depths at midpoints between samples (1 dbar ~ 1 m)
            iface = np.empty(nz + 1)
            iface[1:-1] = 0.5 * (Pc[1:] + Pc[:-1])
            iface[0] = max(0.0, Pc[0] - 0.5 * (Pc[1] - Pc[0]))
            iface[-1] = Pc[-1] + 0.5 * (Pc[-1] - Pc[-2])
            h = np.diff(iface)
            if np.all(h > 0):
                columns.append((h.astype(np.float32), Tc.astype(np.float32), Sc.astype(np.float32)))
        used += 1
    stats["files"][basin] = {"n_prof_in_file": int(nprof), "profiles_used": used,
                             "profiles_skipped": skipped, "good_points": int(npts_file)}
    stats["n_profiles_used"] += used
    stats["n_profiles_skipped"] += skipped
    ds.close()

T = np.concatenate(all_T).astype(np.float32)
S = np.concatenate(all_S).astype(np.float32)
P = np.concatenate(all_P).astype(np.float32)
R = np.concatenate(all_R)
n = T.size

with open(DATA + r"\eos_tsp.bin", "wb") as f:
    np.array([n], dtype=np.int32).tofile(f)
    T.tofile(f); S.tofile(f); P.tofile(f)
R.tofile(DATA + r"\eos_region.u8")

with open(DATA + r"\columns.bin", "wb") as f:
    np.array([len(columns)], dtype=np.int32).tofile(f)
    for h, Tc, Sc in columns:
        np.array([h.size], dtype=np.int32).tofile(f)
        h.tofile(f); Tc.tofile(f); Sc.tofile(f)

stats.update({
    "n_eos_triplets": int(n),
    "n_columns": len(columns),
    "T_min": float(T.min()), "T_max": float(T.max()),
    "S_min": float(S.min()), "S_max": float(S.max()),
    "P_min": float(P.min()), "P_max": float(P.max()),
    "n_polar": int((R == 1).sum()), "n_med": int((R == 2).sum()),
    "n_open": int((R == 0).sum()),
    "col_nz_min": int(min(c[0].size for c in columns)),
    "col_nz_max": int(max(c[0].size for c in columns)),
})
with open(DATA + r"\summary.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
