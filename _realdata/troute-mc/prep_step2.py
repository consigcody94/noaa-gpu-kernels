"""
Build the step-2 (warm-start) input lcr_network_step2.bin.

Takes the step-1 real-data input (lcr_network.bin, all values from NWM
RouteLink + AnA CHRTOUT) plus the step-1 outputs of the UNMODIFIED FP64 CPU
reference (cpu_state_out.bin dumped by troute_mc_final_realdata.exe) and
advances the state one routing step the same way t-route does:

  dp_2  = depth output of step 1 (CPU reference, FP64)
  qd_2  = flow output of step 1 at the segment
  qu_2  = sum over upstream parents of their step-1 flow output
          (parents derived from the same RouteLink 'to' topology, re-derived
           here from the ordered link list written by prep_lcr_network.py)
  ql_2  = unchanged (t-route subdivides hourly CHRTOUT qlat into 12 identical
          300 s values, qts_subdivisions: 12 -- so the same ql is exact)

No fabricated or random values: everything is real data or the deterministic
one-step evolution of real data through the unmodified reference implementation.
"""
import struct
import numpy as np
import netCDF4 as nc

BASE = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\troute-mc"

# ---- read step-1 binary ----
with open(BASE + r"\lcr_network.bin", "rb") as f:
    ns, nr = struct.unpack("<ii", f.read(8))
    P = np.frombuffer(f.read(ns * 8 * 8), dtype="<f8").reshape(ns, 8).copy()
    ql = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()
    qu = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()
    qd = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()
    dp = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()
    rs = np.frombuffer(f.read(nr * 4), dtype="<i4").copy()
    rl = np.frombuffer(f.read(nr * 4), dtype="<i4").copy()
print(f"step-1 input: {ns} segments, {nr} reaches")

# ---- read step-1 CPU-reference outputs ----
with open(BASE + r"\cpu_state_out.bin", "rb") as f:
    (ns2,) = struct.unpack("<i", f.read(4))
    assert ns2 == ns
    q1 = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()  # step-1 outflow
    d1 = np.frombuffer(f.read(ns * 8), dtype="<f8").copy()  # step-1 depth
print(f"step-1 CPU outputs: flow [{q1.min():.4g},{q1.max():.4g}] depth [{d1.min():.4g},{d1.max():.4g}]")

# ---- topology from the ordered id CSV + RouteLink 'to' ----
ids = np.loadtxt(BASE + r"\lcr_network_ids.csv", delimiter=",", skiprows=1,
                 usecols=(1,), dtype=np.int64)
assert len(ids) == ns
rlnc = nc.Dataset(BASE + r"\data\RouteLink.nc")
link_all = np.asarray(rlnc["link"][:], dtype=np.int64)
to_all = np.asarray(rlnc["to"][:], dtype=np.int64)
rlnc.close()
to_map = dict(zip(link_all, to_all))
pos = {l: i for i, l in enumerate(ids)}

qu2 = np.zeros(ns)
for i in range(ns):
    j = pos.get(to_map[ids[i]])
    if j is not None:
        qu2[j] += q1[i]

with open(BASE + r"\lcr_network_step2.bin", "wb") as f:
    f.write(struct.pack("<ii", ns, nr))
    f.write(P.astype("<f8").tobytes())
    f.write(ql.astype("<f8").tobytes())   # same hourly qlat (qts_subdivisions)
    f.write(qu2.astype("<f8").tobytes())  # parents' step-1 outflow
    f.write(q1.astype("<f8").tobytes())   # qd = step-1 outflow at segment
    f.write(d1.astype("<f8").tobytes())   # dp = step-1 depth (warm start)
    f.write(rs.astype("<i4").tobytes())
    f.write(rl.astype("<i4").tobytes())
print("wrote lcr_network_step2.bin")
print(f"warm dp: nonzero {np.count_nonzero(d1)}/{ns}, mean {d1.mean():.4f} m, max {d1.max():.4f} m")
