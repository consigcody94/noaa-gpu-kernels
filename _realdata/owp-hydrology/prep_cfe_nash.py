"""
prep_cfe_nash.py — Convert real CFE config parameters + AORC precipitation from
NOAA-OWP/cfe into inputs for the CFE Nash-cascade kernel in
owp_batched_kernels_realdata.cu.

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/cfe master):
  - configs/cfe_config_cat_87.txt : real calibrated CFE config for NextGen catchment cat-87
        K_nash_subsurface = 0.03,  nash_storage_subsurface = 0.0,0.0  (N=2)
        K_nash_surface    = 0.83089, nash_storage_surface  = 0.0,0.0  (N_nash_surface=2)
  - forcings/cat87_01Dec2015.csv  : 720 hourly AORC records; APCP_surface (mm/h, i.e.
        kg m-2 over the 1 h timestep) -> lateral flux in metres per timestep = APCP/1000.

The Nash cascade kernel routes a lateral flux (m) through N linear reservoirs with
coefficient K. We drive it with the real hourly precipitation depth series as the
inflow flux and the two real calibrated (K, N, initial storage) parameter sets from
the cat-87 config (subsurface and surface Nash cascades).

Output: data/nash_realdata.bin
  int32 magic=0x4E415348, int32 nparamsets, int32 nhours
  per paramset: float32 K, int32 N, float32 init_storage[10]
  float32 precip_m[nhours]
"""
import csv
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
CFE = HERE / "upstream" / "cfe"
OUT = HERE / "data" / "nash_realdata.bin"
MAX_NASH = 10

cfg = {}
for line in (CFE / "configs" / "cfe_config_cat_87.txt").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.split("[")[0].strip()

paramsets = []
# subsurface Nash cascade
ss_stor = [float(x) for x in cfg["nash_storage_subsurface"].split(",")]
paramsets.append((float(cfg["K_nash_subsurface"]), len(ss_stor), ss_stor))
# surface Nash cascade
sf_stor = [float(x) for x in cfg["nash_storage_surface"].split(",")]
assert len(sf_stor) == int(cfg["N_nash_surface"])
paramsets.append((float(cfg["K_nash_surface"]), len(sf_stor), sf_stor))

precip = []
with open(CFE / "forcings" / "cat87_01Dec2015.csv", newline="") as f:
    for row in csv.DictReader(f):
        precip.append(float(row["APCP_surface"]) / 1000.0)  # mm -> m per 1 h step
precip = np.array(precip, dtype="<f4")
print(f"paramsets: {paramsets}")
print(f"precip: n={len(precip)}, total={precip.sum()*1000:.1f} mm, max={precip.max()*1000:.2f} mm/h")

with open(OUT, "wb") as f:
    f.write(struct.pack("<iii", 0x4E415348, len(paramsets), len(precip)))
    for K, N, stor in paramsets:
        f.write(struct.pack("<f", K))
        f.write(struct.pack("<i", N))
        padded = (stor + [0.0] * MAX_NASH)[:MAX_NASH]
        f.write(struct.pack(f"<{MAX_NASH}f", *padded))
    f.write(precip.tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
