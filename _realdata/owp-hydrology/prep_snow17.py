"""
prep_snow17.py — Convert the real Snow17 test case shipped in NOAA-OWP/snow17
(test_cases/ex1) into flat inputs for owp_snow17_lgar_realdata.cu.

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/snow17 master,
test_cases/ex1.tgz extracted in place):
  - ex1/input/forcing/forcing.snow17bmi.HHWM8IL.csv and ...HHWM8IU.csv :
      16,801 daily records (1970-01-01 .. 2015-12-31) of mean-areal precipitation
      (mm/s) and air temperature (degC) for the lower (IL) and upper (IU) elevation
      bands of the South Fork Flathead River watershed above Hungry Horse reservoir,
      Montana (case study developed at NCAR; Mendoza et al. 2017, HESS 21).
  - ex1/input/params/snow17_params.HHWM8.txt :
      real calibrated Snow17 parameters per band (scf, mfmax, mfmin, uadj, si,
      pxtemp, tipm, mbase, plwhc, daygm).

Mapping to kernel structs:
  Snow17Params (order scf, mfmax, mfmin, uadj, si, tipm, mbase, plwhc, daygm, pxtemp)
      <- per-band values from the params file.
  Snow17Forcing: ta = tavg_degc; px = prec_mm_s-1 * 86400 (mm per daily timestep);
      dt_hours = 24 (daily model timestep per ex1 README).
State trajectories (we, liqw, neghs, aesc, tprev) are generated inside the harness by
running the UNTOUCHED CPU reference sequentially over this real series (cold start,
all-zero state, as in a standard model spin-up).

Output: data/snow17_realdata.bin
  int32 magic=0x534E4F57, int32 nhru, int32 nrec
  per hru: 10 float32 params (struct order above)
  per hru: nrec * 2 float32 (ta, px)
"""
import csv
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
EX1 = HERE / "upstream" / "snow17" / "test_cases" / "ex1"
OUT = HERE / "data" / "snow17_realdata.bin"

# --- params (two HRUs: columns IL, IU) ---
ptab = {}
for line in (EX1 / "input" / "params" / "snow17_params.HHWM8.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 3:
        ptab[parts[0]] = parts[1:3]

hrus = ptab["hru_id"]            # ['HHWM8IL', 'HHWM8IU']
ORDER = ["scf", "mfmax", "mfmin", "uadj", "si", "tipm", "mbase", "plwhc", "daygm", "pxtemp"]
params = []
for i, hru in enumerate(hrus):
    params.append([float(ptab[k][i]) for k in ORDER])
    print(hru, dict(zip(ORDER, params[-1])))

# --- forcing ---
forcing = []
nrec = None
for hru in hrus:
    ta, px = [], []
    with open(EX1 / "input" / "forcing" / f"forcing.snow17bmi.{hru}.csv", newline="") as f:
        for row in csv.DictReader(f):
            px.append(float(row["prec_mm_s-1"]) * 86400.0)  # mm per daily step
            ta.append(float(row["tavg_degc"]))
    forcing.append((np.array(ta, dtype="<f4"), np.array(px, dtype="<f4")))
    nrec = len(ta) if nrec is None else min(nrec, len(ta))
    print(f"{hru}: {len(ta)} records, ta [{min(ta):.1f},{max(ta):.1f}] C, "
          f"px max {max(px):.1f} mm/d")

with open(OUT, "wb") as f:
    f.write(struct.pack("<iii", 0x534E4F57, len(hrus), nrec))
    for p in params:
        f.write(struct.pack("<10f", *p))
    for ta, px in forcing:
        inter = np.empty((nrec, 2), dtype="<f4")
        inter[:, 0] = ta[:nrec]
        inter[:, 1] = px[:nrec]
        f.write(inter.tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
