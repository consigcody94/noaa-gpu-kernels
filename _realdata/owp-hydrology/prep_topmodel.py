"""
prep_topmodel.py — Convert the real TOPMODEL test catchment data shipped in
NOAA-OWP/topmodel (data/) into a flat binary for owp_extended_kernels_realdata.cu.

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/topmodel master):
  - data/params.dat   : "Extracted study basin: Taegu Pyungkwang River" calibrated parameters
                        szm t0 td chv rv srmax Q0 sr0 infex xk0 hf dth
  - data/subcat.dat   : 30-ordinate ln(a/tanB) areal distribution (AC, ST pairs)
  - data/inputs.dat   : nstep=950, dt=1.0 h; columns rain(m/dt), pe(m/dt), Qobs(m/dt)
This is the real catchment dataset distributed with the canonical TOPMODEL test case
(Pyungkwang River basin, Korea), inherited from the original Beven TMOD9502 distribution format.

Derived quantities use the model's own init equations (src/topmodel.c):
  tl   = sum_j AC[j] * (ST[j] + ST[j-1])/2      (line ~515, areal trapezoid; ST[0]=first ordinate)
  szq  = exp(t0 + ln(dt) - tl)                  (init_water_balance, line 838-840)
  sbar0 = -szm * ln(Q0 / szq)                   (line 848)

Output: data/topmodel_realdata.bin
  int32 magic=0x544F504D, int32 num_bins, float32 szm, szq, td, sbar0,
  float32 lnaotb[num_bins], int32 nstep, float32 rain[nstep]
"""
import math
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
TOP = HERE / "upstream" / "topmodel" / "data"
OUT = HERE / "data" / "topmodel_realdata.bin"

# --- params.dat ---
lines = (TOP / "params.dat").read_text().splitlines()
vals = lines[1].split()
szm, t0, td, chv, rv, srmax, Q0, sr0, infex, xk0, hf, dth = [float(v) for v in vals[:12]]

# --- subcat.dat ---
sc = (TOP / "subcat.dat").read_text().split("\n")
# line0: "1 1 1", line1: name, line2: "nac area"
nac, area = sc[2].split()
nac = int(nac)
pairs = []
idx = 3
while len(pairs) < nac:
    parts = sc[idx].split()
    pairs.append((float(parts[0]), float(parts[1])))
    idx += 1
AC = np.array([p[0] for p in pairs])
ST = np.array([p[1] for p in pairs])

# tl exactly as topmodel.c lines 508-516: normalize AC by total area, then
# tl = sum_{j=2..nac} AC[j] * (ST[j] + ST[j-1]) / 2   (1-indexed; j=1 term excluded,
# AC[1] is the zero-area upper limit ordinate per the comment in topmodel.c)
ACn = AC / AC.sum()
tl = 0.0
for j in range(1, nac):
    tl += ACn[j] * (ST[j] + ST[j - 1]) / 2.0

# --- inputs.dat ---
tok = (TOP / "inputs.dat").read_text().split()
nstep = int(tok[0])
dt = float(tok[1])
data = np.array(tok[2:2 + 3 * nstep], dtype=np.float64).reshape(nstep, 3)
rain = data[:, 0]  # m per timestep

t0dt = t0 + math.log(dt)
szq = math.exp(t0dt - tl)
sbar0 = -szm * math.log(Q0 / szq)

print(f"nac={nac} tl={tl:.6f} szq={szq:.8g} sbar0={sbar0:.8g} nstep={nstep} dt={dt}")
print(f"rain: min={rain.min():.6g} max={rain.max():.6g} mean={rain.mean():.6g} (m/dt)")

with open(OUT, "wb") as f:
    f.write(struct.pack("<i", 0x544F504D))
    f.write(struct.pack("<i", nac))
    f.write(struct.pack("<4f", szm, szq, td, sbar0))
    f.write(ST.astype("<f4").tobytes())
    f.write(struct.pack("<i", nstep))
    f.write(rain.astype("<f4").tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
