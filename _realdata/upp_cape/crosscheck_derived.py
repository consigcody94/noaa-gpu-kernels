"""
crosscheck_derived.py — Compare the kernel's CPU/GPU CAPE on real IGRA2 soundings
(Norman, OK USM00072357) against NOAA NCEI's own IGRA2 *derived-parameter* CAPE/CIN
for the SAME soundings.

Source: https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por/USM00072357-drvd.txt.zip
(accessed 2026-06-10). Derived header record contains CAPE (cols 146-151, J/kg)
and CIN (cols 152-157, J/kg) computed by NCEI from parcel theory.

NOTE (documented, not tuned): the repo kernel is a SIMPLIFIED CALCAPE — it lifts
the most-unstable parcel moist-adiabatically from its origin level (no dry ascent
to the LCL, parcel treated as saturated from the start), so it systematically
produces LARGER CAPE than NCEI's derived values. This script quantifies the
relationship (correlation/rank order) and prints selected dates for plausibility.
"""
import csv
import zipfile
from pathlib import Path

import numpy as np

WD = Path(r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\upp_cape")
DRVD_ZIP = WD / "data" / "USM00072357-drvd.txt.zip"

# 1. kernel results for OUN soundings, joined via manifest
man = list(csv.DictReader(open(WD / "manifest.csv")))
res = list(csv.DictReader(open(WD / "cape_results.csv")))
assert len(man) == len(res)
ours = {}  # "YYYY-MM-DD HHZ" -> (cpu_cape, cpu_cin, gpu_cape, gpu_cin)
for m, r in zip(man, res):
    assert m["index"] == r["index"]
    if m["station"] == "USM00072357":
        ours[m["datetime"]] = (float(r["cpu_cape"]), float(r["cpu_cin"]),
                               float(r["gpu_cape"]), float(r["gpu_cin"]))
print(f"kernel results for OUN: {len(ours)} soundings")

# 2. NCEI derived CAPE/CIN header records (only need headers; stream the zip)
ncei = {}
with zipfile.ZipFile(DRVD_ZIP) as zf:
    name = zf.namelist()[0]
    with zf.open(name) as f:
        for raw in f:
            if not raw.startswith(b"#"):
                continue
            line = raw.decode("ascii", "replace")
            year = int(line[13:17])
            if year < 2025:
                continue
            month = int(line[18:20]); day = int(line[21:23]); hour = int(line[24:26])
            if hour not in (0, 12):
                continue
            cape_s = line[145:151].strip(); cin_s = line[151:157].strip()
            if not cape_s or not cin_s:
                continue
            cape = int(cape_s); cin = int(cin_s)
            if cape in (-99999, -9999) or cin in (-99999, -9999):
                continue
            ncei[f"{year:04d}-{month:02d}-{day:02d} {hour:02d}Z"] = (float(cape), float(cin))
print(f"NCEI derived records 2025+: {len(ncei)}")

# 3. join and compare
keys = sorted(set(ours) & set(ncei))
print(f"matched soundings: {len(keys)}")
k_cape = np.array([ours[k][0] for k in keys])
g_cape = np.array([ours[k][2] for k in keys])
n_cape = np.array([ncei[k][0] for k in keys])
k_cin = np.array([ours[k][1] for k in keys])
n_cin = np.array([ncei[k][1] for k in keys])

both = (k_cape > 50) & (n_cape > 50)
corr = np.corrcoef(k_cape[both], n_cape[both])[0, 1] if both.sum() > 2 else np.nan
# rank correlation (no scipy needed)
def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
rcorr = np.corrcoef(rank(k_cape[both]), rank(n_cape[both]))[0, 1] if both.sum() > 2 else np.nan

print(f"\nsoundings where both kernel & NCEI CAPE > 50 J/kg: {both.sum()}")
print(f"Pearson r (kernel vs NCEI CAPE):  {corr:.3f}")
print(f"Spearman rank r:                  {rcorr:.3f}")
print(f"median kernel/NCEI CAPE ratio:    {np.median(k_cape[both]/n_cape[both]):.2f}")
print(f"NCEI CAPE=0 -> kernel CAPE median {np.median(k_cape[n_cape==0]):.0f} J/kg "
      f"(n={int((n_cape==0).sum())})")

# contingency: does the kernel discriminate convective vs stable days?
thr = 500.0
tp = int(((k_cape >= thr) & (n_cape >= thr)).sum())
fn = int(((k_cape < thr) & (n_cape >= thr)).sum())
fp = int(((k_cape >= thr) & (n_cape < thr)).sum())
tn = int(((k_cape < thr) & (n_cape < thr)).sum())
print(f"\ncontingency @ {thr:.0f} J/kg: TP={tp} FP={fp} FN={fn} TN={tn}")

# 4. selected dates: top-5 kernel CAPE plus two winter cases
top = np.argsort(-k_cape)[:5]
print("\n top kernel-CAPE OUN soundings        kernelCPU  kernelGPU  NCEI_CAPE  NCEI_CIN  kernelCIN")
for i in top:
    k = keys[i]
    print(f"  {k}                       {k_cape[i]:9.0f} {g_cape[i]:9.0f} {n_cape[i]:9.0f} {n_cin[i]:9.0f} {k_cin[i]:9.1f}")
wint = [k for k in keys if k.startswith("2025-01") or k.startswith("2025-12")][:4]
print("\n winter OUN soundings")
for k in wint:
    i = keys.index(k)
    print(f"  {k}                       {k_cape[i]:9.0f} {g_cape[i]:9.0f} {n_cape[i]:9.0f} {n_cin[i]:9.0f} {k_cin[i]:9.1f}")

with open(WD / "logs" / "crosscheck_oun.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "kernel_cpu_cape", "kernel_gpu_cape", "kernel_cpu_cin",
                "ncei_cape", "ncei_cin"])
    for i, k in enumerate(keys):
        w.writerow([k, k_cape[i], g_cape[i], k_cin[i], n_cape[i], n_cin[i]])
print(f"\nwrote {WD / 'logs' / 'crosscheck_oun.csv'}")
