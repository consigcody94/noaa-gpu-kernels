#!/usr/bin/env python3
"""
Prep script: convert the official COARE 3.6 test dataset into flat binaries
for the coare_flux_realdata.cu harness.

Data provenance (REAL observations, no synthetic values):
  Source repo : https://github.com/NOAA-PSL/COARE-algorithm (official NOAA-PSL release)
  Input file  : Python/COARE3.6/test_36_data.txt
                https://raw.githubusercontent.com/NOAA-PSL/COARE-algorithm/master/Python/COARE3.6/test_36_data.txt
  Gold output : Python/COARE3.6/test_36_output_withnowavesinput_withwarmlayer.txt
                (official expected outputs of coare36vnWarm_et.py on the same data,
                 no-wave-input variant, warm-layer + cool-skin physics enabled)
  Accessed    : 2026-06-10
  Content     : 2165 hourly-ish ship observations, tropical Atlantic
                (lat 13.14..15.86 N, lon 59.06..51.38 W, yearday 9.83..43.22),
                consistent with the RV Ronald H. Brown ATOMIC cruise (Jan-Feb 2020).
                This is the validation dataset shipped with the official COARE 3.6
                distribution (Fairall et al.); used unmodified, full record count.

Input columns (whitespace-separated, 1 header line):
  1 jd  2 u  3 zu  4 ta  5 zt  6 rh  7 zq  8 P  9 tsnk  10 sw_dn  11 lw_dn
  12 lat  13 lon  14 zi  15 rain  16 Ss  17 cp  18 sigH  19 tsg  20 ztsg

Mapping to the kernel's COAREInput struct (10 x float32 per record):
  u     <- col 2  (wind speed m/s, height zu)
  ta    <- col 4  (air temp degC, height zt)
  rh    <- col 6  (relative humidity %, height zq)
  P     <- col 8  (pressure mb)
  ts    <- col 9  (tsnk: "sea snake" near-surface SST at ~0.05 m depth degC)
  sw_dn <- col 10 (downwelling shortwave W/m2; carried in struct, unused by kernel math)
  lw_dn <- col 11 (downwelling longwave  W/m2; carried in struct, unused by kernel math)
  zu    <- col 3  (18 m on this cruise)
  zt    <- col 5  (17 m)
  zq    <- col 7  (17 m)
Ignored real columns (kernel does not take them): jd, lat, lon, zi (=600 m, which
matches the kernel's hard-coded gust scaling Bf*600), rain, Ss, cp, sigH, tsg, ztsg.
6 NaNs exist only in sigH (unused). No NaNs in any used column.

Gold output columns used: tau (2), hsb (3), hlb (4), plus usr (1) for reference.

Output files (little-endian):
  coare_inputs.bin   : int32 n, then n * 10 float32 (struct order above)
  coare_expected.bin : int32 n, then n * 4 float32 (usr, tau, hsb, hlb)
"""
import struct
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "test_36_data.txt")
GOLD = os.path.join(HERE, "test_36_output_withnowavesinput_withwarmlayer.txt")

def read_table(path):
    with open(path) as f:
        header = f.readline().split()
        rows = []
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rows.append([float(p) for p in parts])
    return header, rows

hdr_in, rows_in = read_table(DATA)
hdr_out, rows_out = read_table(GOLD)
assert len(rows_in) == len(rows_out), (len(rows_in), len(rows_out))
n = len(rows_in)

col = {name: i for i, name in enumerate(hdr_in)}
gcol = {name: i for i, name in enumerate(hdr_out)}
print(f"records: {n}")
print(f"input cols: {hdr_in}")
print(f"using ts = tsnk (sea snake, ~0.05 m depth)")

used = ["u", "ta", "rh", "P", "tsnk", "sw_dn", "lw_dn", "zu", "zt", "zq"]
for name in used:
    vals = [r[col[name]] for r in rows_in]
    bad = sum(1 for v in vals if v != v)
    assert bad == 0, f"NaN in used column {name}"
    print(f"  {name:6s} min={min(vals):12.4f} max={max(vals):12.4f}")

with open(os.path.join(HERE, "coare_inputs.bin"), "wb") as f:
    f.write(struct.pack("<i", n))
    for r in rows_in:
        rec = (r[col["u"]], r[col["ta"]], r[col["rh"]], r[col["P"]],
               r[col["tsnk"]], r[col["sw_dn"]], r[col["lw_dn"]],
               r[col["zu"]], r[col["zt"]], r[col["zq"]])
        f.write(struct.pack("<10f", *rec))

with open(os.path.join(HERE, "coare_expected.bin"), "wb") as f:
    f.write(struct.pack("<i", n))
    for r in rows_out:
        rec = (r[gcol["usr"]], r[gcol["tau"]], r[gcol["hsb"]], r[gcol["hlb"]])
        f.write(struct.pack("<4f", *rec))

print("wrote coare_inputs.bin and coare_expected.bin")
