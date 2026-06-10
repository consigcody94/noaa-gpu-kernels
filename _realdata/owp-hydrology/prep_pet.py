"""
prep_pet.py — Convert real AORC meteorological forcing shipped in NOAA-OWP/cfe
into PETForcing records for owp_extended_kernels_realdata.cu (Penman-Monteith kernel).

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/cfe master):
  - forcings/cat87_01Dec2015.csv            : 720 hourly AORC records, NextGen catchment
                                              cat-87 (Sugar Creek, NC area), Dec 2015
  - forcings/Laramie_14Jun09_to_15Apr12.csv : 24,862 hourly AORC records, Laramie River
                                              basin, WY, Jun 2009 - Apr 2012
AORC = NOAA Analysis of Record for Calibration, real gridded reanalysis forcing.

Mapping to PETForcing struct (order: temp_C, pressure_Pa, spec_humidity, wind_speed,
shortwave_W, longwave_W):
  temp_C        = TMP_2maboveground - 273.15      (K -> degC)
  pressure_Pa   = PRES_surface                    (Pa)
  spec_humidity = SPFH_2maboveground              (kg/kg)
  wind_speed    = sqrt(UGRD^2 + VGRD^2)           (m/s, 10 m)
  shortwave_W   = DSWRF_surface                   (W/m2)
  longwave_W    = DLWRF_surface                   (W/m2)

Output: data/pet_realdata.bin
  int32 magic=0x50455446, int32 nrec, then nrec * 6 float32 in struct order.
"""
import csv
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
FORC = HERE / "upstream" / "cfe" / "forcings"
OUT = HERE / "data" / "pet_realdata.bin"

records = []
for fname in ["cat87_01Dec2015.csv", "Laramie_14Jun09_to_15Apr12.csv"]:
    with open(FORC / fname, newline="") as f:
        rd = csv.DictReader(f)
        n0 = len(records)
        for row in rd:
            t = float(row["TMP_2maboveground"]) - 273.15
            p = float(row["PRES_surface"])
            q = float(row["SPFH_2maboveground"])
            u = float(row["UGRD_10maboveground"])
            v = float(row["VGRD_10maboveground"])
            w = (u * u + v * v) ** 0.5
            sw = float(row["DSWRF_surface"])
            lw = float(row["DLWRF_surface"])
            records.append((t, p, q, w, sw, lw))
        print(f"{fname}: {len(records)-n0} records")

arr = np.array(records, dtype="<f4")
print(f"total {len(records)} records; T range [{arr[:,0].min():.1f},{arr[:,0].max():.1f}] C, "
      f"P range [{arr[:,1].min():.0f},{arr[:,1].max():.0f}] Pa")
with open(OUT, "wb") as f:
    f.write(struct.pack("<i", 0x50455446))
    f.write(struct.pack("<i", len(records)))
    f.write(arr.tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
