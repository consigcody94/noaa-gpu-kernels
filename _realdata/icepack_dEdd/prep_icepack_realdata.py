#!/usr/bin/env python3
"""
Prepare REAL ice-column inputs for the Icepack delta-Eddington kernel in
noaa-gpu-kernels/noaa_multi_kernel.cu (harness: icepack_de_realdata.cu).

REAL DATA SOURCES (no synthetic values anywhere in the input path):

1. Ice/snow state (snow depth + co-located ice thickness), per point:
   Itkin, P. et al. (2021): Magnaprobe snow and melt pond depth measurements
   from the 2019-2020 MOSAiC expedition. PANGAEA,
   https://doi.org/10.1594/PANGAEA.937781  (CC-BY-4.0)
   File: Magnaprobe.zip -> Level-3 files magna+gem2-transect-*.csv
   Columns used: Date/Time (UTC), Lon, Lat, "Snow Depth (m)",
   "Ice Thickness 18kHz ip (m)" (preferred; fallbacks documented below).
   Ice thickness in these L3 files is GEM-2 total thickness minus co-located
   Magnaprobe snow depth (i.e., actual sea-ice thickness), per the dataset
   abstract and Itkin et al. (2023), Elementa 11(1):00048.

2. Downwelling shortwave (-> swdn) at the same floe (MOSAiC Central Obs.):
   Uttal, T. et al.: Merged Datasets for the MOSAiC Central Observatory
   (2019-2020) version 2. NSF Arctic Data Center,
   https://doi.org/10.18739/A2WD3Q35Z
   (This is the exact dataset the CICE-Consortium/Icepack wiki points to for
   its MOSAiC forcing: https://github.com/CICE-Consortium/Icepack/wiki/Icepack-Input-Data)
   Files used:
     - MOSAiC_atm_drift1_precip_MDF_20191015_20200731.nc : variable rsds
       (downward SW at surface, W/m2, 1-min cadence) -- Icepack's fsw input.
     - mosshipradS1_b1_subset_20191015_20200918.nc : spn1_total_corr
       (tilt-corrected total SW, shaded SPN1 aboard Polarstern, W/m2, 1-min)
       used only for transect times after drift1 ends (2020-07-31).

DERIVED (deterministic, no random values):
  - coszen: cosine of solar zenith angle computed from each record's UTC
    timestamp + lat/lon using the standard NOAA Global Monitoring Division
    solar position formulas (same role as Icepack's internal compute_coszen).
  - swdn[3]: fsw split into the kernel's 3 bands using Icepack's own
    spectral fractions (configuration/driver/icedrv_forcing.F90):
      frcvdr=0.28, frcvdf=0.24 (visible), frcidr=0.31, frcidf=0.17 (near-IR)
    band0 (vis)   = (0.28+0.24)*fsw = 0.52*fsw
    band1 (NIR-1) = 0.31*fsw
    band2 (NIR-2) = 0.17*fsw
  - nslyr=1, nilyr=7: Icepack default column configuration.
  - snow_grain_r = 500.0 um: Icepack's rsnw_nonmelt constant
    (columnphysics/icepack_shortwave.F90). NOTE: this field is NOT used by
    either the CPU reference or the GPU kernel math in noaa_multi_kernel.cu.

QC FILTERS (documented; records dropped, never altered):
  - unparseable timestamp / non-finite hs, hi, lat, lon
  - hi <= 0 m  (no ice / invalid retrieval)
  - hs <  0 m
  - hs > 3 m or hi > 15 m (gross outliers beyond instrument validity)
  - no valid SW sample within +/-15 min of the record time
  - fsw clamped at >= 0 (nighttime sensor noise can be slightly negative)

OUTPUT
  ice_columns_real.bin : int32 n, then n records matching struct IceColumn
      {f32 snow_depth, f32 ice_thickness, f32 snow_grain_r,
       i32 nslyr, i32 nilyr, f32 coszen, f32 swdn[3]}  (36 bytes/record)
  ice_columns_real.csv : audit trail (per-record provenance)
  prep_summary.txt     : counts per filter and per source file

Access date: 2026-06-10.
"""
import csv
import glob
import io
import math
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAGNA_DIR = os.path.join(DATA, "magnaprobe")
ATM_NC = os.path.join(DATA, "MOSAiC_atm_drift1_precip_MDF_20191015_20200731.nc")
RAD_NC = os.path.join(DATA, "mosshipradS1_b1_subset_20191015_20200918.nc")

# Icepack constants (verified against CICE-Consortium/Icepack main, 2026-06-10)
FRC_VIS = 0.28 + 0.24   # frcvdr + frcvdf
FRC_NIR_DR = 0.31       # frcidr
FRC_NIR_DF = 0.17       # frcidf
NSLYR = 1               # Icepack default snow layers
NILYR = 7               # Icepack default ice layers
RSNW_NONMELT = 500.0    # um, icepack_shortwave.F90 (unused by kernel math)

SW_TOL_MIN = 15.0       # max |dt| for shortwave match, minutes


def coszen_noaa(dt_utc, lat_deg, lon_deg):
    """Cosine of solar zenith angle, NOAA GMD solar position formulas.
    https://gml.noaa.gov/grad/solcalc/solareqns.PDF  (deterministic)."""
    doy = dt_utc.timetuple().tm_yday
    hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    time_offset = eqtime + 4.0 * lon_deg          # minutes (UTC base)
    tst = hour * 60.0 + time_offset               # true solar time, minutes
    ha = math.radians(tst / 4.0 - 180.0)          # hour angle
    lat = math.radians(lat_deg)
    return math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)


def load_sw_series():
    """Concatenate rsds (drift1 MDF) and spn1 (ship radiation, post-drift1)
    into one (epoch_seconds, fsw) series sorted by time. NaNs removed."""
    segs = []
    d = netCDF4.Dataset(ATM_NC)
    t = d.variables["time01"][:].astype(np.float64) * 60.0  # min since 1970 -> s
    v = np.ma.filled(d.variables["rsds"][:].astype(np.float64), np.nan)
    d.close()
    ok = np.isfinite(v)
    segs.append((t[ok], v[ok], np.zeros(ok.sum(), dtype=np.int8)))  # src 0 = rsds
    drift1_end = t.max()

    d = netCDF4.Dataset(RAD_NC)
    base = datetime(2019, 10, 15, tzinfo=timezone.utc).timestamp()
    t2 = d.variables["time"][:].astype(np.float64) * 60.0 + base
    v2 = np.ma.filled(d.variables["spn1_total_corr"][:].astype(np.float64), np.nan)
    d.close()
    ok2 = np.isfinite(v2) & (t2 > drift1_end)
    segs.append((t2[ok2], v2[ok2], np.ones(ok2.sum(), dtype=np.int8)))  # src 1 = spn1

    t_all = np.concatenate([s[0] for s in segs])
    v_all = np.concatenate([s[1] for s in segs])
    s_all = np.concatenate([s[2] for s in segs])
    order = np.argsort(t_all)
    return t_all[order], v_all[order], s_all[order]


def nearest_sw(t_sw, v_sw, s_sw, epoch):
    i = np.searchsorted(t_sw, epoch)
    best, bdiff = -1, None
    for j in (i - 1, i):
        if 0 <= j < len(t_sw):
            diff = abs(t_sw[j] - epoch)
            if bdiff is None or diff < bdiff:
                best, bdiff = j, diff
    if best < 0 or bdiff > SW_TOL_MIN * 60.0:
        return None
    return v_sw[best], s_sw[best], bdiff


# Ice-thickness column preference (in-phase 18 kHz is the channel used by
# Itkin et al. 2023; fallbacks for files with alternate headers)
CH_PREF = ["ice thickness 18khz ip (m)",
           "ice thickness f18325hz_hcp_i (m)",
           "ice thickness 18khz q (m)"]


def pick_columns(header):
    cols = [h.strip().lower() for h in header]
    def find(name):
        for k, c in enumerate(cols):
            if c == name:
                return k
        return None
    idx = {}
    idx["dt"] = find("date/time")
    idx["lon"] = find("lon")
    idx["lat"] = find("lat")
    idx["hs"] = find("snow depth (m)")
    idx["hi"], idx["channel"] = None, None
    for name in CH_PREF:
        k = find(name)
        if k is not None:
            idx["hi"], idx["channel"] = k, name
            break
    return idx


def main():
    t_sw, v_sw, s_sw, = load_sw_series()
    print(f"SW series: {len(t_sw)} samples "
          f"({(s_sw==0).sum()} rsds, {(s_sw==1).sum()} spn1)")

    files = sorted(glob.glob(os.path.join(MAGNA_DIR, "*", "magna+gem2-transect-*.csv")))
    print(f"{len(files)} magna+gem2 L3 transect files")

    stats = {"rows": 0, "bad_parse": 0, "bad_state": 0, "outlier": 0,
             "no_sw": 0, "kept": 0}
    per_file = []
    records = []   # tuples for bin + audit
    for path in files:
        rel = os.path.relpath(path, MAGNA_DIR)
        kept_f = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rdr = csv.reader(fh)
            header = next(rdr)
            idx = pick_columns(header)
            if None in (idx["dt"], idx["lon"], idx["lat"], idx["hs"]) or idx["hi"] is None:
                per_file.append((rel, "SKIPPED (header)", 0))
                continue
            for row in rdr:
                if not row or len(row) <= idx["hi"]:
                    continue
                stats["rows"] += 1
                try:
                    dt = datetime.fromisoformat(row[idx["dt"]].strip()).replace(tzinfo=timezone.utc)
                    lon = float(row[idx["lon"]]); lat = float(row[idx["lat"]])
                    hs = float(row[idx["hs"]]);  hi = float(row[idx["hi"]])
                except (ValueError, IndexError):
                    stats["bad_parse"] += 1
                    continue
                if not all(map(math.isfinite, (lon, lat, hs, hi))):
                    stats["bad_parse"] += 1
                    continue
                if hi <= 0.0 or hs < 0.0:
                    stats["bad_state"] += 1
                    continue
                if hs > 3.0 or hi > 15.0:
                    stats["outlier"] += 1
                    continue
                m = nearest_sw(t_sw, v_sw, s_sw, dt.timestamp())
                if m is None:
                    stats["no_sw"] += 1
                    continue
                fsw_raw, sw_src, sw_dt = m
                fsw = max(fsw_raw, 0.0)
                mu = coszen_noaa(dt, lat, lon)
                sw0, sw1, sw2 = FRC_VIS * fsw, FRC_NIR_DR * fsw, FRC_NIR_DF * fsw
                records.append((hs, hi, RSNW_NONMELT, NSLYR, NILYR, mu,
                                sw0, sw1, sw2,
                                dt.isoformat(), lat, lon, fsw_raw,
                                "rsds" if sw_src == 0 else "spn1", sw_dt,
                                idx["channel"], rel))
                stats["kept"] += 1
                kept_f += 1
        per_file.append((rel, idx["channel"], kept_f))

    print("stats:", stats)

    # binary (matches struct IceColumn layout: all members 4 bytes, no padding)
    bin_path = os.path.join(HERE, "ice_columns_real.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<i", len(records)))
        for r in records:
            f.write(struct.pack("<fffiiffff", r[0], r[1], r[2], r[3], r[4],
                                r[5], r[6], r[7], r[8]))
    print(f"wrote {bin_path} ({len(records)} records)")

    # audit CSV
    csv_path = os.path.join(HERE, "ice_columns_real.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snow_depth_m", "ice_thickness_m", "snow_grain_r_um",
                    "nslyr", "nilyr", "coszen", "swdn0_vis", "swdn1_nir_dr",
                    "swdn2_nir_df", "datetime_utc", "lat", "lon",
                    "fsw_raw_wm2", "sw_source", "sw_match_dt_s",
                    "thickness_channel", "source_file"])
        for r in records:
            w.writerow(r)
    print(f"wrote {csv_path}")

    with open(os.path.join(HERE, "prep_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Icepack dEdd real-data prep summary (access date 2026-06-10)\n")
        f.write(f"stats: {stats}\n\nper-file (file, thickness channel, kept):\n")
        for rel, ch, k in per_file:
            f.write(f"  {rel} | {ch} | {k}\n")
    print("wrote prep_summary.txt")


if __name__ == "__main__":
    sys.exit(main())
