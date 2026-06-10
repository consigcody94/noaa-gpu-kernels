"""
prep_lgar.py — Convert real LGAR (LASAM) site configurations and forcing from
NOAA-OWP/LGAR-C into inputs for the Green-Ampt-variant LGAR kernel in
owp_snow17_lgar_realdata.cu.

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/LGAR-C master):
  - configs/config_lasam_Phillipsburg.txt : Phillipsburg, KS USDA SCAN site;
        layers 44/131/25 cm (soil types 13,14,15 = P-1,P-2,P-3), initial_psi=2000 cm
  - configs/config_lasam_Bushland.txt     : Bushland, TX site;
        layers 18/76/135 cm (soil types 16,17,18 = B-1,B-2,B-3), initial_psi=2000 cm
  - data/vG_default_params.dat            : site-calibrated van Genuchten parameters
        (theta_r, theta_e, alpha [1/cm], n, Ks [cm/h]) for those soil types
  - forcing/forcing_data_resampled_uniform_Phillipsburg.csv : 8,760 hourly P records (mm/h),
        water year 2017 (2016-10-01 ..)
  - forcing/forcing_data_resampled_uniform_Bushland.csv     : 8,760 hourly P records (mm/h),
        water year 2021 (2020-10-01 ..)

Mapping to LGARParams (Ks, porosity, wetting_front_suction, initial_moisture, soil_depth),
using the top soil layer of each site (the layer the kernel's single wetting front enters):
  Ks                    = vG Ks (cm/h) * 0.01/3600          [m/s]
  porosity              = theta_e
  initial_moisture      = van Genuchten retention at the config's initial_psi (2000 cm):
                          theta = theta_r + (theta_e-theta_r)*(1+(alpha*psi)^n)^(-(1-1/n))
                          (the same retention function LGAR-C uses)
  wetting_front_suction = Green-Ampt effective capillary drive G from vG parameters via
                          the Morel-Seytoux et al. (1996, WRR 32) closed form used in
                          GAR-type models:
                            m = 1-1/n
                            G = (1/alpha)*(0.046m + 2.07m^2 + 19.5m^3)/(1 + 4.7m + 16m^2) [cm] -> m
  soil_depth            = sum of config layer thicknesses [m]
  precip_rate           = P (mm/h) * 1e-3/3600              [m/s]

Cumulative-infiltration state trajectories are generated inside the harness by running
the UNTOUCHED CPU reference sequentially over the real hourly series (cold start F=0).

Output: data/lgar_realdata.bin
  int32 magic=0x4C474152, int32 nsite
  per site: 5 float32 params, int32 nrec, float32 precip_rate[nrec]
"""
import csv
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
LG = HERE / "upstream" / "LGAR-C"
OUT = HERE / "data" / "lgar_realdata.bin"

# --- parse vG table (row order = soil type index, 1-based) ---
vg = []
for line in (LG / "data" / "vG_default_params.dat").read_text().splitlines()[1:]:
    parts = line.replace('"', " ").split()
    if len(parts) >= 6:
        name = parts[0]
        vals = [float(x) for x in parts[-5:]]
        vg.append((name, *vals))   # (name, theta_r, theta_e, alpha, n, Ks_cmh)
for i, row in enumerate(vg, 1):
    if i in (13, 16):
        print(f"type {i}: {row}")

sites = []
for cfg_name, forc_name in [
    ("config_lasam_Phillipsburg.txt", "forcing_data_resampled_uniform_Phillipsburg.csv"),
    ("config_lasam_Bushland.txt", "forcing_data_resampled_uniform_Bushland.csv"),
]:
    cfg = {}
    for line in (LG / "configs" / cfg_name).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.split("[")[0].strip()
    thick_cm = [float(x) for x in cfg["layer_thickness"].split(",")]
    types = [int(x) for x in cfg["layer_soil_type"].split(",")]
    psi0_cm = float(cfg["initial_psi"])
    top = vg[types[0] - 1]
    name, theta_r, theta_e, alpha, n, ks_cmh = top
    m = 1.0 - 1.0 / n
    theta0 = theta_r + (theta_e - theta_r) * (1.0 + (alpha * psi0_cm) ** n) ** (-m)
    G_cm = (1.0 / alpha) * (0.046 * m + 2.07 * m**2 + 19.5 * m**3) / (1.0 + 4.7 * m + 16.0 * m**2)
    params = (
        ks_cmh * 0.01 / 3600.0,        # Ks m/s
        theta_e,                        # porosity
        G_cm * 0.01,                    # wetting front suction, m
        theta0,                         # initial moisture
        sum(thick_cm) * 0.01,           # soil depth, m
    )
    precip = []
    with open(LG / "forcing" / forc_name, newline="") as f:
        for row in csv.DictReader(f):
            precip.append(float(row["P(mm/h)"]) * 1e-3 / 3600.0)  # m/s
    precip = np.array(precip, dtype="<f4")
    print(f"{cfg_name}: top soil '{name}' Ks={params[0]:.3e} m/s, porosity={params[1]}, "
          f"psi_f={params[2]*100:.1f} cm, theta0={params[3]:.4f}, depth={params[4]:.2f} m, "
          f"nrec={len(precip)}, wet hours={(precip>0).sum()}")
    sites.append((params, precip))

with open(OUT, "wb") as f:
    f.write(struct.pack("<ii", 0x4C474152, len(sites)))
    for params, precip in sites:
        f.write(struct.pack("<5f", *params))
        f.write(struct.pack("<i", len(precip)))
        f.write(precip.tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
