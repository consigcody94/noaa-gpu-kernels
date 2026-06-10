# OWP Hydrology Kernels — Real-Data Validation (RTX 5070, sm_120)

**Date:** 2026-06-10
**Target kernels:** `water-prediction/owp_batched_kernels.cu` (CFE Nash cascade, NOAH-MP ROSR12 tridiag),
`water-prediction/owp_extended_kernels.cu` (TOPMODEL, Penman-Monteith PET),
`water-prediction/owp_snow17_lgar.cu` (Snow17, LGAR).
**Rule compliance:** kernel math and CPU reference implementations copied **verbatim** into new
`*_realdata.cu` harnesses; only the synthetic generators (`gen_*`) were replaced with real-data
loaders. Originals untouched. All inputs trace to real, public NOAA-OWP datasets (shallow clones,
accessed 2026-06-10). Total downloaded data well under 500 MB.

**Hardware/build:** RTX 5070 12 GB, driver 610.47, CUDA 13.3, `nvcc -O3 -arch=sm_120`, MSVC 2022
BuildTools host (see `build.bat`). GPU shared with sibling agents — **accuracy numbers are
authoritative; timing is indicative only.**

## Summary table

| Kernel | Real dataset (n real records) | Max abs err | Max rel err | Binary's own verdict | Synthetic-run verdict (BLACKWELL_RESULTS.md) |
|---|---|---|---|---|---|
| CFE Nash cascade | cat-87 AORC Dec 2015, 720 h × 2 cascades | 0.0 | 0.0 | **PASS (bit-identical)** at 1,440 / 100k / 1M / 2.7M | PASS, bit-identical (same) |
| NOAH-MP ROSR12 tridiag | Bondville, IL 1998, 17,521 systems | 7.45e-09 | 3.35e-05 (residual 2.98e-08) | **PASS (FP32 rounding)**, 0/27M points >0.01% | **FAIL @≥100k** on rel err (7.38e-01, near-zero-solution artifact) — *real data removes the failure* |
| TOPMODEL | Pyungkwang River, 950 hourly steps, 30 ln(a/tanB) bins | — | qb 5.21e-07 · qo 0.0 · sbar 1.09e-07 | **PASS** at 950 / 100k / 1M / 2.7M | PASS, 5.87e-07 (same magnitude) |
| Penman-Monteith PET | AORC cat-87 + Laramie, 25,582 hourly records | — | 4.31e-04 | **PASS (fast exp)** | PASS (fast exp), 1.44e-04 (real is ~3× larger; see notes) |
| Snow17 | HHWM8 Flathead R. 1970-2015, 33,602 daily records | 7.63e-06 | 5.52e-06 | **PASS** (strict <1e-4 tier) | PASS, 2.10e-05 (real is ~4× smaller) |
| LGAR Green-Ampt | Phillipsburg WY2017 + Bushland WY2021, 17,520 hourly records | 0.0 | 0.0 | **PASS (bit-identical)** | PASS, bit-identical (same) |

All six kernels **pass their binaries' own acceptance criteria on real data** at every batch size
tested, including full-CONUS NWM scale (2.7 M catchments/columns).

---

## 1. CFE Nash cascade (`owp_batched_kernels_realdata.cu`)

**Provenance.** github.com/NOAA-OWP/cfe (shallow clone, master, 2026-06-10):
- `configs/cfe_config_cat_87.txt` — real calibrated CFE config for NextGen catchment cat-87:
  subsurface cascade `K_nash_subsurface=0.03`, `nash_storage_subsurface=0.0,0.0` (N=2); surface
  cascade `K_nash_surface=0.83089`, `nash_storage_surface=0.0,0.0` (N=2).
- `forcings/cat87_01Dec2015.csv` — 720 hourly AORC (Analysis of Record for Calibration)
  records, Dec 2015. `APCP_surface` (mm over the 1-h step) → lateral flux in m = APCP/1000.
  Real total: 269.2 mm over the month; max 37.7 mm/h.

**Mapping.** `flux_lat` ← hourly precipitation depth (m); `K_nash`/`N_nash`/initial `storage` ← the
two real cascade configs. Because the cascades start empty (the config's real cold-start state), a
single step would be degenerate; the harness builds a library of 1,440 (state, flux) pairs by
running the **untouched CPU reference** sequentially through the 720-h real series for each of the
2 parameter sets, recording the pre-step storage each hour. Batches tile this library (`c % 1440`)
— replication documented, no values invented. Note: in CFE proper the cascade inflow is the soil
lateral/direct-runoff flux; using the real precipitation depth as inflow exercises identical math
with realistic real-world magnitudes without reimplementing CFE's soil reservoir (which would be
new model code outside the kernel under test).

**Result.** CPU and GPU **bit-identical** (max abs = max rel = 0) at 1,440 (pure real), 100k, 1M,
2.7M. Same as synthetic run.

## 2. NOAH-MP ROSR12 tridiagonal solver

**Provenance.** github.com/NOAA-OWP/noah-owp-modular (sparse shallow clone, master, 2026-06-10):
- `data/bondville.dat` — 17,521 30-min observed meteorological records, Bondville, IL
  (40.01 N, −88.37 E), calendar year 1998; precipitation column kg m⁻² s⁻¹.
- `run/namelist.input` — official point-test config: nsoil=4, dz=[0.1,0.3,0.6,1.0] m,
  initial sh2o=0.30, isltyp=1, dt=1800 s, drainage option 8.
- `parameters/SOILPARM.TBL` STAS class 1 (SAND): BEXP=2.79, MAXSMC=0.339, SATDK=4.66e-5 m/s,
  SATDW=2.65e-5 m²/s, DRYSMC=0.010; `parameters/GENPARM.TBL` SLOPE_DATA(1)=0.1.

**Mapping.** `prep_noahmp_tridiag.py` assembles one Richards-equation tridiagonal system per
forcing record using the model's **own** equations transcribed from `src/SoilWaterMovement.f90`
(SRT, with WDFCND2 from `src/SoilWaterRetentionCoeff.f90`, OPT_INF=2, no soil ice, OPT_DRN=8 →
QDRAIN = SLOPE·WCND(nsoil)) and the SSTEP scaling `A=AI·dt, B=1+BI·dt, C=CI·dt, D=RHSTT·dt` —
exactly the system NOAH-MP hands to ROSR12. Surface flux PDDUM = observed precipitation rate;
ETRANI and QSEVA set to 0 (bondville.dat contains no observed ET partitioning — documented
simplification affecting only RHS magnitude). The soil-moisture state is advanced through the full
year with the model's own implicit update (Thomas solve, clipped to [DRYSMC, MAXSMC]), so all
17,521 systems are distinct and real-forcing-driven. Batches >17,521 tile the year (`col % 17521`);
nsoil=4 for every column (real config; the synthetic run used random 4–10).

**Result.** Max abs 7.45e-09, max rel 3.35e-05, max residual |Ax−D| 2.98e-08, 0 NaN, 0/27,000,000
points above the binary's 0.01 % gate at every size up to 2.7 M → **PASS (FP32 rounding)**.

**Delta vs synthetic — the headline finding:** on synthetic data this binary reported **FAIL at
≥100k columns** (max rel 7.38e-01) because random RHS values produce near-zero solution components
where relative error explodes while residuals stay at FP32 level. On real Bondville Richards
systems (diagonally dominant, B = 1+BI·dt ≥ 1, solution components are physically scaled moisture
increments) the pathology disappears entirely: worst relative error 3.35e-05. The synthetic FAIL is
confirmed to be a test-data artifact, not a kernel defect.

## 3. TOPMODEL (`owp_extended_kernels_realdata.cu`)

**Provenance.** github.com/NOAA-OWP/topmodel (shallow clone, master, 2026-06-10), `data/` —
the repo's canonical real test catchment, **"Extracted study basin: Taegu Pyungkwang River"**
(Korea), in the original Beven TMOD9502 distribution format. (Note: the historical Slapton Wood
files were replaced upstream by this basin; it is the real dataset the NOAA-OWP test suite runs.)
- `data/subcat.dat` — 30-ordinate ln(a/tanB) areal distribution (AC, ST pairs).
- `data/params.dat` — calibrated szm=0.032, t0=5.0, td=50, Q0=3.28e-05, sr0=0.002, …
- `data/inputs.dat` — nstep=950, dt=1.0 h; real rain/pe/Qobs columns (m per step).

**Mapping.** `lnaotb` ← the 30 real ST ordinates; `szm`, `td` ← params.dat; derived exactly per
`src/topmodel.c`: tl = Σ ACⱼ·(STⱼ+STⱼ₋₁)/2 (area-normalised) = 5.45498, `szq` = exp(t0+ln dt − tl)
= 0.63446, initial `sbar` = −szm·ln(Q0/szq) = 0.31584. `precip[c]` ← real hourly rain. The harness
builds the 950-step sbar state trajectory by running the **untouched CPU reference** sequentially
over the real rain series; batch entry c gets the real (precip, sbar) pair `t = c % 950`.
(The kernel's equal-area-bin treatment of the histogram is the kernel's own documented
simplification of upstream TOPMODEL; the inputs themselves are the real distribution values.)

**Result.** Max rel: baseflow 5.21e-07, overland flow 0.0, sbar 1.09e-07; 0 NaN → **PASS** (strict
tier) at 950 / 100k / 1M / 2.7M. Synthetic run: 5.87e-07 — same FP32/`__expf` rounding magnitude.
Real rain produces saturated bins (qo > 0) at many steps, so the overland-flow branch is exercised
(bit-exact, as it contains no transcendentals).

## 4. Penman-Monteith PET

**Provenance.** Real AORC hourly forcing from github.com/NOAA-OWP/cfe (2026-06-10):
- `forcings/cat87_01Dec2015.csv` — 720 records (Dec 2015).
- `forcings/Laramie_14Jun09_to_15Apr12.csv` — 24,862 records, Laramie River basin, WY,
  Jun 2009 – Apr 2012 (includes −37.5 °C winter extremes through +31.2 °C).
(The NOAA-OWP/evapotranspiration repo was also cloned; it documents the same AORC forcing format.)

**Mapping** (struct order): temp_C = TMP_2m − 273.15; pressure_Pa = PRES_surface;
spec_humidity = SPFH_2m; wind_speed = √(UGRD²+VGRD²); shortwave_W = DSWRF; longwave_W = DLWRF.
25,582 real records; larger batches tile (`c % 25582`).

**Result.** Max rel 4.31e-04, 0 NaN → **PASS (fast exp)** at all sizes. Synthetic run: 1.44e-04.
The ~3× larger real-data error comes from the Laramie cold extremes (T to −37.5 °C) driving the
Tetens `__expf` argument further negative than the synthetic generator's −10…+30 °C range — still
two orders of magnitude inside the binary's 1e-2 fast-exp gate.

## 5. Snow17 (`owp_snow17_lgar_realdata.cu`)

**Provenance.** github.com/NOAA-OWP/snow17 (shallow clone, master, 2026-06-10),
`test_cases/ex1.tgz` (extracted) — South Fork Flathead River above Hungry Horse reservoir, MT,
two elevation bands; case study developed at NCAR (Mendoza et al. 2017, HESS 21):
- `ex1/input/params/snow17_params.HHWM8.txt` — real calibrated parameters per band
  (HHWM8IL: scf 2.15177, mfmax 0.930472, …; HHWM8IU: scf 1.86124, …); si=1515 mm.
- `ex1/input/forcing/forcing.snow17bmi.HHWM8{IL,IU}.csv` — 16,801 daily records each
  (1970-01-01…2015), precip mm/s and T °C.

**Mapping.** `Snow17Params` ← the 10 calibrated values per band; `Snow17Forcing`: ta ← tavg_degc,
px ← prec·86400 (mm/day); dt_hours = 24 (daily case study; original harness used 6). State
(we, liqw, neghs, aesc, tprev) trajectories generated by running the **untouched CPU reference**
sequentially over the full 46-year series from a cold start, yielding 33,602 real
(state, params, forcing) tuples; larger batches tile.

**Result.** Max abs 7.63e-06 mm, max rel 5.52e-06, 0 NaN, 0 points >0.01 % → **PASS** (strict
<1e-4 tier, better than the synthetic run's 2.10e-05 "PASS"). The real trajectories build deep
multi-year snowpacks (we ≫ 0) and exercise accumulation, melt, rain-on-snow, heat-deficit and
areal-extent branches.

## 6. LGAR Green-Ampt infiltration

**Provenance.** github.com/NOAA-OWP/LGAR-C (shallow clone, master, 2026-06-10):
- `configs/config_lasam_Phillipsburg.txt` (Phillipsburg, KS; layers 44/131/25 cm, soil types
  13,14,15, initial ψ 2000 cm) and `configs/config_lasam_Bushland.txt` (Bushland, TX; layers
  18/76/135 cm, types 16,17,18).
- `data/vG_default_params.dat` — site-calibrated van Genuchten parameters (P-1: θr 0.0648,
  θe 0.4513, α 0.0031297 cm⁻¹, n 1.6858, Ks 0.45 cm/h; B-1: θr 0.0649, θe 0.4481, α 0.009567,
  n 1.3579, Ks 0.07 cm/h).
- `forcing/forcing_data_resampled_uniform_{Phillipsburg,Bushland}.csv` — 8,760 real hourly
  precipitation records each (water years 2017 and 2021).

**Mapping** (top layer of each site, the layer the kernel's single wetting front occupies):
Ks ← vG Ks (cm/h→m/s); porosity ← θe; initial_moisture ← vG retention θ(ψ=2000 cm) (the same
retention function LGAR-C uses); wetting_front_suction ← Green-Ampt effective capillary drive from
the vG parameters via the Morel-Seytoux et al. (1996, WRR 32) closed form
G = (1/α)(0.046m+2.07m²+19.5m³)/(1+4.7m+16m²), m=1−1/n → 96.2 cm (P-1) / 16.0 cm (B-1)
(documented derivation — the kernel is a Green-Ampt variant, while LGAR-C itself parameterises in
vG space); soil_depth ← Σ layer thicknesses (2.00 m / 2.29 m); precip_rate ← P mm/h → m/s;
dt = 3600 s (real forcing resolution; original harness used 300 s). Cumulative-infiltration state
trajectories from the **untouched CPU reference** over each real series (cold start) → 17,520 real
tuples; larger batches tile.

**Result.** CPU and GPU **bit-identical** (max abs = max rel = 0, 0 NaN) at 17,520 / 100k / 1M /
2.7M. Same as synthetic run.

---

## Behaviour differences vs the synthetic-data run

1. **NOAH-MP tridiag: synthetic FAIL → real PASS.** The binary's own verdict flips from FAIL
   (max rel 0.738 at ≥100k columns, caused by random RHS producing near-zero solution components)
   to PASS (max rel 3.35e-05) on real Richards systems. Strong evidence the synthetic failure is a
   test-data artifact, not a kernel bug; worth noting wherever the synthetic FAIL was reported.
2. **PET max rel grew 1.44e-04 → 4.31e-04** — real Laramie winter extremes (−37.5 °C) stress
   `__expf` harder than the synthetic range; still passes its gate by >20×.
3. **Snow17 improved 2.10e-05 → 5.52e-06** — real calibrated parameters and physically consistent
   states are gentler than random parameter/state combinations.
4. **Nash, TOPMODEL, LGAR: unchanged** (bit-identical / same error magnitude).
5. Real data fixes batch-composition statistics: nsoil is uniformly 4 (real config) instead of
   random 4–10; Nash N is uniformly 2 (real configs) instead of random 2–10 — the variable-length
   loop paths for larger N/ns are therefore *not* exercised by the real configs (they were by the
   synthetic run).

## Documented simplifications / replication

- Batch sizes above each real-record library tile it with `c % n_real` (every input value remains
  a real record; replication factors up to ~1,875× for Nash at 2.7 M).
- State trajectories (Nash storage, TOPMODEL sbar, Snow17 pack state, LGAR cumulative
  infiltration) are produced by the **verbatim CPU reference implementations** driven by the real
  forcing series — no synthetic state values anywhere.
- NOAH-MP: ETRANI/QSEVA = 0 (no observed ET in bondville.dat); matrix assembly equations
  transcribed 1:1 from `SoilWaterMovement.f90`/`SoilWaterRetentionCoeff.f90` into
  `prep_noahmp_tridiag.py`.
- Nash cascade inflow uses real AORC precipitation depth as the lateral-flux proxy (CFE's internal
  soil-reservoir flux would require running full CFE, i.e. new model code outside the kernel).
- LGAR wetting-front suction derived from real vG parameters via the standard Morel-Seytoux (1996)
  closed form.
- The diffusive-wave kernel in `owp_extended_kernels.cu` was **not** ported: its benchmark-only
  simplified inputs (constant alpha, ad-hoc diffusivity/celerity fields) have no corresponding real
  dataset in the upstream repos, and fabricating one was out of scope (originals untouched).

## Timing (indicative only — GPU shared with sibling agents)

At 2.7 M batch: Nash 13.8 ms (≈1.2×CPU; memcpy-reset dominated), tridiag 1.14 ms (≈80×),
TOPMODEL 1.45 ms (≈91×), PET 0.12 ms (≈379×), Snow17 8.5 ms (≈6×; H2D state reset in loop),
LGAR 1.38 ms (≈29×). Consistent with the prior sm_120 synthetic run.

## Artifacts

- Harnesses: `owp_batched_kernels_realdata.cu`, `owp_extended_kernels_realdata.cu`,
  `owp_snow17_lgar_realdata.cu` (+ built `.exe`), `build.bat`
- Prep scripts: `prep_cfe_nash.py`, `prep_noahmp_tridiag.py`, `prep_topmodel.py`, `prep_pet.py`,
  `prep_snow17.py`, `prep_lgar.py`
- Data: `data/*.bin` (3.7 MB total); upstream clones under `upstream/`
- Run logs: `logs/run_batched_realdata.log`, `logs/run_extended_realdata.log`,
  `logs/run_snow17_lgar_realdata.log`
