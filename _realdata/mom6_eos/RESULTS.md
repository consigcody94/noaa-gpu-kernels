# MOM6 EOS + triDiagTS kernels — real-data validation (Argo)

Target: `ocean/mom6/mom6_vortex.cu` (EOS section, plus optional triDiagTS probe)
Date: 2026-06-10. GPU: RTX 5070 12GB (sm_120), CUDA 13.3, MSVC 2022 host.
Per project rules, kernel math and CPU references were copied **verbatim**; only the
synthetic generators were replaced with file loaders (`*_realdata.cu`).

## 1. Data provenance (real, public, authoritative)

Argo GDAC (Ifremer mirror), daily geographic profile files for **2026-05-15**,
accessed **2026-06-10**:

| File | URL | Size |
|---|---|---|
| atlantic_ocean_20260515_prof.nc | https://data-argo.ifremer.fr/geo/atlantic_ocean/2026/05/20260515_prof.nc | 20.3 MB |
| pacific_ocean_20260515_prof.nc | https://data-argo.ifremer.fr/geo/pacific_ocean/2026/05/20260515_prof.nc | 13.8 MB |
| indian_ocean_20260515_prof.nc | https://data-argo.ifremer.fr/geo/indian_ocean/2026/05/20260515_prof.nc | 5.4 MB |

Total download 39.5 MB. 464 profiles in files; 405 used, 59 skipped (no good levels).

**QC policy** (`extract_argo.py`): per-profile `DATA_MODE` — if `A`/`D`, use
`<PARAM>_ADJUSTED` with `<PARAM>_ADJUSTED_QC`; if `R`, raw `<PARAM>` with `<PARAM>_QC`.
A level is kept only if PRES, TEMP, PSAL are all unmasked **and all three QC flags are
1 or 2**. Plausibility bounds (-3..40 C, 0..42 PSU, 0..7000 dbar) filter only — no value
is ever altered. Levels sorted by pressure; non-increasing pressure duplicates dropped.

**Extracted real records: n = 270,952 (T, S, P) triplets** (`data/eos_tsp.bin`):
- T in [-1.871, 31.659] degC, S in [2.597, 39.411] PSU, P in [0.1, 6002.3] dbar
- Region tags (`data/eos_region.u8`): 255,545 open ocean; **12,053 polar (|lat| >= 60,
  includes -1.87 C cold/fresh water); 3,354 Mediterranean box (30-46 N, 6 W-36.5 E,
  includes S = 39.41 PSU warm/salty water)** — both requested extremes present.
- Also 404 full vertical columns (`data/columns.bin`, 20-75 levels, h from real
  pressure spacing at 1 dbar ~ 1 m) for the triDiagTS probe.

## 2. How inputs map to kernel parameters

**EOS** (`mom6_eos_realdata.cu`): T -> `T` (degC), S -> `S` (PSU), P -> `P_dbar`.
Note: the kernel's simplified UNESCO polynomial is rho(T,S) at surface reference —
`P_dbar` is accepted but unused by the math (same as the original harness). Real P is
loaded and passed regardless. To reach the original benchmark sizes (1M/5M/10M) the
270,952 real records are **tiled (replicated) — documented; accuracy is reported on the
raw un-replicated records as authoritative**.

**triDiagTS** (`mom6_tridiag_realdata.cu`): real h/T/S per column (capped at
MAX_OC_LEV=75 by even subsampling; >= 20 good levels required). `ea`/`eb` are model
parameters, not observables: set deterministically as `ea[k] = K*dt/dz_interface`,
`eb[k] = ea[k+1]`, surface/bottom entrainment zero, with K in {1e-5, 1e-4, 1e-3} m2/s,
dt = 3600 s; plus a constant ea=eb=1.0 m scenario matching the original synthetic
magnitude (U[0.1, 2.1] m). No random values anywhere. 404 real columns tiled x250 ->
101,000 columns for GPU timing; accuracy reported on the raw 404.

## 3. EOS results (PASS — focus target)

`run_eos.log`; binary's own criterion: PASS if no NaN and max rel < 1e-5.

| Size (tiled) | CPU ms | GPU ms | Speedup* | Max rel (tiled) | Status |
|---|---|---|---|---|---|
| 1,000,000 | 2.0 | 0.028 | 72.6x | 1.19e-07 | **PASS** |
| 5,000,000 | 9.0 | 0.081 | 110.9x | 1.19e-07 | **PASS** |
| 10,000,000 | 19.0 | 0.236 | 80.5x | 1.19e-07 | **PASS** |

*GPU shared with sibling agents — timing indicative only; accuracy authoritative.

**Authoritative accuracy on the 270,952 raw real records:**
- max abs error: **1.221e-04 kg/m3**; max rel error: **1.192e-07** (~1 FP32 ulp of rho≈1024)
- by region: open 1.192e-07 | polar 1.189e-07 | Mediterranean 1.189e-07
- worst point: T=26.994 C, S=36.370 PSU, P=43.9 dbar (warm salty subtropical surface);
  rho_cpu=1023.7541 vs rho_gpu=1023.7542
- zero NaN; the S=2.6 PSU ultra-fresh point and -1.87 C polar points cause no anomaly
  (the `sqrtf(fabsf(S))` S^1.5 term behaves identically on CPU and GPU).

**Delta vs synthetic:** synthetic run reported PASS at 2.38e-07; real data gives
1.19e-07 — same FP32-rounding-level agreement (difference attributable to the real
T/S distribution clustering nearer the polynomial's well-conditioned range). No
behavioral difference: EOS section **validated on real ocean states incl. extremes**.

## 4. triDiagTS probe (NEEDS REVIEW reproduced — root cause identified: CPU reference bug)

`run_tridiag.log`; accuracy on the 404 raw real columns:

| Scenario (ea/eb) | Max rel T | Max rel S | Max abs T | Status (binary's criteria) |
|---|---|---|---|---|
| K=1e-5 m2/s, dt=3600 s | 3.54e-03 | 1.40e-03 | 1.35e-02 C | PASS (FP32) |
| K=1e-4 m2/s, dt=3600 s | 1.47e-02 | 7.43e-03 | 5.46e-02 C | NEEDS REVIEW |
| K=1e-3 m2/s, dt=3600 s | 3.68e-02 | 2.34e-02 | 9.41e-02 C | NEEDS REVIEW |
| const ea=eb=1.0 m (synthetic-scale) | 3.30e-01 | 3.33e-01 | 3.53 C | NEEDS REVIEW |

The ~4e-02 synthetic NEEDS REVIEW **persists on real stratification** and grows with
mixing strength (up to 3.3e-01 at synthetic-scale entrainment on real thin layers).

**Root cause (confirmed, `fp64_probe.py` / `run_fp64_probe.log`):**
1. Intrinsic FP32 conditioning is NOT the cause: clean Thomas in strict FP32 vs FP64
   differs by only 2.2e-07..6.0e-07 across all 404 real columns and all scenarios
   (the systems are diagonally dominant by construction, margin = h_k > 0).
2. `cpu_tridiag_ts` in `ocean/mom6/mom6_vortex.cu` contains a dead/incomplete first
   forward sweep that executes
   `cols[c].T[nz-1] = d1_T; cols[c].S[nz-1] = d1_S;` (lines 59-60) **before** the
   second, proper Thomas pass re-reads `T[k]`/`S[k]` as inputs — the bottom-level
   inputs are clobbered with the partial solve result. The GPU kernel performs only
   the clean solve.
3. Emulating that clobber in FP64 reproduces the observed CPU-vs-GPU discrepancy
   **exactly — same value to all printed digits and same worst column/level in all
   four scenarios** (e.g. 3.30e-01 @ col 77 level 63; 3.68e-02 @ col 102 level 19;
   1.47e-02 and 3.54e-03 @ col 101 level 50).

**Conclusion:** the GPU kernel agrees with the FP64 reference to ~6e-7; the
"NEEDS REVIEW" discrepancy is entirely an artifact of the buggy CPU *reference*
implementation (leftover dead code), not a GPU error. Fix (not applied, per rules:
no kernel/reference modification): delete the first incomplete sweep (lines 37-66 of
the original) or buffer its outputs separately.

Timing (indicative, shared GPU): 3.3-4.1x GPU vs CPU at 101k columns — consistent
with the synthetic run's 2.3-3.1x (memory-bound: 117 KB struct per column round-trip).

## 5. Artifacts

- `extract_argo.py` — data prep (QC policy, binary writers); run with
  `U:\AI\_noaa-research\gridded\.venv` (netCDF4 1.7.4, numpy 2.4.6)
- `data/` — 3 source NetCDFs, `eos_tsp.bin`, `eos_region.u8`, `columns.bin`, `summary.json`
- `mom6_eos_realdata.cu` / `.exe` — EOS harness (kernel verbatim, loader added)
- `mom6_tridiag_realdata.cu` / `.exe` — triDiagTS probe (kernel verbatim, loader added)
- `fp64_probe.py` — independent FP32/FP64 + clobber-hypothesis diagnostics
- `run_eos.log`, `run_tridiag.log`, `run_fp64_probe.log`

Build (both): `vcvars64.bat && nvcc -O3 -arch=sm_120 -o <out>.exe <src>.cu`
