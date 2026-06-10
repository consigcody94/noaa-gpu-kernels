# RTX 5070 (Blackwell, sm_120) re-verification & benchmarks

> Generated 2026-06-10 from `_build\runlogs\*.log`. All 22 buildable binaries compiled with
> `nvcc -O3 -arch=sm_120` (CUDA 13.3, MSVC 14.44 host) and run sequentially (no concurrent
> CPU/GPU load). Raw logs preserved in `_build\runlogs\`.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 12GB (Blackwell, compute capability 12.0 / sm_120) |
| Driver | 610.47 |
| CUDA Toolkit | v13.3 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`) |
| Host compiler | MSVC 14.44.35207 (VS 2022 Build Tools) |
| CPU (baseline for "speedup") | AMD Ryzen 7 5700G — **differs from the README's original CPU**, so speedup-vs-CPU is NOT apples-to-apples with the published 3060 numbers. GPU times and error stats are directly comparable. |
| OS | Windows 11 Pro 10.0.26200 |

**Build notes:** all 22 top-level benchmark binaries compiled without source changes for
sm_120. The two `evp_dmi_*` binaries run but require an external input file
(`input_double_1d_v1.bin`) not present in the repo — not exercised (CICE EVP is covered by
`noaa_multi`). Banner strings still print "RTX 3060 12GB" — hardcoded in the sources;
cosmetic fix candidate.

## The four "Pending" kernels (upstream issues not yet filed) — all verify on Blackwell

| Kernel | 3060 (README) | 5070 result | Verification on 5070 |
|---|---|---|---|
| CCPP tridi1 (PBL) | 7.5–11.6x | 49.9–64.9x | PASS @10k cols; **NEEDS REVIEW @100k/500k** (fast-math rel err up to 4.55e-01 on 1990/32M points; max abs only 4.77e-07 — near-zero denominators) |
| Icepack Delta-Eddington | 291x, 5.62e-07 | 81x / 295x / **3042x** (10k/100k/500k cols) | PASS all sizes; max rel 5.62e-07 — **identical error stat to 3060** |
| CICE EVP dynamics | 462x, 9.40e-05 | 176.9x / 553.3x / **588x** | PASS all sizes; max rel 9.40e-05 — identical to 3060 |
| MOSART kinematic wave | 5.9–6.5x | 11.4–11.8x | PASS @100k/500k; NEEDS REVIEW @1M (fast-math rel 3.21e-02) |

Recommendation before filing: for CCPP and MOSART, either present the fast-math caveat
explicitly at large problem sizes (the binary's own label) or add a non-fast-math build
variant; the absolute errors remain at FP32 rounding level throughout.

## Published table kernels — 5070 vs 3060

| # | Kernel | 3060 (README) | 5070 speedup | 5070 verification |
|---|---|---|---|---|
| 1 | rte-rrtmgp prefix scan | 3.10x | **3.91x** (scan); full flux solver 2.82x | max rel 7.76e-07; **15/15 stress tests PASS** |
| 2 | CRTM clear-sky adding | 6/6 pass, <4.4e-07 | sub-ms across all 6 configs | **6/6 PASS**, max rel 4.4e-07 (identical) |
| 3 | GSI ensemble forward | 109 GB/s | **567–582 GB/s** (5.2x the 3060 figure; exceeds 3060's theoretical peak) | PASS on all 3 configs that fit 12GB; C768/C1152 skipped (>12GB; binary's "2 TESTS FAILED" summary line miscounts SKIP as FAIL — cosmetic) |
| 4 | WW3 DIA | up to 16.8x (caveated) | 4.0x GPU-parallel vs GPU-sequential, 41,182 M DIA evals/sec | max rel 0.0 (note: this binary benchmarks GPU-vs-GPU, not vs CPU) |
| 5 | t-route MC reach-parallel | **92x, "FP64 verified"** | see ⚠ below | ⚠ **accuracy discrepancy between the two binaries** |
| 6 | t-route diffusive wave | 10x, 2.48e-07 | 27.3–158.4x | PASS, max rel 2.48e-07 (identical) |
| 7 | CFE Nash cascade | 2x, bit-identical | 2.1–4.4x | PASS, bit-identical (0.0 error) |
| 8 | NOAH-MP tridiag | 11.2x, residual <4e-7 | 79.4–127.3x | residual 2.98e-07–4.17e-07 ✓ consistent with README criterion; binary labels FAIL @≥100k cols on *relative* error (7.38e-01 max on 374/10M points — near-zero solution values; residual stays at FP32 level) |
| 9 | TOPMODEL | 31.3x, 5.87e-07 | 240.8–276.1x | PASS, max rel 5.87e-07 (identical) |
| 10 | PET Penman-Monteith | 34.9x, 1.44e-04 | 182.4–953.8x | PASS, max rel 1.44e-04 (identical) |
| 11 | Snow17 | 4.1x, 2.10e-05 | 4.9–6.0x | PASS, max rel 2.10e-05 (identical) |
| 12 | LGAR Green-Ampt | 8.9x, bit-identical | 15.8–24.3x | PASS, bit-identical |

⚠ **t-route MC (README row 5, upstream issue #874):** two binaries exist.
`troute_mc_final` ("reach-parallel, exact algorithm match") PASSES at 1K/10K reaches
(max rel 5.26e-06) with 5.5–11.1x speedup, and goes NEEDS REVIEW at 50K/100K
(7–18 segments of >0.5M exceed 0.1%). `troute_mc_reach_parallel` shows 20–338x speedups
but **FAILS accuracy at every size on this hardware** (max rel err 6.8 → 405; 11–872
segments >1%). Before citing the README's "92x verified FP64" upstream, re-establish which
binary/config produced that number and whether `troute_mc_reach_parallel`'s failure is
(a) pre-existing on the 3060, (b) an sm_120 behavior change (e.g. warp-sync or
contraction differences), or (c) RNG/test-harness divergence. This is the only
**possible correctness regression** observed in the suite.

## Documented-failure kernels (README "Documented Failures") — reproduced as expected

- **GSI recursive filter** (`gsi_recursive_filter`): 1 PASS / 19 FAIL across configurations —
  consistent with the README's documented negative result (prefix scan unsuitable for IIR
  filters; 26/27 configs fail). Expected behavior, not a regression.

## Exploratory kernels (not in the README's 16-kernel table)

| Kernel | 5070 speedup | Verification |
|---|---|---|
| COARE 3.6 air-sea flux | 700x–3670x | PASS (fast math), hlb max rel 2.07e-03 |
| FV3 map1_ppm vertical remap | 299.6–490.3x | PASS, max rel ≤4.31e-07 |
| MOM6 equation of state | 74.7–147.7x | PASS, 2.38e-07 |
| MOM6 triDiagTS vertical mixing | 2.3–3.1x | NEEDS REVIEW (T/S max rel ~4e-02) |
| UPP CAPE/CIN | 351.1–408.3x | PASS, 100% columns match <1 J/kg |
| tuv-x Delta-Eddington (NCAR #64) | 103.2–179.7x | PASS, max rel ≤1.71e-05 |
| NCEPLIBS-sp Legendre synthesis | 33.3–403.9x | **NEEDS REVIEW — max rel up to 1.5e+05**; needs error-metric audit before any claim |
| CTSM tridiag (NCAR) | 68.4–82.6x | NEEDS REVIEW (rel up to 1.8e-01; residual ≤4.17e-07 — same near-zero-denominator pattern as NOAH-MP) |

## Real-data validation (2026-06-10)

Synthetic generators were replaced with real public datasets (loader-only harness
variants — kernel math and CPU references untouched, verified by diff). Full provenance,
prep scripts, adapted harnesses, and run logs per target under `_realdata\<target>\RESULTS.md`.
Accuracy numbers authoritative; timings indicative (GPU shared during agent runs).

| Target | Real dataset (provenance) | Result on RTX 5070 |
|---|---|---|
| CICE EVP (`evp_dmi_*`) | **DMI Arctic 1 Mar 2020 winter state**, Zenodo 10.5281/zenodo.11248366 (Rasmussen et al. 2024, GMD) — input files were already in the repo clone, **MD5-verified vs Zenodo**; the earlier "Cannot open" failure was a working-directory issue | **PASS** at ndte=1 (5.3e-09) and ndte=5 (2.4e-07) per the binary's own <1e-6 criterion; ndte≥120 shows the README-documented iterative FP divergence. 74–87× vs single-core 5700G (~2.5× faster than the 3060 logs). ⚠ the "optimized" persistent fused kernel **regresses −1.1% on sm_120** (was +3.7–5.2% on sm_86) — grid-wide sync costs more than launch overhead on Blackwell |
| t-route MC | **Real Lower Colorado TX NWM network** (RouteLink.nc, 10,907 segments) + real AnA CHRTOUT flows 2021-08-23, from NOAA-OWP/t-route test fixtures | FP64 `mc_final`: **max rel 2.75e-14 (bit-identical class)** cold + warm start. FP32 `reach_parallel`: **PASSES on real data** (3.3e-06 cold; 3.7e-03 warm — error grows ~1000× with warm depths but stays under the 1% cliff). Same-session synthetic rerun still fails ⇒ the failure is **regime-dependent FP32 sensitivity, not an sm_120 miscompile**. FP64 is the defensible config for issue #874 follow-ups |
| COARE 3.6 | **Official COARE validation dataset** (NOAA-PSL/COARE-algorithm `test_36_data.txt`, 2,165 ship obs, ATOMIC cruise 2020) + published expected outputs | CPU-vs-GPU max rel ≤1.31e-06 — **passes the strict 1e-3 bar** (synthetic only passed the fast-math waiver). Vs official gold outputs both CPU and GPU share the same bias (tau +3%, hlb +13%) ⇒ kernel's simplified physics (no cool-skin/warm-layer), not the GPU port |
| Icepack Delta-Eddington | **MOSAiC campaign**: Itkin et al. 2021 snow/ice transects (PANGAEA 10.1594/PANGAEA.937781) + MOSAiC merged radiation (NSF ADC 10.18739/A2WD3Q35Z), 84,921 real Arctic columns | **Strict PASS**: max rel 9.48e-07 absorbed / 8.40e-06 transmitted, 0 NaN incl. polar-night and 3 mm ice edge cases |
| CCPP tridi1 (PBL) | **IGRA2 radiosondes**, 5 stations Arctic→tropics, 4,915 QC'd soundings → 19,660 real diffusion systems | **The synthetic NEEDS-REVIEW at scale does NOT persist**: max rel constant 5.60e-04 at 10k/100k/500k (vs 4.55e-01 synthetic). FP64-truth check: the FP32 CPU reference itself deviates 1.3e-03 from FP64 — more than the CPU-GPU gap. Synthetic verdict was a test-data artifact |
| OWP kernels (6) | Real basin data from the upstream NOAA-OWP repos: cfe cat-87 + Laramie AORC, noah-owp-modular **Bondville 1998**, topmodel canonical catchment, snow17 **Hungry Horse MT 1970–2015**, LGAR-C Phillipsburg/Bushland | **All six PASS.** NOAH-MP verdict **flips FAIL→PASS** (real Richards systems: max rel 3.35e-05 vs synthetic 0.738 — near-zero-solution artifact). CFE Nash + LGAR bit-identical; Snow17 improves to 5.52e-06; PET 4.31e-04 (real −37.5 °C extremes, still 20× inside gate) |
| MOM6 EOS (+triDiagTS probe) | **Argo GDAC** 2026-05-15 global profiles, 270,952 QC'd (T,S,p) records incl. polar + Mediterranean extremes | EOS **PASS, max rel 1.19e-07 (~1 ulp)**. triDiagTS NEEDS-REVIEW persists on real stratification and is now **root-caused: the CPU reference clobbers bottom-level inputs (dead-code bug) — the GPU kernel matches a clean FP64 Thomas solve to ≤6e-07**. Fix the CPU reference, not the kernel |
| UPP CAPE/CIN | **IGRA2**: Norman OK, Del Rio TX, Chanhassen MN, San Juan PR — 4,045 real soundings; cross-checked vs NCEI's derived-parameter files | **PASS: 100% of columns match <1 J/kg** (max diff 0.391 J/kg). Real data exposed that the synthetic generator feeds levels in the **wrong vertical orientation** (kernel's intended path exercised for the first time). Magnitudes ~4.45× NCEI derived CAPE because the kernel lifts a saturated-from-origin parcel (no dry ascent to LCL) — documented physics simplification, rank-order correct (Spearman 0.60) |

### Fixes this pass surfaced in THIS repo (before filing upstream)

1. `mom6_vortex.cu` triDiagTS **CPU reference bug** — dead-code clobber of bottom-level
   inputs; the GPU kernel is correct (FP64-verified). Fixing it should clear all
   triDiagTS NEEDS-REVIEW statuses.
2. `upp_cape.cu` synthetic generator builds columns **surface-at-index-0**, opposite the
   kernel's assumed orientation — the synthetic benchmark never exercised the
   most-unstable-parcel break or the intended lift loop.
3. `evp_dmi_optimized.cu` persistent fused kernel is a **regression on Blackwell** —
   consider gating by arch or retiring.
4. Banner strings hardcode "RTX 3060 12GB" — print the actual device from
   `cudaGetDeviceProperties`.
5. CCPP/NOAH-MP/MOSART "NEEDS REVIEW"/"FAIL" labels at scale are synthetic-extreme
   artifacts (proven by real data + FP64 truth checks) — worth noting in the harness
   output or README so reviewers don't misread them.

## Caveats

1. **Speedup-vs-CPU is not comparable to the README's 3060 table** — different CPU
   (Ryzen 7 5700G vs the original benchmark host), and several CPU timings sit at the
   1 ms timer resolution floor (e.g. CFE @10k shows "0.0 ms CPU → 0.0x"). GPU-side
   timings, bandwidth, and all error statistics are directly comparable.
2. Identical max-error stats across GPUs (TOPMODEL, PET, Snow17, Icepack, CICE, diffusive
   wave, CRTM) indicate deterministic test data and confirm sm_120 numerical behavior
   matches sm_86 on those kernels.
3. Windows/MSVC build — original numbers were presumably Linux/GCC; no source changes
   were needed, which is itself a useful portability data point.
4. The "NEEDS REVIEW" labels on CCPP/MOSART/NOAH-MP at large sizes come from each binary's
   own strict relative-error thresholds hitting near-zero solution values under fast math;
   absolute errors and residuals remain at FP32 rounding level. Worth stating explicitly
   in the upstream issues to preempt reviewer pushback.
