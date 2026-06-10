# t-route Muskingum-Cunge GPU kernels on a REAL river network

**Target:** `water-prediction/t-route/troute_mc_final.cu` (FP64) and
`troute_mc_reach_parallel.cu` (FP32) validated against a real NWM river network
instead of the synthetic generator.

**Key question:** does `troute_mc_reach_parallel`'s synthetic-data accuracy failure
on sm_120 reproduce on a real network, and does `troute_mc_final` stay clean?

**Answer:** the failure does **not** reproduce on the real network — both binaries
**PASS** their own criteria on real data (cold and warm start). `troute_mc_final`
stays essentially bit-identical (max rel 3.86e-11). However, the FP32 binary's
max relative error grows ~3 orders of magnitude when going from cold start
(3.32e-06) to a warm-start second routing step (3.73e-03), confirming the FP32
secant-path divergence mechanism is real and regime-dependent — the synthetic
inputs (deep warm depths everywhere, 10–100x more segments, broader random
geometry) push it over the 1% threshold; this real 10.9K-segment network does not.
The synthetic FAIL was re-reproduced in the same session on the same GPU
immediately before the real-data runs, so the contrast is not a driver/GPU-state
artifact.

## Data provenance (REAL data only)

| Item | Value |
|---|---|
| Source repo | https://github.com/NOAA-OWP/t-route (NOAA Office of Water Prediction) |
| Commit | `12a8eae0cdfed437143c590659fa7077605a5e70` (master) |
| Access date | 2026-06-10 |
| Network file | `test/LowerColorado_TX/domain/RouteLink.nc` — NWM RouteLink subset, Lower Colorado River basin, TX. Real NHDPlus-derived channel geometry and topology (11,248 features) |
| Flow state file | `test/LowerColorado_TX/channel_forcing/202108231300.CHRTOUT_DOMAIN1` — real NWM Analysis-and-Assimilation channel output, valid **2021-08-23 13:00 UTC** (Hurricane-season AnA test case) |
| Downloads | 1.5 MB total (RouteLink.nc 1,117,660 B; CHRTOUT 264,692 B; plus HYDRO_RST 144,072 B fetched but NOT used, see below) |
| Local copies | `data/` in this directory |

Subsetting performed (documented, no value modification):
- **341 waterbody-internal links excluded** — their CHRTOUT `streamflow` is
  `_FillValue` (NWM does not channel-route links inside lakes; all 341 have
  `NHDWaterbodyComID` set). t-route's own `test_AnA.yaml` removes them the same
  way (`break_network_at_waterbodies: True`). Routed domain: **10,907 segments**.
- The shipped WRF-Hydro restart (`HYDRO_RST.2021-08-23_12:00_DOMAIN1`) was
  evaluated and **rejected**: it has 11,141 links vs RouteLink's 11,248 and no ID
  variable; t-route's `nhd_io.get_channel_restart_from_wrf_hydro` joins
  positionally ("order is simply the same as the Route-Link file"), which cannot
  be correct with mismatched lengths. Notably t-route's own `test_AnA.yaml` leaves
  the restart line commented out and cold-starts — we follow the same convention.

**n records: 10,907 real channel segments** (from 11,248 RouteLink features),
organized into **7,592 reaches** (max length 25 segments, mean 1.44). No
replication or augmentation. Every input value is real data (or, for the step-2
scenario, the deterministic one-step evolution of real data through the
unmodified FP64 CPU reference).

## Input -> kernel parameter mapping

| Kernel input | Real source |
|---|---|
| `dx` | RouteLink `Length` (m) |
| `bw` | RouteLink `BtmWdth` (m) |
| `tw` | RouteLink `TopWdth` (m) |
| `twcc` | RouteLink `TopWdthCC` (m) |
| `n_ch` | RouteLink `n` (Manning) |
| `ncc` | RouteLink `nCC` |
| `cs` | RouteLink `ChSlp` |
| `s0` | RouteLink `So` |
| `ql` (lateral inflow) | CHRTOUT `qBucket + qSfcLatRunoff` — t-route's preferred qlat composition (`nhd_io.py`) |
| `qd` (prev. flow at segment) | CHRTOUT `streamflow` |
| `qu` (prev. upstream flow) | sum of CHRTOUT `streamflow` over upstream parents via RouteLink `to` topology (0 for headwaters; lake outflow boundary set to 0, documented) |
| `dp` (prev. depth) | step 1: **0** (cold start, exactly as `test_AnA.yaml`); step 2: FP64 CPU-reference depth output of step 1 |
| reaches (`rs`,`rl`) | maximal linear chains of `to` topology; new reach wherever in-degree != 1 (t-route's junction-break concept) |
| `dt` | 300 s (per `test_AnA.yaml` `dt: 300`) |
| `uq` (scalar head inflow) | 0.0 (kernel signature takes one scalar for all reaches; the original used a synthetic 10.0 — we inject nothing) |

Real-data ranges vs the synthetic generator (notably different regime):
`dx` 1–61,296 m (synthetic 500–5,500), `bw` 0.196–73 m (synthetic 5–55),
`s0` 1e-5–3.85 (synthetic 1e-4–5.1e-3), `cs` 0.14–1.91, `n` 0.05–0.06,
`ql` 0–0.74 m³/s, `qu`/`qd` 0–303 m³/s (mean 3.4; synthetic 5–55 everywhere),
804 segments with all-zero inflow (synthetic: none).

## Harness adaptation (hard-rule compliance)

- `troute_mc_final_realdata.cu` / `troute_mc_reach_parallel_realdata.cu`: copies
  of the originals where **only** the synthetic generator (`gen` / `gen_data`)
  and the `main()` driver loop were replaced by a flat-binary loader.
  `mc_solve`, `hgeo`, `kernel_mc_gpu`, `cpu_mc`, `mc_secant_solve`,
  `hydraulic_geometry_d`, `kernel_mc_reach_parallel`, `cpu_mc_sequential` were
  verified **byte-identical** to the originals (`cmp` over the code regions).
  The FP64 driver additionally dumps the CPU-reference outputs
  (`cpu_state_out.bin`) to enable the step-2 input; this does not touch the
  CPU/GPU comparison. Accuracy checks and pass criteria are unchanged.
- The FP32 harness loads the same FP64 binary and casts to `float`, mirroring
  the original FP32 design.
- Build: CUDA 13.3, `nvcc -O3 -arch=sm_120`, MSVC 2022 BuildTools host,
  RTX 5070 (sm_120), driver 610.47, Windows 11.

## Results (accuracy authoritative; timing indicative — GPU may be shared)

### Real network, step 1 (cold start, 2021-08-23 13:00 UTC)

| Binary | CPU | GPU | Max abs err | Max rel err | Fail count | Own criterion | Status |
|---|---|---|---|---|---|---|---|
| `troute_mc_final` (FP64) | 5.0 ms | 2.82 ms | **5.68e-14** | **2.75e-14** | 0/10,907 >0.1% | rel<1e-10 = bit-identical | **PASS (bit-identical)** |
| `troute_mc_reach_parallel` (FP32) | 3.0 ms | 0.21 ms | **3.05e-05** | **3.32e-06** | 0/10,907 >1% | max rel<0.01 | **PASS** |

### Real network, step 2 (warm start: state = step-1 output of unmodified FP64 CPU reference; depths 0–8.17 m, 10,106/10,907 nonzero)

| Binary | CPU | GPU | Max abs err | Max rel err | Fail count | Status |
|---|---|---|---|---|---|---|
| `troute_mc_final` (FP64) | 7.0 ms | 2.87 ms | **1.30e-10** | **3.86e-11** | 0/10,907 >0.1% | **PASS** |
| `troute_mc_reach_parallel` (FP32) | 5.0 ms | 0.17 ms | **2.02e-01** | **3.73e-03** | 0/10,907 >1% | **PASS** (but see below) |

### Synthetic baseline, same session, same GPU (`_build/troute_mc_reach_parallel.exe`)

FAIL at every size, identical to the previously logged run: max rel err 6.84 (1K),
19.1 (10K), 13.0 (50K), 405 (100K); 11/88/415/872 segments >1%.
(`run_synthetic_baseline_reach_parallel.log`)

## Delta vs synthetic run and interpretation

1. **`troute_mc_final` (FP64) is cleaner on real data than on synthetic.** On
   synthetic it goes "NEEDS REVIEW" at 50K/100K reaches (a handful of >0.1%
   segments); on the real network it is bit-identical-class at both cold
   (2.75e-14) and warm (3.86e-11) starts. FP64 secant iteration takes the same
   convergence path on CPU and GPU for every one of the 10,907 real segments.
2. **`troute_mc_reach_parallel` (FP32) does not fail on the real network.** The
   synthetic FAIL (rel err up to 405) did not reproduce — 0 segments above the
   1% threshold in either scenario.
3. **But the underlying FP32 divergence mechanism is visible in real data:**
   cold→warm start moves max rel err from 3.32e-06 to 3.73e-03 (~1000x). With a
   warm depth state the secant solver iterates through depth regimes where CPU
   `powf` vs GPU `powf` ULP differences can flip iteration counts. The synthetic
   regime — every segment warm (0.5–3.5 m), flows 5–55 m³/s everywhere, broader
   geometry randomization, and 10x–100x more segments sampling the tail —
   produces occasional full convergence-path divergence (different secant root),
   hence rel errors >>1. The real Lower Colorado network at one AnA timestep
   (mostly small flows, mean depth 0.07 m) stays ~3 ms below that cliff.
4. Practical read for the repo's open question (BLACKWELL_RESULTS.md): the
   reach_parallel FAIL is not an sm_120 miscompilation that corrupts all
   workloads — it is FP32 secant-path sensitivity that real, moderate-flow
   networks may not trigger at this scale, but which warm states demonstrably
   amplify. FP64 (`troute_mc_final`) is the defensible configuration; on this
   real network it is also only ~14x slower GPU-side than the FP32 kernel
   (2.87 ms vs 0.17–0.21 ms) and still beats the (small) CPU job.
5. Speedups on the real network are modest (1.4–2.4x FP64, 14–30x FP32) versus
   the synthetic 20–370x — expected: 10.9K segments / 7.6K mostly 1-segment
   reaches is a far smaller and less parallel workload than the synthetic 1M+
   segments with mean reach length ~10.

## Caveats

- `uq` (current upstream inflow at each reach head) is one scalar in the kernel
  signature; set to 0.0 for all reaches (the original injected a synthetic
  10 m³/s). Identical for CPU and GPU, so the comparison is unaffected.
- Outflow from the 28 lakes is not represented (children of excluded waterbody
  links get qu only from surviving parents). A reservoir module would supply
  this in production t-route; irrelevant to CPU-vs-GPU consistency.
- Timing is indicative only (GPU possibly shared with sibling agents); CPU
  timings use `clock()` on a few-ms job.
- Step-2 state is model-evolved real data (t-route's own state-advance pattern),
  not an independent observation.

## Files

- `prep_lcr_network.py`, `prep_run.log` — extraction RouteLink/CHRTOUT -> `lcr_network.bin` (+ `lcr_network_ids.csv` traceability)
- `prep_step2.py`, `prep_step2.log` — warm-start input `lcr_network_step2.bin` from `cpu_state_out.bin`
- `troute_mc_final_realdata.cu`, `troute_mc_reach_parallel_realdata.cu` (+ `.exe`) — adapted harnesses (loader-only changes)
- `run_mc_final_realdata.log`, `run_mc_reach_parallel_realdata.log` — step-1 runs
- `run_mc_final_realdata_step2.log`, `run_mc_reach_parallel_realdata_step2.log` — step-2 runs
- `run_synthetic_baseline_reach_parallel.log` — same-session synthetic FAIL baseline
- `data/` — RouteLink.nc, 202108231300.CHRTOUT_DOMAIN1.nc, HYDRO_RST (unused, kept for the record)
- `nhd_io.py`, `test_AnA.yaml` — upstream reference files documenting t-route's own conventions (qlat composition, cold start, dt=300)
