# CCPP tridi1 (PBL tridiagonal solver) — real-data validation

**Target:** `kernel_tridi1` / `cpu_tridi1` in `noaa_gpu-kernels/noaa_multi_kernel.cu`
(Thomas algorithm for implicit PBL vertical diffusion, from NCAR/ccpp-physics
`physics/PBL/tridi.f`).

**Key question:** the synthetic-data run goes **NEEDS REVIEW at 100k/500k columns**
(max rel err 7.13e-02 / 4.55e-01). Do those relative-error blowups persist when the
tridiagonal systems are built from **real atmospheric profiles**, or were they
artifacts of synthetic extremes?

**Answer: they do NOT persist. All sizes PASS on real data** (binary's own
"PASS (fast math)" label, max rel 5.60e-04 at every size, error does not grow with
column count). The synthetic NEEDS REVIEW was an artifact of random-coefficient
systems producing ever-more-extreme near-zero solution values as the sample grows.

---

## 1. Data provenance (real, public, authoritative)

- **Dataset:** NOAA NCEI Integrated Global Radiosonde Archive v2 (IGRA2),
  year-to-date files (observations 2025-01-01 through 2026-06-09).
- **URL:** `https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/`
  (files `<STATION>-data-beg2025.txt.zip`); station metadata from
  `.../doc/igra2-station-list.txt`.
- **Accessed:** 2026-06-10.
- **Stations (5, spanning Arctic / mid-latitude / subtropical / tropical):**

| IGRA2 ID | Station | Lat / Lon / Elev | Soundings parsed | Accepted by QC |
|---|---|---|---|---|
| USM00070026 | Barrow/W. Post W. Rogers, AK (Arctic) | 71.29N 156.78W 12 m | 1948 | 1772 |
| USM00072250 | Brownsville Intl, TX | 25.92N 97.42W 7 m | 1051 | 1039 |
| USM00072403 | Sterling, VA (Washington Dulles) | 38.98N 77.49W 88 m | 1046 | 1040 |
| GMM00010868 | Oberschleissheim (Munich), Germany | 48.24N 11.55E 484 m | 1063 | 29 |
| RQM00078526 | San Juan Intl, Puerto Rico (tropics) | 18.43N 65.99W 4 m | 1044 | 1035 |
| **Total** | | | **6152** | **4915** |

Total download: ~21.6 MB (5 zips). No fabricated or random values anywhere in the
input path; every cl/cm/cu/r1 value derives from measured radiosonde
pressure/geopotential/temperature/humidity/wind plus documented physical constants.

**QC gates** (per sounding): surface pressure >= 850 hPa; >= 25 levels with valid
T and geopotential; >= 15 levels with valid wind; >= 10 levels with valid humidity
(RH or dewpoint depression); valid T/Z and wind coverage up to 0.20*psfc; monotone
geopotential after interpolation. Interpolation only *between* valid levels of the
same sounding — no extrapolation, no cross-sounding gap-filling.
*Munich note:* most GMM00010868 ascents are GTS-sourced and carry geopotential at
only ~13 standard levels, so they fail the >= 25-valid-levels gate — correctly
rejected as too coarse for a 64-level grid, leaving 29 high-resolution ascents.

## 2. Real records used

- **4915 accepted soundings -> 19,660 real tridiagonal systems (columns)** of 64
  levels each: per sounding, 4 systems — (K_h matrix, T), (K_h, q), (K_m, u),
  (K_m, v) — exactly as a PBL scheme solves heat/moisture/momentum diffusion.
- Binary input `tridi_real.bin` (20.1 MB): `int32 ncol=19660, int32 nlev=64`,
  then `float32 cl/cm/cu/r1 [ncol*nlev]`.

## 3. How real profiles map to kernel inputs

(All preprocessing — standard PBL column physics, **not** kernel math; see
`prep_igra2_tridi.py` for the exact code.)

1. **Parse** IGRA2 fixed-width records (PRESS Pa, GPH m, TEMP 0.1 C, RH 0.1 %,
   DPDP 0.1 C, WDIR deg, WSPD 0.1 m/s; -9999/-8888 missing).
2. **Vertical grid:** 64 sigma layers, interfaces `s_if[j] = 1 - 0.8*(j/64)^1.3`
   (sigma 1.00 -> 0.20, denser near the surface). T, q, u, v, z interpolated to
   layer-midpoint pressures `p_k = sigma_k * psfc`, linear in ln(p). q from RH
   (preferred) or dewpoint depression via Magnus e_s over water,
   `q = 0.622 e / (p - 0.378 e)`.
3. **Eddy diffusivity** at interior interfaces — bulk-Richardson local K closure
   (Louis-1979-type stability functions, Blackadar mixing length; the standard
   first-order local scheme used for PBL preprocessing):
   - `Ri = (g/thv_m) * d(thv) * dz / (du^2 + dv^2 + 1e-9)`
   - `l = 0.4 z / (1 + 0.4 z / 150 m)` ; `S = |dU/dz|`
   - `f_m = sqrt(1-16Ri)` (Ri<0) ; `1/(1+5Ri)^2` (Ri>=0)
   - `K_m = l^2 S f_m`, clipped to [0.01, 1000] m^2/s
   - `K_h = K_m / Pr`, `Pr = 0.85` (unstable) or `min(1+2.1Ri, 3)` (stable;
     Kim & Mahrt 1992)
4. **Tridiagonal assembly** — implicit-Euler vertical diffusion with dt = 600 s,
   matching the solver's system structure (cl[0]=0, cu[n-1]=0, negative
   off-diagonals, diagonally dominant cm — same structure `gen_tridi` produces):
   - `cl[k] = -dt K_(k-1/2) / (dz_layer[k] * dz_interface[k-1/2])`
   - `cu[k] = -dt K_(k+1/2) / (dz_layer[k] * dz_interface[k+1/2])`
   - `cm[k] = 1 - cl[k] - cu[k]` ; `r1[k] = T_k | q_k | u_k | v_k`
   - Measured ranges: |cl| up to 64.3, cm up to 118, r1 in [-76.3, 308.1];
     diagonal-dominance margin `cm - |cl| - |cu|` >= 0.999996 everywhere.

## 4. Harness adaptation and replication

- `ccpp_tridi1_realdata.cu`: `cpu_tridi1`, `kernel_tridi1`, and `compare()` are
  **verbatim copies** from `noaa_multi_kernel.cu` (kernel math untouched). Only
  `gen_tridi()` is replaced by a file loader + column tiler, plus a post-comparison
  diagnostic that does not affect pass/fail.
- **Replication (documented):** real columns are tiled modulo 19,660 to reach the
  original benchmark sizes — 10k = first 10,000 real columns (no replication),
  100k = 5.09x tiling, 500k = 25.43x tiling. Replication adds no new distinct
  systems, so error statistics saturate once all 19,660 real columns are covered.
- **Build:** `nvcc -O3 -arch=sm_120` (CUDA 13.3, MSVC 14.44 host) — identical
  flags to the original `noaa_multi.exe` build (`build.cmd`). No
  `-use_fast_math`; the "(fast math)" label is the binary's own name for the
  1e-4..1e-2 relative-error band (driven mainly by FMA contraction on GPU).

## 5. Results — RTX 5070 (sm_120), 2026-06-10

GPU possibly shared with sibling agents: **accuracy authoritative, timing indicative.**

| Columns | Synthetic (run log 2026-06-10) | **Real IGRA2 data** |
|---|---|---|
| 10,000 | max abs 3.58e-07, max rel 2.22e-03, 35/640k >1e-4 — PASS (fast math) | max abs 4.12e-04, **max rel 5.60e-04**, 2/640k >1e-4 — **PASS (fast math)** |
| 100,000 | max abs 4.77e-07, **max rel 7.13e-02**, 428/6.4M — **NEEDS REVIEW** | max abs 6.10e-04, **max rel 5.60e-04**, 11/6.4M — **PASS (fast math)** |
| 500,000 | max abs 4.77e-07, **max rel 4.55e-01**, 1990/32M — **NEEDS REVIEW** | max abs 6.10e-04, **max rel 5.60e-04**, 52/32M — **PASS (fast math)** |

Indicative timing (real data): 66.1x / 55.6x / 48.4x GPU speedup vs single-core
CPU at 10k/100k/500k — essentially identical to the synthetic run (66.9/56.8/48.3x),
i.e. real-magnitude coefficients do not change kernel performance.

**Pass/fail vs the binary's own criteria:** PASS at all three sizes (label
"PASS (fast math)": 0 NaN and max rel < 1e-2). The synthetic run's NEEDS REVIEW
at 100k/500k is eliminated.

## 6. Behavior differences vs synthetic, and why

1. **Relative error no longer grows with scale.** Synthetic: 2.2e-03 -> 7.1e-02 ->
   4.6e-01 as columns increase (more random draws -> more extreme near-zero
   solution values). Real: constant 5.60e-04 at all sizes.
2. **Worst point is a physically real near-zero wind.** The harness diagnostic
   pins the max rel error (all sizes) to real column 646 — a **u-wind** system
   from a San Juan sounding — at level k=50, where the solution crosses zero:
   CPU -1.849234e-05 vs GPU -1.850270e-05 m/s (abs diff 1.0e-08). Same
   near-zero-denominator mechanism as synthetic, but bounded ~3 orders of
   magnitude lower because real profiles don't produce the adversarial
   coefficient/RHS combinations uniform-random data does.
3. **Absolute error is larger on real data (6.1e-04 vs 4.8e-07) — expected and
   benign:** real solutions are O(100) (temperature in K) instead of O(1), so
   FP32 rounding at ~1e-7 relative shows up as ~1e-4 absolute.
4. **Both implementations sit at inherent FP32 accuracy.** Solving the same
   19,660 real systems in FP64 (numpy Thomas, same operation order) shows the
   FP32 *CPU reference itself* deviates from FP64 truth by up to **1.30e-03**
   relative (again at a near-zero wind value) — more than twice the CPU-vs-GPU
   disagreement of 5.60e-04. The GPU adds no error beyond FP32
   reordering/FMA-contraction noise.

**Conclusion:** the fast-math NEEDS REVIEW flags at 100k+ columns were artifacts
of synthetic extremes, not a kernel defect. On real atmospheric diffusion systems
the GPU kernel matches the CPU reference to FP32 rounding at every tested scale,
including 500k columns (25x replication of 19,660 real columns). For upstream
reporting, the honest framing is: "relative-error spikes occur only at near-zero
solution values (wind components crossing zero); absolute errors stay at FP32
rounding level; on real radiosonde-derived systems max rel = 5.6e-04."

## 7. Artifacts

| File | Description |
|---|---|
| `prep_igra2_tridi.py` | IGRA2 parser + sigma-grid interpolation + bulk-Ri K(z) + tridiagonal assembly |
| `prep_run.log` | Prep-script run log (acceptance counts, coefficient ranges) |
| `data/*-data-beg2025.txt.zip`, `data/*-data.txt` | Raw IGRA2 downloads (5 stations) + unzipped |
| `data/igra2-station-list.txt` | IGRA2 station metadata |
| `tridi_real.bin` | 19,660 real systems x 64 levels (cl/cm/cu/r1, float32) |
| `tridi_real_meta.json` | Machine-readable provenance/configuration |
| `ccpp_tridi1_realdata.cu` | Adapted harness (verbatim kernels, file loader) |
| `build.cmd` | Build script (same nvcc flags as original) |
| `ccpp_tridi1_realdata.exe` | Built binary (sm_120) |
| `run_realdata.log` | GPU run log (the results above) |
