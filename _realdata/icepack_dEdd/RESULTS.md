# Icepack delta-Eddington shortwave kernel — REAL DATA validation

**Target:** Icepack delta-Eddington section of `noaa_multi_kernel.cu` (consigcody94/noaa-gpu-kernels)
**Harness:** `icepack_de_realdata.cu` (this directory) — kernel `kernel_dEdd`, CPU reference `cpu_dEdd`,
`struct IceColumn`, and `compare()` copied **verbatim** from `../../noaa_multi_kernel.cu` (diff-verified
identical); the only functional change is replacing the synthetic generator `gen_ice()` with a binary
file loader `load_ice()`.
**Date:** 2026-06-10. GPU: RTX 5070 (sm_120), CUDA 13.3, MSVC 2022 BuildTools host, `-O3 -arch=sm_120`.

## Verdict

**PASS (strict)** on 84,921 real MOSAiC ice columns — by the binary's own criteria
(0 NaN, max rel < 1e-4 → "PASS", not merely "PASS (fast math)").

| Output | Max abs err (W/m²) | Max rel err | NaN | records > 1e-4 rel | Status |
|---|---|---|---|---|---|
| absorbed (the original harness's comparison) | 1.22e-04 | **9.48e-07** | 0 | 0 / 84,921 | **PASS** |
| transmitted (additional check, same compare()) | 1.07e-04 | 8.40e-06 | 0 | 0 / 84,921 | **PASS** |

Timing (indicative only — GPU possibly shared with sibling agents; CPU is Ryzen 7 5700G, 1 ms timer):
- 84,921 real columns: CPU 10.0 ms, GPU 0.013 ms → ~792x
- 679,368 columns (real set tiled ×8, replication documented below, timing only): CPU 77.0 ms, GPU 0.018 ms → ~4348x

Full output: `run_log_realdata.txt`.

## Data provenance (all public, authoritative; no fabricated or random values in the input path)

1. **Ice/snow state (snow depth + co-located sea-ice thickness), per point**
   Itkin, P. et al. (2021): *Magnaprobe snow and melt pond depth measurements from the
   2019-2020 MOSAiC expedition*. PANGAEA, https://doi.org/10.1594/PANGAEA.937781 (CC-BY-4.0).
   - File downloaded: `Magnaprobe.zip` (20.1 MB) via
     `https://download.pangaea.de/dataset/937781/files/Magnaprobe.zip` — accessed 2026-06-10.
   - Subset used: the 157 Level-3 `magna+gem2-transect-*.csv` files (Magnaprobe snow depth merged
     with GEM-2 electromagnetic-induction total thickness; the "Ice Thickness" columns are
     **total thickness minus co-located snow depth**, i.e., actual sea-ice thickness, per the dataset
     abstract and Itkin et al. 2023, *Elementa* 11(1):00048).
   - Coverage: MOSAiC Central Observatory drift, Arctic Ocean, 2019-10-24 → 2020-09-30,
     79.4–89.1 °N. Ice-thickness channel: "Ice Thickness 18kHz ip (m)" preferred
     (fallbacks "f18325Hz_hcp_i", "18kHz q" for files with alternate headers; per-file channel
     recorded in `prep_summary.txt` and per-record in `ice_columns_real.csv`).

2. **Downwelling shortwave (→ `swdn`), same floe/expedition**
   *Merged Data Files (MDF) for the MOSAiC Central Observatory, version 2*. NSF Arctic Data Center,
   https://doi.org/10.18739/A2WD3Q35Z — accessed 2026-06-10. This is the exact dataset the
   CICE-Consortium points to for Icepack's MOSAiC forcing
   (https://github.com/CICE-Consortium/Icepack/wiki/Icepack-Input-Data).
   - `MOSAiC_atm_drift1_precip_MDF_20191015_20200731.nc` (22.4 MB): variable `rsds`
     (downward surface shortwave, W/m², 1-min cadence) — Icepack's `fsw` input. Used for all
     records ≤ 2020-07-31.
   - `mosshipradS1_b1_subset_20191015_20200918.nc` (11.8 MB): `spn1_total_corr`
     (tilt-corrected total SW, shaded SPN1 radiometer aboard Polarstern, 1-min) — used only for
     records after drift1 ends (2020-07-31 → 2020-09-18).
   - Nearest-time match within ±15 min, NaN samples excluded; records without a valid match dropped.

3. **Icepack consortium forcing tarball** (Zenodo, https://doi.org/10.5281/zenodo.3728287,
   `Icepack_data-20200326.tar.gz`, 0.46 MB) was downloaded and inspected (CFSv2/ISPOL/NICE/SHEBA);
   it predates the MOSAiC forcing and was **not** used in the final input path — the ADC MDF above
   is the consortium's MOSAiC-era forcing source.

Total downloads ≈ 56 MB (< 500 MB budget).

## Input → kernel-parameter mapping (`prep_icepack_realdata.py`)

| `IceColumn` field | Source | Notes |
|---|---|---|
| `snow_depth` (m) | Magnaprobe "Snow Depth (m)" | measured, per point |
| `ice_thickness` (m) | GEM-2 "Ice Thickness 18kHz ip (m)" | measured total minus snow, per point |
| `coszen` | computed from each record's UTC timestamp + lat/lon | NOAA GMD solar-position formulas (deterministic; same role as Icepack's internal `compute_coszen`). Cross-check: the 50,969 records with coszen > 0 are exactly the records with measured fsw > 1 W/m² |
| `swdn[0]` (vis) | 0.52 × fsw | Icepack `frcvdr+frcvdf = 0.28+0.24` (`configuration/driver/icedrv_forcing.F90`, verified against main on 2026-06-10) |
| `swdn[1]` (NIR-1) | 0.31 × fsw | Icepack `frcidr` |
| `swdn[2]` (NIR-2) | 0.17 × fsw | Icepack `frcidf` |
| `nslyr` | 1 | Icepack default column configuration |
| `nilyr` | 7 | Icepack default column configuration |
| `snow_grain_r` (µm) | 500.0 | Icepack `rsnw_nonmelt` (`columnphysics/icepack_shortwave.F90`); **not used by either the CPU reference or the GPU kernel math** |

fsw = max(measured value, 0) (nighttime radiometer noise can be slightly negative — clamp documented).

## Record counts and QC (documented drops only; values never altered)

| Stage | Count |
|---|---|
| Rows in 157 L3 transect CSVs | 92,279 |
| Dropped: unparseable / non-finite | 298 |
| Dropped: hi ≤ 0 or hs < 0 (no-ice/invalid retrievals) | 5,862 |
| Dropped: gross outliers (hs > 3 m or hi > 15 m) | 0 |
| Dropped: no valid SW sample within ±15 min (e.g., transects after 2020-09-18) | 1,198 |
| **Kept (real records, before any replication)** | **84,921** |

Resulting real-state ranges: snow 0–1.74 m (mean 0.163 m); ice 0.0003–12.92 m (mean 2.25 m, ridges
included); coszen −0.434–0.521 (polar night through Arctic summer); per-band swdn up to 356 W/m²
(total fsw up to ~684 W/m²). The ×8 replication case (679,368 columns) tiles the same real records
purely for large-size timing; accuracy is identical because inputs repeat exactly.

## Behavior vs. the synthetic-data run

- **Accuracy:** synthetic run on this same GPU (BLACKWELL_RESULTS.md): max rel 5.62e-07 on absorbed,
  PASS. Real data: 9.48e-07 — same order of magnitude, still strict PASS with zero records above
  the 1e-4 threshold. The slight increase is expected: real columns span far more extreme optical
  depths than the synthetic generator (synthetic: snow 0–0.1 m, ice 0.5–3.5 m, coszen 0.1–0.9;
  real: snow up to 1.74 m, ice up to 12.9 m, and 40% of records in polar night where coszen is
  clamped to 0.01, making τ/µ0 huge so transmittance underflows identically on both paths).
- **Transmitted** (not compared by the original harness; added here using the same verbatim
  `compare()`): max rel 8.40e-06 — larger than absorbed because transmitted values under thick
  snow/ice are tiny, but still 100× inside the strict criterion.
- **No NaN/Inf** on real data, including the polar-night fsw = 0 records and near-zero ice
  thicknesses (3 mm) — the `fmaxf` clamps behave identically on CPU and GPU.
- **Speedup scaling** matches the synthetic pattern (81x @10k → 3042x @500k synthetic;
  792x @85k → 4348x @679k real) — the kernel is launch-latency-bound at small n. Timing is
  indicative only (shared GPU; CPU clock at 1 ms resolution).
- The CPU reference computes per-layer `expf` products while the GPU collapses to a single
  `__expf(-Στ/µ0)`; on real data this algebraic identity holds to FP32 rounding exactly as it
  did synthetically — real-world optical depths do not expose any new divergence.

## Files in this directory

- `prep_icepack_realdata.py` — data-prep (provenance, QC, mapping; run with
  `U:\AI\_noaa-research\hydrotools\.venv` python: numpy + netCDF4)
- `ice_columns_real.bin` — 84,921 records, `struct IceColumn` layout (int32 n header + 36 B/record)
- `ice_columns_real.csv` — per-record audit (state, forcing, timestamps, lat/lon, SW source,
  match distance, thickness channel, source file)
- `prep_summary.txt` — QC counts and per-file channel/record breakdown
- `icepack_de_realdata.cu` / `icepack_de_realdata.exe` — adapted harness (kernel math unchanged)
- `run_log_realdata.txt` — full run output
- `data/` — raw downloads (Magnaprobe.zip + extracted CSVs, two MDF netCDFs, Icepack Zenodo tarball,
  PANGAEA/Zenodo metadata snapshots)

Build command:

```
vcvars64.bat && nvcc -O3 -arch=sm_120 -o icepack_de_realdata.exe icepack_de_realdata.cu
```

Run: `icepack_de_realdata.exe ice_columns_real.bin 8`
