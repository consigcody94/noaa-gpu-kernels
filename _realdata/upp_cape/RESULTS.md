# UPP CAPE/CIN kernel — real-data validation (IGRA2 radiosondes)

Target: `post-processing/upp/upp_cape.cu` (NOAA-EMC UPP CALCAPE-style most-unstable CAPE/CIN, one thread per column)
Validation harness: `upp_cape_realdata.cu` (this directory) — CPU reference and GPU kernel **byte-identical** to the original; only the synthetic generator `gen_columns()` was replaced by a file loader.
GPU: RTX 5070 12 GB (sm_120), CUDA 13.3, driver 610.47. Date of run: 2026-06-10.

## Verdict

**PASS** on real radiosonde data by the binary's own criteria:

| run | columns | Max rel CAPE | Max rel CIN | Max abs CAPE | Max abs CIN | NaN | Match(<1 J/kg) | status line |
|---|---|---|---|---|---|---|---|---|
| real IGRA2 | 10,000 | 1.56e-03 | 1.35e-04 | 0.391 J/kg | 0.136 J/kg | 0 | **10000/10000** | PASS |
| real IGRA2 | 100,000 | 1.56e-03 | 1.35e-04 | 0.391 J/kg | 0.136 J/kg | 0 | **100000/100000** | PASS |
| synthetic baseline (same GPU, unmodified original) | 10k/100k/500k | 5.0–6.1e-06 | 0 | — | — | 0 | 100% | PASS |

Indicative timing only (GPU shared with sibling agents): real-data speedup 431x (10k) / 406x (100k); synthetic baseline on the same GPU 367–371x.

## Data provenance (real data only — nothing synthesized in the input path)

- Dataset: **Integrated Global Radiosonde Archive v2 (IGRA2)**, NOAA NCEI (public).
- Sounding files (`data-y2d`, period 2025-01-01 → present), accessed **2026-06-10**, from
  `https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/<ID>-data-beg2025.txt.zip`:
  | station ID | site | role | accepted/total 00Z+12Z soundings |
  |---|---|---|---|
  | USM00072357 | Norman / Max Westheimer, OK (OUN), 35.18N 97.44W, 344.9 m | Great Plains convective | 1008 / 1038 |
  | USM00072261 | Del Rio Intl, TX (DRT), 29.37N 100.92W, 314.0 m | subtropical high-CAPE | 982 / 1025 |
  | USM00072649 | Chanhassen, MN (MPX), 44.85N 93.56W, 290.2 m | midlatitude | 1026 / 1042 |
  | RQM00078526 | San Juan Intl, Puerto Rico (TJSJ), 18.43N 65.99W, 4.0 m | tropical | 1029 / 1038 |
- Station metadata: `.../doc/igra2-station-list.txt` (in `data/`).
- Plausibility cross-check: NCEI's own **IGRA2 derived parameters** (includes NCEI-computed CAPE/CIN per sounding),
  `.../access/derived-por/USM00072357-drvd.txt.zip` (127 MB), accessed 2026-06-10.
- Total download ≈ 150 MB (22 MB soundings + 127 MB derived cross-check).

**n records = 4,045 real soundings** (Jan 2025 – Jun 2026, 00Z/12Z). To reach the harness column counts they are **replicated by cycling** (`cols[c] = sounding[c % 4045]`): 2.5x for 10k columns, 24.7x for 100k. Every value in every column originates from a real observed sounding.

## Input mapping (IGRA2 → kernel `Column` struct) — `prep_igra2.py`

The kernel's logic (read first, unmodified) requires surface at index `nlev-1` (max pressure) and column top at index 0; the most-unstable search starts at `l = nlev-1` and breaks once `pmid[nlev-1] - pmid[l] > 30000 Pa`.

- `pmid[]`: uniform 1000-Pa grid from observed surface pressure up to ≤100 hPa (85–93 levels per sounding, < MAX_LEV=128).
- `T[]`: linear interpolation in ln(p) of observed IGRA2 temperatures.
- `Q[]`: specific humidity from observed dewpoint depression (Td = T − DPDP) using **the kernel's own Tetens formula** (es = 611.2·exp(17.67·tc/(tc+243.5)); q = 0.622e/(p−0.378e)); RH used when DPDP missing; q = 0 outside the span of valid moisture obs (typically the dry stratosphere).
- `zint[]`: observed geopotential heights interpolated in ln(p) to grid midpoints, interfaces = midpoints between adjacent levels; `zint[nlev]` = surface geopotential height; `zint[0]` extrapolated half a layer above the top level.
- Acceptance (reject, never fill): nominal 00Z/12Z, valid surface level (LVLTYP2=1) with PRESS/TEMP/GPH, ≥30 valid levels, profile reaching 100 hPa, strictly monotone heights. 97.3% of candidate soundings accepted; binary: `cape_input.bin` (5.8 MB), per-sounding metadata: `manifest.csv`.

## CPU-vs-GPU accuracy on real data (authoritative)

- Max |CAPE_gpu − CAPE_cpu| = **0.391 J/kg** (mean 0.041 J/kg over 4,045 unique soundings); max |CIN diff| = **0.136 J/kg**.
- Max relative CAPE error **1.56e-03**, CIN **1.35e-04**; 0 NaNs.
- Binary's strict criterion **Match(<1 J/kg) = 100%** of 100,000 columns; status logic prints **PASS** (max rel < 0.01).
- Per-sounding values: `cape_results.csv` (CPU & GPU CAPE/CIN for all 4,045 soundings).

## Differences vs the synthetic-data run

1. **Larger (but still passing) float divergence.** Synthetic max-rel error was ~6e-06; real data gives 1.56e-03 (max abs 0.391 J/kg). Real soundings drive the `__expf/__powf/__logf` fast-math intrinsics across much wider T/q ranges (inversions, dry layers, tropical moisture, −80 °C tropopauses) and longer lifting loops, so CPU/GPU rounding differences accumulate more — yet every column still matches within 1 J/kg.
2. **The real-data run exercises the kernel's intended code path for the first time.** The original `gen_columns()` builds columns with the **surface at index 0** (pmid[0]=P_sfc, decreasing with l), opposite the orientation the kernel's logic assumes. With synthetic data the 300-hPa break (`pmid[nlev-1]-pmid[l] > 30000`) can never fire (pmid[nlev-1] is the column top there) and the "lifting" loop actually integrates downward. CPU and GPU share the logic, so the synthetic test is self-consistent, but it is an algebraic agreement test, not a CAPE test. Real data was supplied in the kernel-assumed orientation (surface at `nlev-1`), so the most-unstable 300-hPa search, the upward moist-adiabatic lift, and the CIN accumulation genuinely run as designed — and still agree CPU-vs-GPU.
3. **CIN is actually exercised.** Synthetic CIN was essentially always 0 (max rel printed 0.0e+00). Real winter/stable soundings produce deep CIN (down to ≈ −2.4e4 J/kg when no LFC is found and the kernel accumulates all negative buoyancy), tropical soundings produce the classic near-zero CIN. GPU matched CPU to ≤0.136 J/kg throughout.

## Meteorological plausibility (documented, not tuned)

Cross-check vs NCEI's derived CAPE/CIN for the same Norman OK soundings (475 matched, `crosscheck_derived.py`, `logs/crosscheck_oun.csv`):

- Seasonality and rank order are right: Spearman r = 0.60 vs NCEI CAPE (CAPE>50 J/kg subset); at a 500 J/kg threshold the kernel produced **zero misses** (FN=0; TP=257, FP=133, TN=85). Station climatology is sensible: San Juan (tropical) median 3,671 J/kg with tiny CIN; Chanhassen (MN) median 24 J/kg with strong summer tail; Del Rio > Norman > Chanhassen, as expected.
- **Magnitudes are systematically high — a kernel simplification, not a data issue.** Median kernel/NCEI CAPE ratio ≈ **4.45**; e.g. OUN 2025-08-09 00Z: kernel 19,623 J/kg vs NCEI 4,453 J/kg (a genuinely extreme-CAPE day); OUN 2025-07-31 00Z: 20,938 vs 1,762. Cause (visible in the unmodified source): the parcel is lifted **moist-adiabatically from its origin level and treated as saturated from the start** (`Tv_parcel` uses `qsat(Tp,p)`; there is no dry ascent to the LCL — `Q_parcel`/`P_parcel` are set but never used, which the compiler also flags). A saturated-from-origin parcel is far too warm aloft, inflating CAPE several-fold. Winter soundings correctly give ≈0 CAPE (e.g. 2025-01-05 00Z: kernel 3 J/kg, NCEI 0).
- Per the rules, kernel math was left untouched; the bias is reported, not corrected.

## Artifacts (this directory)

- `prep_igra2.py` — IGRA2 parser/grid prep (provenance + mapping documented in header)
- `cape_input.bin`, `manifest.csv` — 4,045-sounding kernel input + metadata
- `upp_cape_realdata.cu`, `upp_cape_realdata.exe` — adapted harness (loader only; kernel math identical)
- `upp_cape_synthetic_baseline.exe` — unmodified original compiled for same-GPU baseline
- `cape_results.csv` — per-sounding CPU/GPU CAPE & CIN
- `crosscheck_derived.py`, `logs/crosscheck_oun.csv`, `logs/crosscheck.log` — NCEI derived-CAPE comparison
- `logs/run_realdata.log`, `logs/run_synthetic_baseline.log`, `logs/prep.log`
- `data/` — original downloaded IGRA2 archives (sounding zips, station list, OUN derived-por zip)
