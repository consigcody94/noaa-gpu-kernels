# COARE 3.6 GPU Kernel — Real-Data Validation

Target: `ocean/coare/coare_flux.cu` (COARE 3.6 bulk air-sea flux, simplified subset)
Harness: `coare_flux_realdata.cu` (verbatim copy; only the synthetic generator
`gen_coare` + size-sweep `main` replaced by a binary-file loader and gold comparison —
kernel math and CPU reference untouched).
Date: 2026-06-10. GPU: RTX 5070 12GB (sm_120), CUDA 13.3, MSVC 2022 BuildTools host.

## Verdict

**PASS (strict).** On 2165 real ship observations, CPU-vs-GPU max relative error is
6.5e-07 (tau), 4.3e-07 (hsb), 1.3e-06 (hlb), 0 NaN — well inside the harness's strict
1e-3 criterion. This is *better* than the synthetic-data run, which only passed under
the "PASS (fast math)" waiver (tau max rel < 1e-1).

## Data provenance (real observations, no synthetic values anywhere in the input path)

- **Source**: official NOAA-PSL COARE algorithm repository,
  https://github.com/NOAA-PSL/COARE-algorithm (master branch), accessed **2026-06-10**.
- **Input file**: `Python/COARE3.6/test_36_data.txt` (914 KB)
  https://raw.githubusercontent.com/NOAA-PSL/COARE-algorithm/master/Python/COARE3.6/test_36_data.txt
- **Gold standard**: `Python/COARE3.6/test_36_output_withnowavesinput_withwarmlayer.txt`
  (2.4 MB) — the repo's published expected outputs of the official Python
  `coare36vnWarm_et.py` (no wave input, warm-layer + cool-skin enabled) on the same data.
- **Content**: **2165** ship observations (1 header + 2165 rows), tropical Atlantic,
  lat 13.14–15.86 N, lon 59.06–51.38 W, yearday 9.83–43.22 — the COARE 3.6 release
  validation cruise dataset (timing/location consistent with the RV Ronald H. Brown
  ATOMIC campaign, Jan–Feb 2020). Used unmodified, full record count, no subsetting.
- **Data quality**: 6 NaNs exist, all in `sigH` (significant wave height), a column the
  kernel does not use. Every used column is NaN-free (verified by `prep_realdata.py`).

## Input mapping (data columns → `COAREInput` struct)

| Kernel field | Data column | Real-data range |
|---|---|---|
| `u` (wind speed m/s) | `u` | 2.28 – 13.34 |
| `ta` (air temp °C) | `ta` | 22.92 – 26.68 |
| `rh` (%) | `rh` | 61.3 – 89.1 |
| `P` (mb) | `P` | 1012.1 – 1020.4 |
| `ts` (SST °C) | `tsnk` (sea-snake, ~0.05 m depth) | 26.06 – 27.79 |
| `sw_dn` (W/m²) | `sw_dn` (carried in struct; unused by kernel math) | 0 – 967 |
| `lw_dn` (W/m²) | `lw_dn` (carried in struct; unused by kernel math) | 367 – 440 |
| `zu` (m) | `zu` | 18.0 (constant) |
| `zt` (m) | `zt` | 17.0 (constant) |
| `zq` (m) | `zq` | 17.0 (constant) |

Real columns the kernel has no inputs for (documented, ignored): `jd, lat, lon, rain,
Ss, cp, sigH, tsg, ztsg`. The dataset's `zi` is 600 m for every record, which happens to
match the kernel's hard-coded gust scaling (`Bf * 600`), so no defaulting mismatch there.

Pipeline: `prep_realdata.py` parses both text files and writes
`coare_inputs.bin` (int32 n + n×10 float32, exact `COAREInput` layout) and
`coare_expected.bin` (int32 n + n×4 float32: usr, tau, hsb, hlb).

## Results (run log: `run_log.txt`)

### CPU-vs-GPU agreement on real data (authoritative)

| Output | max rel err | max abs err | criterion | result |
|---|---|---|---|---|
| tau (wind stress) | 6.48e-07 | 1.19e-07 N/m² | < 1e-3 | PASS |
| hsb (sensible heat) | 4.34e-07 | 1.14e-05 W/m² | < 1e-3 | PASS |
| hlb (latent heat) | 1.31e-06 | 3.36e-04 W/m² | < 1e-3 | PASS |
| NaN count | 0 | — | == 0 | PASS |

Status line from the binary's own criteria: **PASS** (strict tier, not the fast-math tier).

### Kernel (CPU and GPU identical to ~1e-6) vs official expected outputs (informative)

Mean official magnitudes on this cruise: tau ≈ 0.104 N/m², hsb ≈ 8.7 W/m², hlb ≈ 175 W/m².

| Output | mean bias (kernel − official) | mean abs err | max abs err |
|---|---|---|---|
| tau | +0.0028 N/m² (~3%) | 0.0049 | 0.0242 |
| hsb | +2.79 W/m² | 2.79 | 4.54 |
| hlb | +22.5 W/m² (~13%) | 22.5 | 30.9 |

These are *physics* differences, not numerical bugs: the kernel implements a simplified
COARE subset — bulk SST with **no cool-skin / warm-layer correction**, `zot = zoq`,
fixed-coefficient roughness (no wave/Charnock wind dependence), no Webb correction, no
rain heat flux. The official gold run applies cool-skin physics, which lowers the skin
temperature ~0.2–0.3 K below the sea-snake bulk reading; using the warmer bulk `ts`
systematically inflates qs → hlb (+13%) and dT → hsb (+2.8 W/m²), exactly the sign and
size seen. GPU and CPU show *identical* bias vs gold (agree with each other to 1e-6),
confirming the GPU port faithfully reproduces the reference math on real inputs.

### Timing (indicative only — GPU shared with sibling agents)

- 2165 real points: CPU 7.0 ms, GPU 0.028 ms.
- 1,000,230 points (the 2165 real records tiled ×462 — replication used for **timing
  only**, all accuracy numbers come from the unreplicated real records): 0.508 ms/launch,
  ~1.97 Gpts/s.

## Differences vs the synthetic-data run

1. **Stricter pass tier.** Synthetic run passed only as "PASS (fast math)" (tau max rel
   error up to ~1e-1 waived); on real data all three outputs meet the strict 1e-3 bar
   with ~1e-6 to spare. Cause: the synthetic generator sweeps u ∈ [1,21] m/s,
   ta ∈ [10,30] °C, ts−ta ∈ [−3,+3] K, hitting regimes (very light wind / strongly
   stable, near-zero-flux cancellation) where `__expf/__logf/__powf` intrinsic error is
   amplified; this trade-wind cruise (u 2.3–13.3 m/s, unstable ts>ta throughout) stays in
   the well-conditioned regime, where fast-math intrinsics are essentially exact.
2. **Gold-standard anchor.** Real data comes with official expected outputs, so the
   kernel's simplifications are now quantified (tau +3%, hlb +13% vs full COARE 3.6
   with skin physics) instead of unknown.
3. Measurement heights are 17–18 m here vs the synthetic 10/2/2 m, exercising the
   profile-correction terms over a different (taller) range — no issues observed.

## Artifacts (all in `U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\coare\`)

- `test_36_data.txt`, `test_36_output_withnowavesinput_withwarmlayer.txt`,
  `coare36_python_README.md` — downloaded official files
- `prep_realdata.py` — column mapping + binary writer (provenance header inside)
- `coare_inputs.bin`, `coare_expected.bin` — packed real data
- `coare_flux_realdata.cu` — adapted harness (originals untouched)
- `coare_flux_realdata.exe` — built `nvcc -O3 -arch=sm_120`
- `run_log.txt` — full run output
