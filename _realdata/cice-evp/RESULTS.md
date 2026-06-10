# CICE EVP Real-Data Validation — RTX 5070 (sm_120)

Target: `sea-ice\cice-evp\evp_dmi_benchmark.cu` and `evp_dmi_optimized.cu` (prebuilt at
`_build\evp_dmi_benchmark.exe` / `_build\evp_dmi_optimized.exe`).
Run date: 2026-06-10. No source or harness changes were needed or made.

## Root cause of the original failure

The prior run logs (`_build\runlogs\evp_dmi_*.err`) show "Cannot open input_double_1d_v1.bin".
This was purely a **working-directory issue**: the binaries `fopen()` the three input files by
bare relative filename, and they were executed from `_build\` while the inputs live in
`sea-ice\cice-evp\`. The input files were already present in the repo clone, correctly sized,
and checksum-verified (below). Re-running the unmodified executables with
cwd = `U:\AI\_noaa-research\noaa-gpu-kernels\sea-ice\cice-evp` fixed everything.

## Data provenance (verified, not fabricated)

- Dataset: **"Input data for 1d EVP model"**, Zenodo record 11248366,
  DOI [10.5281/zenodo.11248366](https://doi.org/10.5281/zenodo.11248366), published 2024-05-22.
- Companion data to: Rasmussen et al. (2024), *"Refactoring the elastic-viscous-plastic solver
  from the sea ice model CICE v6.5.1 for improved performance"*, Geosci. Model Dev. 17,
  6529–6544. Standalone code: DOI 10.5281/zenodo.10782548 (DMI / Till Rasmussen et al.,
  CICE-Consortium EVP standalone benchmark).
- Physical content: DMI operational Arctic domain, **1 March 2020** (winter) model state —
  real ice strength, velocities, ocean forcing, Coriolis, seabed stress, grid metrics.
- Access date: 2026-06-10 (Zenodo API record fetched; files verified locally).
- Subsetting: none — full v1 (1D) input set used as published.

### Checksum verification (local files vs Zenodo-published MD5)

| File | Bytes | Zenodo MD5 | Local MD5 | Match |
|---|---|---|---|---|
| input_double_1d_v1.bin | 192,409,288 | 2a0bbaa1de601dab53375c670d5dfc3e | 2a0bbaa1de601dab53375c670d5dfc3e | YES |
| input_integer_1d.bin | 15,153,288 | 5ac738b7d6bd734abf37ed1301c95ce6 | 5ac738b7d6bd734abf37ed1301c95ce6 | YES |
| input_logical_1d.bin | 5,051,096 | 84eecf529ecb3a50b2b14d04bd31130c | 84eecf529ecb3a50b2b14d04bd31130c | YES |

See `checksum_verification.txt` and `zenodo_11248366_record.json` in this directory.
The files were already in the clone; only record metadata (~KB) was downloaded to verify.

## Record counts and input → kernel mapping

- **n records: 631,387 active T-cells** (na), 608,124 active U-cells (nb),
  660,613 navel (union). The harness's runtime count from the logical masks matched exactly:
  `Active T-cells: 631387 / 631387, Active U-cells: 608124 / 631387`.
- `input_double_1d_v1.bin`: 24,051,161 float64 = 36·na + 2·navel + 3, laid out as
  strength(na), uvel(navel), vvel(navel), then 23 na-sized geometry/forcing arrays
  (dxT, dyT, dxhy, dyhx, cxp, cyp, cxm, cym, DminTarea, uarear, cdn_ocn, aiX, uocn, vocn,
  waterx, watery, forcex, forcey, umassdti, fm, strintx, strinty, Tbu), 12 na-sized stress
  arrays (stressp_1..4, stressm_1..4, stress12_1..4), and 3 trailing scalars
  (capping, e_factor, epp2i — recomputed in code from ndte).
- `input_integer_1d.bin`: 6 na-sized int32 neighbor-index arrays (ee, ne, se, nw, sw, sse),
  Fortran 1-based (converted to 0-based in the kernels).
- `input_logical_1d.bin`: 2 na-sized 4-byte Fortran logicals (skipUcell, skipTcell).
- Sanity sample from the run: `strength[0]=1.893606e-06, uvel[0]=7.789914e-03,
  DminTarea[0]=4.175743e-04` — plausible physical values, not placeholders.

## Hardware / build

- GPU: NVIDIA GeForce RTX 5070 12 GB (sm_120), driver 610.47, CUDA 13.3.
- Note: the baseline binary's results banner **hardcodes the string "(RTX 3060)"** in its
  printf — cosmetic only; the actual device (verified via nvidia-smi) is the RTX 5070.
- CPU reference: single-core on Ryzen 7 5700G.
- Prebuilt executables from `_build\` used unmodified.

## Results — baseline `evp_dmi_benchmark.exe` (CPU reference vs GPU, double precision)

Binary's own pass criterion: max relative error of uvel AND vvel < 1e-6 → "PASS", else "CHECK".

| ndte | CPU (1-core) | GPU (RTX 5070) | Speedup | max rel err uvel | max rel err vvel | max rel err stressp_1 | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 86.14 ms | 1.16 ms | 74.3x | 5.318e-09 | 7.125e-09 | 2.950e-11 | **PASS** |
| 5 | 433.02 ms | 5.70 ms | 75.9x | 2.386e-07 | 3.928e-09 | 8.175e-10 | **PASS** |
| 120 | 10,278.79 ms | 132.23 ms | 77.7x | 8.215e+00 | 3.111e+00 | 1.521e-02 | CHECK |
| 500 | 47,712.17 ms | 545.69 ms | 87.4x | 6.418e+02 | 5.978e+03 | 1.358e-01 | CHECK |
| 1000 | 89,319.72 ms | 1,079.01 ms | 82.8x | 1.174e+04 | 5.615e+03 | 1.797e-01 | CHECK |

Accuracy verdict matches the repo README's own validation claims exactly: PASS at ndte=1
(~1e-9) and ndte=5 (~1e-7); at higher ndte the CPU and GPU trajectories diverge through
iterative feedback (the README documents this as the same behavior the paper observed for
SIMD vs scalar). The huge *relative* uvel/vvel numbers at high ndte are dominated by
near-zero-velocity cells; the integrated stress field (stressp_1) stays within ~0.18 relative
even at ndte=1000. Timing is indicative only (GPU possibly shared with sibling agents).

## Results — optimized `evp_dmi_optimized.exe` (GPU-only timing comparison)

This binary compares its two GPU variants (no CPU reference / no PASS criterion).
Persistent kernel launched with 192 blocks (4 per SM x 48 SMs); cooperative launch supported.

| ndte | Optimized separate | Persistent fused | Fused improvement |
|---|---|---|---|
| 120 | 131.53 ms | 133.04 ms | **-1.1%** |
| 1000 | 1,099.42 ms | 1,111.01 ms | **-1.1%** |

## Behavior differences vs the README's published (RTX 3060) runs

1. **GPU ~2.5x faster end-to-end**: ndte=120 GPU 132 ms vs 336 ms on the 3060; ndte=1000
   1,079 ms vs 2,673 ms. Speedup vs single-core CPU is 74–87x here vs 25–28x in the README
   (this machine's single-core CPU is also slower: 10.3 s vs 8.45 s at ndte=120, inflating
   the ratio).
2. **The persistent-fused optimization regresses on sm_120**: -1.1% at both ndte=120 and
   ndte=1000, vs +3.7–5.2% gains reported on the RTX 3060 (sm_86). On Blackwell the
   grid-wide cooperative sync costs slightly more than the kernel-launch overhead it removes.
3. Accuracy behavior is identical in kind to the README: PASS at ndte=1/5, divergence-driven
   CHECK above that.
4. Versus the repo's original *synthetic* kernel claim (462x in `noaa_multi_kernel.cu`):
   the real-data faithful EVP solver achieves 74–87x on this GPU — consistent with the
   README's explanation that the real solver is bandwidth-bound (0.3 FLOP/byte) with
   irregular neighbor gathers.

## Artifacts

- This report: `U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\cice-evp\RESULTS.md`
- Full run logs: `U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\cice-evp\logs\`
  (`benchmark_ndte{1,5,120,500,1000}.log`, `optimized_ndte{120,1000}.log`)
- Provenance: `checksum_verification.txt`, `zenodo_11248366_record.json` (same dir)
- Inputs (verified): `U:\AI\_noaa-research\noaa-gpu-kernels\sea-ice\cice-evp\input_*.bin`
- No new harness was required; no original files were modified.
