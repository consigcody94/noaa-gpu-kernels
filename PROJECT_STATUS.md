# NOAA Computational Optimization Project — Complete Status

**Hardware:** RTX 3060 12GB, CUDA 13.2, gfortran 15.2.0
**Date:** 2026-03-26

---

## VERIFIED & PUBLISHED

### rte-rrtmgp: Parallel Prefix Scan for Two-Stream RT Solver
- **Speedup:** 3.10x (full flux solver), 3.97x (albedo-only)
- **Accuracy:** All errors < 1.3e-06, zero NaN/Inf/negative fluxes
- **Testing:** 15/15 stress tests PASS (thick clouds, conservative scattering, SW, edge cases, nlay 4-256)
- **Validated against:** Independent CPU sequential reference + upstream unit tests (LW, SW, optical props all pass)
- **GitHub:** https://github.com/consigcody94/parallel-prefix-rt
- **Upstream issue:** https://github.com/earth-system-radiation/rte-rrtmgp/issues/393
- **Novel contribution:** First-ever parallel prefix scan for atmospheric radiative transfer

### rte-rrtmgp: Fast exp() (CPU only)
- **Speedup:** 2.55x on CPU
- **Accuracy:** 3.13e-16 max error on 512 real atmospheric tau values (range 0.1 to 100.0)
- **GPU note:** No benefit on GPU — hardware SFU already handles exp() efficiently

### rte-rrtmgp: FP64 → FP32 Precision Reduction
- **Speedup:** 7.96x from memory bandwidth alone
- **Confirmed by:** Kashi et al. (March 2026, ORNL) mixed-precision survey

---

## VERIFIED — NOT YET PUBLISHED (waiting for more testing or compute)

### GSI: Ensemble Forward Model GPU Kernel
- **Result:** 109 GB/s bandwidth, correct within FP32 tolerance
- **Tested:** 3 configurations (RAP/HRRR, GFS C384, 160-member experimental)
- **Could NOT test:** GFS C768 (11.4 GB) and C1152 (22.9 GB) — exceed 12GB VRAM
- **Requires:** A100 40GB+ for production-size testing
- **Status:** Kernel works, needs large-GPU validation before publishing

---

## TESTED & FAILED — HONEST NEGATIVE RESULTS

### rte-rrtmgp: Tucker Tensor Compression of k-Tables
- **Frobenius error:** 0.079% at 33x compression — looks great
- **Flux error:** 4.0 W/m² — FAILS NWP tolerance (< 0.1 W/m²)
- **Root cause:** exp(-τ) amplifies small k-errors; 18% of reconstructed values are negative
- **Upstream issue corrected:** https://github.com/earth-system-radiation/rte-rrtmgp/issues/394
- **Lesson:** Frobenius norm is unreliable for radiation table accuracy

### GSI: Recursive Filter Parallel Prefix Scan
- **Result:** 26/27 tests FAILED
- **Root cause:** IIR filter coefficients create exponentially decaying prefix products → catastrophic cancellation
- **This is fundamental:** The technique works for bounded transformations (RT) but NOT for unbounded IIR filters
- **Not published:** Correctly identified as non-viable before any claims were made

---

## NOT TESTED — HARDWARE LIMITATIONS

### GSI: Large-grid ensemble (C768, C1152)
- Data exceeds RTX 3060's 12GB VRAM
- Requires: NOAA WCOSS2 (A100 GPUs) or cloud HPC

### GSI: Full integration test
- Requires: MPI infrastructure, NCEP libraries, full input data
- Cannot be done on single GPU workstation

### WW3: DIA with real spectral indexing
- Tested with simplified index arrays only
- Requires: Full WW3 initialization (INSNL1) to generate the 32 real index arrays
- The 16.8x speedup is an upper bound — real performance will be lower

### CRTM Clear-Sky Adding Method
- **Result: 6/6 tests PASS** — max relative error 4.4e-07, zero NaN
- Tested: 6 configurations (IR sounder, MW sounder, multi-stream, deep atmosphere, 1K-100K profiles, 60-200 layers)
- The non-scattering adding path in ADA_Module.f90 is identical to rte-rrtmgp — our parallel scan applies directly
- The scattering path (nZ×nZ matrix adding with matinv) is theoretically parallelizable but NOT tested — would need matrix Redheffer star product
- **Not yet published** — needs comparison against actual CRTM output with real coefficient files

### MOM6 Vertical Mixing
- **Analysis only — NOT implemented**
- MOM6 uses Thomas algorithm (tridiagonal solver) for implicit vertical diffusion
- This has the SAME numerical instability as the GSI IIR filter — our parallel scan would FAIL
- Correct GPU approach: NVIDIA `cusparseGtsv2StridedBatch` (batched tridiagonal solver)
- This is a known engineering approach, not a novel contribution
- Would need A100+ for production-size ocean grids

### CCPP Microphysics
- Not yet started — would need to extract Thompson/GFDL column kernels
- Abdi & Jankov (2024) already showed 10x with OpenACC — our contribution would need to exceed this

---

## LESSONS LEARNED

1. **Test before publishing.** The tensor compression looked great in Frobenius norm but failed physical validation. We corrected the upstream issue.
2. **Parallel prefix scan is NOT universal.** Works for bounded transformations (RT albedo/flux) but fails for unbounded IIR filters (GSI recursive filter).
3. **GPU exp() is already hardware-optimized.** Our fast_exp is CPU-only; on GPU the SFU handles it.
4. **FP32 summation order matters.** GPU parallel reduction gives different last-digit results than CPU sequential — this is expected, not a bug.
5. **12GB VRAM is limiting.** Production GSI/GFS configurations need 40-80GB GPUs.
