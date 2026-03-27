# GSI GPU Optimization — Honest Results

## Hardware: RTX 3060 12GB (CUDA 13.2)

---

## 1. Ensemble Forward Model (`ensemble_forward_model`)

**Source:** `hybrid_ensemble_isotropic.F90` lines 1977-2000

**What it does:** Weighted sum of 30-160 ensemble member perturbations at every 3D grid point. This is 30-50% of GSI wall-time in hybrid EnVar mode.

**Result: WORKS — trivially parallel, memory-bandwidth bound**

### Tested Configurations

| Config | Grid | Ens Members | Data Size | GPU Time | Bandwidth | Status |
|---|---|---|---|---|---|---|
| RAP/HRRR | 192×192×64 | 30 | 540 MB | 5.20 ms | 109 GB/s | **PASS** |
| GFS C384 | 384×192×64 | 80 | 2,880 MB | 27.8 ms | 109 GB/s | **PASS** |
| Experimental | 192×192×64 | 160 | 2,880 MB | 27.8 ms | 109 GB/s | **PASS** |

### Could NOT Test (insufficient VRAM)

| Config | Grid | Ens Members | Data Size | Requires |
|---|---|---|---|---|
| GFS C768 | 384×384×127 | 80 | **11.4 GB** | A100 40GB+ |
| GFS C1152 | 768×384×127 | 80 | **22.9 GB** | A100 80GB or H100 |

These large configurations are what NOAA actually runs on WCOSS2. They would need testing on an A100/H100 to get production-relevant benchmarks.

### Accuracy

- Max absolute error: 3.8e-06 to 1.5e-05 (expected FP32 summation order difference)
- Zero NaN, zero Inf
- Errors are from different summation order (GPU parallel vs CPU sequential), not algorithmic error
- Expected theoretical error: sqrt(n_ens) × machine_epsilon × value_range ≈ 2e-06 for n_ens=80

### Notes

- Kernel is **memory-bandwidth bound** at 109 GB/s (30% of RTX 3060 peak ~360 GB/s)
- On WCOSS2 A100 (2 TB/s HBM bandwidth): expect ~6x faster than our 3060 results
- No algorithmic change needed — this is a straightforward data-parallel kernel
- The CPU code already has `!$omp parallel do` but it's OFF by default in GSI builds

---

## 2. Recursive Anisotropic Filter (`one_color4`)

**Source:** `raflib.f90` lines 2224-2357

**What it does:** Forward-backward IIR (Infinite Impulse Response) recursive filter applied to gathered strings of grid points. This is 20-40% of GSI wall-time in 3DVAR/4DVAR mode.

**Result: FAILED — parallel prefix scan has numerical instability for IIR filters**

### Why It Fails

The IIR filter recurrence `y[i] = x[i] - a[i]*y[i-1]` with coefficients `|a[i]| ∈ [0.1, 0.6]` creates prefix products that decay exponentially:

```
a₁ × a₂ × ... × a₁₂₈ ≈ 0.3^128 ≈ 1e-67
```

This causes catastrophic cancellation in the affine tuple accumulator during the parallel prefix scan. The intermediate values underflow to zero, losing all information about the input signal.

**This is fundamentally different from the RT adding method**, where:
- The Möbius transformation keeps albedo bounded in [0, 1]
- The matrix condition numbers stay bounded
- No exponential decay in the prefix products

### Test Results

| Config | String Length | Max Rel Error | Status |
|---|---|---|---|
| 1000 strings × 64 pts | 64 | 7.3e-04 | Marginal |
| 1000 strings × 128 pts | 128 | 1.2e-03 | **FAIL** |
| 1000 strings × 256 pts | 256 | 4.0e-03 | **FAIL** |
| 8000 strings × 256 pts | 256 | 3.4e-02 | **FAIL** |

**26 out of 27 test configurations failed.**

### Potential Fixes (NOT implemented, would need further research)

1. **Blocked scan**: Sequential within blocks of 16-32 elements, scan across blocks
2. **Double precision accumulator**: FP64 for scan intermediates, FP32 for final values
3. **Renormalization**: Periodically rescale affine tuples to prevent underflow
4. **Completely different approach**: Frequency-domain parallelization instead of time-domain scan

### Honest Assessment

The parallel prefix scan technique that works for radiative transfer **does NOT generalize** to arbitrary IIR filters. The numerical conditioning is fundamentally different. This is a real limitation, not a bug.

---

## 3. What We Could NOT Test At All

### Multigrid Filter (`filtering_fast_bkg` in `mg_filtering.f90`)
- Same IIR structure as `one_color4` — would likely have the same numerical instability
- Not tested because the IIR scan failed first

### Normal Mode Constraint (`rtlnmc_version3.f90`)
- 5-point stencil (`forward_op`) — trivially parallel, would work on GPU
- ADI tridiagonal solver (`relax`) — would need batched cuSPARSE, not tested
- Lower priority (called less frequently than filter/ensemble)

### Full GSI Integration Test
- Requires MPI infrastructure, NCEP libraries, and full input data
- Cannot be done on a single GPU workstation
- Would need WCOSS2 or NOAA cloud environment

---

## Summary

| Component | % of GSI | GPU Viable? | Tested? | Result |
|---|---|---|---|---|
| Ensemble forward model | 30-50% | **YES** | ✅ 3 configs | PASS (109 GB/s) |
| Recursive filter | 20-40% | **NO** (scan unstable) | ✅ 27 configs | 26 FAIL |
| Multigrid filter | 15-30% | Probably NO | ❌ Not tested | — |
| Normal mode constraint | 5-10% | Likely YES | ❌ Not tested | — |
| QC module | 5% | NO (branch-heavy) | ❌ Not tested | — |

The ensemble forward model is the clear GPU win. The recursive filter needs a fundamentally different parallelization approach than what we used for rte-rrtmgp.
