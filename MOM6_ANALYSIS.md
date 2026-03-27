# MOM6 GPU Analysis

## Finding: Tridiagonal Solver — Same Issue as GSI IIR Filter

MOM6's vertical mixing uses `solve_tridiagonal_system` (Thomas algorithm) in `regrid_solvers.F90`. This is a forward-backward sweep:

```fortran
! Forward: X(k) = (R(k) - Al(k)*X(k-1)) * I_pivot
! Backward: X(k) = X(k) - c1(K)*X(k+1)
```

This has the **same numerical structure** as the GSI recursive filter — an affine recurrence where the prefix product decays exponentially. Our parallel prefix scan would fail here for the same reason (26/27 GSI tests failed).

## Correct Approach for MOM6

**Use NVIDIA's `cusparseGtsv2StridedBatch`** — a batched tridiagonal solver optimized for GPU. This solves thousands of independent tridiagonal systems (one per ocean column) in parallel without the numerical instability of our custom scan.

This is a library call, not a custom kernel. It would work but is a straightforward engineering task rather than a novel algorithmic contribution.

## What Our Parallel Scan CAN Do in MOM6

The parallel prefix scan technique works for **bounded transformations** (albedo, transmissivity). In MOM6, the closest analog would be:

- Radiative transfer through ocean layers (light attenuation) — bounded, would work
- Vertical coordinate remapping with PPM — bounded, would work
- Actual mixing/diffusion — tridiagonal, needs cuSPARSE

## Not Tested

We did not implement or benchmark any MOM6 kernels because:
1. The primary bottleneck (tridiagonal solver) is not suited for our technique
2. The correct GPU approach (batched cuSPARSE) is well-known and not novel
3. The secondary opportunities (remapping, light attenuation) are smaller targets

## Hardware Limitation

MOM6 ocean grids at operational resolution (0.25° or finer) would likely exceed 12GB VRAM for the full state vector. Testing would need an A100 or H100.
