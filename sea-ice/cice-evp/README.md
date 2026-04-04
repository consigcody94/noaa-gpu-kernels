# CICE EVP GPU Benchmark -- Real DMI Arctic Data

GPU CUDA port of the EVP (Elastic-Viscous-Plastic) sea ice dynamics solver from CICE v6.5.1, benchmarked against real Arctic ice conditions from the published Rasmussen et al. (2024) dataset.

## Data Source

- **Paper:** Rasmussen et al., "Refactoring the elastic-viscous-plastic solver from the sea ice model CICE v6.5.1 for improved performance", GMD 17, 6529-6544 (2024)
- **Standalone code:** https://doi.org/10.5281/zenodo.10782548
- **Input data:** https://doi.org/10.5281/zenodo.11248366
- **Domain:** DMI operational Arctic, 1 March 2020 (winter)
- **Active cells:** 631,387 T-cells, 608,124 U-cells, 660,613 total (navel)

## Results (RTX 3060, double precision)

### Baseline kernel (evp_dmi_benchmark.cu)

| ndte | CPU (single-core) | GPU | Speedup |
|------|------------------|-----|---------|
| 1 | 77 ms | 2.9 ms | 26.6x |
| 5 | 376 ms | 14.1 ms | 26.7x |
| 120 | 8,451 ms | 336 ms | 25.1x |
| 500 | 35,508 ms | 1,332 ms | 26.7x |
| 1000 | 74,736 ms | 2,673 ms | 28.0x |

### Optimized kernel (evp_dmi_optimized.cu)

Persistent fused kernel using cooperative groups grid sync, __launch_bounds__, and constant memory for EVP parameters.

| ndte | Baseline GPU | Optimized GPU | Improvement |
|------|-------------|--------------|-------------|
| 120 | 336 ms | 319 ms | 5.2% |
| 500 | 1,332 ms | 1,290 ms | 3.2% |
| 1000 | 2,673 ms | 2,573 ms | 3.7% |

### Comparison with Rasmussen et al. (2024)

| Platform | Speedup vs single-core baseline |
|----------|-------------------------------|
| AVX-512 (single core, 3rd Gen Xeon) | 5.1x |
| Full-node DDR (4th Gen Xeon, 112 cores) | 13x |
| **RTX 3060 GPU (this work)** | **26-29x** |
| Full-node HBM (Xeon Max, 112 cores) | 35x |

The GPU sits between CPU full-node DDR and HBM results. This is expected for a bandwidth-bound kernel (0.3 FLOP/byte) running on a consumer GPU (360 GB/s) vs HBM (1.6 TB/s peak).

## Validation

- ndte=1: max relative error ~1e-9 (PASS, double precision rounding)
- ndte=5: max relative error ~1e-7 (PASS, expected iterative divergence)
- Higher ndte: FP divergence from iterative feedback, same behavior as SIMD vs scalar in the paper

## Why 26-29x and not 462x

The original simplified kernel (noaa_multi_kernel.cu) reported 462x on synthetic data. The real EVP solver is fundamentally different:

1. **Irregular neighbor access** via ee/ne/se/nw/sw/sse index arrays defeats memory coalescing
2. **12-component stress tensor** with 8 str combinations increases memory traffic
3. **Bandwidth-bound at 0.3 FLOP/byte** as identified in the paper
4. **Register-only tricks don't apply** when the stencil requires global memory neighbor lookups

## Proposed Bug Fix: shared_mem_1d in CICE (PR #1062)

**Status: Proposed fix based on code analysis. Not yet tested in a full CICE simulation. Requires validation by the CICE team through their test suite and realistic coupled runs before merging.**

See `cice_evp1d_bugfix.patch` for the proposed fix. The `gather_dyn` subroutine in `ice_dyn_evp1d.F90` (lines 714-727) calls `gather_global_ext` without passing `spc_val=c0` for 11 stepu-related fields (cdn_ocn, aiu, uocn, vocn, waterxU, wateryU, forcexU, forceyU, umassdti, fmU, Tbu). When cells are not covered by any distributed block, they retain the default `spval_dbl = 1.0e30` instead of `0.0`. The stress, strength, uvel, and vvel fields in the same function already pass `c0` (lines 698-712, 728-729), so the 11 missing calls appear to be an oversight. This was likely safe before PR #1062 changed HaloUpdate boundary behavior, but now stale or uninitialized boundary values can propagate through the neighbor stencil during stepu_1d.

The fix adds `,c0` to those 11 calls to match the pattern used for all other fields in the same subroutine.

## Building

Requires CUDA Toolkit and a C++ compiler (MSVC on Windows, gcc on Linux).

```bash
# Baseline
nvcc -O3 -arch=sm_86 -o evp_dmi_benchmark evp_dmi_benchmark.cu

# Optimized (requires cooperative groups)
nvcc -O3 -arch=sm_86 --std=c++17 -rdc=true -o evp_dmi_optimized evp_dmi_optimized.cu -lcudadevrt
```

## Running

Download the Zenodo input data into the same directory:

```bash
curl -L -o input_double_1d_v1.bin "https://zenodo.org/api/records/11248366/files/input_double_1d_v1.bin/content"
curl -L -o input_integer_1d.bin "https://zenodo.org/api/records/11248366/files/input_integer_1d.bin/content"
curl -L -o input_logical_1d.bin "https://zenodo.org/api/records/11248366/files/input_logical_1d.bin/content"

./evp_dmi_benchmark 120    # 120 subcycles
./evp_dmi_optimized 1000   # 1000 subcycles
```

## Attribution

Author: Cody Churchwell, April 2026

This work was developed using a custom-configured Claude Code environment with specialized CUDA research skills, benchmarking pipelines, and GPU tooling. This is not a default Claude/Claude Code installation. The Co-Authored-By tag indicates AI-assisted development; the research methodology, hardware validation, and decision to benchmark against published data is the human work.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
