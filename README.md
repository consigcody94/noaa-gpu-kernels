# NOAA GPU Kernels

GPU-accelerated computational kernels for NOAA weather, ocean, wave, and water prediction models. All benchmarked on RTX 3060 12GB with accuracy validated against CPU reference implementations.

## Results Summary

| # | Model | Kernel | Speedup | Max Error | Status | Upstream Issue |
|---|-------|--------|---------|-----------|--------|----------------|
| 1 | rte-rrtmgp | Parallel prefix scan (adding method) | **3.10x** | 15/15 stress tests pass | Verified | [#393](https://github.com/earth-system-radiation/rte-rrtmgp/issues/393) |
| 2 | CRTM | Clear-sky adding scan | 6/6 pass | < 4.4e-07 | Verified | [#298](https://github.com/JCSDA/CRTMv3/issues/298) |
| 3 | GSI | Ensemble forward model | 109 GB/s | 3/3 configs pass | Verified | [#994](https://github.com/NOAA-EMC/GSI/issues/994) |
| 4 | WW3 | DIA four-wave interaction | up to 16.8x | simplified indexing | Caveated | [#1583](https://github.com/NOAA-EMC/WW3/issues/1583) |
| 5 | t-route | MC reach-parallel routing | **92x** | FP64 verified | Verified | [#874](https://github.com/NOAA-OWP/t-route/issues/874) |
| 6 | t-route | Diffusive wave tridiag | **10x** | 2.48e-07 | Verified | (in #874) |
| 7 | CFE | Nash cascade | 2x | bit-identical | Verified | [#169](https://github.com/NOAA-OWP/cfe/issues/169) |
| 8 | NOAH-MP | Tridiag soil water solver | **11.2x** | residual < 4e-7 | Verified | [#128](https://github.com/NOAA-OWP/noah-owp-modular/issues/128) |
| 9 | TOPMODEL | Runoff generation | **31.3x** | 5.87e-07 | Verified | [#60](https://github.com/NOAA-OWP/topmodel/issues/60) |
| 10 | PET | Penman-Monteith evapotranspiration | **34.9x** | 1.44e-04 | Verified | [#57](https://github.com/NOAA-OWP/evapotranspiration/issues/57) |
| 11 | Snow17 | Snow accumulation/melt | **4.1x** | 2.10e-05 | Verified | [#63](https://github.com/NOAA-OWP/snow17/issues/63) |
| 12 | LGAR | Green-Ampt infiltration | **8.9x** | bit-identical | Verified | [#51](https://github.com/NOAA-OWP/LGAR-C/issues/51) |
| 13 | CCPP | PBL tridiag solver (tridi1) | **7.5-11.6x** | FP32 rounding | Verified | Pending |
| 14 | Icepack | Delta-Eddington shortwave | **291x** | 5.62e-07 | Verified | Pending |
| 15 | CICE | EVP ice dynamics | **462x** | 9.40e-05 | Verified | Pending |
| 16 | MOSART | Kinematic wave routing | **5.9-6.5x** | fast math | Verified | Pending |

## Documented Failures (included for transparency)

- **GSI recursive filter**: Parallel prefix scan fails for IIR filters (26/27 configurations fail due to exponential decay in prefix products)
- **Tensor compression of RRTMGP k-tables**: Frobenius error does not predict flux accuracy. 33x compression fails at 4+ W/m2 broadband error
- **t-route MC prefix scan within-reach**: Operator splitting introduces 1000%+ error due to nonlinear Muskingum-Cunge coefficient coupling

## Repository Structure

```
radiation/
  rte-rrtmgp/     # Parallel prefix scan for two-stream adding method
  crtm/            # Clear-sky satellite RT scan
data-assimilation/
  gsi/             # Ensemble forward model + recursive filter analysis
waves/
  ww3/             # DIA four-wave interaction
water-prediction/
  t-route/         # Muskingum-Cunge + diffusive wave routing
  cfe/             # Nash cascade (in owp_batched_kernels.cu)
  noah-mp/         # Tridiag soil water (in owp_batched_kernels.cu)
  topmodel/        # Runoff generation (in owp_extended_kernels.cu)
  evapotranspiration/ # PET (in owp_extended_kernels.cu)
  snow17/          # Snow model (in owp_snow17_lgar.cu)
  lgar/            # Infiltration (in owp_snow17_lgar.cu)
sea-ice/
  icepack/         # Delta-Eddington shortwave (in noaa_multi_kernel.cu)
  cice-evp/        # EVP dynamics (in noaa_multi_kernel.cu)
atmosphere/
  ccpp-pbl/        # PBL tridiag solver (in noaa_multi_kernel.cu)
river-routing/
  mosart/          # Kinematic wave (in noaa_multi_kernel.cu)
```

## Building

Requires NVIDIA CUDA Toolkit and a GPU with compute capability 8.6+ (RTX 3060 or newer).

```bash
# Example: build t-route MC routing kernel
nvcc -O3 -arch=sm_86 -o troute_mc water-prediction/t-route/troute_mc_final.cu

# Example: build OWP batched kernels (CFE + NOAH-MP)
nvcc -O3 -arch=sm_86 -o owp_batched water-prediction/owp_batched_kernels.cu

# Example: build multi-model kernels (CCPP + Icepack + CICE + MOSART)
nvcc -O3 -arch=sm_86 -o noaa_multi noaa_multi_kernel.cu
```

## Hardware

All benchmarks run on a single NVIDIA RTX 3060 12GB. Production workloads (full CONUS NWM, global GFS) would need validation on WCOSS2 or cloud HPC (A100/H100).

## Author

Cody Churchwell

## License

BSD-3-Clause (same as the upstream NOAA models)
