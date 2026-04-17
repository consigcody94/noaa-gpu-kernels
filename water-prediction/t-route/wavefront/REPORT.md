# Wavefront GPU Muskingum-Cunge — benchmark report

Follow-up to [NOAA-OWP/t-route#874](https://github.com/NOAA-OWP/t-route/issues/874) after maintainer feedback that a reach-parallel kernel which ignores the dependency graph is not an honest routing benchmark.

This kernel **respects the tree dependency**: reaches are grouped into topological levels (level 0 = headwater), and we launch one CUDA kernel per level. All reaches in a level are independent and run in parallel. Between levels, we sync so downstream reaches see upstream results.

Each reach runs the same full MUSKINGCUNGE secant solver as the t-route Fortran reference. FP32 throughout (t-route `REAL` default).

## Hardware
- GPU: NVIDIA RTX 3060 12 GB (consumer Ampere, 28 SMs)
- CPU: measured single-thread (same machine)
- CUDA 13.2, nvcc `-O3 -arch=sm_86`

## Network

Synthetic NWM-like topology with:
- Binary-tree backbone with Galton-Watson-style confluences (fan-in 2 or 3)
- Poisson-distributed chain extensions between confluences (mean chain length 3)
- Poisson(12) segments per reach
- Realistic level-size distribution — widest level has most reaches, tapering geometrically
- Tree depth ~34-69 levels at 1K-1M reach scale (matches CONUS NWM depth band)

Why synthetic: we do not have JoshCu's actual 309K reach configuration. We asked for it in the issue thread but have not received it yet. The synthetic network intentionally has *worse-than-real* parallel efficiency at deeper levels — the binary tree with exponentially decreasing level sizes is a harder case for wavefront scheduling than the real NWM, which has long dendritic branches that keep many reaches in the same level.

## Accuracy

Against an FP64 CPU reference on the same topology/schedule:

| n_timesteps | GPU-FP32 vs FP64 | CPU-FP32 vs FP64 |
|-------------|------------------|------------------|
| 4 | 99.53% within 1%, 99.87% within 10% | 99.53% within 1%, 99.87% within 10% |
| 24 | 95.93% within 1%, 98.82% within 10% | 95.83% within 1%, 98.81% within 10% |

GPU-FP32 and CPU-FP32 track each other within 0.1 percentage point at every percentile. The secant method is genuinely FP32-sensitive — near rerror ≈ 0.01 and aerror ≈ 0.01 the iteration count can flip between runs with different FMA scheduling. This is **inherent to FP32 iterative secant, not a GPU artifact**: the CPU FP32 kernel diverges from FP64 truth by the same amount.

For higher-fidelity runs you would compile with `double` everywhere — the CU file is a single template-style change away from FP64, at the cost of 30-60x slower GPU performance on Ampere consumer cards (1/32 FP64 rate).

## Benchmarks

All numbers are single-timestep averages. GPU includes the per-timestep H→D qlat copy. CPU-FP32 is single-thread.

| target reaches | actual reaches | segments | levels | t/steps | GPU FP32 | CPU FP32 | CPU FP64 | GPU vs CPU-FP32 | GPU vs CPU-FP64 |
|---------------:|---------------:|---------:|-------:|--------:|---------:|---------:|---------:|----------------:|----------------:|
|           1K  |          879  |  10 K   |   34  |   24 |  7.02 ms |  3.74 ms |  5.97 ms |  **0.53x**  |  0.85x |
|          10K  |        8 547  | 103 K   |   39  |   24 | 11.81 ms | 37.17 ms | 60.91 ms |  **3.15x**  |  5.16x |
|         100K  |       87 527  |   1.05 M |   59  |   24 | 21.21 ms | 376.5 ms | 623.9 ms |  **17.7x**  | 29.4x |
|         309K  |      271 190  |   3.27 M |   60  |   24 | 29.42 ms | 1169.6 ms | 1931.7 ms | **39.8x** | 65.7x |
|         500K  |      439 030  |   5.27 M |   67  |    4 | 50.82 ms | 2155.6 ms | 3605.2 ms | **42.4x** | 70.9x |
|           1M  |      878 020  |  10.6 M  |   69  |    4 | 70.71 ms | 4468.6 ms | 7202.8 ms | **63.2x** | 101.9x |

### Key observations

- **GPU crosses over CPU around 10K reaches** (first where speedup > 1x). Below that, launch overhead dominates.
- **Strong scaling through 1M reaches.** GPU per-timestep time scales roughly linearly with segment count (×10 reaches ≈ ×3.5 GPU time), while single-thread CPU scales linearly with segment count AND no parallel speedup, so the speedup ratio widens.
- **At 309K reaches (JoshCu's published test size)**: 29.4 ms/timestep on RTX 3060. JoshCu's 32-core Rust kernel reports 5.32s for 309K × 24 timesteps = **222 ms/timestep**, meaning this kernel is ~**7.6x faster than a 32-core AMD 9950X3D running rs_route** on a comparable topology.
- **Projection to CONUS (2.7M reaches)**: linear extrapolation from 1M (70.7 ms/ts) gives ~190 ms/ts on RTX 3060. A100 has ~6x the FP32 throughput of a 3060 and more SMs, so ~30-50 ms/ts is realistic.

## What this does not show

- No linearized/sparse-matrix MC variant yet. [DeepGroundwater/ddr](https://github.com/DeepGroundwater/ddr) and [geoglows/river-route](https://github.com/geoglows/river-route) use a linearized per-timestep form `(I − C₁N) Q_{t+1} = C₂NQ_t + C₃Q_t + C₄Q'` that avoids the 100-iteration secant entirely, replacing it with a single sparse triangular solve. That formulation should be strictly faster than this kernel on the same hardware because cuSPARSE's batched triangular solve is hand-tuned. That's the next comparison.
- The `max` outlier (up to ~10⁴ relative error) is a handful of segments where the FP64 reference converges to a tiny flow (~1e-4) and the FP32 run converges to something ~1 — the iteration paths diverged. This affects <0.1% of segments and happens equally on CPU FP32 and GPU FP32.
- Single-thread CPU only. I have not yet run the Rust rs_route or an OpenMP Fortran build as a multi-thread CPU baseline on the same machine — the 32-core/222 ms number comes from JoshCu's published issue comment on a different machine.

## Reproducing

```bash
# On Windows with CUDA 13.2 and MSVC 2022 Build Tools installed
python gen_network.py --reaches 309000 --timesteps 24 --out net_309k
cmd.exe /c .\build.cmd
./wavefront_mc.exe net_309k
```

Full results are in `results/run_*.txt`.
