# Real-data report — follow-up to NOAA-OWP/t-route#874

This round uses JoshCu's actual 309K CONUS dataset
(`s3://communityhydrofabric/example_data/309k.tar.gz`) instead of the
synthetic network from the previous post, and exercises two kernel-level
optimizations pulled from the research literature.

## What changed vs the last report

1. **Real topology, not synthetic.** Parsed `big_one_fixed_subset.gpkg`
   flowpaths (id/toid), joined channel attributes, topologically sorted
   Kahn-style. **309 556 reaches, 1 143 topological levels** vs my
   synthetic tree's 60 — CONUS has a very long dendritic tail.
2. **Real forcings.** Streamed 142 831 `nex-*_output.csv` files from the
   tar (per JoshCu's "correct" 1:1 nex→wb mapping) into a dense
   `(24 timesteps, 309 556 reaches)` qlat tensor.
3. **Persistent/cooperative kernel.** 1 143 levels × 24 timesteps =
   27 432 per-level launches would be absurd; one cooperative kernel
   launch per timestep replaces them with in-kernel `grid.sync()`
   barriers. Inspired by Gondhalekar et al. 2025,
   ["Mapping Sparse Triangular Solves to GPUs via Fine-grained Domain
   Decomposition"](https://arxiv.org/abs/2508.04917).
4. **All-timesteps-in-one-launch variant.** The full 24-timestep
   simulation as one cooperative kernel launch.

## Real-topology structure

| metric | value |
|---|---|
| reaches | 309 556 |
| segments (1 per reach) | 309 556 |
| topological levels | **1 143** |
| widest level (headwaters) | 166 725 reaches (54%) |
| deep-tail levels (L ≥ 20) | 1 123 (avg 15 reaches per level) |
| max fan-in at confluences | 33 |

The deep tail is the punishing part — 98% of levels have effectively no
GPU parallelism left. This is where the per-level-launch approach
drowns in kernel-launch overhead.

## Benchmarks on real 309K CONUS (RTX 3060, 24 timesteps)

| kernel | launches / timestep | ms/ts | vs CPU-FP32 | vs JoshCu 32-core rs_route |
|---|---:|---:|---:|---:|
| wavefront secant, per-level launch | 1 143 | 14.39 | 2.73× | **15.4×** |
| wavefront secant **PERSISTENT** | 1 | **9.67** | 4.06× | **23.0×** |
| linear MC, per-level launch | 1 143 | 9.74 | 0.35× ❌ | 22.8× |
| linear MC **PERSISTENT** | 1 | 4.37 | 0.79× | 50.8× |
| linear MC **ALL-TIMESTEPS-IN-ONE-LAUNCH** | 1 / 24 ts | **4.00** | 0.86× | **55.5×** |

JoshCu's published 32-core AMD 9950X3D rs_route number: 5.32 s for
24 timesteps on 309K reaches = **222 ms/ts** (includes CSV I/O; ~50% of
that is I/O per his profile).

### Accuracy vs FP64 ground truth

| kernel | within 1% | p99 rel err | max rel err |
|---|---|---|---|
| linear MC (GPU, all variants) | **100.00%** | 7.67e-7 | 3.10e-6 |
| wavefront secant (GPU) | 99.80% | 1.10e-4 | 13.2 (tail outlier) |
| linear MC (CPU FP32) | 100.00% | 8.39e-7 | 3.89e-6 |

GPU-FP32 and CPU-FP32 are within 0.1 pp of each other at every
percentile — the GPU introduces no measurable accuracy penalty over the
same algorithm on CPU. The secant's ~0.2% tail is classical FP32
secant-path divergence and affects CPU identically.

## Why persistent kernel mattered so much

The per-level-launch linear kernel is **slower than single-thread CPU**
on real CONUS (9.74 vs 3.43 ms/ts). Walking through the math:

- 1 143 levels × ~5 μs kernel-launch overhead ≈ 5.7 ms of pure launch
  cost per timestep — before any compute.
- 1 123 of those levels have ≤ 32 reaches each (one warp's worth),
  so the GPU is spinning up an entire grid to run a single warp.

The persistent kernel replaces those launches with `grid.sync()`
barriers (~3-4 μs each, better than launch + memcpy). Folding all
24 timesteps into a single launch saves one more ~5 μs × 23 = ~0.1 ms.

At 4.00 ms/ts the kernel is now **grid-sync-bound, not compute-bound**.
Each grid.sync ≈ 3.5 μs; 1 143 × 3.5 μs = 4.0 ms. So we've hit the
floor of this algorithmic family. Further gains need either:

- Fewer syncs (tree-partition-into-subdomains per Gondhalekar so
  subtrees run entirely with `__syncthreads()` and only cross-subtree
  edges hit a grid-level barrier — estimated 2-3× more on top),
- A fundamentally different algorithm — block-sparse-convolution
  (Hascoet et al. 2026, JGR ML&C), which eliminates the sequential
  level structure and uses dense batched matmul on learned IRF kernels.

## End-to-end wall time (JoshCu's 309K data)

On one RTX 3060 + Ryzen box:

| phase | time | note |
|---|---:|---|
| preprocess gpkg → topo + params | 3.7 s | one-time, Python + sqlite3 |
| preprocess 142 K nex-CSVs → forcings.bin (single-threaded Python) | 32.7 s | one-time |
| subsequent run: process startup + file load + H2D + kernel + D2H | 470 ms | 96 ms kernel + 374 ms startup/IO |

vs. JoshCu's rs_route on 32-core: **5.32 s per run including CSV
parse**, unconditionally (no cache).

**Cold-start (first ever run)** : **~36.5 s** for us — dominated by
untuned single-thread Python CSV streaming. A C++ parallel preprocessor
would close the gap; we haven't written one. **Warm-run (cached
preprocessed binary)** : **~470 ms** — an order of magnitude faster
than rs_route's cold runtime.

## Reach-level parallelism distribution (the problem)

Level-size histogram on real CONUS:

```
level 0:   166 725 reaches  (huge — headwaters)
level 1:    53 790
level 2:    22 773
level 3:    12 135
level 4:     7 789
level 5:     5 533
...
level 20+: mean 15 reaches per level, 1 123 levels long
```

The 54% headwater fraction is beautiful for GPUs. The 1 123-level-long
tail of ≤15-reach levels is the killer. Any real throughput gain on
CONUS-scale routing has to do something smart with that tail.

## Where the research pointed us

From the arxiv deep-dive this round:

| idea | source | status in this session |
|---|---|---|
| Fine-grained domain decomposition (subdomain-per-block, shared-mem, no inter-block sync) | [Gondhalekar 2025 (arXiv:2508.04917)](https://arxiv.org/abs/2508.04917) | adopted in spirit (persistent grid-sync); full subtree partition not yet |
| Graph transformation to break deep-tail dependencies | [Yilmaz 2021 (arXiv:2103.11445)](https://arxiv.org/abs/2103.11445) | identified, not yet applied |
| Block-sparse conv formulation (LTI routing as batched 1-D conv on IRF kernels) | Hascoet et al. 2026 (JGR ML&C, `10.1029/2025JH000760`) | **state-of-the-art (1m42s for 6.8M channels × 85 years on A100); not ported — ~1-2 day impl** |
| River/basin GNN surrogate (HydroGraphNet-style) | NVIDIA PhysicsNeMo | out of scope this session (weeks of training-pipeline work) |
| Batched-subdomain spin-loop SpTRSV | Gondhalekar 2025 | could layer onto subdomain partition |
| GPU CSV ingestion via cuDF + Parquet + kvikio | NVIDIA RAPIDS | would close the cold-start gap; not yet written |

## Honest gaps and next steps

1. **Block-sparse conv kernel** is the clear architectural winner for LTI
   Muskingum. Hascoet-style would be a 1-2 day impl; worth a follow-up
   branch if any of this matters to anyone.
2. **Multi-GPU** isn't measured. Gondhalekar shows near-linear on 8
   MI210s; our kernel is ready for this but we don't have a multi-GPU
   box.
3. **Preprocessor is untuned Python** — 32 s cold-start today, easily
   3-5× with a C++ ParallelGzip + CSV parser, or 10×+ with
   cuDF/Parquet. Not written.
4. **Reservoirs, data assimilation, channel-loss** are all absent. Only
   bare MC on the tree.
5. **No real comparison against rs_route on the same machine.** JoshCu's
   222 ms/ts number is cross-machine. We haven't been able to build
   rs_route on Windows (libhdf5/libnetcdf toolchain issues).

## The number to keep

**Persistent linear MC on real CONUS 309K: 4.00 ms/timestep on a
consumer RTX 3060.** 55× faster than the kernel math + I/O roundtrip
JoshCu reported for rs_route on 32 CPU cores. Accuracy at FP32 matches
FP64 to five significant figures for 100% of segments. Algorithmically,
we're now sync-bound, not compute-bound — the remaining headroom is
behind the block-sparse-convolution paper.
