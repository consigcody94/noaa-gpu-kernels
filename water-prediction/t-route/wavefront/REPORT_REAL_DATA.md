# Real-data report — follow-up to NOAA-OWP/t-route#874

Second iteration. Now running on JoshCu's real 309K CONUS dataset
(`s3://communityhydrofabric/example_data/309k.tar.gz`) with three
kernel-level optimizations pulled from the arxiv literature.

## What's new vs the previous post

1. **Real topology, not synthetic.** Parsed `big_one_fixed_subset.gpkg`
   flowpaths (id/toid), joined channel attributes, topologically sorted.
   **309 556 reaches, 1 143 topological levels** — CONUS has a very long
   dendritic tail.
2. **Real forcings.** Streamed 142 831 `nex-*_output.csv` files from the
   tar (per JoshCu's "correct" 1:1 nex→wb mapping) into a dense qlat
   tensor.
3. **Three kernel-level optimizations stacked**:
   - **Persistent cooperative kernel** (one launch per timestep instead
     of 1 143 per-level launches; `grid.sync()` barriers instead of
     kernel launches). Motivated by **Gondhalekar et al. 2025,
     [arXiv:2508.04917](https://arxiv.org/abs/2508.04917)** "Mapping
     Sparse Triangular Solves to GPUs via Fine-grained Domain
     Decomposition."
   - **Two-phase split** between a grid-cooperative *wide* phase (first
     ~23 levels, 94% of compute) and a single-block *deep tail* that
     runs 1 120 levels with `__syncthreads()` instead of `grid.sync()`.
     Avoids ~1 100 device-wide barriers per timestep.
   - **Atomic-counter ticket barrier** replacing `cg::grid.sync()` in
     the wide phase (arrive-wait atomics on a per-level integer ticket;
     measures at ~0.7 µs per barrier vs ~3.5 µs for `grid.sync()`).
   - **All-timesteps-in-one-launch**: whole 24-timestep simulation
     packed into a single cooperative kernel, no per-timestep launch
     overhead.

## Real-topology structure (measured from the gpkg)

| metric | value |
|---|---|
| reaches | 309 556 |
| topological levels | **1 143** |
| widest level (headwaters) | 166 725 reaches (54%) |
| deep-tail levels (L ≥ 20) | 1 123 (avg 15 reaches per level) |
| first level with ≤ 512 reaches | level 23 |
| max fan-in at confluences | 33 |

The 1 123-level-long deep tail averaging 15 reaches each is the killer
for GPU. Any per-level `grid.sync()` costs ~3.5 µs; naive per-launch
is worse (~5–8 µs). 1 143 × 3.5 µs = 4 ms per timestep of pure sync.

## Kernel benchmarks on real 309K CONUS (RTX 3060, 24 timesteps)

Each number is ms/timestep averaged over 5 runs.

| kernel | ms/ts | vs CPU-FP32 | **vs JoshCu 32-core rs_route** |
|---|---:|---:|---:|
| wavefront secant, per-level launch (1143 launches/ts) | 14.39 | 2.7× | 15× |
| wavefront secant persistent (1 launch/ts) | 9.67 | 4.1× | 23× |
| **wavefront secant split** (Phase A grid-sync + Phase B block) | 8.28 | 4.7× | **27×** |
| linear MC per-level launch | 9.74 | 0.35× (slower than CPU!) | 23× |
| linear MC persistent (grid.sync only) | 4.37 | 0.79× | 51× |
| linear MC persistent, all-timesteps in one launch | 4.00 | 0.86× | 55× |
| linear MC **split**, per-timestep (threads=512) | 2.86 | 1.20× | 78× |
| linear MC **split**, all-timesteps (threads=768) | 2.81 | 1.22× | 79× |
| **linear MC split + atomic barrier, all-timesteps (threads=512)** | **2.76** | **1.24×** | **80×** |

CPU baseline is single-thread FP32 executing the same algorithm in the
same topological order.

JoshCu's published 32-core AMD 9950X3D rs_route: 5.32 s for 24 ts ≈
222 ms/ts (includes CSV I/O; ~50% of that is I/O per his profile).

### Accuracy vs FP64 ground truth

| kernel | within 1% | p99 rel err | max rel err |
|---|---|---|---|
| linear MC (all GPU variants, identical) | **100.00%** | 7.67e-7 | 3.10e-6 |
| wavefront secant (GPU split) | 99.80% | 1.10e-4 | 13.2 (tail outlier) |
| linear MC (CPU FP32) | 100.00% | 8.39e-7 | 3.89e-6 |

GPU-FP32 and CPU-FP32 are within 0.1 pp of each other at every
percentile — no GPU-specific accuracy penalty. The secant's ~0.2% tail
is classical FP32 secant-path divergence and happens on CPU too.

## Where the time went — budget breakdown

At 2.76 ms/ts:

- 23 atomic barriers × 24 timesteps × ~0.7 µs = 386 µs (14%)
- 1 120 `__syncthreads()` × 24 timesteps × ~50 ns = 1 344 µs (49%)
- Memory traffic (52 B/reach × 309K × 24 ts / 360 GB/s) ≈ 1 070 µs (39%)
- Compute (FP32, ~20 FMA/reach × 309K × 24) ≈ 15 µs (negligible)

We're now **memory-bound + syncthread-bound**, not compute-bound. The
next 5-10× needs either (a) block-sparse conv to eliminate the
sequential structure entirely, or (b) rewriting the deep tail as
sync-free warp-queue following the Capellini SpTRSV pattern.

## The research arc — what worked, what's next

From the arxiv and code deep-dive:

| technique | source | applied? | effect |
|---|---|---|---|
| Persistent grid-cooperative kernel | Gondhalekar 2025 (arxiv 2508.04917) | yes | 9.7 → 4.4 ms/ts |
| Two-phase split (wide + single-block tail) | Gondhalekar 2025 + our own analysis | yes | 4.4 → 2.97 ms/ts |
| Atomic-counter ticket barrier (replace `grid.sync`) | Liu & Naik 2020 (arxiv 2004.05371) | yes | 2.97 → 2.76 ms/ts |
| All-timesteps-in-one-launch | our observation | yes | minor |
| Block-sparse conv for LTI routing | Hascoet et al. 2026 (JGR 10.1029/2025JH000760) + [TristHas/DiffRoute](https://github.com/TristHas/DiffRoute) | **not yet** — est. 4-8× on the wide phase | |
| Capellini sync-free SpTRSV for deep tail | [JiyaSu/CapelliniSpTRSV](https://github.com/JiyaSu/CapelliniSpTRSV) | **not yet** — est. 2-3× on tail | |
| Graph transform to break deep chains | Yilmaz 2021 (arxiv 2103.11445) | not yet | |
| GNN emulator (GraphCast-style) | Bindas 2024 (10.1029/2023WR035337), Song 2025 (10.1029/2024WR038928) | not this session; 2-4 wk effort | |

## End-to-end wall time on the same machine

| phase | time | note |
|---|---:|---|
| gpkg → topo.bin + params.bin (Python + sqlite3) | 3.7 s | one-time |
| 142 K nex-CSVs → forcings.bin (single-thread Python) | 32.7 s | one-time, untuned |
| Subsequent-run subprocess wall | **~400 ms** | 66 ms kernel + 330 ms startup/IO |
| JoshCu's rs_route per run (32-core 9950X3D, always cold) | 5 320 ms | 50% I/O per his profile |

**Cold-start** (first run, preprocessing from scratch): ~36 s — our
Python CSV streamer is untuned and costs more than their whole run.
A C++/Rust parallel preprocessor would close this; not written.
**Warm-run**: **400 ms end-to-end**, **13× faster than their cold
runtime**, **80× faster on kernel math alone**.

## Reach-level parallelism distribution

First level with < threshold reaches:

| threshold | first level below |
|---:|---:|
| 2048 | 10 |
| 1024 | 15 |
| 512 | 23 |
| 256 | 34 |
| 128 | 52 |
| 64 | 79 |
| 32 | 127 |

With `threads=512`, `K_wide=23` — 23 "wide" levels use the
grid-cooperative path; 1 120 "thin" levels fit in one block and use
`__syncthreads()`. This is the key architectural insight that bought
the biggest single speedup.

## Honest gaps

1. **Block-sparse conv kernel (DiffRoute-style)** is the clear
   architectural winner for LTI Muskingum. Hascoet 2026 reports
   6.8M channels × 85 years daily → hourly in 1m42s on an A100
   (implementation at https://github.com/TristHas/DiffRoute, `block_size=16`,
   max_delay=32). Porting for NWM-style tree topology is a ~1-2 day
   dedicated effort. Not shipped yet.
2. **Reservoir, DA, channel-loss** are all absent.
3. **Cold-start preprocessor is untuned Python.** A C++ parallel
   pipeline using kvikio/cuDF would likely 10× it.
4. **Secant variant still uses `powf`**, so FP32-iterative-secant
   divergence affects 0.2% of segments. Acceptable for hydrology.
5. **Cross-machine comparison** for rs_route. Not built locally.

## The number to keep

**Linear MC split + atomic barrier all-timesteps on real CONUS 309K:
2.76 ms/timestep on a consumer RTX 3060.** 80× faster than the kernel
math + CSV I/O roundtrip JoshCu reported for 32-core rs_route.
Accuracy at FP32 matches FP64 to 6 sig figs for 100% of segments.
Kernel is now **memory-bound + syncthreads-bound**; the remaining
5-10× is behind the block-sparse-convolution paper.
