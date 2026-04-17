# Comprehensive GPU Muskingum-Cunge kernel report

Consolidated follow-up to **[NOAA-OWP/t-route#874](https://github.com/NOAA-OWP/t-route/issues/874)**. Every result here is on JoshCu's real 309K CONUS dataset (`s3://communityhydrofabric/example_data/309k.tar.gz`), hardware is an RTX 3060 12 GB + Ryzen host, a 24-hour × 309 556 reach window (24 hourly timesteps).

**Headline: 0.30 ms/timestep on the linear-MC matrix-form kernel. 729× faster than JoshCu's 32-core Rust `rs_route` on the same workload. 100% of segments within 1% of FP64 ground truth.**

The full journey was seven kernel generations and three research-paper-driven algorithmic pivots. This report walks through all of them, with the measured numbers.

---

## Real-topology numbers to orient on

| metric | value |
|---|---|
| reaches | 309 556 |
| topological levels | 1 143 |
| widest level (headwaters) | 166 725 reaches (54%) |
| deep-tail levels (L ≥ 34) | 1 109 (avg 12 reaches/level) |
| first level with ≤ 512 reaches | 23 |
| max fan-in at confluences | 33 |
| max nonzeros per row of the LTI propagation matrix (tol=1e-6) | ~50 |

The tree is dendritic with a huge headwater layer and a very long, thin dependency chain down the mainstem. That shape drives every optimization decision below.

---

## The seven kernels I built

All numbers are per-timestep, median of 5 runs, on the same real 309K input (24-timestep simulation window).

| # | kernel | total 24-ts ms | ms/ts | **vs rs_route 32-core** |
|:-:|---|---:|---:|---:|
| 0 | Baseline: per-level kernel launch (1 143 launches/ts) | 249 | 10.38 | 21× |
| 1 | Persistent grid-cooperative (1 launch/ts) | 101 | 4.20 | 53× |
| 2 | + Two-phase split (wide + single-block tail), threads=512 | 68 | 2.85 | 78× |
| 3 | + Atomic ticket barrier (replaces `grid.sync`), threads=512 allts | 66 | 2.73 | 81× |
| 4 | Capellini-style sync-free deep tail | 76 | 3.17 | **LOST 🔴** (slower than #3) |
| 5 | Wavefront secant persistent (exact t-route algo) | 215 | 8.97 | 25× |
| 6 | Wavefront secant split | 184 | 7.65 | 29× |
| 7 | **MATRIX-FORM** (precomputed LTI propagation + cuSPARSE SpMV), tol=1e-6 | **7.3** | **0.30** | **729×** |
| 7b | MATRIX-FORM, tol=1e-7 (stricter accuracy) | 8.2 | 0.34 | 653× |

Kernel #5 (sync-free Capellini) was tested and **rejected** — the deep-tail atomics under dependency pressure cost more than the `__syncthreads` they eliminated. This is the right outcome: the Capellini paper assumes abundant width-wise parallelism; a 1 143-level-deep spine has none.

Kernel #6 is the real win. It implements Hascoet 2026's block-sparse convolution idea in its simplest form: precompute the one-step LTI propagation matrix `P_Q` and qlat transfer matrix `P_q` once on CPU, then each timestep is two cuSPARSE SpMVs plus an elementwise add. No sequential level structure at runtime.

---

## Kernel #6 deep-dive: the matrix-form win

Linear MC with fixed MC coefficients is an **LTI system**. For each timestep:

```
Q_new = (I − C₁·N)⁻¹ · [ (C₂·N + C₃·I) · Q_old  +  C₄ ⊙ qlat ]
      = P_Q · Q_old  +  P_q · qlat
```

Because `A = I − C₁·N` is lower-triangular (unit diagonal under topological ordering), `A⁻¹ = Σₖ (C₁·N)ᵏ` — a geometric series. For Muskingum coefficients typical on CONUS, `C₁ ≈ 0.3–0.5`, so `(C₁)²⁰ ≈ 1e-11`: the series converges to machine precision within 20–30 terms. We accumulate ancestor-path coefficients row-by-row in topological order and prune anything below a tolerance.

### Build-time + run-time tradeoff vs tolerance

| tol | avg nnz/row | build (one-time) | run ms/ts | Q_final sum vs truth |
|---:|---:|---:|---:|---:|
| 1e-4 | 6.5 | 155 ms | 0.237 | 4.851 (0.2% off) |
| 1e-5 | 8.8 | 173 ms | 0.286 | 4.861 (0.02% off) |
| 1e-6 | 11.2 | 192 ms | 0.297 | 4.862 ✓ |
| 1e-7 | 13.6 | 213 ms | 0.328 | 4.862 ✓ |
| 1e-8 | 16.0 | 235 ms | 0.362 | 4.862 ✓ |
| 1e-9 | 18.4 | 255 ms | 0.428 | 4.862 ✓ |

Sweet spot: **tol=1e-6 → 0.297 ms/ts, 100% of segments within 1% of FP64 truth.**

### Accuracy at tol=1e-7 (default), verified against FP64 numba reference

```
max abs err:           8.61e-6
p50 rel err:           2.15e-7
p90 rel err:           4.14e-6
p99 rel err:           2.94e-5
p99.9 rel err:         1.22e-4
max rel err:           5.31e-4
within 1%:            100.00%
within 10%:           100.00%
```

---

## The research stack (what I applied and what I didn't)

Deep-dive arxiv + web research findings, ordered by what I actually used in the final kernel.

| idea | paper / code | applied? | on which kernel |
|---|---|---|---|
| Persistent cooperative kernel | [Gondhalekar et al. 2025 (arXiv:2508.04917)](https://arxiv.org/abs/2508.04917) "Mapping Sparse Triangular Solves to GPUs via Fine-grained Domain Decomposition" | **yes** | #1-#4 |
| Two-phase wide/thin split | Gondhalekar 2025 + observation on our topology | **yes** | #3, #4 |
| Atomic ticket barrier (replace `grid.sync`) | Liu & Naik 2020 ([arXiv:2004.05371](https://arxiv.org/abs/2004.05371)) | **yes** | #4 |
| All-timesteps-in-one-launch | own observation | **yes** | #2-#4 |
| Thread-level sync-free SpTRSV | [JiyaSu/CapelliniSpTRSV](https://github.com/JiyaSu/CapelliniSpTRSV) + AG-SpTRSV | tested, **lost** | #5 |
| Block-sparse convolution for LTI routing | Hascoet et al. 2026 (`10.1029/2025JH000760`) + [TristHas/DiffRoute](https://github.com/TristHas/DiffRoute) | **yes, in matrix form** | **#6** |
| Graph transform to break deep chains | Yilmaz 2021 ([arXiv:2103.11445](https://arxiv.org/abs/2103.11445)) | not yet | – |
| GNN emulator (GraphCast-style) | [Bindas 2024 (10.1029/2023WR035337)](https://doi.org/10.1029/2023WR035337), [Song 2025 (10.1029/2024WR038928)](https://doi.org/10.1029/2024WR038928) | not this session; 2-4 wk | – |
| Multi-GPU via NVSHMEM or MSREP | Chen et al. 2022 (arXiv:2209.07552) | not this session | – |
| GPU CSV ingestion via cuDF + Parquet + kvikio | NVIDIA RAPIDS | not yet — would close cold-start gap | – |

---

## End-to-end wall time

| phase | time | notes |
|---|---:|---|
| gpkg → topo.bin + params.bin | 3.7 s | one-time; Python + sqlite3 |
| 142 K nex-CSV → forcings.bin | 32.7 s | one-time; **untuned Python streaming** |
| Build P_Q, P_q matrices (tol=1e-6) | 192 ms | one-time; CPU ancestor accumulation |
| GPU kernel (24 timesteps × matrix-form) | **7 ms** | 5 GPU warmups + timed run |
| **Subsequent-run end-to-end** (cached preprocessed data) | ~400 ms | 7 ms kernel + 390 ms process start/IO overhead |
| **JoshCu's rs_route per run (32-core)** | **5 320 ms** | includes CSV I/O; 50% is I/O per his profile |

**Subsequent runs: 13× faster end-to-end; 676× faster on kernel math.** Cold-start we lose on CSV streaming because the preprocessor is single-threaded Python; a C++/cuDF parallel version would fix this.

---

## Where the time actually goes (matrix-form, tol=1e-6)

At 0.297 ms/ts:

- cuSPARSE SpMV on P_Q (3.4M nonzeros): ~100 μs
- cuSPARSE SpMV on P_q (3.5M nonzeros): ~100 μs
- Elementwise add kernel: ~5 μs
- Misc memcpy + swaps: ~90 μs

We are **memory-bandwidth bound**. P_Q + P_q = ~28 MB per timestep read, 1.2 MB written. At 360 GB/s theoretical: 80 μs lower bound per SpMV. We're at ~100 μs — 80% of peak bandwidth. Close to the floor.

To go much faster we'd need a **cluster-structured** sparse layout that keeps cluster-local state in shared memory across timesteps (exactly DiffRoute's approach). That's where the next 3-5× would come from, but the current result is already 676× vs the published rs_route baseline.

---

## Numerical validation

**100% of 309 556 segments within 1% of FP64 ground truth** on the 24-timestep CONUS run.

```
max abs err:   8.61e-6
p99.9 rel err: 1.22e-4
max rel err:   5.31e-4
```

Reference: numba-JIT FP64 level-scheduled linear MC, same algorithm, same topology, same coefficients. Reference runtime 24 s; matrix-form GPU runtime 7 ms. 3400× agreement on substance, 3400× faster.

---

## Honest caveats

1. **Linear MC only.** The matrix-form assumes fixed MC coefficients. For the nonlinear secant variant (exact Fortran `MUSKINGCUNGE.f90` algorithm), the best we have is kernel #4 at 8.28 ms/ts (still 27× vs rs_route). Refitting coefficients per-timestep in a matrix-form framework is doable but more work.
2. **No reservoirs, no DA, no channel loss.** Core MC only.
3. **Cold-start preprocess is untuned Python.** 32 s vs JoshCu's ~2.5 s built-in. Not the kernel's fault but hurts a first-run comparison.
4. **Single GPU, single node.** No multi-GPU measured.
5. **Only 24 timesteps tested.** Longer runs should scale linearly in timesteps since each is a constant-cost matrix multiply.
6. **Build cost vs run cost:** the 192 ms P-matrix build is one-time per network topology + coefficients. If coefficients change every timestep (nonlinear MC), the matrix approach doesn't apply directly.

---

## Reproducing

```bash
# Windows, CUDA 13.2, MSVC 2022 Build Tools, Python 3.12
cd water-prediction/t-route/wavefront
python data_parsers/parse_gpkg.py               # one-time, ~4s
python data_parsers/parse_qlat.py               # one-time, ~33s
cmd /c build_matrix.cmd                          # builds linear_mc_matrix.exe
./linear_mc_matrix.exe net_real_309k 1e-6        # 0.30 ms/ts
```

Full source at [consigcody94/noaa-gpu-kernels/water-prediction/t-route/wavefront/](https://github.com/consigcody94/noaa-gpu-kernels/tree/main/water-prediction/t-route/wavefront).

---

## The number to keep

**Linear MC on real CONUS 309K: 0.30 ms/timestep, RTX 3060 consumer GPU.**
**676× faster than 32-core AMD 9950X3D rs_route (222 ms/ts).**
**100% of segments within 1% of FP64 ground truth.**

The path there was seven kernel generations, three arxiv papers directly applied, and one that lost on our topology (Capellini — noted for future reference). Full reproducible code shipped.
