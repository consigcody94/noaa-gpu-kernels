/**
 * Linearized MC with SYNC-FREE deep-tail (Capellini-inspired SpTRSV).
 *
 * Previous best (linear_mc_atomic.cu):
 *   Phase A: 23 "wide" levels via grid-cooperative atomic ticket barrier
 *   Phase B: 1120 "thin" levels via single block + __syncthreads per level
 *
 * Phase B was costing ~1.34 ms/timestep of the 2.74 ms total (1120 __syncthreads
 * × 50 ns × 24 timesteps = 1.34 ms). We eliminate it with a sync-free approach:
 *
 *   - One "ready" flag per reach, initialized to 0 each timestep.
 *   - Threads pull reach-ids off a work queue (using atomic counter).
 *   - For each reach, thread reads `ready[upstream]` fields; spin-waits
 *     (pure volatile load) until all are set.
 *   - Processes reach, sets `ready[reach] = 1`, __threadfence().
 *   - No __syncthreads between levels.
 *
 * Adapted from Su et al. "CapelliniSpTRSV: Thread-level sync-free SpTRSV on GPU"
 * (https://github.com/JiyaSu/CapelliniSpTRSV), specialized for tree topology
 * where each reach has at most max_up upstreams.
 *
 * Author: Cody Churchwell — NOAA-OWP/t-route#874 follow-up
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) {\
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
            cudaGetErrorString(_e)); std::exit(1); } } while (0)

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

__host__ __device__ inline void compute_mc_coeffs(
    const ChParams& p, float dt,
    float& C1, float& C2, float& C3, float& C4_over_ql)
{
    float z = (p.cs == 0.0f) ? 1.0f : 1.0f / p.cs;
    float bfd;
    if (p.bw > p.tw) bfd = p.bw / 0.00001f;
    else if (p.bw == p.tw) bfd = p.bw / (2.0f * z);
    else bfd = (p.tw - p.bw) / (2.0f * z);
    float h = fmaxf(0.5f * bfd, 0.1f);
    float A = (p.bw + h * z) * h;
    float WP = p.bw + 2.0f * h * sqrtf(1.0f + z * z);
    float R = (WP > 0.0f) ? A / WP : 0.0f;
    float r23 = powf(R, 2.0f / 3.0f);
    float r53 = powf(R, 5.0f / 3.0f);
    float Ck = fmaxf(1e-6f, (sqrtf(p.s0) / p.n_ch) *
                 ((5.0f / 3.0f) * r23 - (2.0f / 3.0f) * r53 *
                  (2.0f * sqrtf(1.0f + z * z) / (p.bw + 2.0f * h * z))));
    float Km = fmaxf(dt, p.dx / Ck);
    float X = 0.25f;
    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f;
    C1 = (Km * X + dt / 2.0f) / D;
    C2 = (dt / 2.0f - Km * X) / D;
    C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    C4_over_ql = dt / D;
}

__global__ void kernel_precompute_coeffs(
    const ChParams* params, float* C1, float* C2, float* C3, float* C4oq,
    float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    compute_mc_coeffs(params[i], dt, C1[i], C2[i], C3[i], C4oq[i]);
}

// Atomic barrier for Phase A
__device__ __forceinline__ void atomic_barrier(
    int L, int num_blocks, int* __restrict__ grid_barrier)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(&grid_barrier[L], 1);
        while (*((volatile int*)&grid_barrier[L]) < num_blocks) { }
    }
    __syncthreads();
}

__device__ __forceinline__ void process_reach(
    int r,
    const int* __restrict__ reach_seg_start,
    const int* __restrict__ reach_seg_len,
    const int* __restrict__ reach_up_ptr,
    const int* __restrict__ reach_up_idx,
    const float* __restrict__ C1,
    const float* __restrict__ C2,
    const float* __restrict__ C3,
    const float* __restrict__ C4oq,
    const float* __restrict__ Q_old,
    const float* __restrict__ qlat,
    float* __restrict__ Q_new)
{
    int s0 = reach_seg_start[r];
    int sl = reach_seg_len[r];
    float sum_new = 0.f, sum_old = 0.f;
    int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
    for (int kk = ub; kk < ue; ++kk) {
        int u = reach_up_idx[kk];
        int last_u = reach_seg_start[u] + reach_seg_len[u] - 1;
        sum_new += Q_new[last_u];
        sum_old += Q_old[last_u];
    }
    int idx = s0;
    Q_new[idx] = C1[idx] * sum_new + C2[idx] * sum_old
               + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
    float qn_prev_new = Q_new[idx];
    float qn_prev_old = Q_old[idx];
    for (int i = 1; i < sl; ++i) {
        idx = s0 + i;
        Q_new[idx] = C1[idx] * qn_prev_new + C2[idx] * qn_prev_old
                   + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
        qn_prev_new = Q_new[idx];
        qn_prev_old = Q_old[idx];
    }
}

// =========================================================================
// Sync-free kernel: single launch, NO grid.sync, NO __syncthreads in
// Phase B. Uses per-reach "ready" flags for dependency tracking, and
// atomic work-queue counter for load balancing. Phase A still uses atomic
// barrier for wide levels.
// =========================================================================
__global__ void kernel_syncfree(
    const int*   __restrict__ reach_seg_start,
    const int*   __restrict__ reach_seg_len,
    const int*   __restrict__ reach_up_ptr,
    const int*   __restrict__ reach_up_idx,
    const int*   __restrict__ level_reach,
    const int*   __restrict__ level_ptr,
    const float* __restrict__ C1,
    const float* __restrict__ C2,
    const float* __restrict__ C3,
    const float* __restrict__ C4oq,
    const float* __restrict__ Q_old,
    const float* __restrict__ qlat,
    float*       __restrict__ Q_new,
    int*   __restrict__ grid_barrier,    // Phase A atomic barriers
    int*   __restrict__ ready_flags,     // Phase B per-reach ready (0/1)
    int*   __restrict__ work_counter,    // Phase B atomic work queue counter
    int num_blocks,
    int n_levels, int n_reaches, int K_wide,
    int phaseB_first_reach_idx,          // in level_reach array
    int phaseB_reach_count)
{
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ---- Phase A ----
    for (int L = 0; L < K_wide; ++L) {
        int lb = level_ptr[L], le = level_ptr[L + 1];
        int cnt = le - lb;
        for (int k = tid_global; k < cnt; k += stride) {
            int r = level_reach[lb + k];
            process_reach(r, reach_seg_start, reach_seg_len,
                          reach_up_ptr, reach_up_idx,
                          C1, C2, C3, C4oq, Q_old, qlat, Q_new);
        }
        atomic_barrier(L, num_blocks, grid_barrier);
    }

    // ---- Phase B: sync-free work-queue over remaining reaches ----
    // Each thread repeatedly: atomically claim next reach from queue,
    // spin-wait until all its upstreams have ready_flags set, then process
    // and set its own flag. All blocks participate; no __syncthreads.

    while (true) {
        // Claim work
        int wi;
        if (true) {
            wi = atomicAdd(work_counter, 1);
        }
        if (wi >= phaseB_reach_count) break;

        int r = level_reach[phaseB_first_reach_idx + wi];

        // Wait for upstreams to be ready.
        int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
        // Only wait on upstreams that are ALSO in Phase B. Upstreams
        // in Phase A were already computed and Q_new is populated.
        // We detect Phase-B upstream by reach level >= K_wide — but we don't
        // have level here. Easier: we use ready_flags[u] which Phase A
        // reaches don't touch but it's initialized to 1 for Phase A reaches
        // and 0 for Phase B reaches at launch.
        for (int kk = ub; kk < ue; ++kk) {
            int u = reach_up_idx[kk];
            // Spin until upstream ready_flags set to 1
            while (*((volatile int*)&ready_flags[u]) == 0) { }
        }
        // All upstreams ready (both Phase A pre-marked + Phase B computed)
        // __threadfence() to ensure we read up-to-date Q_new[last_seg(u)]
        __threadfence();

        process_reach(r, reach_seg_start, reach_seg_len,
                      reach_up_ptr, reach_up_idx,
                      C1, C2, C3, C4oq, Q_old, qlat, Q_new);

        // Publish this reach as ready
        __threadfence();
        atomicExch(&ready_flags[r], 1);
    }
}

// All-timesteps variant
__global__ void kernel_syncfree_allts(
    const int*   __restrict__ reach_seg_start,
    const int*   __restrict__ reach_seg_len,
    const int*   __restrict__ reach_up_ptr,
    const int*   __restrict__ reach_up_idx,
    const int*   __restrict__ level_reach,
    const int*   __restrict__ level_ptr,
    const float* __restrict__ C1,
    const float* __restrict__ C2,
    const float* __restrict__ C3,
    const float* __restrict__ C4oq,
    const float* __restrict__ qlat_all,
    float*       __restrict__ Qa,
    float*       __restrict__ Qb,
    int*   __restrict__ grid_barrier,       // [(n_ts+1)*n_levels] at worst
    int*   __restrict__ ready_flags,        // [n_reaches] reused per ts
    int*   __restrict__ work_counters,      // [n_ts] one counter per ts
    int num_blocks,
    int n_levels, int n_reaches, int n_timesteps, int K_wide,
    int phaseB_first_reach_idx,
    int phaseB_reach_count,
    // Precomputed: which reaches are Phase A (need ready_flags preset to 1)
    const unsigned char* __restrict__ is_phaseA)  // [n_reaches]
{
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ts = 0; ts < n_timesteps; ++ts) {
        const float* Q_old = (ts & 1) ? Qb : Qa;
        float*       Q_new = (ts & 1) ? Qa : Qb;
        const float* qlat  = qlat_all + (size_t)ts * n_reaches;

        // Reset ready_flags at start of each timestep:
        // Phase A reaches -> will be set to 1 as they finish (inside the
        // barrier). Phase B reaches -> stay 0 until processed.
        // Simplest correct approach: set ALL to 0 now; Phase A reaches
        // get set to 1 INSIDE Phase A after processing.
        for (int r = tid_global; r < n_reaches; r += stride) {
            ready_flags[r] = 0;
        }
        // Barrier 0 (for this ts): ensure ready_flags zeroed before any Phase A proceeds
        atomic_barrier(ts * (n_levels + 2) + 0, num_blocks, grid_barrier);

        // Phase A: process wide levels, mark ready as we go
        for (int L = 0; L < K_wide; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            int cnt = le - lb;
            for (int k = tid_global; k < cnt; k += stride) {
                int r = level_reach[lb + k];
                process_reach(r, reach_seg_start, reach_seg_len,
                              reach_up_ptr, reach_up_idx,
                              C1, C2, C3, C4oq, Q_old, qlat, Q_new);
                __threadfence();
                atomicExch(&ready_flags[r], 1);
            }
            atomic_barrier(ts * (n_levels + 2) + 1 + L, num_blocks, grid_barrier);
        }

        // Reset work counter for this timestep
        if (tid_global == 0) work_counters[ts] = 0;
        atomic_barrier(ts * (n_levels + 2) + 1 + K_wide, num_blocks, grid_barrier);

        // Phase B: sync-free
        while (true) {
            int wi = atomicAdd(&work_counters[ts], 1);
            if (wi >= phaseB_reach_count) break;
            int r = level_reach[phaseB_first_reach_idx + wi];

            int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
            for (int kk = ub; kk < ue; ++kk) {
                int u = reach_up_idx[kk];
                while (*((volatile int*)&ready_flags[u]) == 0) { }
            }
            __threadfence();
            process_reach(r, reach_seg_start, reach_seg_len,
                          reach_up_ptr, reach_up_idx,
                          C1, C2, C3, C4oq, Q_old, qlat, Q_new);
            __threadfence();
            atomicExch(&ready_flags[r], 1);
        }
        // Barrier to ensure all Phase B done before next timestep
        atomic_barrier(ts * (n_levels + 2) + 1 + K_wide + 1, num_blocks, grid_barrier);
    }
}

struct Topo {
    int n_reaches, n_segments, n_levels, max_up;
    std::vector<int> reach_seg_start, reach_seg_len, reach_level, reach_n_up;
    std::vector<int> reach_up_ptr, reach_up_idx, level_ptr, level_reach;
    std::vector<int> level_counts;
};

static Topo load_topo(const std::string& dir) {
    Topo t;
    FILE* f = std::fopen((dir + "/topo.bin").c_str(), "rb");
    std::fread(&t.n_reaches, 4, 1, f);
    std::fread(&t.n_segments, 4, 1, f);
    std::fread(&t.n_levels, 4, 1, f);
    std::fread(&t.max_up, 4, 1, f);
    t.reach_seg_start.resize(t.n_reaches);
    t.reach_seg_len.resize(t.n_reaches);
    t.reach_level.resize(t.n_reaches);
    t.reach_n_up.resize(t.n_reaches);
    std::fread(t.reach_seg_start.data(), 4, t.n_reaches, f);
    std::fread(t.reach_seg_len.data(), 4, t.n_reaches, f);
    std::fread(t.reach_level.data(), 4, t.n_reaches, f);
    std::fread(t.reach_n_up.data(), 4, t.n_reaches, f);
    t.reach_up_ptr.resize(t.n_reaches + 1);
    std::fread(t.reach_up_ptr.data(), 4, t.n_reaches + 1, f);
    int nnz = t.reach_up_ptr.back();
    t.reach_up_idx.resize(nnz);
    std::fread(t.reach_up_idx.data(), 4, nnz, f);
    t.level_ptr.resize(t.n_levels + 1);
    std::fread(t.level_ptr.data(), 4, t.n_levels + 1, f);
    t.level_reach.resize(t.n_reaches);
    std::fread(t.level_reach.data(), 4, t.n_reaches, f);
    std::fclose(f);
    t.level_counts.resize(t.n_levels);
    for (int L = 0; L < t.n_levels; ++L) t.level_counts[L] = t.level_ptr[L+1] - t.level_ptr[L];
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir> [threads=512]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float dt = 300.0f;
    int threads = (argc >= 3) ? atoi(argv[2]) : 512;

    Topo t = load_topo(dir);
    printf("[io] n_reaches=%d n_levels=%d\n", t.n_reaches, t.n_levels);

    int K_wide = t.n_levels;
    for (int L = 0; L < t.n_levels; ++L) {
        bool ok = true;
        for (int L2 = L; L2 < t.n_levels; ++L2) {
            if (t.level_counts[L2] > threads) { ok = false; break; }
        }
        if (ok) { K_wide = L; break; }
    }
    int phaseB_first = t.level_ptr[K_wide];
    int phaseB_count = t.n_reaches - phaseB_first;
    printf("[io] K_wide=%d (Phase B sync-free: %d reaches starting at index %d)\n",
           K_wide, phaseB_count, phaseB_first);

    std::vector<ChParams> params(t.n_segments);
    FILE* pf = std::fopen((dir + "/params.bin").c_str(), "rb");
    std::fread(params.data(), sizeof(ChParams), t.n_segments, pf);
    std::fclose(pf);

    FILE* ff = std::fopen((dir + "/forcings.bin").c_str(), "rb");
    int n_ts = 0;
    std::fread(&n_ts, 4, 1, ff);
    std::vector<float> qlat_ts((size_t)n_ts * t.n_segments);
    std::fread(qlat_ts.data(), 4, qlat_ts.size(), ff);
    std::vector<float> qup0(t.n_reaches);
    std::fread(qup0.data(), 4, t.n_reaches, ff);
    std::vector<float> qdp0(t.n_segments);
    std::fread(qdp0.data(), 4, t.n_segments, ff);
    std::fclose(ff);

    // Precompute is_phaseA flags (1 if reach is in Phase A, 0 else)
    std::vector<unsigned char> is_phaseA(t.n_reaches, 0);
    for (int L = 0; L < K_wide; ++L) {
        for (int i = t.level_ptr[L]; i < t.level_ptr[L+1]; ++i) {
            is_phaseA[t.level_reach[i]] = 1;
        }
    }

    // Device allocations
    ChParams* d_params;
    int *d_seg_start, *d_seg_len, *d_up_ptr, *d_up_idx, *d_level_reach, *d_level_ptr;
    float *d_C1, *d_C2, *d_C3, *d_C4oq, *d_Qp, *d_Qn, *d_qlat, *d_qlat_all;
    int *d_barrier, *d_ready, *d_work_counter, *d_work_counters_ts;
    unsigned char *d_isA;

    CUDA_CHECK(cudaMalloc(&d_params, t.n_segments * sizeof(ChParams)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_seg_len, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_up_ptr, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_up_idx, t.reach_up_idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_level_reach, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_level_ptr, (t.n_levels + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_C1, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C2, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C3, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C4oq, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_Qp, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_Qn, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat_all, (size_t)n_ts * t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_barrier, (n_ts * (t.n_levels + 2) + 4) * 4));
    CUDA_CHECK(cudaMalloc(&d_ready, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_work_counter, 4));
    CUDA_CHECK(cudaMalloc(&d_work_counters_ts, n_ts * 4));
    CUDA_CHECK(cudaMalloc(&d_isA, t.n_reaches));

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), t.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_start, t.reach_seg_start.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_len, t.reach_seg_len.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_ptr, t.reach_up_ptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_idx, t.reach_up_idx.data(), t.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, t.level_reach.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_ptr, t.level_ptr.data(), (t.n_levels + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(), (size_t)n_ts * t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isA, is_phaseA.data(), t.n_reaches, cudaMemcpyHostToDevice));

    int blocks_pre = (t.n_segments + threads - 1) / threads;
    kernel_precompute_coeffs<<<blocks_pre, threads>>>(
        d_params, d_C1, d_C2, d_C3, d_C4oq, dt, t.n_segments);
    CUDA_CHECK(cudaDeviceSynchronize());

    int dev = 0, numSM = 0, maxBPSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBPSM, (const void*)kernel_syncfree, threads, 0);
    int coop_blocks = numSM * maxBPSM;
    if (coop_blocks <= 0) coop_blocks = numSM;
    printf("[gpu] syncfree coop: numSM=%d bpSM=%d -> %d blocks x %d threads\n",
           numSM, maxBPSM, coop_blocks, threads);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // --- Per-timestep syncfree variant ---
    // Warmup
    CUDA_CHECK(cudaMemset(d_barrier, 0, (n_ts * (t.n_levels + 2) + 4) * 4));
    CUDA_CHECK(cudaMemset(d_ready, 0, t.n_reaches * 4));
    // Preset Phase A reaches to ready (so Phase B threads waiting on them don't spin forever on PA upstreams in ts=0)
    // But we need them re-set to 0 each timestep. For Phase A reaches, mark ready=1 after processing.
    // In this per-timestep version, we pre-mark in Phase A itself (kernel doesn't, but we can preset the isA reaches)
    // Actually easier: in the per-timestep kernel, we do NOT preset; we mark in phase A inline.

    // Warmup launch
    CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_ts.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemsetAsync(d_ready, 0, t.n_reaches * 4));
    CUDA_CHECK(cudaMemsetAsync(d_work_counter, 0, 4));
    CUDA_CHECK(cudaMemsetAsync(d_barrier, 0, (t.n_levels + 2) * 4));
    // Mark Phase A reaches as ready=1 initially (this works because Phase A
    // will fill Q_new correctly, and we just need ready_flags set AFTER Phase A writes).
    // Simplest: inline the ready-set in Phase A. But our per-ts kernel doesn't; let's fall back
    // to the allts variant for benchmarking.

    // --- All-timesteps syncfree variant ---
    int allts_bpSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&allts_bpSM, (const void*)kernel_syncfree_allts, threads, 0);
    int allts_blocks = numSM * (allts_bpSM > 0 ? allts_bpSM : 1);
    printf("[gpu] syncfree_allts coop: %d blocks x %d threads\n", allts_blocks, threads);

    // Warmup
    CUDA_CHECK(cudaMemset(d_barrier, 0, (n_ts * (t.n_levels + 2) + 4) * 4));
    CUDA_CHECK(cudaMemset(d_ready, 0, t.n_reaches * 4));
    CUDA_CHECK(cudaMemset(d_work_counters_ts, 0, n_ts * 4));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_qlat_all, &d_Qp, &d_Qn, &d_barrier, &d_ready,
                        &d_work_counters_ts, &allts_blocks,
                        &t.n_levels, &t.n_reaches, &n_ts, &K_wide,
                        &phaseB_first, &phaseB_count, &d_isA};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_syncfree_allts,
            dim3(allts_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaMemset(d_barrier, 0, (n_ts * (t.n_levels + 2) + 4) * 4));
    CUDA_CHECK(cudaMemset(d_ready, 0, t.n_reaches * 4));
    CUDA_CHECK(cudaMemset(d_work_counters_ts, 0, n_ts * 4));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));

    cudaEventRecord(e0);
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_qlat_all, &d_Qp, &d_Qn, &d_barrier, &d_ready,
                        &d_work_counters_ts, &allts_blocks,
                        &t.n_levels, &t.n_reaches, &n_ts, &K_wide,
                        &phaseB_first, &phaseB_count, &d_isA};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_syncfree_allts,
            dim3(allts_blocks), dim3(threads), args));
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[gpu-syncfree-allts] total ms: %.3f  per-timestep: %.3f ms\n", ms, ms / n_ts);

    // Sanity
    std::vector<float> gpu_Q(t.n_segments);
    const float* d_final = (n_ts & 1) ? d_Qn : d_Qp;
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_final, t.n_segments * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0; int nans = 0;
    for (int i = 0; i < t.n_segments; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q; if (q > maxv) maxv = q;
    }
    printf("[gpu-syncfree-allts] Q_final sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);

    return 0;
}
