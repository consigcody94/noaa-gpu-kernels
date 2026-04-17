/**
 * Linearized MC with a TWO-PHASE split kernel:
 *
 *  Phase A (wide levels 0..K_wide):  cooperative grid with N blocks,
 *      one grid.sync() per level. Good for levels with enough reaches
 *      to saturate the grid.
 *
 *  Phase B (thin levels K_wide..end): ONE thread block processes all
 *      remaining levels with __syncthreads() between levels. Avoids
 *      ~1100 grid.sync() calls (each ~3.5 us) in the deep dendritic
 *      tail of the CONUS tree.
 *
 * K_wide is chosen at runtime: the first level whose reach count drops
 * below a threshold (e.g. 256, one block-worth). If every subsequent
 * level also stays below that threshold, Phase B is correct.
 *
 * On the real 309K CONUS tree: 1143 total levels, 50 "wide" levels
 * (count >= 128), 1093 "thin" levels (count <= 134). Expected speedup
 * vs per-level-launch or all-grid-sync: ~5-20x.
 *
 * Author: Cody Churchwell — NOAA-OWP/t-route#874 deep-dive follow-up
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

// Process one reach's full chain (1-segment version)
__device__ __forceinline__ void process_reach(
    int r,
    const int* reach_seg_start, const int* reach_seg_len,
    const int* reach_up_ptr, const int* reach_up_idx,
    const float* C1, const float* C2, const float* C3, const float* C4oq,
    const float* Q_old, const float* qlat, float* Q_new)
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
// Two-phase split kernel. Launched as ONE cooperative kernel per timestep.
//   Phase A: wide levels 0..K_wide-1, grid-wide parallel + grid.sync
//   Phase B: thin levels K_wide..n_levels-1, ONE block with __syncthreads
// Thin-levels compute done by block 0 only; other blocks busy-wait at
// the outer grid.sync at the end.
// =========================================================================
__global__ void kernel_split_timestep(
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
    int n_levels, int n_reaches, int K_wide)
{
    cg::grid_group grid = cg::this_grid();
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ---- Phase A: wide levels, grid-parallel ----
    for (int L = 0; L < K_wide; ++L) {
        int lb = level_ptr[L], le = level_ptr[L + 1];
        int cnt = le - lb;
        for (int k = tid_global; k < cnt; k += stride) {
            int r = level_reach[lb + k];
            process_reach(r, reach_seg_start, reach_seg_len,
                          reach_up_ptr, reach_up_idx,
                          C1, C2, C3, C4oq, Q_old, qlat, Q_new);
        }
        grid.sync();
    }

    // ---- Phase B: thin tail, single block (block 0) ----
    if (blockIdx.x == 0) {
        int tid = threadIdx.x;
        for (int L = K_wide; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            int cnt = le - lb;
            if (tid < cnt) {
                int r = level_reach[lb + tid];
                process_reach(r, reach_seg_start, reach_seg_len,
                              reach_up_ptr, reach_up_idx,
                              C1, C2, C3, C4oq, Q_old, qlat, Q_new);
            }
            __syncthreads();
        }
    }
    // All blocks converge here (implicit on kernel exit for this launch).
}

// All-timesteps: one launch, N timesteps internal, two-phase each
__global__ void kernel_split_allts(
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
    int n_levels, int n_reaches, int n_timesteps, int K_wide)
{
    cg::grid_group grid = cg::this_grid();
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ts = 0; ts < n_timesteps; ++ts) {
        const float* Q_old = (ts & 1) ? Qb : Qa;
        float*       Q_new = (ts & 1) ? Qa : Qb;
        const float* qlat  = qlat_all + (size_t)ts * n_reaches;

        // Phase A: wide levels
        for (int L = 0; L < K_wide; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            int cnt = le - lb;
            for (int k = tid_global; k < cnt; k += stride) {
                int r = level_reach[lb + k];
                process_reach(r, reach_seg_start, reach_seg_len,
                              reach_up_ptr, reach_up_idx,
                              C1, C2, C3, C4oq, Q_old, qlat, Q_new);
            }
            grid.sync();
        }

        // Phase B: thin tail in block 0
        if (blockIdx.x == 0) {
            int tid = threadIdx.x;
            for (int L = K_wide; L < n_levels; ++L) {
                int lb = level_ptr[L], le = level_ptr[L + 1];
                int cnt = le - lb;
                if (tid < cnt) {
                    int r = level_reach[lb + tid];
                    process_reach(r, reach_seg_start, reach_seg_len,
                                  reach_up_ptr, reach_up_idx,
                                  C1, C2, C3, C4oq, Q_old, qlat, Q_new);
                }
                __syncthreads();
            }
        }
        grid.sync();   // sync before next timestep's Phase A (reads Q_new as Q_old)
    }
}

// =========================================================================
// I/O
// =========================================================================
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
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir> [threads=256]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float dt = 300.0f;
    int threads = (argc >= 3) ? atoi(argv[2]) : 256;

    Topo t = load_topo(dir);
    printf("[io] n_reaches=%d n_levels=%d\n", t.n_reaches, t.n_levels);

    // Determine K_wide: the first level L such that for all L' >= L, count[L'] <= threads.
    int K_wide = t.n_levels;
    for (int L = 0; L < t.n_levels; ++L) {
        bool ok = true;
        for (int L2 = L; L2 < t.n_levels; ++L2) {
            if (t.level_counts[L2] > threads) { ok = false; break; }
        }
        if (ok) { K_wide = L; break; }
    }
    printf("[io] K_wide=%d (Phase A: %d wide levels, Phase B: %d thin levels fit in one block of %d)\n",
           K_wide, K_wide, t.n_levels - K_wide, threads);

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

    // Device allocs
    ChParams* d_params;
    int *d_seg_start, *d_seg_len, *d_up_ptr, *d_up_idx, *d_level_reach, *d_level_ptr;
    float *d_C1, *d_C2, *d_C3, *d_C4oq, *d_Qp, *d_Qn, *d_qlat, *d_qlat_all;
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

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), t.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_start, t.reach_seg_start.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_len, t.reach_seg_len.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_ptr, t.reach_up_ptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_idx, t.reach_up_idx.data(), t.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, t.level_reach.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_ptr, t.level_ptr.data(), (t.n_levels + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(), (size_t)n_ts * t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));

    int blocks_pre = (t.n_segments + threads - 1) / threads;
    kernel_precompute_coeffs<<<blocks_pre, threads>>>(
        d_params, d_C1, d_C2, d_C3, d_C4oq, dt, t.n_segments);
    CUDA_CHECK(cudaDeviceSynchronize());

    int dev = 0, numSM = 0, maxBPSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBPSM, (const void*)kernel_split_timestep, threads, 0);
    int coop_blocks = numSM * maxBPSM;
    if (coop_blocks <= 0) coop_blocks = numSM;
    printf("[gpu] coop: numSM=%d bpSM=%d -> %d blocks x %d threads\n", numSM, maxBPSM, coop_blocks, threads);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // --- Per-timestep split variant ---
    // Warmup
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_Qp, &d_qlat, &d_Qn, &t.n_levels, &t.n_reaches, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_timestep,
            dim3(coop_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    for (int ts = 0; ts < n_ts; ++ts) {
        const float* qlat_host = qlat_ts.data() + (size_t)ts * t.n_segments;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_host, t.n_segments * 4, cudaMemcpyHostToDevice));
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_Qp, &d_qlat, &d_Qn, &t.n_levels, &t.n_reaches, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_timestep,
            dim3(coop_blocks), dim3(threads), args));
        std::swap(d_Qp, d_Qn);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_per_ts = 0.f;
    cudaEventElapsedTime(&ms_per_ts, e0, e1);
    printf("[gpu-split-per-ts] total ms: %.3f per-timestep: %.3f ms\n", ms_per_ts, ms_per_ts / n_ts);

    // --- All-timesteps-in-one-launch ---
    int allts_bpSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&allts_bpSM, (const void*)kernel_split_allts, threads, 0);
    int allts_blocks = numSM * (allts_bpSM > 0 ? allts_bpSM : 1);
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_qlat_all, &d_Qp, &d_Qn,
                        &t.n_levels, &t.n_reaches, &n_ts, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_allts,
            dim3(allts_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_qlat_all, &d_Qp, &d_Qn,
                        &t.n_levels, &t.n_reaches, &n_ts, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_allts,
            dim3(allts_blocks), dim3(threads), args));
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_allts = 0.f;
    cudaEventElapsedTime(&ms_allts, e0, e1);
    printf("[gpu-split-allts] total ms: %.3f per-timestep: %.3f ms (1 launch)\n",
           ms_allts, ms_allts / n_ts);

    // Read back, sanity check
    std::vector<float> gpu_Q(t.n_segments);
    const float* d_final = (n_ts & 1) ? d_Qn : d_Qp;
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_final, t.n_segments * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0;
    int nans = 0;
    for (int i = 0; i < t.n_segments; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q; if (q > maxv) maxv = q;
    }
    printf("[gpu-split-allts] Q_final sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);

    return 0;
}
