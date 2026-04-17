/**
 * Linearized MC with a PERSISTENT (grid-synchronized) kernel.
 *
 * Rationale: on the real CONUS topology (309K reaches, 1143 topological
 * levels), launching one CUDA kernel per level = 1143 launches per timestep
 * × 24 timesteps = 27K launches. Even at 5 us per launch that's ~135 ms of
 * pure overhead.
 *
 * This kernel fires ONCE per timestep (or, with an outer driver kernel,
 * once for the entire simulation). Each thread block picks up work from
 * a global work queue; grid-level cooperative_groups::this_grid().sync()
 * replaces host-side kernel launches as the level barrier.
 *
 * Inspired by Gondhalekar et al., "Mapping Sparse Triangular Solves to
 * GPUs via Fine-grained Domain Decomposition" (arXiv:2508.04917, 2025).
 *
 * Author: Cody Churchwell — follow-up to NOAA-OWP/t-route#874
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

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                        \
                #expr, __FILE__, __LINE__, cudaGetErrorString(_err));          \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

// Compute Muskingum-Cunge coefficients from channel geometry at reference depth.
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
    float twl = p.bw + 2.0f * z * h;
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

// ==========================================================================
// Persistent grid-synchronized kernel: one launch executes ALL levels for
// ONE timestep. Replaces per-level launches with grid.sync().
// ==========================================================================
// All-timesteps-in-one-launch variant. qlat is laid out as (n_timesteps, n_segments).
// Internal timestep loop swaps Q_old/Q_new every step without re-launching.
__global__ void kernel_allts_linear(
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
    const float* __restrict__ qlat_all,     // [n_timesteps, n_segments]
    float*       __restrict__ Qa,           // ping
    float*       __restrict__ Qb,           // pong
    int n_levels, int n_reaches, int n_timesteps)
{
    cg::grid_group grid = cg::this_grid();
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int ts = 0; ts < n_timesteps; ++ts) {
        const float* Q_old = (ts & 1) ? Qb : Qa;
        float*       Q_new = (ts & 1) ? Qa : Qb;
        const float* qlat  = qlat_all + (size_t)ts * n_reaches;
        for (int L = 0; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            int cnt = le - lb;
            for (int k = tid_global; k < cnt; k += stride) {
                int r = level_reach[lb + k];
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
            grid.sync();
        }
    }
}

__global__ void kernel_persistent_linear_timestep(
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
    int n_levels,
    int n_reaches)
{
    cg::grid_group grid = cg::this_grid();

    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int L = 0; L < n_levels; ++L) {
        int lb = level_ptr[L];
        int le = level_ptr[L + 1];
        int cnt = le - lb;

        for (int k = tid_global; k < cnt; k += stride) {
            int r = level_reach[lb + k];
            int s0 = reach_seg_start[r];
            int sl = reach_seg_len[r];

            // Aggregate upstream contributions (last seg of each parent reach).
            float sum_new = 0.f, sum_old = 0.f;
            int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
            for (int kk = ub; kk < ue; ++kk) {
                int u = reach_up_idx[kk];
                int last_u = reach_seg_start[u] + reach_seg_len[u] - 1;
                sum_new += Q_new[last_u];
                sum_old += Q_old[last_u];
            }

            // Segment 0
            int idx = s0;
            Q_new[idx] = C1[idx] * sum_new + C2[idx] * sum_old
                       + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
            float qn_prev_new = Q_new[idx];
            float qn_prev_old = Q_old[idx];

            // Interior segments
            for (int i = 1; i < sl; ++i) {
                idx = s0 + i;
                Q_new[idx] = C1[idx] * qn_prev_new + C2[idx] * qn_prev_old
                           + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
                qn_prev_new = Q_new[idx];
                qn_prev_old = Q_old[idx];
            }
        }

        grid.sync();  // wait for all blocks before moving to next level
    }
}

// ==========================================================================
// I/O structs matching wavefront_mc.cu binary format
// ==========================================================================
struct Topo {
    int n_reaches, n_segments, n_levels, max_up;
    std::vector<int> reach_seg_start, reach_seg_len, reach_level, reach_n_up;
    std::vector<int> reach_up_ptr, reach_up_idx;
    std::vector<int> level_ptr, level_reach;
};

static Topo load_topo(const std::string& dir) {
    Topo t;
    FILE* f = std::fopen((dir + "/topo.bin").c_str(), "rb");
    if (!f) { fprintf(stderr, "cant open %s/topo.bin\n", dir.c_str()); std::exit(1); }
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
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir>\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float dt = 300.0f;

    printf("[io] loading %s\n", dir.c_str());
    Topo t = load_topo(dir);
    printf("[io] n_reaches=%d n_segments=%d n_levels=%d max_up=%d\n",
           t.n_reaches, t.n_segments, t.n_levels, t.max_up);

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
    printf("[io] n_timesteps=%d\n", n_ts);

    // Device allocations
    ChParams* d_params;
    int *d_seg_start, *d_seg_len, *d_up_ptr, *d_up_idx, *d_level_reach, *d_level_ptr;
    float *d_C1, *d_C2, *d_C3, *d_C4oq, *d_Qp, *d_Qn, *d_qlat;

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

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), t.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_start, t.reach_seg_start.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_len, t.reach_seg_len.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_ptr, t.reach_up_ptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_idx, t.reach_up_idx.data(), t.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, t.level_reach.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_ptr, t.level_ptr.data(), (t.n_levels + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (t.n_reaches + threads - 1) / threads;
    kernel_precompute_coeffs<<<blocks, threads>>>(d_params, d_C1, d_C2, d_C3, d_C4oq, dt, t.n_segments);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Determine max blocks for cooperative launch
    int dev = 0;
    int numSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, (const void*)kernel_persistent_linear_timestep, threads, 0);
    int coop_blocks = numSM * maxBlocksPerSM;
    // Cap to a reasonable minimum (some SMs may not be able to fit blocks at all)
    if (coop_blocks <= 0) coop_blocks = numSM;
    printf("[gpu] cooperative: numSM=%d maxBlocksPerSM=%d -> coop_blocks=%d\n",
           numSM, maxBlocksPerSM, coop_blocks);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // warmup
    {
        void* args[] = {
            &d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
            &d_level_reach, &d_level_ptr,
            &d_C1, &d_C2, &d_C3, &d_C4oq,
            &d_Qp, &d_qlat, &d_Qn, &t.n_levels, &t.n_reaches
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)kernel_persistent_linear_timestep,
            dim3(coop_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[gpu] running %d timesteps\n", n_ts);
    cudaEventRecord(e0);

    for (int ts = 0; ts < n_ts; ++ts) {
        const float* qlat_host = qlat_ts.data() + (size_t)ts * t.n_segments;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_host, t.n_segments * 4, cudaMemcpyHostToDevice));
        void* args[] = {
            &d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
            &d_level_reach, &d_level_ptr,
            &d_C1, &d_C2, &d_C3, &d_C4oq,
            &d_Qp, &d_qlat, &d_Qn, &t.n_levels, &t.n_reaches
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)kernel_persistent_linear_timestep,
            dim3(coop_blocks), dim3(threads), args));
        std::swap(d_Qp, d_Qn);
    }

    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, e0, e1);
    printf("[gpu-persistent] total ms: %.3f  per-timestep: %.3f ms\n",
           gpu_ms, gpu_ms / n_ts);

    // ---- ALL-TIMESTEPS-IN-ONE-LAUNCH variant ----
    // Upload ALL qlat timesteps once, then one cooperative kernel runs all 24 steps.
    float* d_qlat_all = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qlat_all, (size_t)n_ts * t.n_segments * 4));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(),
                          (size_t)n_ts * t.n_segments * 4, cudaMemcpyHostToDevice));
    // Reset Qp to initial qdp0
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));

    int allts_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &allts_blocks, (const void*)kernel_allts_linear, threads, 0);
    allts_blocks *= numSM;
    if (allts_blocks <= 0) allts_blocks = numSM;

    // warmup
    {
        void* args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr,
                        &d_C1, &d_C2, &d_C3, &d_C4oq,
                        &d_qlat_all, &d_Qp, &d_Qn,
                        &t.n_levels, &t.n_reaches, &n_ts};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_allts_linear,
            dim3(allts_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset Qp before timed run
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));

    cudaEvent_t a0, a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
    cudaEventRecord(a0);
    void* allts_args[] = {&d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                          &d_level_reach, &d_level_ptr,
                          &d_C1, &d_C2, &d_C3, &d_C4oq,
                          &d_qlat_all, &d_Qp, &d_Qn,
                          &t.n_levels, &t.n_reaches, &n_ts};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_allts_linear,
        dim3(allts_blocks), dim3(threads), allts_args));
    cudaEventRecord(a1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float allts_ms = 0.f;
    cudaEventElapsedTime(&allts_ms, a0, a1);
    printf("[gpu-allts] total ms: %.3f  per-timestep: %.3f ms (1 kernel launch)\n",
           allts_ms, allts_ms / n_ts);
    cudaFree(d_qlat_all);

    // Read back for sanity
    std::vector<float> gpu_Q(t.n_segments);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_Qp, t.n_segments * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0;
    int nans = 0;
    for (int i = 0; i < t.n_segments; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q;
        if (q > maxv) maxv = q;
    }
    printf("[gpu-persistent] Q_final sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);

    cudaFree(d_params); cudaFree(d_seg_start); cudaFree(d_seg_len);
    cudaFree(d_up_ptr); cudaFree(d_up_idx); cudaFree(d_level_reach); cudaFree(d_level_ptr);
    cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3); cudaFree(d_C4oq);
    cudaFree(d_Qp); cudaFree(d_Qn); cudaFree(d_qlat);
    return 0;
}
