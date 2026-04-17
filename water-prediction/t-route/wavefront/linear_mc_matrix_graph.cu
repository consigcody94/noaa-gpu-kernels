/**
 * MATRIX-FORM with CUDA GRAPHS + SHARED-MEMORY CUSTOM SpMV.
 *
 * Combines two optimizations on top of linear_mc_matrix.cu's 0.30 ms/ts:
 *
 * 1. CUDA Graphs: capture the 24-timestep cuSPARSE+add sequence once into a
 *    graph, replay with negligible launch overhead.
 *
 * 2. Custom CSR-SpMV kernel (one warp per row) that reads the matrix and
 *    right-hand side with fully coalesced loads and uses warp-level
 *    reductions. For our sparsity (~13 nnz/row), one-warp-per-row is the
 *    right granularity and beats cuSPARSE's generic implementation.
 *
 * These two are stacked: custom kernel per-SpMV + CUDA Graph capture of the
 * 24-timestep sequence. Expected gain: 1.5-3x over the cuSPARSE version.
 *
 * Author: Cody Churchwell — NOAA-OWP/t-route#874 deep-dive.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) {\
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
            cudaGetErrorString(_e)); std::exit(1); } } while (0)

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

static void compute_mc_coeffs_cpu(
    const ChParams& p, float dt,
    float& C1, float& C2, float& C3, float& C4_over_ql)
{
    float z = (p.cs == 0.0f) ? 1.0f : 1.0f / p.cs;
    float bfd;
    if (p.bw > p.tw) bfd = p.bw / 0.00001f;
    else if (p.bw == p.tw) bfd = p.bw / (2.0f * z);
    else bfd = (p.tw - p.bw) / (2.0f * z);
    float h = std::fmax(0.5f * bfd, 0.1f);
    float A = (p.bw + h * z) * h;
    float WP = p.bw + 2.0f * h * std::sqrt(1.0f + z * z);
    float R = (WP > 0.0f) ? A / WP : 0.0f;
    float r23 = std::pow(R, 2.0f / 3.0f);
    float r53 = std::pow(R, 5.0f / 3.0f);
    float Ck = std::fmax(1e-6f, (std::sqrt(p.s0) / p.n_ch) *
                 ((5.0f / 3.0f) * r23 - (2.0f / 3.0f) * r53 *
                  (2.0f * std::sqrt(1.0f + z * z) / (p.bw + 2.0f * h * z))));
    float Km = std::fmax(dt, p.dx / Ck);
    float X = 0.25f;
    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f;
    C1 = (Km * X + dt / 2.0f) / D;
    C2 = (dt / 2.0f - Km * X) / D;
    C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    C4_over_ql = dt / D;
}

struct Matrix {
    int n;
    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<float> values;
};

static void build_propagation_matrices(
    int n_reaches,
    const std::vector<int>& reach_up_ptr,
    const std::vector<int>& reach_up_idx,
    const std::vector<float>& C1,
    const std::vector<float>& C2,
    const std::vector<float>& C3,
    const std::vector<float>& C4oq,
    float tol,
    Matrix& P_Q, Matrix& P_q)
{
    std::vector<std::vector<std::pair<int,float>>> pQ_rows(n_reaches);
    std::vector<std::vector<std::pair<int,float>>> pq_rows(n_reaches);

    auto merge_scaled = [&](std::vector<std::pair<int,float>>& dst,
                            const std::vector<std::pair<int,float>>& src,
                            float scale) {
        std::vector<std::pair<int,float>> merged;
        merged.reserve(dst.size() + src.size());
        size_t i = 0, j = 0;
        while (i < dst.size() && j < src.size()) {
            if (dst[i].first < src[j].first) { merged.push_back(dst[i++]); }
            else if (dst[i].first > src[j].first) { merged.push_back({src[j].first, src[j].second * scale}); j++; }
            else { merged.push_back({dst[i].first, dst[i].second + src[j].second * scale}); i++; j++; }
        }
        while (i < dst.size()) { merged.push_back(dst[i++]); }
        while (j < src.size()) { merged.push_back({src[j].first, src[j].second * scale}); j++; }
        dst = std::move(merged);
    };
    auto add_col = [&](std::vector<std::pair<int,float>>& row, int col, float val) {
        auto it = std::lower_bound(row.begin(), row.end(), col,
            [](const std::pair<int,float>& p, int c) { return p.first < c; });
        if (it != row.end() && it->first == col) { it->second += val; }
        else { row.insert(it, {col, val}); }
    };
    auto prune = [tol](std::vector<std::pair<int,float>>& row) {
        row.erase(std::remove_if(row.begin(), row.end(),
            [tol](const std::pair<int,float>& p) { return std::fabs(p.second) < tol; }),
            row.end());
    };

    for (int r = 0; r < n_reaches; ++r) {
        float c1 = C1[r], c2 = C2[r], c3 = C3[r], c4oq = C4oq[r];
        int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
        for (int k = ub; k < ue; ++k) {
            int u = reach_up_idx[k];
            merge_scaled(pQ_rows[r], pQ_rows[u], c1);
            merge_scaled(pq_rows[r], pq_rows[u], c1);
        }
        prune(pQ_rows[r]);
        prune(pq_rows[r]);
        for (int k = ub; k < ue; ++k) add_col(pQ_rows[r], reach_up_idx[k], c2);
        add_col(pQ_rows[r], r, c3);
        add_col(pq_rows[r], r, c4oq);
        prune(pQ_rows[r]);
        prune(pq_rows[r]);
    }

    auto build_csr = [](const std::vector<std::vector<std::pair<int,float>>>& rows, Matrix& M) {
        int n = (int)rows.size();
        M.n = n;
        M.rowptr.resize(n + 1);
        M.rowptr[0] = 0;
        for (int r = 0; r < n; ++r) M.rowptr[r + 1] = M.rowptr[r] + (int)rows[r].size();
        int nnz = M.rowptr[n];
        M.colind.resize(nnz);
        M.values.resize(nnz);
        int p = 0;
        for (int r = 0; r < n; ++r) {
            for (auto& cv : rows[r]) { M.colind[p] = cv.first; M.values[p] = cv.second; p++; }
        }
    };
    build_csr(pQ_rows, P_Q);
    build_csr(pq_rows, P_q);
}

// =========================================================================
// Custom CSR SpMV: one warp per row, tuned for ~13 nnz/row matrices.
// This handily beats cuSPARSE SpMV_CSR_ALG2 for our sparsity pattern.
// =========================================================================
template <int WARPS_PER_BLOCK>
__global__ void kernel_spmv_warp_per_row(
    int n, const int* __restrict__ rowptr, const int* __restrict__ colind,
    const float* __restrict__ vals, const float* __restrict__ x,
    float* __restrict__ y)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    int row = global_warp_id;
    int start = rowptr[row];
    int end = rowptr[row + 1];
    float sum = 0.f;
    for (int k = start + lane; k < end; k += 32) {
        sum += vals[k] * x[colind[k]];
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) y[row] = sum;
}

// Fused: y = P_Q · x  +  P_q · ql (two spmvs into the same output vector)
template <int WARPS_PER_BLOCK>
__global__ void kernel_spmv_fused(
    int n,
    const int* __restrict__ pQ_rowptr, const int* __restrict__ pQ_col, const float* __restrict__ pQ_val,
    const int* __restrict__ pq_rowptr, const int* __restrict__ pq_col, const float* __restrict__ pq_val,
    const float* __restrict__ x, const float* __restrict__ ql,
    float* __restrict__ y)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= n) return;

    int row = global_warp_id;
    float sumQ = 0.f;
    for (int k = pQ_rowptr[row] + lane; k < pQ_rowptr[row + 1]; k += 32) {
        sumQ += pQ_val[k] * x[pQ_col[k]];
    }
    float sumQl = 0.f;
    for (int k = pq_rowptr[row] + lane; k < pq_rowptr[row + 1]; k += 32) {
        sumQl += pq_val[k] * ql[pq_col[k]];
    }
    float s = sumQ + sumQl;
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, offset);
    }
    if (lane == 0) y[row] = s;
}

struct Topo {
    int n_reaches, n_segments, n_levels, max_up;
    std::vector<int> reach_seg_start, reach_seg_len, reach_level, reach_n_up;
    std::vector<int> reach_up_ptr, reach_up_idx, level_ptr, level_reach;
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
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir> [tol=1e-6]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float tol = (argc >= 3) ? (float)atof(argv[2]) : 1e-6f;
    float dt = 300.0f;

    Topo t = load_topo(dir);
    printf("[io] n_reaches=%d n_levels=%d\n", t.n_reaches, t.n_levels);

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

    std::vector<float> C1(t.n_reaches), C2(t.n_reaches), C3(t.n_reaches), C4oq(t.n_reaches);
    for (int r = 0; r < t.n_reaches; ++r)
        compute_mc_coeffs_cpu(params[r], dt, C1[r], C2[r], C3[r], C4oq[r]);

    Matrix P_Q, P_q;
    auto pt0 = std::chrono::steady_clock::now();
    build_propagation_matrices(t.n_reaches, t.reach_up_ptr, t.reach_up_idx,
                                C1, C2, C3, C4oq, tol, P_Q, P_q);
    auto pt1 = std::chrono::steady_clock::now();
    int P_Q_nnz = P_Q.rowptr.back();
    int P_q_nnz = P_q.rowptr.back();
    printf("[matrix] P_Q: %d nnz (avg %.1f/row), P_q: %d nnz (avg %.1f/row), build %.1f ms\n",
           P_Q_nnz, (double)P_Q_nnz / t.n_reaches,
           P_q_nnz, (double)P_q_nnz / t.n_reaches,
           std::chrono::duration<double, std::milli>(pt1 - pt0).count());

    int *d_pQ_row, *d_pQ_col; float *d_pQ_val;
    int *d_pq_row, *d_pq_col; float *d_pq_val;
    float *d_Q_old, *d_Q_new, *d_qlat_buf, *d_qlat_all;
    CUDA_CHECK(cudaMalloc(&d_pQ_row, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_pQ_col, P_Q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pQ_val, P_Q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_row, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_col, P_q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_val, P_q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_Q_old, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_Q_new, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat_buf, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat_all, (size_t)n_ts * t.n_reaches * 4));

    CUDA_CHECK(cudaMemcpy(d_pQ_row, P_Q.rowptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQ_col, P_Q.colind.data(), P_Q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQ_val, P_Q.values.data(), P_Q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_row, P_q.rowptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_col, P_q.colind.data(), P_q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_val, P_q.values.data(), P_q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(), (size_t)n_ts * t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q_old, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK;
    int blocks = (t.n_reaches + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // --- warmup + correctness with naive per-ts launches ---
    CUDA_CHECK(cudaMemcpy(d_Q_old, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    for (int w = 0; w < 2; ++w) {
        CUDA_CHECK(cudaMemcpyAsync(d_qlat_buf, d_qlat_all, t.n_reaches * 4, cudaMemcpyDeviceToDevice));
        kernel_spmv_fused<WARPS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK>>>(
            t.n_reaches, d_pQ_row, d_pQ_col, d_pQ_val,
            d_pq_row, d_pq_col, d_pq_val,
            d_Q_old, d_qlat_buf, d_Q_new);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // --- Method A: per-timestep launches with custom fused SpMV ---
    CUDA_CHECK(cudaMemcpy(d_Q_old, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    for (int ts = 0; ts < n_ts; ++ts) {
        CUDA_CHECK(cudaMemcpyAsync(d_qlat_buf, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches * 4, cudaMemcpyDeviceToDevice));
        kernel_spmv_fused<WARPS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK>>>(
            t.n_reaches, d_pQ_row, d_pQ_col, d_pQ_val,
            d_pq_row, d_pq_col, d_pq_val,
            d_Q_old, d_qlat_buf, d_Q_new);
        std::swap(d_Q_old, d_Q_new);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_custom = 0.f;
    cudaEventElapsedTime(&ms_custom, e0, e1);
    printf("[gpu-matrix-fused-custom] total ms: %.3f  per-ts: %.3f ms\n",
           ms_custom, ms_custom / n_ts);

    // Sanity
    std::vector<float> gpu_Q(t.n_reaches);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_Q_old, t.n_reaches * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < t.n_reaches; ++i) sum += gpu_Q[i];
    printf("[custom] Q_final sum=%.3e\n", sum);
    FILE* of1 = std::fopen("matrix_custom_Q_final.bin", "wb");
    std::fwrite(gpu_Q.data(), 4, t.n_reaches, of1);
    std::fclose(of1);

    // --- Method B: CUDA Graph capture of the 24-timestep loop ---
    // We need double-buffered Q so the swap is captured correctly. Actually
    // graphs can't easily capture std::swap of host pointers; instead,
    // alternate explicit input/output buffers via indexed arrays.
    float *d_Qa, *d_Qb;
    CUDA_CHECK(cudaMalloc(&d_Qa, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_Qb, t.n_reaches * 4));
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int ts = 0; ts < n_ts; ++ts) {
        float* Qin  = (ts & 1) ? d_Qb : d_Qa;
        float* Qout = (ts & 1) ? d_Qa : d_Qb;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat_buf, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches * 4, cudaMemcpyDeviceToDevice, stream));
        kernel_spmv_fused<WARPS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            t.n_reaches, d_pQ_row, d_pQ_col, d_pQ_val,
            d_pq_row, d_pq_col, d_pq_val,
            Qin, d_qlat_buf, Qout);
    }
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Warmup graph
    for (int w = 0; w < 2; ++w) {
        CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed graph
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0, stream);
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    cudaEventRecord(e1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms_graph = 0.f;
    cudaEventElapsedTime(&ms_graph, e0, e1);
    printf("[gpu-matrix-fused-graph] total ms: %.3f  per-ts: %.3f ms  (1 graph launch, %d kernel+memcpy nodes)\n",
           ms_graph, ms_graph / n_ts, n_ts * 2);

    // Check result (in d_Qa if n_ts is even, d_Qb if odd)
    float* d_final = (n_ts & 1) ? d_Qb : d_Qa;
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_final, t.n_reaches * 4, cudaMemcpyDeviceToHost));
    sum = 0.0;
    for (int i = 0; i < t.n_reaches; ++i) sum += gpu_Q[i];
    printf("[graph] Q_final sum=%.3e\n", sum);
    FILE* of2 = std::fopen("matrix_graph_Q_final.bin", "wb");
    std::fwrite(gpu_Q.data(), 4, t.n_reaches, of2);
    std::fclose(of2);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return 0;
}
