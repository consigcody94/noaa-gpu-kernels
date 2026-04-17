/**
 * MATRIX-FORM linearized MC via precomputed LTI propagation matrix.
 *
 * For linear MC with fixed coefficients, each timestep is:
 *   Q_new = P_Q · Q_old  +  P_q · qlat
 *
 * where:
 *   P_Q = A^(-1) · (C2·N + C3·I)     [single-timestep propagation]
 *   P_q = A^(-1) · diag(C4)          [qlat propagation]
 *   A   = I - C1·N
 *   N   = upstream adjacency (N[r,u] = 1 iff u is upstream of r)
 *
 * Both P_Q and P_q are lower-triangular. For a tree, row r of these
 * matrices has nonzeros only at r's ancestors. We precompute them on
 * CPU via the iterative expansion of A^(-1) = sum_k (C1·N)^k truncated
 * at a tolerance.
 *
 * Per timestep: two cuSPARSE SpMV calls. No level synchronization needed.
 *
 * This is the DiffRoute-style approach (Hascoet 2026), specialized for
 * a single-timestep transition rather than multi-step IRF convolution.
 *
 * Author: Cody Churchwell — NOAA-OWP/t-route#874 deep-dive.
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
#include <cusparse.h>

__global__ void kernel_add_inplace(float* __restrict__ dst,
                                   const float* __restrict__ src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) {\
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
            cudaGetErrorString(_e)); std::exit(1); } } while (0)

#define CUSPARSE_CHECK(expr) do { cusparseStatus_t _s = (expr); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(1); } } while (0)

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

// Same MC coefficient computation
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

// Precompute P_Q and P_q matrices on CPU.
// For a tree topology with topologically-ordered reaches:
//   P_Q[r, r]     = C3[r]                                (diag)
//   P_Q[r, u]     = C2[r] * path_coef(r, u) for each path r ← ... ← u
//              through C1 chain
//
// We compute by recursion:
//   P_Q[r, u] = C3[r] * [u==r] + C2[r] * [u ∈ up(r)] + C1[r] * P_Q[up(r), u]
//
// Actually more precisely:
//   A·X = C2·N·X + C3·X → X = A^(-1) · (C2·N·X + C3·X)
//
// For a single row (since A = I - C1·N, A·X = X - C1·N·X, so X = C1·N·X + rhs):
//   X[r] = C1[r] * sum(X[u] for u in up(r)) + rhs[r]
//
// For our P_Q, rhs[r] = C2[r] * (sum delta if upstream) + C3[r] * delta
// That is, the column-u-th column of P_Q is the single-timestep response
// to an impulse at u.
//
// Easier approach: iterate (I + C1·N + (C1·N)^2 + ... + (C1·N)^K) · (C2·N + C3·I)
// Truncate nonzeros below threshold.
//
// To keep things simple, we'll CONSTRUCT P_Q and P_q row-by-row in
// topological order, accumulating nonzeros along ancestor paths.

struct Matrix {
    int n;
    std::vector<int> rowptr;  // CSR
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
    // For each reach r (in topological order):
    //   P_Q[r, *] = C1[r] · (sum over upstreams of P_Q[u, *]) + C2[r] · sum(delta[upstream]) + C3[r] · delta[r]
    //   P_q[r, *] = C1[r] · (sum over upstreams of P_q[u, *]) + C4[r] · delta[r]
    //
    // Each row is a sparse vector. We store per-row a map: col -> value.
    // After computing, we'll prune entries below tol and build CSR.

    std::vector<std::vector<std::pair<int,float>>> pQ_rows(n_reaches);
    std::vector<std::vector<std::pair<int,float>>> pq_rows(n_reaches);

    auto merge_scaled = [&](std::vector<std::pair<int,float>>& dst,
                            const std::vector<std::pair<int,float>>& src,
                            float scale) {
        // dst is sorted by col; src is sorted by col; merge in place
        std::vector<std::pair<int,float>> merged;
        merged.reserve(dst.size() + src.size());
        size_t i = 0, j = 0;
        while (i < dst.size() && j < src.size()) {
            if (dst[i].first < src[j].first) {
                merged.push_back(dst[i++]);
            } else if (dst[i].first > src[j].first) {
                merged.push_back({src[j].first, src[j].second * scale});
                j++;
            } else {
                float v = dst[i].second + src[j].second * scale;
                merged.push_back({dst[i].first, v});
                i++; j++;
            }
        }
        while (i < dst.size()) { merged.push_back(dst[i++]); }
        while (j < src.size()) { merged.push_back({src[j].first, src[j].second * scale}); j++; }
        dst = std::move(merged);
    };

    auto add_col = [&](std::vector<std::pair<int,float>>& row, int col, float val) {
        // Insert (col, val) keeping sorted-by-col order, merging if exists.
        auto it = std::lower_bound(row.begin(), row.end(), col,
            [](const std::pair<int,float>& p, int c) { return p.first < c; });
        if (it != row.end() && it->first == col) {
            it->second += val;
        } else {
            row.insert(it, {col, val});
        }
    };

    auto prune = [tol](std::vector<std::pair<int,float>>& row) {
        row.erase(std::remove_if(row.begin(), row.end(),
            [tol](const std::pair<int,float>& p) { return std::fabs(p.second) < tol; }),
            row.end());
    };

    for (int r = 0; r < n_reaches; ++r) {
        float c1 = C1[r], c2 = C2[r], c3 = C3[r], c4oq = C4oq[r];
        int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];

        // Inherit from upstreams with scale C1
        for (int k = ub; k < ue; ++k) {
            int u = reach_up_idx[k];
            merge_scaled(pQ_rows[r], pQ_rows[u], c1);
            merge_scaled(pq_rows[r], pq_rows[u], c1);
        }
        prune(pQ_rows[r]);
        prune(pq_rows[r]);

        // Add C2·(sum upstream delta) => pQ_rows[r][u] += C2 for u in up(r)
        for (int k = ub; k < ue; ++k) {
            int u = reach_up_idx[k];
            add_col(pQ_rows[r], u, c2);
        }
        // Add C3·delta[r]
        add_col(pQ_rows[r], r, c3);
        // Add C4/ql·delta[r] in pq
        add_col(pq_rows[r], r, c4oq);

        prune(pQ_rows[r]);
        prune(pq_rows[r]);
    }

    // Build CSR for both matrices
    auto build_csr = [](const std::vector<std::vector<std::pair<int,float>>>& rows, Matrix& M) {
        int n = (int)rows.size();
        M.n = n;
        M.rowptr.resize(n + 1);
        M.rowptr[0] = 0;
        for (int r = 0; r < n; ++r)
            M.rowptr[r + 1] = M.rowptr[r] + (int)rows[r].size();
        int nnz = M.rowptr[n];
        M.colind.resize(nnz);
        M.values.resize(nnz);
        int p = 0;
        for (int r = 0; r < n; ++r) {
            for (auto& cv : rows[r]) {
                M.colind[p] = cv.first;
                M.values[p] = cv.second;
                p++;
            }
        }
    };
    build_csr(pQ_rows, P_Q);
    build_csr(pq_rows, P_q);
}

// =========================================================================
// I/O helpers
// =========================================================================
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
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir> [tol=1e-7]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float tol = (argc >= 3) ? (float)atof(argv[2]) : 1e-7f;
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

    // Compute C1..C4 per reach
    std::vector<float> C1(t.n_reaches), C2(t.n_reaches), C3(t.n_reaches), C4oq(t.n_reaches);
    for (int r = 0; r < t.n_reaches; ++r) {
        compute_mc_coeffs_cpu(params[r], dt, C1[r], C2[r], C3[r], C4oq[r]);
    }

    // Build propagation matrices
    printf("[matrix] building P_Q and P_q matrices (tol=%.0e)...\n", tol);
    auto pt0 = std::chrono::steady_clock::now();
    Matrix P_Q, P_q;
    build_propagation_matrices(t.n_reaches, t.reach_up_ptr, t.reach_up_idx,
                                C1, C2, C3, C4oq, tol, P_Q, P_q);
    auto pt1 = std::chrono::steady_clock::now();
    double pt_ms = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
    int P_Q_nnz = P_Q.rowptr.back();
    int P_q_nnz = P_q.rowptr.back();
    printf("[matrix] P_Q: %d nnz (avg %.1f per row), P_q: %d nnz (avg %.1f per row)\n",
           P_Q_nnz, (double)P_Q_nnz / t.n_reaches,
           P_q_nnz, (double)P_q_nnz / t.n_reaches);
    printf("[matrix] build time: %.1f ms (one-time)\n", pt_ms);

    // Upload matrices to GPU
    int *d_pQ_row, *d_pQ_col; float *d_pQ_val;
    int *d_pq_row, *d_pq_col; float *d_pq_val;
    float *d_Q_old, *d_Q_new, *d_qlat_buf, *d_tmp, *d_qlat_all;

    CUDA_CHECK(cudaMalloc(&d_pQ_row, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_pQ_col, P_Q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pQ_val, P_Q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_row, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_col, P_q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_pq_val, P_q_nnz * 4));
    CUDA_CHECK(cudaMalloc(&d_Q_old, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_Q_new, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat_buf, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_tmp, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat_all, (size_t)n_ts * t.n_reaches * 4));

    CUDA_CHECK(cudaMemcpy(d_pQ_row, P_Q.rowptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQ_col, P_Q.colind.data(), P_Q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQ_val, P_Q.values.data(), P_Q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_row, P_q.rowptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_col, P_q.colind.data(), P_q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_val, P_q.values.data(), P_q_nnz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(), (size_t)n_ts * t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q_old, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t mat_pQ, mat_pq;
    cusparseDnVecDescr_t vec_Qold, vec_Qnew, vec_qlat, vec_tmp;
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_pQ, t.n_reaches, t.n_reaches, P_Q_nnz,
        d_pQ_row, d_pQ_col, d_pQ_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_pq, t.n_reaches, t.n_reaches, P_q_nnz,
        d_pq_row, d_pq_col, d_pq_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Qold, t.n_reaches, d_Q_old, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Qnew, t.n_reaches, d_Q_new, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_qlat, t.n_reaches, d_qlat_buf, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_tmp, t.n_reaches, d_tmp, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_sz1 = 0, buf_sz2 = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_pQ, vec_Qold, &beta, vec_Qnew, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG2, &buf_sz1));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_pq, vec_qlat, &beta, vec_tmp, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG2, &buf_sz2));
    size_t buf_sz = std::max(buf_sz1, buf_sz2);
    void* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Warmup
    for (int w = 0; w < 2; ++w) {
        CUDA_CHECK(cudaMemcpy(d_qlat_buf, d_qlat_all, t.n_reaches * 4, cudaMemcpyDeviceToDevice));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pQ, vec_Qold, &beta, vec_Qnew, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pq, vec_qlat, &beta, vec_tmp, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaMemcpy(d_Q_old, qdp0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    for (int ts = 0; ts < n_ts; ++ts) {
        // Load qlat for this timestep
        CUDA_CHECK(cudaMemcpyAsync(d_qlat_buf, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches * 4, cudaMemcpyDeviceToDevice));
        // Q_new = P_Q · Q_old
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pQ, vec_Qold, &beta, vec_Qnew, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        // tmp = P_q · qlat
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pq, vec_qlat, &beta, vec_tmp, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        // Q_new += tmp (custom elementwise-add kernel)
        {
            int blk = (t.n_reaches + 255) / 256;
            kernel_add_inplace<<<blk, 256>>>(d_Q_new, d_tmp, t.n_reaches);
        }
        // swap Q_old, Q_new
        std::swap(d_Q_old, d_Q_new);
        cusparseDnVecSetValues(vec_Qold, d_Q_old);
        cusparseDnVecSetValues(vec_Qnew, d_Q_new);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[gpu-matrix] total ms: %.3f  per-timestep: %.3f ms  (2x SpMV + AXPY per ts)\n",
           ms, ms / n_ts);

    // Sanity
    std::vector<float> gpu_Q(t.n_reaches);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_Q_old, t.n_reaches * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0; int nans = 0;
    for (int i = 0; i < t.n_reaches; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q; if (q > maxv) maxv = q;
    }
    printf("[gpu-matrix] Q_final sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);

    // Write output for external diffing against linear_mc's reference
    FILE* of = std::fopen("matrix_Q_final.bin", "wb");
    std::fwrite(gpu_Q.data(), 4, t.n_reaches, of);
    std::fclose(of);
    printf("[gpu-matrix] wrote matrix_Q_final.bin (%d reaches)\n", t.n_reaches);

    cusparseDestroySpMat(mat_pQ);
    cusparseDestroySpMat(mat_pq);
    cusparseDestroyDnVec(vec_Qold);
    cusparseDestroyDnVec(vec_Qnew);
    cusparseDestroyDnVec(vec_qlat);
    cusparseDestroyDnVec(vec_tmp);
    cusparseDestroy(handle);
    return 0;
}
