/**
 * MATRIX-FORM with cuSPARSE + CUDA Graphs.
 *
 * Same math as linear_mc_matrix.cu but the 24 per-timestep (memcpy + 2×SpMV +
 * add) ops are captured once into a cudaGraph and replayed with one launch.
 * cuSPARSE SpMV is already ~memory-bound on this matrix size, so this just
 * kills the per-launch API overhead.
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
#include <cusparse.h>

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) {\
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
            cudaGetErrorString(_e)); std::exit(1); } } while (0)
#define CUSPARSE_CHECK(expr) do { cusparseStatus_t _s = (expr); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(1); } } while (0)

__global__ void add_inplace(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

static void compute_mc_coeffs(const ChParams& p, float dt,
    float& C1, float& C2, float& C3, float& C4oq) {
    float z = (p.cs == 0.0f) ? 1.0f : 1.0f / p.cs;
    float bfd;
    if (p.bw > p.tw) bfd = p.bw / 0.00001f;
    else if (p.bw == p.tw) bfd = p.bw / (2.0f * z);
    else bfd = (p.tw - p.bw) / (2.0f * z);
    float h = std::fmax(0.5f * bfd, 0.1f);
    float A = (p.bw + h * z) * h;
    float WP = p.bw + 2.0f * h * std::sqrt(1.0f + z * z);
    float R = (WP > 0.0f) ? A / WP : 0.0f;
    float Ck = std::fmax(1e-6f, (std::sqrt(p.s0) / p.n_ch) *
                 ((5.0f / 3.0f) * std::pow(R, 2.0f/3.0f) - (2.0f / 3.0f) * std::pow(R, 5.0f/3.0f) *
                  (2.0f * std::sqrt(1.0f + z * z) / (p.bw + 2.0f * h * z))));
    float Km = std::fmax(dt, p.dx / Ck);
    float X = 0.25f;
    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f;
    C1 = (Km * X + dt / 2.0f) / D;
    C2 = (dt / 2.0f - Km * X) / D;
    C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    C4oq = dt / D;
}

struct Matrix {
    int n;
    std::vector<int> rowptr, colind;
    std::vector<float> values;
};

static void build_matrices(int n_reaches, const std::vector<int>& up_ptr,
    const std::vector<int>& up_idx, const std::vector<float>& C1,
    const std::vector<float>& C2, const std::vector<float>& C3,
    const std::vector<float>& C4oq, float tol, Matrix& P_Q, Matrix& P_q)
{
    std::vector<std::vector<std::pair<int,float>>> pQ(n_reaches), pq(n_reaches);
    auto merge = [&](std::vector<std::pair<int,float>>& dst,
                     const std::vector<std::pair<int,float>>& src, float sc) {
        std::vector<std::pair<int,float>> m;
        m.reserve(dst.size() + src.size());
        size_t i = 0, j = 0;
        while (i < dst.size() && j < src.size()) {
            if (dst[i].first < src[j].first) m.push_back(dst[i++]);
            else if (dst[i].first > src[j].first) { m.push_back({src[j].first, src[j].second*sc}); j++; }
            else { m.push_back({dst[i].first, dst[i].second + src[j].second*sc}); i++; j++; }
        }
        while (i < dst.size()) m.push_back(dst[i++]);
        while (j < src.size()) { m.push_back({src[j].first, src[j].second*sc}); j++; }
        dst = std::move(m);
    };
    auto add = [&](std::vector<std::pair<int,float>>& row, int col, float v) {
        auto it = std::lower_bound(row.begin(), row.end(), col,
            [](const std::pair<int,float>& p, int c) { return p.first < c; });
        if (it != row.end() && it->first == col) it->second += v;
        else row.insert(it, {col, v});
    };
    auto prune = [tol](std::vector<std::pair<int,float>>& r) {
        r.erase(std::remove_if(r.begin(), r.end(),
            [tol](auto& p) { return std::fabs(p.second) < tol; }), r.end());
    };
    for (int r = 0; r < n_reaches; ++r) {
        for (int k = up_ptr[r]; k < up_ptr[r+1]; ++k) {
            merge(pQ[r], pQ[up_idx[k]], C1[r]);
            merge(pq[r], pq[up_idx[k]], C1[r]);
        }
        prune(pQ[r]); prune(pq[r]);
        for (int k = up_ptr[r]; k < up_ptr[r+1]; ++k) add(pQ[r], up_idx[k], C2[r]);
        add(pQ[r], r, C3[r]);
        add(pq[r], r, C4oq[r]);
        prune(pQ[r]); prune(pq[r]);
    }
    auto to_csr = [](const std::vector<std::vector<std::pair<int,float>>>& rows, Matrix& M) {
        int n = (int)rows.size();
        M.n = n; M.rowptr.resize(n+1); M.rowptr[0] = 0;
        for (int r = 0; r < n; ++r) M.rowptr[r+1] = M.rowptr[r] + rows[r].size();
        M.colind.resize(M.rowptr[n]); M.values.resize(M.rowptr[n]);
        int p = 0;
        for (int r = 0; r < n; ++r)
            for (auto& cv : rows[r]) { M.colind[p] = cv.first; M.values[p] = cv.second; p++; }
    };
    to_csr(pQ, P_Q); to_csr(pq, P_q);
}

struct Topo {
    int n_reaches, n_segments, n_levels, max_up;
    std::vector<int> reach_seg_start, reach_seg_len, reach_level, reach_n_up;
    std::vector<int> reach_up_ptr, reach_up_idx, level_ptr, level_reach;
};
static Topo load_topo(const std::string& dir) {
    Topo t; FILE* f = std::fopen((dir + "/topo.bin").c_str(), "rb");
    std::fread(&t.n_reaches, 4, 1, f); std::fread(&t.n_segments, 4, 1, f);
    std::fread(&t.n_levels, 4, 1, f); std::fread(&t.max_up, 4, 1, f);
    t.reach_seg_start.resize(t.n_reaches); t.reach_seg_len.resize(t.n_reaches);
    t.reach_level.resize(t.n_reaches); t.reach_n_up.resize(t.n_reaches);
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
    std::fclose(f); return t;
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
    std::fread(params.data(), sizeof(ChParams), t.n_segments, pf); std::fclose(pf);

    FILE* ff = std::fopen((dir + "/forcings.bin").c_str(), "rb");
    int n_ts = 0; std::fread(&n_ts, 4, 1, ff);
    std::vector<float> qlat_ts((size_t)n_ts * t.n_segments);
    std::fread(qlat_ts.data(), 4, qlat_ts.size(), ff);
    std::vector<float> qup0(t.n_reaches);
    std::fread(qup0.data(), 4, t.n_reaches, ff);
    std::vector<float> qdp0(t.n_segments);
    std::fread(qdp0.data(), 4, t.n_segments, ff);
    std::fclose(ff);

    std::vector<float> C1(t.n_reaches), C2(t.n_reaches), C3(t.n_reaches), C4oq(t.n_reaches);
    for (int r = 0; r < t.n_reaches; ++r)
        compute_mc_coeffs(params[r], dt, C1[r], C2[r], C3[r], C4oq[r]);

    Matrix P_Q, P_q;
    auto tb0 = std::chrono::steady_clock::now();
    build_matrices(t.n_reaches, t.reach_up_ptr, t.reach_up_idx, C1, C2, C3, C4oq, tol, P_Q, P_q);
    auto tb1 = std::chrono::steady_clock::now();
    int pQ_nnz = P_Q.rowptr.back(), pq_nnz = P_q.rowptr.back();
    printf("[matrix] P_Q %d nnz (%.1f/row), P_q %d nnz (%.1f/row), build %.1f ms\n",
           pQ_nnz, (double)pQ_nnz/t.n_reaches, pq_nnz, (double)pq_nnz/t.n_reaches,
           std::chrono::duration<double, std::milli>(tb1-tb0).count());

    int *d_pQr, *d_pQc, *d_pqr, *d_pqc;
    float *d_pQv, *d_pqv, *d_Qa, *d_Qb, *d_ql, *d_tmp, *d_qlat_all;
    CUDA_CHECK(cudaMalloc(&d_pQr, (t.n_reaches+1)*4));
    CUDA_CHECK(cudaMalloc(&d_pQc, pQ_nnz*4));
    CUDA_CHECK(cudaMalloc(&d_pQv, pQ_nnz*4));
    CUDA_CHECK(cudaMalloc(&d_pqr, (t.n_reaches+1)*4));
    CUDA_CHECK(cudaMalloc(&d_pqc, pq_nnz*4));
    CUDA_CHECK(cudaMalloc(&d_pqv, pq_nnz*4));
    CUDA_CHECK(cudaMalloc(&d_Qa, t.n_reaches*4));
    CUDA_CHECK(cudaMalloc(&d_Qb, t.n_reaches*4));
    CUDA_CHECK(cudaMalloc(&d_ql, t.n_reaches*4));
    CUDA_CHECK(cudaMalloc(&d_tmp, t.n_reaches*4));
    CUDA_CHECK(cudaMalloc(&d_qlat_all, (size_t)n_ts * t.n_reaches*4));

    CUDA_CHECK(cudaMemcpy(d_pQr, P_Q.rowptr.data(), (t.n_reaches+1)*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQc, P_Q.colind.data(), pQ_nnz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pQv, P_Q.values.data(), pQ_nnz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pqr, P_q.rowptr.data(), (t.n_reaches+1)*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pqc, P_q.colind.data(), pq_nnz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pqv, P_q.values.data(), pq_nnz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qlat_all, qlat_ts.data(), (size_t)n_ts * t.n_reaches*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice));

    // cuSPARSE setup: we need 2 SpMV descriptors per timestep orientation (in/out)
    // Trick: create per-timestep descriptors pre-bound to the right Qin/Qout.
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t mat_pQ, mat_pq;
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_pQ, t.n_reaches, t.n_reaches, pQ_nnz,
        d_pQr, d_pQc, d_pQv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_pq, t.n_reaches, t.n_reaches, pq_nnz,
        d_pqr, d_pqc, d_pqv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Pre-create per-ts dense vector descriptors to avoid mid-graph creation.
    std::vector<cusparseDnVecDescr_t> vIn(n_ts), vOut(n_ts);
    std::vector<cusparseDnVecDescr_t> vQl(n_ts);
    cusparseDnVecDescr_t vTmp;
    for (int ts = 0; ts < n_ts; ++ts) {
        float* Qin  = (ts & 1) ? d_Qb : d_Qa;
        float* Qout = (ts & 1) ? d_Qa : d_Qb;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vIn[ts], t.n_reaches, Qin, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vOut[ts], t.n_reaches, Qout, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vQl[ts], t.n_reaches, d_ql, CUDA_R_32F));
    }
    CUSPARSE_CHECK(cusparseCreateDnVec(&vTmp, t.n_reaches, d_tmp, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_sz1 = 0, buf_sz2 = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_pQ, vIn[0], &beta, vOut[0], CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG2, &buf_sz1));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_pq, vQl[0], &beta, vTmp, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG2, &buf_sz2));
    size_t buf_sz = std::max(buf_sz1, buf_sz2);
    void* d_buf; CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // ===== Method A: per-timestep cuSPARSE calls (baseline) =====
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice));
    // Warmup
    for (int w = 0; w < 2; ++w) {
        for (int ts = 0; ts < n_ts; ++ts) {
            CUDA_CHECK(cudaMemcpyAsync(d_ql, d_qlat_all + (size_t)ts * t.n_reaches,
                t.n_reaches*4, cudaMemcpyDeviceToDevice, stream));
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, mat_pQ, vIn[ts], &beta, vOut[ts], CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, d_buf));
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, mat_pq, vQl[ts], &beta, vTmp, CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2, d_buf));
            float* Qout = (ts & 1) ? d_Qa : d_Qb;
            int blk = (t.n_reaches + 255) / 256;
            add_inplace<<<blk, 256, 0, stream>>>(Qout, d_tmp, t.n_reaches);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice));
    }

    // Timed (per-ts)
    cudaEventRecord(e0, stream);
    for (int ts = 0; ts < n_ts; ++ts) {
        CUDA_CHECK(cudaMemcpyAsync(d_ql, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches*4, cudaMemcpyDeviceToDevice, stream));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pQ, vIn[ts], &beta, vOut[ts], CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pq, vQl[ts], &beta, vTmp, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        float* Qout = (ts & 1) ? d_Qa : d_Qb;
        int blk = (t.n_reaches + 255) / 256;
        add_inplace<<<blk, 256, 0, stream>>>(Qout, d_tmp, t.n_reaches);
    }
    cudaEventRecord(e1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms_perts = 0.f;
    cudaEventElapsedTime(&ms_perts, e0, e1);
    printf("[cusparse-perts] total %.3f ms  per-ts %.3f ms\n", ms_perts, ms_perts / n_ts);

    // ===== Method B: capture as CUDA Graph, launch once =====
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int ts = 0; ts < n_ts; ++ts) {
        CUDA_CHECK(cudaMemcpyAsync(d_ql, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches*4, cudaMemcpyDeviceToDevice, stream));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pQ, vIn[ts], &beta, vOut[ts], CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pq, vQl[ts], &beta, vTmp, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        float* Qout = (ts & 1) ? d_Qa : d_Qb;
        int blk = (t.n_reaches + 255) / 256;
        add_inplace<<<blk, 256, 0, stream>>>(Qout, d_tmp, t.n_reaches);
    }
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Warmup graph
    for (int w = 0; w < 2; ++w) {
        CUDA_CHECK(cudaMemcpyAsync(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed graph
    CUDA_CHECK(cudaMemcpy(d_Qa, qdp0.data(), t.n_reaches*4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0, stream);
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    cudaEventRecord(e1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms_graph = 0.f;
    cudaEventElapsedTime(&ms_graph, e0, e1);
    printf("[cusparse-graph] total %.3f ms  per-ts %.3f ms  (1 graph launch)\n",
           ms_graph, ms_graph / n_ts);

    // Sanity check
    std::vector<float> gpu_Q(t.n_reaches);
    float* d_final = (n_ts & 1) ? d_Qb : d_Qa;
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_final, t.n_reaches*4, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < t.n_reaches; ++i) sum += gpu_Q[i];
    printf("[final] Q_final sum=%.3e\n", sum);
    FILE* of = std::fopen("matrix_v2_Q_final.bin", "wb");
    std::fwrite(gpu_Q.data(), 4, t.n_reaches, of);
    std::fclose(of);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return 0;
}
