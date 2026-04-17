/**
 * Linearized sparse-matrix Muskingum-Cunge on GPU, matching the formulation
 * used by DeepGroundwater/ddr and geoglows/river-route:
 *
 *   (I - C1·N) · Q_{t+1} = C2·(N·Q_t) + C3·Q_t + C4·Q'
 *
 * where N is the segment adjacency matrix (N[i,j] = 1 iff j is an upstream
 * segment of i). Because the segments are topologically ordered,
 * (I - C1·N) is lower-triangular with unit diagonal, so Q_{t+1} comes from
 * a single sparse forward-substitution per timestep — no per-segment secant
 * iteration.
 *
 * Coefficients C1..C4 are precomputed once from fixed channel geometry
 * (same channel params as the nonlinear secant kernel uses). That matches
 * the assumption of fixed celerity used in rapid/linearized MC kernels.
 *
 * Parallelism is level-scheduled forward substitution: all segments at
 * topological level L are independent of each other and run in parallel;
 * levels L+1 sync-await.
 *
 * CPU reference: scipy-style forward substitution in pure C++ (no deps).
 *
 * Author: Cody Churchwell — for NOAA-OWP/t-route#874 follow-up
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

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                        \
                #expr, __FILE__, __LINE__, cudaGetErrorString(_err));          \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct ChParams {
    float dx, bw, tw, twcc, n_ch, ncc, cs, s0;
};

// ==========================================================================
// Precompute MC coefficients C1..C4 from channel params at a nominal flow.
// We use the reference-depth half-bankfull approximation: depth ≈ 0.5*bfd.
// This gives a representative celerity for the linearized form. The same
// thing the classical "linearized Muskingum-Cunge" does.
// ==========================================================================
__host__ __device__ inline void compute_mc_coeffs(
    const ChParams& p, float dt,
    float& C1, float& C2, float& C3, float& C4_over_ql)
{
    float z = (p.cs == 0.0f) ? 1.0f : 1.0f / p.cs;
    float bfd;
    if (p.bw > p.tw) bfd = p.bw / 0.00001f;
    else if (p.bw == p.tw) bfd = p.bw / (2.0f * z);
    else bfd = (p.tw - p.bw) / (2.0f * z);

    // Reference depth: half bankfull (capped small to avoid Ck=0 issues)
    float h = fmaxf(0.5f * bfd, 0.1f);

    // Hydraulic radius at reference depth
    float twl = p.bw + 2.0f * z * h;
    float A = (p.bw + h * z) * h;
    float WP = p.bw + 2.0f * h * sqrtf(1.0f + z * z);
    float R = (WP > 0.0f) ? A / WP : 0.0f;

    // Wave celerity at reference
    float r23 = powf(R, 2.0f / 3.0f);
    float r53 = powf(R, 5.0f / 3.0f);
    float Ck = fmaxf(1e-6f, (sqrtf(p.s0) / p.n_ch) *
                 ((5.0f / 3.0f) * r23 - (2.0f / 3.0f) * r53 *
                  (2.0f * sqrtf(1.0f + z * z) / (p.bw + 2.0f * h * z))));

    float Km = fmaxf(dt, p.dx / Ck);
    // Weighting factor X at reference: use classical Cunge formula with Qj=0
    // for a representative nominal flow
    float tw_use = (h > bfd && p.twcc > 0.0f) ? p.twcc : twl;
    float X = 0.25f;
    if (Ck > 0.0f && tw_use * p.s0 * Ck * p.dx > 0.0f) {
        X = fminf(0.5f, fmaxf(0.0f, 0.5f * (1.0f - 0.0f /
                 (2.0f * tw_use * p.s0 * Ck * p.dx))));
    }

    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f;

    C1 = (Km * X + dt / 2.0f) / D;
    C2 = (dt / 2.0f - Km * X) / D;
    C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    C4_over_ql = dt / D;   // multiply by ql at runtime to get C4
}

__global__ void kernel_precompute_coeffs(
    const ChParams* __restrict__ params,
    float* __restrict__ C1,
    float* __restrict__ C2,
    float* __restrict__ C3,
    float* __restrict__ C4_over_ql,
    float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    compute_mc_coeffs(params[i], dt, C1[i], C2[i], C3[i], C4_over_ql[i]);
}

// ==========================================================================
// GPU kernel: REACH-LEVEL scheduling of the sparse MC forward substitution.
// One thread per reach. Within a reach, segments are processed sequentially
// (they form a chain, so there's no intra-reach parallelism to be had
// anyway). Across reaches in the same topological level, they're
// independent and run in parallel.
//
// For segment 0 of reach r:
//   Q_new[0] = C3·Q_old[0] + C4/ql·qlat[0]
//            + C1·sum(Q_new[last_seg(u)] for u in up(r))
//            + C2·sum(Q_old[last_seg(u)] for u in up(r))
// For segment i > 0 in reach r:
//   Q_new[i] = C3·Q_old[i] + C4/ql·qlat[i]
//            + C1·Q_new[i-1] + C2·Q_old[i-1]
// ==========================================================================
__global__ void kernel_linear_level_reach(
    const int*   __restrict__ reach_seg_start,
    const int*   __restrict__ reach_seg_len,
    const int*   __restrict__ reach_up_ptr,
    const int*   __restrict__ reach_up_idx,
    const int*   __restrict__ level_reach_slice,
    const float* __restrict__ C1,
    const float* __restrict__ C2,
    const float* __restrict__ C3,
    const float* __restrict__ C4oq,
    const float* __restrict__ Q_old,
    const float* __restrict__ qlat,
    float*       __restrict__ Q_new,
    int level_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;
    int r = level_reach_slice[tid];

    int s0 = reach_seg_start[r];
    int sl = reach_seg_len[r];

    // Aggregate upstream contribution for segment 0 of this reach.
    float sum_new = 0.f, sum_old = 0.f;
    int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
    // Need last-segment index of each upstream reach.
    // We stored upstream reach IDs in reach_up_idx; to fetch its last-segment,
    // we use reach_seg_start[u] + reach_seg_len[u] - 1.
    for (int k = ub; k < ue; ++k) {
        int u = reach_up_idx[k];
        int last_u = reach_seg_start[u] + reach_seg_len[u] - 1;
        sum_new += Q_new[last_u];
        sum_old += Q_old[last_u];
    }

    // Segment 0: upstream = external (aggregated above)
    float qn_prev_new = 0.f, qn_prev_old = 0.f;
    {
        int idx = s0;
        Q_new[idx] = C1[idx] * sum_new + C2[idx] * sum_old
                   + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
        qn_prev_new = Q_new[idx];
        qn_prev_old = Q_old[idx];
    }
    // Interior segments: upstream = previous segment in same reach
    for (int i = 1; i < sl; ++i) {
        int idx = s0 + i;
        Q_new[idx] = C1[idx] * qn_prev_new + C2[idx] * qn_prev_old
                   + C3[idx] * Q_old[idx] + C4oq[idx] * qlat[idx];
        qn_prev_new = Q_new[idx];
        qn_prev_old = Q_old[idx];
    }
}

// ==========================================================================
// CPU reference (single-thread, level order, linear MC)
// ==========================================================================
static void cpu_linear_mc(
    const float* C1, const float* C2, const float* C3, const float* C4oq,
    const int* reach_seg_start, const int* reach_seg_len,
    const int* reach_up_ptr, const int* reach_up_idx,
    const int* level_ptr, const int* level_reach,
    int n_reaches, int n_levels, int n_segments,
    const float* qlat_ts, const float* Q0,
    float* Q_final,
    int n_timesteps)
{
    std::vector<float> Qp(Q0, Q0 + n_segments);
    std::vector<float> Qn(n_segments, 0.f);

    for (int t = 0; t < n_timesteps; ++t) {
        const float* qlat = qlat_ts + (size_t)t * n_segments;
        for (int L = 0; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            for (int kk = lb; kk < le; ++kk) {
                int r = level_reach[kk];
                int s0 = reach_seg_start[r], sl = reach_seg_len[r];
                float sum_new = 0.f, sum_old = 0.f;
                int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
                for (int k = ub; k < ue; ++k) {
                    int u = reach_up_idx[k];
                    int last_u = reach_seg_start[u] + reach_seg_len[u] - 1;
                    sum_new += Qn[last_u];
                    sum_old += Qp[last_u];
                }
                // segment 0
                int idx = s0;
                Qn[idx] = C1[idx] * sum_new + C2[idx] * sum_old
                        + C3[idx] * Qp[idx] + C4oq[idx] * qlat[idx];
                float qn_prev_new = Qn[idx];
                float qn_prev_old = Qp[idx];
                for (int i = 1; i < sl; ++i) {
                    idx = s0 + i;
                    Qn[idx] = C1[idx] * qn_prev_new + C2[idx] * qn_prev_old
                            + C3[idx] * Qp[idx] + C4oq[idx] * qlat[idx];
                    qn_prev_new = Qn[idx];
                    qn_prev_old = Qp[idx];
                }
            }
        }
        Qp.swap(Qn);
        std::fill(Qn.begin(), Qn.end(), 0.f);
    }

    std::memcpy(Q_final, Qp.data(), n_segments * sizeof(float));
}

// FP64 ground-truth CPU reference
static void cpu_linear_mc_fp64(
    const float* C1_f, const float* C2_f, const float* C3_f, const float* C4oq_f,
    const int* reach_seg_start, const int* reach_seg_len,
    const int* reach_up_ptr, const int* reach_up_idx,
    const int* level_ptr, const int* level_reach,
    int n_reaches, int n_levels, int n_segments,
    const float* qlat_ts, const float* Q0,
    double* Q_final,
    int n_timesteps)
{
    std::vector<double> C1(n_segments), C2(n_segments), C3(n_segments), C4oq(n_segments);
    for (int i = 0; i < n_segments; ++i) {
        C1[i] = (double)C1_f[i]; C2[i] = (double)C2_f[i];
        C3[i] = (double)C3_f[i]; C4oq[i] = (double)C4oq_f[i];
    }
    std::vector<double> Qp(n_segments), Qn(n_segments, 0.0);
    for (int i = 0; i < n_segments; ++i) Qp[i] = Q0[i];

    for (int t = 0; t < n_timesteps; ++t) {
        const float* qlat = qlat_ts + (size_t)t * n_segments;
        for (int L = 0; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            for (int kk = lb; kk < le; ++kk) {
                int r = level_reach[kk];
                int s0 = reach_seg_start[r], sl = reach_seg_len[r];
                double sum_new = 0.0, sum_old = 0.0;
                int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
                for (int k = ub; k < ue; ++k) {
                    int u = reach_up_idx[k];
                    int last_u = reach_seg_start[u] + reach_seg_len[u] - 1;
                    sum_new += Qn[last_u];
                    sum_old += Qp[last_u];
                }
                int idx = s0;
                Qn[idx] = C1[idx] * sum_new + C2[idx] * sum_old
                        + C3[idx] * Qp[idx] + C4oq[idx] * (double)qlat[idx];
                double qn_prev_new = Qn[idx], qn_prev_old = Qp[idx];
                for (int i = 1; i < sl; ++i) {
                    idx = s0 + i;
                    Qn[idx] = C1[idx] * qn_prev_new + C2[idx] * qn_prev_old
                            + C3[idx] * Qp[idx] + C4oq[idx] * (double)qlat[idx];
                    qn_prev_new = Qn[idx];
                    qn_prev_old = Qp[idx];
                }
            }
        }
        Qp.swap(Qn);
        std::fill(Qn.begin(), Qn.end(), 0.0);
    }

    for (int i = 0; i < n_segments; ++i) Q_final[i] = Qp[i];
}

// ==========================================================================
// Build segment-level adjacency & levels from the reach-level topology we
// already generated (same files as wavefront kernel).
// ==========================================================================
struct ReachTopo {
    int n_reaches, n_segments, n_reach_levels, max_up;
    std::vector<int> reach_seg_start, reach_seg_len, reach_level, reach_n_up;
    std::vector<int> reach_up_ptr, reach_up_idx;
    std::vector<int> reach_level_ptr, reach_level_reach;
};

static ReachTopo load_reach_topo(const std::string& dir)
{
    ReachTopo t;
    std::string path = dir + "/topo.bin";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path.c_str()); std::exit(1); }
    std::fread(&t.n_reaches, 4, 1, f);
    std::fread(&t.n_segments, 4, 1, f);
    std::fread(&t.n_reach_levels, 4, 1, f);
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

    t.reach_level_ptr.resize(t.n_reach_levels + 1);
    std::fread(t.reach_level_ptr.data(), 4, t.n_reach_levels + 1, f);
    t.reach_level_reach.resize(t.n_reaches);
    std::fread(t.reach_level_reach.data(), 4, t.n_reaches, f);

    std::fclose(f);
    return t;
}

// Segment-level topology: expand reach-level adjacency to per-segment.
struct SegTopo {
    int n_segments = 0, n_levels = 0;
    std::vector<int> seg_up_ptr;       // CSR [n_segments+1]
    std::vector<int> seg_up_idx;       // [nnz]
    std::vector<int> seg_level;        // [n_segments]
    std::vector<int> level_ptr;        // [n_levels+1]
    std::vector<int> level_seg;        // [n_segments]
};

static SegTopo build_seg_topo(const ReachTopo& r)
{
    SegTopo s;
    s.n_segments = r.n_segments;
    s.seg_up_ptr.assign(r.n_segments + 1, 0);
    s.seg_level.assign(r.n_segments, 0);

    // First pass: count upstreams per segment.
    // For segment idx in reach R:
    //   if idx == reach_seg_start[R]: upstream = last segments of each upstream reach
    //   else: upstream = idx - 1 (the preceding segment in the same reach)
    std::vector<int> deg(r.n_segments, 0);
    for (int R = 0; R < r.n_reaches; ++R) {
        int s0 = r.reach_seg_start[R], sl = r.reach_seg_len[R];
        deg[s0] = r.reach_n_up[R];
        for (int i = 1; i < sl; ++i) deg[s0 + i] = 1;
    }
    for (int i = 0; i < r.n_segments; ++i) s.seg_up_ptr[i + 1] = s.seg_up_ptr[i] + deg[i];
    int nnz = s.seg_up_ptr.back();
    s.seg_up_idx.resize(nnz);

    // Fill adjacency.
    for (int R = 0; R < r.n_reaches; ++R) {
        int s0 = r.reach_seg_start[R], sl = r.reach_seg_len[R];
        // first segment: upstreams = last segment of each parent reach
        int pos = s.seg_up_ptr[s0];
        for (int k = r.reach_up_ptr[R]; k < r.reach_up_ptr[R + 1]; ++k) {
            int U = r.reach_up_idx[k];
            int u_last = r.reach_seg_start[U] + r.reach_seg_len[U] - 1;
            s.seg_up_idx[pos++] = u_last;
        }
        // interior segments: upstream = prev segment
        for (int i = 1; i < sl; ++i) {
            int idx = s0 + i;
            s.seg_up_idx[s.seg_up_ptr[idx]] = idx - 1;
        }
    }

    // Topological levels via Kahn's: level(i) = 1 + max(level of upstreams)
    // Since segments are already in topological order (reaches ordered, segments in reach sequential),
    // we can do this in a single pass.
    int max_lvl = 0;
    for (int i = 0; i < r.n_segments; ++i) {
        int lvl = 0;
        for (int k = s.seg_up_ptr[i]; k < s.seg_up_ptr[i + 1]; ++k) {
            int u = s.seg_up_idx[k];
            lvl = std::max(lvl, s.seg_level[u] + 1);
        }
        s.seg_level[i] = lvl;
        if (lvl > max_lvl) max_lvl = lvl;
    }
    s.n_levels = max_lvl + 1;

    // Build level CSR
    std::vector<int> counts(s.n_levels, 0);
    for (int i = 0; i < r.n_segments; ++i) counts[s.seg_level[i]]++;
    s.level_ptr.assign(s.n_levels + 1, 0);
    for (int L = 0; L < s.n_levels; ++L) s.level_ptr[L + 1] = s.level_ptr[L] + counts[L];
    s.level_seg.resize(r.n_segments);
    std::vector<int> cur(s.n_levels, 0);
    for (int L = 0; L < s.n_levels; ++L) cur[L] = s.level_ptr[L];
    for (int i = 0; i < r.n_segments; ++i) {
        int L = s.seg_level[i];
        s.level_seg[cur[L]++] = i;
    }
    return s;
}

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t k = (size_t)((p / 100.0) * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <net_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    float dt = 300.0f;

    printf("[io] loading from %s\n", dir.c_str());
    ReachTopo r = load_reach_topo(dir);
    printf("[io] n_reaches=%d n_segments=%d reach-levels=%d\n",
           r.n_reaches, r.n_segments, r.n_reach_levels);

    // Load params
    std::vector<ChParams> params(r.n_segments);
    FILE* pf = std::fopen((dir + "/params.bin").c_str(), "rb");
    std::fread(params.data(), sizeof(ChParams), r.n_segments, pf);
    std::fclose(pf);

    // Load forcings
    FILE* ff = std::fopen((dir + "/forcings.bin").c_str(), "rb");
    int n_ts = 0;
    std::fread(&n_ts, 4, 1, ff);
    std::vector<float> qlat_ts((size_t)n_ts * r.n_segments);
    std::fread(qlat_ts.data(), 4, qlat_ts.size(), ff);
    std::vector<float> qup0(r.n_reaches);
    std::fread(qup0.data(), 4, r.n_reaches, ff);
    std::vector<float> qdp0(r.n_segments);
    std::fread(qdp0.data(), 4, r.n_segments, ff);
    std::vector<float> dp0(r.n_segments);
    std::fread(dp0.data(), 4, r.n_segments, ff);
    std::fclose(ff);

    printf("[io] n_timesteps=%d\n", n_ts);

    // ---- GPU allocations (reach-level) ----
    ChParams* d_params;
    int *d_reach_seg_start, *d_reach_seg_len;
    int *d_reach_up_ptr, *d_reach_up_idx, *d_level_reach;
    float *d_C1, *d_C2, *d_C3, *d_C4oq, *d_Qp, *d_Qn, *d_qlat;

    CUDA_CHECK(cudaMalloc(&d_params, r.n_segments * sizeof(ChParams)));
    CUDA_CHECK(cudaMalloc(&d_reach_seg_start, r.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_seg_len, r.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_up_ptr, (r.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_up_idx, r.reach_up_idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_level_reach, r.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_C1, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C2, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C3, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_C4oq, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_Qp, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_Qn, r.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat, r.n_segments * 4));

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), r.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_seg_start, r.reach_seg_start.data(), r.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_seg_len, r.reach_seg_len.data(), r.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_up_ptr, r.reach_up_ptr.data(), (r.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_up_idx, r.reach_up_idx.data(), r.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, r.reach_level_reach.data(), r.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Qp, qdp0.data(), r.n_segments * 4, cudaMemcpyHostToDevice));

    // Precompute MC coeffs on GPU
    int threads = 256, blocks = (r.n_segments + threads - 1) / threads;
    kernel_precompute_coeffs<<<blocks, threads>>>(
        d_params, d_C1, d_C2, d_C3, d_C4oq, dt, r.n_segments);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy coefficients back for CPU reference
    std::vector<float> C1(r.n_segments), C2(r.n_segments), C3(r.n_segments), C4oq(r.n_segments);
    CUDA_CHECK(cudaMemcpy(C1.data(), d_C1, r.n_segments * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C2.data(), d_C2, r.n_segments * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C3.data(), d_C3, r.n_segments * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C4oq.data(), d_C4oq, r.n_segments * 4, cudaMemcpyDeviceToHost));

    // ---- GPU benchmark (reach-level scheduling) ----
    printf("[gpu] running %d timesteps x %d reach-levels\n", n_ts, r.n_reach_levels);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    // warmup
    kernel_linear_level_reach<<<1, 32>>>(
        d_reach_seg_start, d_reach_seg_len,
        d_reach_up_ptr, d_reach_up_idx,
        d_level_reach,
        d_C1, d_C2, d_C3, d_C4oq, d_Qp, d_qlat, d_Qn, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e0);

    for (int ts = 0; ts < n_ts; ++ts) {
        const float* qlat_host = qlat_ts.data() + (size_t)ts * r.n_segments;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_host, r.n_segments * 4, cudaMemcpyHostToDevice));

        for (int L = 0; L < r.n_reach_levels; ++L) {
            int lb = r.reach_level_ptr[L], le = r.reach_level_ptr[L + 1];
            int cnt = le - lb;
            if (cnt <= 0) continue;
            int blk = (cnt + threads - 1) / threads;
            kernel_linear_level_reach<<<blk, threads>>>(
                d_reach_seg_start, d_reach_seg_len,
                d_reach_up_ptr, d_reach_up_idx,
                d_level_reach + lb,
                d_C1, d_C2, d_C3, d_C4oq,
                d_Qp, d_qlat, d_Qn, cnt);
        }
        std::swap(d_Qp, d_Qn);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, e0, e1);
    printf("[gpu] total ms: %.3f  per-timestep: %.3f ms\n", gpu_ms, gpu_ms / n_ts);

    std::vector<float> gpu_Q(r.n_segments);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_Qp, r.n_segments * 4, cudaMemcpyDeviceToHost));

    // ---- CPU FP32 reference ----
    printf("[cpu-fp32] running CPU FP32 linear MC...\n");
    std::vector<float> cpu_Q(r.n_segments);
    auto c0 = std::chrono::steady_clock::now();
    cpu_linear_mc(C1.data(), C2.data(), C3.data(), C4oq.data(),
                  r.reach_seg_start.data(), r.reach_seg_len.data(),
                  r.reach_up_ptr.data(), r.reach_up_idx.data(),
                  r.reach_level_ptr.data(), r.reach_level_reach.data(),
                  r.n_reaches, r.n_reach_levels, r.n_segments,
                  qlat_ts.data(), qdp0.data(),
                  cpu_Q.data(), n_ts);
    auto c1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    printf("[cpu-fp32] total ms: %.3f  per-timestep: %.3f ms\n", cpu_ms, cpu_ms / n_ts);

    // ---- CPU FP64 ground truth ----
    printf("[cpu-fp64] running CPU FP64 ground truth...\n");
    std::vector<double> truth(r.n_segments);
    auto d0 = std::chrono::steady_clock::now();
    cpu_linear_mc_fp64(C1.data(), C2.data(), C3.data(), C4oq.data(),
                       r.reach_seg_start.data(), r.reach_seg_len.data(),
                       r.reach_up_ptr.data(), r.reach_up_idx.data(),
                       r.reach_level_ptr.data(), r.reach_level_reach.data(),
                       r.n_reaches, r.n_reach_levels, r.n_segments,
                       qlat_ts.data(), qdp0.data(),
                       truth.data(), n_ts);
    auto d1 = std::chrono::steady_clock::now();
    double truth_ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
    printf("[cpu-fp64] total ms: %.3f\n", truth_ms);

    // Accuracy vs FP64 truth
    auto stats = [&](const float* pred, const char* name) {
        std::vector<double> rel;
        rel.reserve(r.n_segments);
        double max_abs = 0.0;
        int nonzero = 0;
        for (int i = 0; i < r.n_segments; ++i) {
            double t = truth[i];
            double p = (double)pred[i];
            double ae = std::fabs(p - t);
            max_abs = std::fmax(max_abs, ae);
            if (std::fabs(t) > 1e-3) { rel.push_back(ae / std::fabs(t)); nonzero++; }
        }
        if (rel.empty()) { printf("[acc:%s] no nonzero truth\n", name); return; }
        std::vector<double> tmp = rel;
        double p50 = pct(tmp, 50.0); tmp = rel;
        double p90 = pct(tmp, 90.0); tmp = rel;
        double p99 = pct(tmp, 99.0); tmp = rel;
        double p999 = pct(tmp, 99.9);
        double pmax = *std::max_element(rel.begin(), rel.end());
        int under1 = 0, under10 = 0;
        for (double v : rel) { if (v < 0.01) under1++; if (v < 0.1) under10++; }
        printf("[acc:%s vs FP64] max_abs=%.3e p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e max=%.2e "
               "within-1%%=%.2f%% within-10%%=%.2f%%\n",
               name, max_abs, p50, p90, p99, p999, pmax,
               100.0 * under1 / nonzero, 100.0 * under10 / nonzero);
    };
    stats(gpu_Q.data(), "GPU-FP32");
    stats(cpu_Q.data(), "CPU-FP32");

    printf("[sum] Timings: GPU %.2f ms | CPU-FP32 %.2f ms | CPU-FP64 %.2f ms\n",
           gpu_ms, cpu_ms, truth_ms);
    printf("[sum] Speedup GPU vs CPU-FP32: %.2fx\n", cpu_ms / gpu_ms);
    printf("[sum] Speedup GPU vs CPU-FP64: %.2fx\n", truth_ms / gpu_ms);

    cudaFree(d_params);
    cudaFree(d_reach_seg_start); cudaFree(d_reach_seg_len);
    cudaFree(d_reach_up_ptr); cudaFree(d_reach_up_idx); cudaFree(d_level_reach);
    cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3); cudaFree(d_C4oq);
    cudaFree(d_Qp); cudaFree(d_Qn); cudaFree(d_qlat);
    return 0;
}
