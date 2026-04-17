/**
 * MATRIX-FORM LINEAR MC with SECANT-DERIVED COEFFICIENTS.
 *
 * Combines the exact-Fortran MUSKINGCUNGE secant solver (to derive
 * C1..C4 at a reference flow) with the 0.30 ms/ts matrix-form runtime.
 *
 * Workflow:
 *   1) Run the secant solver on CPU for each reach at a reference flow
 *      (by default: the steady-state approximation from qdp0 + upstream
 *      sum). This gives dynamically-fit C1..C4 per reach, rather than
 *      my simpler half-bankfull heuristic used in linear_mc_matrix.cu.
 *   2) Build propagation matrices P_Q and P_q from those coefficients.
 *   3) Run matrix-form SpMV per timestep on GPU.
 *
 * Tradeoff: accuracy is much closer to the full nonlinear secant (which
 * re-derives coefficients every timestep) without paying the 7.65 ms/ts
 * secant kernel cost. On CONUS-scale flows where coefficients change
 * slowly, this is a good approximation.
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

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) {\
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
            cudaGetErrorString(_e)); std::exit(1); } } while (0)

#define CUSPARSE_CHECK(expr) do { cusparseStatus_t _s = (expr); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(1); } } while (0)

__global__ void kernel_add_inplace(float* __restrict__ dst,
                                   const float* __restrict__ src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

struct ChParams { float dx, bw, tw, twcc, n_ch, ncc, cs, s0; };

// Full MUSKINGCUNGE secant solver (matches Fortran exactly, FP32).
// Given reference qup, quc, qdp, ql, and initial depth depthp, returns
// the converged C1..C4 coefficients via the solver's final iteration.
static void mc_secant_solve_cpu(
    float dt, float qup, float quc, float qdp, float ql,
    float dx, float bw, float tw, float twcc,
    float n_ch, float ncc, float cs, float s0,
    float depthp,
    float& C1_out, float& C2_out, float& C3_out, float& C4oq_out)
{
    auto hgeo = [](float h, float bfd, float bw, float twcc, float z,
                   float& twl, float& R, float& A, float& AC, float& WP, float& WPC) {
        twl = bw + 2.0f*z*h;
        float hgb = std::fmax(h-bfd, 0.0f), hlb = std::fmin(bfd, h);
        if (hgb > 0.0f && twcc <= 0.0f) { hgb = 0.0f; hlb = h; }
        A = (bw + hlb*z) * hlb;
        WP = bw + 2.0f*hlb*std::sqrt(1.0f + z*z);
        AC = twcc * hgb;
        WPC = (hgb > 0.0f) ? (twcc + 2.0f*hgb) : 0.0f;
        R = (WP+WPC > 0.0f) ? (A+AC)/(WP+WPC) : 0.0f;
    };

    float z = (cs == 0.0f) ? 1.0f : 1.0f / cs;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f*z);
    else bfd = (tw - bw) / (2.0f*z);

    // Default fallback
    auto fallback = [&]() {
        // Half-bankfull reference (same as linear_mc_matrix.cu)
        float h = std::fmax(0.5f * bfd, 0.1f);
        float A = (bw + h * z) * h;
        float WP = bw + 2.0f * h * std::sqrt(1.0f + z * z);
        float R = (WP > 0.0f) ? A / WP : 0.0f;
        float r23 = std::pow(R, 2.0f/3.0f);
        float r53 = std::pow(R, 5.0f/3.0f);
        float Ck = std::fmax(1e-6f, (std::sqrt(s0) / n_ch) *
                 ((5.0f/3.0f) * r23 - (2.0f/3.0f) * r53 *
                  (2.0f * std::sqrt(1.0f + z * z) / (bw + 2.0f * h * z))));
        float Km = std::fmax(dt, dx / Ck);
        float X = 0.25f;
        float D = Km * (1.0f - X) + dt / 2.0f;
        if (D == 0.0f) D = 1.0f;
        C1_out = (Km * X + dt / 2.0f) / D;
        C2_out = (dt / 2.0f - Km * X) / D;
        C3_out = (Km * (1.0f - X) - dt / 2.0f) / D;
        C4oq_out = dt / D;
    };

    if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f) {
        fallback();
        return;
    }
    if (ql <= 0.0f && qup <= 0.0f && quc <= 0.0f && qdp <= 0.0f) {
        fallback();
        return;
    }

    const float mindepth = 0.01f;
    float depthc = std::fmax(depthp, 0.0f);
    float h = depthc*1.33f + mindepth;
    float h_0 = depthc*0.67f;
    float C1=0, C2=0, C3=0, C4=0, X=0.25f, Qj_0=0.f, Qj=0.f;
    int maxiter = 100;

    for (int att = 0; att < 5; att++) {
        float aerror = 0.01f, rerror = 1.0f;
        int iter = 0;
        float c1_0=0,c2_0=0,c3_0=0,c4_0=0;
        while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
            float twl0,R0,A0,AC0,WP0,WPC0;
            hgeo(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);
            float Ck0 = 0.f;
            if (h_0 > 0.f) {
                float r23 = std::pow(R0, 2.0f/3.0f), r53 = std::pow(R0, 5.0f/3.0f);
                Ck0 = std::fmax(0.f, (std::sqrt(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*std::sqrt(1.0f+z*z)/(bw+2.0f*h_0*z))));
            }
            float Km0 = (Ck0>0.f) ? std::fmax(dt, dx/Ck0) : dt;
            float twu0 = (h_0>bfd && twcc>0.f) ? twcc : twl0;
            float X0 = (Ck0>0.f && twu0*s0*Ck0*dx>0.f) ? std::fmin(0.5f, std::fmax(0.f, 0.5f*(1.0f - Qj_0/(2.0f*twu0*s0*Ck0*dx)))) : 0.5f;
            float D0 = Km0*(1.0f-X0) + dt/2.0f; if (D0==0.f) D0=1.f;
            c1_0 = (Km0*X0 + dt/2.0f)/D0;
            c2_0 = (dt/2.0f - Km0*X0)/D0;
            c3_0 = (Km0*(1.0f-X0) - dt/2.0f)/D0;
            c4_0 = (ql*dt)/D0;
            float Qmn0 = (WP0+WPC0>0.f) ? (1.0f/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*std::pow(R0,2.0f/3.0f)*std::sqrt(s0) : 0.f;
            Qj_0 = (c1_0*qup + c2_0*quc + c3_0*qdp + c4_0) - Qmn0;

            float twl1,R1,A1,AC1,WP1,WPC1;
            hgeo(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            float Ck1 = 0.f;
            if (h > 0.f) {
                float r23 = std::pow(R1, 2.0f/3.0f), r53 = std::pow(R1, 5.0f/3.0f);
                Ck1 = std::fmax(0.f, (std::sqrt(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*std::sqrt(1.0f+z*z)/(bw+2.0f*h*z))));
            }
            float Km1 = (Ck1>0.f) ? std::fmax(dt, dx/Ck1) : dt;
            float twu1 = (h>bfd && twcc>0.f) ? twcc : twl1;
            X = (Ck1>0.f && twu1*s0*Ck1*dx>0.f) ? std::fmin(0.5f, std::fmax(0.25f, 0.5f*(1.0f - (c1_0*qup + c2_0*quc + c3_0*qdp + c4_0)/(2.0f*twu1*s0*Ck1*dx)))) : 0.5f;
            float D1 = Km1*(1.0f-X) + dt/2.0f; if (D1==0.f) D1=1.f;
            C1 = (Km1*X + dt/2.0f)/D1;
            C2 = (dt/2.0f - Km1*X)/D1;
            C3 = (Km1*(1.0f-X) - dt/2.0f)/D1;
            C4 = (ql*dt)/D1;
            if (C4 < 0.f && std::fabs(C4) > C1*qup + C2*quc + C3*qdp) C4 = -(C1*qup + C2*quc + C3*qdp);
            float Qmn1 = (WP1+WPC1>0.f) ? (1.0f/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*std::pow(R1,2.0f/3.0f)*std::sqrt(s0) : 0.f;
            Qj = (C1*qup + C2*quc + C3*qdp + C4) - Qmn1;
            float h_1 = (Qj_0-Qj != 0.f) ? h - (Qj*(h_0-h))/(Qj_0-Qj) : h;
            if (h_1 < 0.f) h_1 = h;
            if (h > 0.f) { rerror = std::fabs((h_1-h)/h); aerror = std::fabs(h_1-h); }
            else { rerror=0.f; aerror=0.9f; }
            h_0 = std::fmax(0.f, h); h = std::fmax(0.f, h_1);
            iter++;
            if (h < mindepth) break;
        }
        if (iter < maxiter) break;
        h *= 1.33f; h_0 *= 0.67f; maxiter += 25;
    }

    // Report secant-derived coefficients
    C1_out = C1;
    C2_out = C2;
    C3_out = C3;
    // C4_out in our matrix-form needs to be C4/ql (so we can multiply by qlat at runtime)
    // In the secant solver, C4 = ql * dt / D. So C4/ql = dt/D. Reconstruct from the
    // last D1 computation. Since we exit the secant with C1, C2, C3, C4 set, and
    // C1+C2+C3 + (something) = 1 approximately, we can invert:
    //   dt / D = C4 / ql (if ql > 0)
    // If ql==0 at reference, we use fallback computation:
    if (std::fabs(ql) > 1e-12f) {
        C4oq_out = C4 / ql;
    } else {
        // Approximate via: dt/D where D = Km*(1-X) + dt/2
        // Km can be extracted from C1 via: C1 = (Km*X + dt/2)/D  -> C1*D = Km*X + dt/2
        // Alternative: just use fallback
        float fb_C1, fb_C2, fb_C3, fb_C4oq;
        float save_C1=C1_out, save_C2=C2_out, save_C3=C3_out;
        fallback();
        C4oq_out = C4oq_out;  // keep fallback
        C1_out=save_C1; C2_out=save_C2; C3_out=save_C3;  // but keep secant C1..C3
    }
}

// Same propagation-matrix builder as linear_mc_matrix.cu
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
        for (int k = ub; k < ue; ++k) {
            int u = reach_up_idx[k];
            add_col(pQ_rows[r], u, c2);
        }
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
    std::vector<float> dp0(t.n_segments);
    std::fread(dp0.data(), 4, t.n_segments, ff);
    std::fclose(ff);

    // ========== SECANT COEFFICIENT DERIVATION ==========
    printf("[secant] deriving C1..C4 per reach via Fortran-equivalent secant solver...\n");
    auto s0 = std::chrono::steady_clock::now();
    std::vector<float> C1(t.n_reaches), C2(t.n_reaches), C3(t.n_reaches), C4oq(t.n_reaches);
    // Reference flow: use a small but nonzero ql and qdp to trigger secant convergence.
    // For CONUS-like data, use the mean qlat in the first timestep as ql_ref.
    double mean_qlat = 0.0;
    for (int i = 0; i < t.n_segments; ++i) mean_qlat += qlat_ts[i];
    mean_qlat /= t.n_segments;
    float ql_ref = std::fmax(0.1f, (float)mean_qlat);

    // Per reach, call secant solver with REALISTIC reference flows.
    // We use the steady-state approximation: upstream ancestors contribute
    // roughly sum of headwater qlats. In a tree, the reference downstream
    // flow is ~ (qlat * area-of-drainage). Approximate with simple sum of
    // reach's own qlat plus qlat-mean * num_ancestors.
    // This gives coefficients valid at typical-flow, not cold-start.
    std::vector<float> ref_Q(t.n_reaches, 0.0f);
    // Approximate ref_Q via one pass through topological levels (cheap,
    // since we're on CPU and only need this once).
    // ref_Q[r] = qlat[r] + sum(ref_Q[u] for u in up(r))
    // Plus scale by mean_qlat to get more realistic magnitude.
    for (int L = 0; L < t.n_levels; ++L) {
        int lb = t.level_ptr[L], le = t.level_ptr[L + 1];
        for (int ii = lb; ii < le; ++ii) {
            int r = t.level_reach[ii];
            float sum_up = 0.f;
            for (int k = t.reach_up_ptr[r]; k < t.reach_up_ptr[r+1]; ++k) {
                sum_up += ref_Q[t.reach_up_idx[k]];
            }
            float own_ql = qlat_ts[r];   // timestep 0 qlat per reach
            ref_Q[r] = sum_up + own_ql + (float)mean_qlat;
        }
    }
    double ref_sum = 0.0, ref_max = 0.0;
    for (int r = 0; r < t.n_reaches; ++r) {
        ref_sum += ref_Q[r];
        if (ref_Q[r] > ref_max) ref_max = ref_Q[r];
    }
    printf("[secant] reference flow: mean=%.3e max=%.3e (accumulated qlat)\n",
           ref_sum / t.n_reaches, ref_max);

    #pragma omp parallel for
    for (int r = 0; r < t.n_reaches; ++r) {
        const ChParams& p = params[r];
        // Reference: upstream-accumulated flow, own qlat as lateral
        float qdp_ref = std::fmax(ref_Q[r], 0.1f);
        // Estimate upstream flow = sum of upstream ref_Q values
        float qup_ref = 0.0f;
        for (int k = t.reach_up_ptr[r]; k < t.reach_up_ptr[r+1]; ++k) {
            qup_ref += ref_Q[t.reach_up_idx[k]];
        }
        qup_ref = std::fmax(qup_ref, 0.1f);
        float quc_ref = qup_ref;  // steady state: qup == quc
        float ql_ref_r = std::fmax(qlat_ts[r], 0.01f);
        // Estimate depth from flow via Manning's (rough): depth ~ (Q/W)^0.6
        float depthp_ref = std::fmax(0.1f, std::pow(qdp_ref / (p.tw + 1.0f), 0.6f));
        mc_secant_solve_cpu(dt, qup_ref, quc_ref, qdp_ref, ql_ref_r,
                            p.dx, p.bw, p.tw, p.twcc,
                            p.n_ch, p.ncc, p.cs, p.s0,
                            depthp_ref,
                            C1[r], C2[r], C3[r], C4oq[r]);
    }
    auto s1 = std::chrono::steady_clock::now();
    printf("[secant] solve done in %.1f ms (one-time)\n",
           std::chrono::duration<double, std::milli>(s1 - s0).count());

    // Build P matrices
    printf("[matrix] building P_Q, P_q with secant-derived coefficients (tol=%.0e)...\n", tol);
    auto pt0 = std::chrono::steady_clock::now();
    Matrix P_Q, P_q;
    build_propagation_matrices(t.n_reaches, t.reach_up_ptr, t.reach_up_idx,
                                C1, C2, C3, C4oq, tol, P_Q, P_q);
    auto pt1 = std::chrono::steady_clock::now();
    int P_Q_nnz = P_Q.rowptr.back();
    int P_q_nnz = P_q.rowptr.back();
    printf("[matrix] P_Q: %d nnz (avg %.1f/row), P_q: %d nnz (avg %.1f/row), build %.1f ms\n",
           P_Q_nnz, (double)P_Q_nnz / t.n_reaches,
           P_q_nnz, (double)P_q_nnz / t.n_reaches,
           std::chrono::duration<double, std::milli>(pt1 - pt0).count());

    // GPU upload
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
        CUDA_CHECK(cudaMemcpyAsync(d_qlat_buf, d_qlat_all + (size_t)ts * t.n_reaches,
            t.n_reaches * 4, cudaMemcpyDeviceToDevice));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pQ, vec_Qold, &beta, vec_Qnew, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_pq, vec_qlat, &beta, vec_tmp, CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2, d_buf));
        {
            int blk = (t.n_reaches + 255) / 256;
            kernel_add_inplace<<<blk, 256>>>(d_Q_new, d_tmp, t.n_reaches);
        }
        std::swap(d_Q_old, d_Q_new);
        cusparseDnVecSetValues(vec_Qold, d_Q_old);
        cusparseDnVecSetValues(vec_Qnew, d_Q_new);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[gpu-matrix-secant] total ms: %.3f  per-timestep: %.3f ms\n", ms, ms / n_ts);

    std::vector<float> gpu_Q(t.n_reaches);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_Q_old, t.n_reaches * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0; int nans = 0;
    for (int i = 0; i < t.n_reaches; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q; if (q > maxv) maxv = q;
    }
    printf("[gpu-matrix-secant] Q_final sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);
    FILE* of = std::fopen("matrix_secant_Q_final.bin", "wb");
    std::fwrite(gpu_Q.data(), 4, t.n_reaches, of);
    std::fclose(of);

    return 0;
}
