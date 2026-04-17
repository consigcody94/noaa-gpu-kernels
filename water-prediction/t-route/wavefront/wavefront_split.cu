/**
 * Two-phase split kernel for the WAVEFRONT SECANT MC (exact t-route Fortran
 * algorithm). Applies the same Phase A (grid-cooperative) + Phase B
 * (single-block __syncthreads on thin tail) idea as linear_mc_split.cu,
 * but with the full nonlinear secant per-segment solve.
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

__device__ inline void hgeo(
    float h, float bfd, float bw, float twcc, float z,
    float& twl, float& R, float& A, float& AC, float& WP, float& WPC)
{
    twl = bw + 2.0f*z*h;
    float hgb = fmaxf(h-bfd, 0.0f), hlb = fminf(bfd, h);
    if (hgb > 0.0f && twcc <= 0.0f) { hgb = 0.0f; hlb = h; }
    A = (bw + hlb*z) * hlb;
    WP = bw + 2.0f*hlb*sqrtf(1.0f + z*z);
    AC = twcc * hgb;
    WPC = (hgb > 0.0f) ? (twcc + 2.0f*hgb) : 0.0f;
    R = (WP+WPC > 0.0f) ? (A+AC)/(WP+WPC) : 0.0f;
}

__device__ inline float mc_secant(
    float dt, float qup, float quc, float qdp, float ql,
    float dx, float bw, float tw, float twcc,
    float n_ch, float ncc, float cs, float s0,
    float depthp, float& depthc_out, float& velc_out)
{
    float z = (cs == 0.0f) ? 1.0f : 1.0f / cs;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f*z);
    else bfd = (tw - bw) / (2.0f*z);
    if (n_ch<=0.0f||s0<=0.0f||z<=0.0f||bw<=0.0f) { depthc_out=0.f; velc_out=0.f; return 0.f; }
    if (ql<=0.0f && qup<=0.0f && quc<=0.0f && qdp<=0.0f) { depthc_out=0.f; velc_out=0.f; return 0.f; }

    const float mindepth = 0.01f;
    float depthc = fmaxf(depthp, 0.0f);
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
                float r23 = powf(R0, 2.0f/3.0f), r53 = powf(R0, 5.0f/3.0f);
                Ck0 = fmaxf(0.f, (sqrtf(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h_0*z))));
            }
            float Km0 = (Ck0>0.f) ? fmaxf(dt, dx/Ck0) : dt;
            float twu0 = (h_0>bfd && twcc>0.f) ? twcc : twl0;
            float X0 = (Ck0>0.f && twu0*s0*Ck0*dx>0.f) ? fminf(0.5f, fmaxf(0.f, 0.5f*(1.0f - Qj_0/(2.0f*twu0*s0*Ck0*dx)))) : 0.5f;
            float D0 = Km0*(1.0f-X0) + dt/2.0f; if (D0==0.f) D0=1.f;
            c1_0 = (Km0*X0 + dt/2.0f)/D0;
            c2_0 = (dt/2.0f - Km0*X0)/D0;
            c3_0 = (Km0*(1.0f-X0) - dt/2.0f)/D0;
            c4_0 = (ql*dt)/D0;
            float Qmn0 = (WP0+WPC0>0.f) ? (1.0f/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*powf(R0,2.0f/3.0f)*sqrtf(s0) : 0.f;
            Qj_0 = (c1_0*qup + c2_0*quc + c3_0*qdp + c4_0) - Qmn0;

            float twl1,R1,A1,AC1,WP1,WPC1;
            hgeo(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            float Ck1 = 0.f;
            if (h > 0.f) {
                float r23 = powf(R1, 2.0f/3.0f), r53 = powf(R1, 5.0f/3.0f);
                Ck1 = fmaxf(0.f, (sqrtf(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z))));
            }
            float Km1 = (Ck1>0.f) ? fmaxf(dt, dx/Ck1) : dt;
            float twu1 = (h>bfd && twcc>0.f) ? twcc : twl1;
            X = (Ck1>0.f && twu1*s0*Ck1*dx>0.f) ? fminf(0.5f, fmaxf(0.25f, 0.5f*(1.0f - (c1_0*qup + c2_0*quc + c3_0*qdp + c4_0)/(2.0f*twu1*s0*Ck1*dx)))) : 0.5f;
            float D1 = Km1*(1.0f-X) + dt/2.0f; if (D1==0.f) D1=1.f;
            C1 = (Km1*X + dt/2.0f)/D1;
            C2 = (dt/2.0f - Km1*X)/D1;
            C3 = (Km1*(1.0f-X) - dt/2.0f)/D1;
            C4 = (ql*dt)/D1;
            if (C4 < 0.f && fabsf(C4) > C1*qup + C2*quc + C3*qdp) C4 = -(C1*qup + C2*quc + C3*qdp);
            float Qmn1 = (WP1+WPC1>0.f) ? (1.0f/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*powf(R1,2.0f/3.0f)*sqrtf(s0) : 0.f;
            Qj = (C1*qup + C2*quc + C3*qdp + C4) - Qmn1;
            float h_1 = (Qj_0-Qj != 0.f) ? h - (Qj*(h_0-h))/(Qj_0-Qj) : h;
            if (h_1 < 0.f) h_1 = h;
            if (h > 0.f) { rerror = fabsf((h_1-h)/h); aerror = fabsf(h_1-h); } else { rerror=0.f; aerror=0.9f; }
            h_0 = fmaxf(0.f, h); h = fmaxf(0.f, h_1);
            iter++;
            if (h < mindepth) break;
        }
        if (iter < maxiter) break;
        h *= 1.33f; h_0 *= 0.67f; maxiter += 25;
    }
    float Qmc = C1*qup + C2*quc + C3*qdp + C4;
    float qdc;
    if (Qmc < 0.f) {
        if (C4 < 0.f && fabsf(C4) > C1*qup + C2*quc + C3*qdp) qdc = 0.f;
        else qdc = fmaxf(C1*qup + C2*quc + C4, C1*qup + C3*qdp + C4);
    } else { qdc = Qmc; }

    float twl = bw + 2.0f*z*h;
    float Rv = (h*(bw+twl)/2.0f)/(bw + 2.0f*sqrtf(((twl-bw)/2.0f)*((twl-bw)/2.0f) + h*h));
    if (Rv < 0.f) Rv = 0.f;
    velc_out = (1.0f/n_ch) * powf(Rv, 2.0f/3.0f) * sqrtf(s0);
    depthc_out = h;
    return qdc;
}

__device__ __forceinline__ void process_reach_secant(
    int r, float dt,
    const ChParams* __restrict__ params,
    const int* __restrict__ reach_seg_start,
    const int* __restrict__ reach_seg_len,
    const int* __restrict__ reach_up_ptr,
    const int* __restrict__ reach_up_idx,
    const float* __restrict__ qlat,
    const float* __restrict__ qdp_prev_seg,
    const float* __restrict__ dp_prev_seg,
    const float* __restrict__ reach_last_prev,
    const float* __restrict__ reach_last_curr,
    float* __restrict__ qdc_seg_out,
    float* __restrict__ vel_seg_out,
    float* __restrict__ dep_seg_out,
    float* __restrict__ reach_last_curr_out)
{
    float qup_reach_prev = 0.f, quc_reach_curr = 0.f;
    int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
    for (int j = ub; j < ue; ++j) {
        int u = reach_up_idx[j];
        qup_reach_prev += reach_last_prev[u];
        quc_reach_curr += reach_last_curr[u];
    }
    int seg_start = reach_seg_start[r];
    int seg_len = reach_seg_len[r];
    float quc_next = 0.f, qdc_last = 0.f;
    for (int i = 0; i < seg_len; ++i) {
        int idx = seg_start + i;
        const ChParams& p = params[idx];
        float qup_i = (i == 0) ? qup_reach_prev : qdp_prev_seg[idx - 1];
        float quc_i = (i == 0) ? quc_reach_curr : quc_next;
        float depthc, velc;
        float qdc = mc_secant(
            dt, qup_i, quc_i, qdp_prev_seg[idx], qlat[idx],
            p.dx, p.bw, p.tw, p.twcc, p.n_ch, p.ncc, p.cs, p.s0,
            dp_prev_seg[idx], depthc, velc);
        qdc_seg_out[idx] = qdc;
        vel_seg_out[idx] = velc;
        dep_seg_out[idx] = depthc;
        quc_next = qdc;
        qdc_last = qdc;
    }
    reach_last_curr_out[r] = qdc_last;
}

__global__ void kernel_split_secant(
    const ChParams* __restrict__ params,
    const int*   __restrict__ reach_seg_start,
    const int*   __restrict__ reach_seg_len,
    const int*   __restrict__ reach_up_ptr,
    const int*   __restrict__ reach_up_idx,
    const int*   __restrict__ level_reach,
    const int*   __restrict__ level_ptr,
    const float* __restrict__ qlat,
    const float* __restrict__ qdp_prev_seg,
    const float* __restrict__ dp_prev_seg,
    const float* __restrict__ reach_last_prev,
    float*       __restrict__ reach_last_curr,
    float*       __restrict__ qdc_seg_out,
    float*       __restrict__ vel_seg_out,
    float*       __restrict__ dep_seg_out,
    float dt, int n_levels, int K_wide)
{
    cg::grid_group grid = cg::this_grid();
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int L = 0; L < K_wide; ++L) {
        int lb = level_ptr[L], le = level_ptr[L + 1];
        int cnt = le - lb;
        for (int k = tid_global; k < cnt; k += stride) {
            int r = level_reach[lb + k];
            process_reach_secant(r, dt, params, reach_seg_start, reach_seg_len,
                                 reach_up_ptr, reach_up_idx, qlat, qdp_prev_seg, dp_prev_seg,
                                 reach_last_prev, reach_last_curr,
                                 qdc_seg_out, vel_seg_out, dep_seg_out, reach_last_curr);
        }
        grid.sync();
    }

    if (blockIdx.x == 0) {
        int tid = threadIdx.x;
        for (int L = K_wide; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            int cnt = le - lb;
            if (tid < cnt) {
                int r = level_reach[lb + tid];
                process_reach_secant(r, dt, params, reach_seg_start, reach_seg_len,
                                     reach_up_ptr, reach_up_idx, qlat, qdp_prev_seg, dp_prev_seg,
                                     reach_last_prev, reach_last_curr,
                                     qdc_seg_out, vel_seg_out, dep_seg_out, reach_last_curr);
            }
            __syncthreads();
        }
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
    if (argc < 2) { fprintf(stderr, "usage: %s <net_dir>\n", argv[0]); return 1; }
    std::string dir = argv[1];
    float dt = 300.0f;
    int threads = 256;

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
    printf("[io] K_wide=%d\n", K_wide);

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

    std::vector<float> reach_last0(t.n_reaches);
    for (int r = 0; r < t.n_reaches; ++r)
        reach_last0[r] = qdp0[t.reach_seg_start[r] + t.reach_seg_len[r] - 1];

    ChParams* d_params;
    int *d_seg_start, *d_seg_len, *d_up_ptr, *d_up_idx, *d_level_reach, *d_level_ptr;
    float *d_qlat, *d_qdp_prev, *d_dp_prev, *d_reach_last_prev, *d_reach_last_curr;
    float *d_qdc_out, *d_vel_out, *d_dep_out;
    CUDA_CHECK(cudaMalloc(&d_params, t.n_segments * sizeof(ChParams)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_seg_len, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_up_ptr, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_up_idx, t.reach_up_idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_level_reach, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_level_ptr, (t.n_levels + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qdp_prev, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_dp_prev, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_last_prev, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_last_curr, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qdc_out, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_vel_out, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_dep_out, t.n_segments * 4));

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), t.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_start, t.reach_seg_start.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_len, t.reach_seg_len.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_ptr, t.reach_up_ptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_idx, t.reach_up_idx.data(), t.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, t.level_reach.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_ptr, t.level_ptr.data(), (t.n_levels + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qdp_prev, qdp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dp_prev, dp0.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_last_prev, reach_last0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    int dev = 0, numSM = 0, maxBPSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBPSM, (const void*)kernel_split_secant, threads, 0);
    int coop_blocks = numSM * maxBPSM;
    if (coop_blocks <= 0) coop_blocks = numSM;
    printf("[gpu] coop: numSM=%d bpSM=%d -> %d blocks x %d threads\n", numSM, maxBPSM, coop_blocks, threads);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // warmup
    {
        void* args[] = {&d_params, &d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_qlat, &d_qdp_prev, &d_dp_prev,
                        &d_reach_last_prev, &d_reach_last_curr,
                        &d_qdc_out, &d_vel_out, &d_dep_out, &dt, &t.n_levels, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_secant,
            dim3(coop_blocks), dim3(threads), args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[gpu] running %d timesteps (split secant)\n", n_ts);
    cudaEventRecord(e0);
    for (int ts = 0; ts < n_ts; ++ts) {
        const float* qlat_host = qlat_ts.data() + (size_t)ts * t.n_segments;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_host, t.n_segments * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_reach_last_curr, 0, t.n_reaches * 4));
        void* args[] = {&d_params, &d_seg_start, &d_seg_len, &d_up_ptr, &d_up_idx,
                        &d_level_reach, &d_level_ptr, &d_qlat, &d_qdp_prev, &d_dp_prev,
                        &d_reach_last_prev, &d_reach_last_curr,
                        &d_qdc_out, &d_vel_out, &d_dep_out, &dt, &t.n_levels, &K_wide};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_split_secant,
            dim3(coop_blocks), dim3(threads), args));
        std::swap(d_qdp_prev, d_qdc_out);
        std::swap(d_dp_prev, d_dep_out);
        std::swap(d_reach_last_prev, d_reach_last_curr);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[gpu-split-secant] total ms: %.3f per-ts: %.3f ms\n", ms, ms / n_ts);

    std::vector<float> gpu_Q(t.n_segments);
    CUDA_CHECK(cudaMemcpy(gpu_Q.data(), d_qdp_prev, t.n_segments * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0, maxv = 0.0;
    int nans = 0;
    for (int i = 0; i < t.n_segments; ++i) {
        float q = gpu_Q[i];
        if (std::isnan(q) || std::isinf(q)) { nans++; continue; }
        sum += q; if (q > maxv) maxv = q;
    }
    printf("[gpu-split-secant] sum=%.3e max=%.3e nans=%d\n", sum, maxv, nans);
    return 0;
}
