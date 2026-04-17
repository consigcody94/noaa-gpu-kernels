/**
 * Wavefront (level-scheduled) GPU Muskingum-Cunge for real NWM tree topology.
 *
 * Unlike the earlier reach_parallel kernel (which treated reaches as
 * independent), this kernel respects the network dependency graph:
 *
 *   - Reaches are grouped into topological LEVELS (level 0 = headwater).
 *   - All reaches in a level are independent, so we launch them in parallel.
 *   - Between levels, we sync so downstream reaches see upstream results.
 *
 * Each reach runs the EXACT same full MUSKINGCUNGE secant solver as the
 * Fortran reference (FP64, 100-iter max, secant bisection fallback).
 * The CPU reference executes the same algorithm in the same topological
 * order — so GPU and CPU outputs are comparable to within FP rounding.
 *
 * Author: Cody Churchwell
 * For: NOAA-OWP/t-route Issue #874
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
    // Kept as float — matches t-route Fortran REAL precision.
    float dx, bw, tw, twcc, n_ch, ncc, cs, s0;
};

// ==========================================================================
// Device/host Muskingum-Cunge secant solver (FP32; mirrors Fortran REAL).
// RTX 3060 has 1/32 FP64 rate, so FP64 tanks performance. t-route's own
// Fortran uses REAL (FP32) by default, so FP32 is the honest reference.
// ==========================================================================
__host__ __device__ inline void hgeo_inline(
    float h, float bfd, float bw, float twcc, float z,
    float& twl, float& R, float& A, float& AC, float& WP, float& WPC)
{
    twl = bw + 2.0f * z * h;
    float hgb = fmaxf(h - bfd, 0.0f);
    float hlb = fminf(bfd, h);
    if (hgb > 0.0f && twcc <= 0.0f) { hgb = 0.0f; hlb = h; }
    A = (bw + hlb * z) * hlb;
    WP = bw + 2.0f * hlb * sqrtf(1.0f + z * z);
    AC = twcc * hgb;
    WPC = (hgb > 0.0f) ? (twcc + 2.0f * hgb) : 0.0f;
    R = (WP + WPC > 0.0f) ? (A + AC) / (WP + WPC) : 0.0f;
}

__host__ __device__ inline float mc_solve(
    float dt, float qup, float quc, float qdp, float ql,
    float dx, float bw, float tw, float twcc,
    float n_ch, float ncc, float cs, float s0,
    float depthp, float& depthc_out, float& velc_out)
{
    float z = (cs == 0.0f) ? 1.0f : 1.0f / cs;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f * z);
    else bfd = (tw - bw) / (2.0f * z);

    if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f) {
        depthc_out = 0.0f; velc_out = 0.0f; return 0.0f;
    }

    if (ql <= 0.0f && qup <= 0.0f && quc <= 0.0f && qdp <= 0.0f) {
        depthc_out = 0.0f; velc_out = 0.0f; return 0.0f;
    }

    const float mindepth = 0.01f;
    float depthc = fmaxf(depthp, 0.0f);
    float h = depthc * 1.33f + mindepth;
    float h_0 = depthc * 0.67f;

    float C1 = 0.0f, C2 = 0.0f, C3 = 0.0f, C4 = 0.0f, X = 0.25f;
    float Qj_0 = 0.0f, Qj = 0.0f;
    int maxiter = 100;

    for (int attempt = 0; attempt < 5; attempt++) {
        float aerror = 0.01f, rerror = 1.0f;
        int iter = 0;
        float c1_0 = 0.0f, c2_0 = 0.0f, c3_0 = 0.0f, c4_0 = 0.0f;

        while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
            float twl0, R0, A0, AC0, WP0, WPC0;
            hgeo_inline(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);
            float Ck0 = 0.0f;
            if (h_0 > 0.0f) {
                float r23 = powf(R0, 2.0f/3.0f);
                float r53 = powf(R0, 5.0f/3.0f);
                Ck0 = fmaxf(0.0f, (sqrtf(s0) / n_ch) *
                    ((5.0f / 3.0f) * r23 - (2.0f / 3.0f) * r53 *
                     (2.0f * sqrtf(1.0f + z * z) / (bw + 2.0f * h_0 * z))));
            }
            float Km0 = (Ck0 > 0.0f) ? fmaxf(dt, dx / Ck0) : dt;
            float twu0 = (h_0 > bfd && twcc > 0.0f) ? twcc : twl0;
            float X0 = (Ck0 > 0.0f && twu0 * s0 * Ck0 * dx > 0.0f)
                ? fminf(0.5f, fmaxf(0.0f, 0.5f * (1.0f - Qj_0 / (2.0f * twu0 * s0 * Ck0 * dx))))
                : 0.5f;
            float D0 = Km0 * (1.0f - X0) + dt / 2.0f;
            if (D0 == 0.0f) D0 = 1.0f;
            c1_0 = (Km0 * X0 + dt / 2.0f) / D0;
            c2_0 = (dt / 2.0f - Km0 * X0) / D0;
            c3_0 = (Km0 * (1.0f - X0) - dt / 2.0f) / D0;
            c4_0 = (ql * dt) / D0;
            float Qmn0 = (WP0 + WPC0 > 0.0f)
                ? (1.0f / (((WP0 * n_ch) + (WPC0 * ncc)) / (WP0 + WPC0)))
                  * (A0 + AC0) * powf(R0, 2.0f/3.0f) * sqrtf(s0)
                : 0.0f;
            Qj_0 = (c1_0 * qup + c2_0 * quc + c3_0 * qdp + c4_0) - Qmn0;

            float twl1, R1, A1, AC1, WP1, WPC1;
            hgeo_inline(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            float Ck1 = 0.0f;
            if (h > 0.0f) {
                float r23 = powf(R1, 2.0f/3.0f);
                float r53 = powf(R1, 5.0f/3.0f);
                Ck1 = fmaxf(0.0f, (sqrtf(s0) / n_ch) *
                    ((5.0f / 3.0f) * r23 - (2.0f / 3.0f) * r53 *
                     (2.0f * sqrtf(1.0f + z * z) / (bw + 2.0f * h * z))));
            }
            float Km1 = (Ck1 > 0.0f) ? fmaxf(dt, dx / Ck1) : dt;
            float twu1 = (h > bfd && twcc > 0.0f) ? twcc : twl1;
            X = (Ck1 > 0.0f && twu1 * s0 * Ck1 * dx > 0.0f)
                ? fminf(0.5f, fmaxf(0.25f, 0.5f * (1.0f -
                    (c1_0 * qup + c2_0 * quc + c3_0 * qdp + c4_0)
                    / (2.0f * twu1 * s0 * Ck1 * dx))))
                : 0.5f;
            float D1 = Km1 * (1.0f - X) + dt / 2.0f;
            if (D1 == 0.0f) D1 = 1.0f;
            C1 = (Km1 * X + dt / 2.0f) / D1;
            C2 = (dt / 2.0f - Km1 * X) / D1;
            C3 = (Km1 * (1.0f - X) - dt / 2.0f) / D1;
            C4 = (ql * dt) / D1;
            if (C4 < 0.0f && fabsf(C4) > C1 * qup + C2 * quc + C3 * qdp) {
                C4 = -(C1 * qup + C2 * quc + C3 * qdp);
            }
            float Qmn1 = (WP1 + WPC1 > 0.0f)
                ? (1.0f / (((WP1 * n_ch) + (WPC1 * ncc)) / (WP1 + WPC1)))
                  * (A1 + AC1) * powf(R1, 2.0f/3.0f) * sqrtf(s0)
                : 0.0f;
            Qj = (C1 * qup + C2 * quc + C3 * qdp + C4) - Qmn1;

            float h_1 = (Qj_0 - Qj != 0.0f) ? h - (Qj * (h_0 - h)) / (Qj_0 - Qj) : h;
            if (h_1 < 0.0f) h_1 = h;

            if (h > 0.0f) { rerror = fabsf((h_1 - h) / h); aerror = fabsf(h_1 - h); }
            else { rerror = 0.0f; aerror = 0.9f; }

            h_0 = fmaxf(0.0f, h);
            h = fmaxf(0.0f, h_1);
            iter++;
            if (h < mindepth) break;
        }
        if (iter < maxiter) break;
        h *= 1.33f;
        h_0 *= 0.67f;
        maxiter += 25;
    }

    float Qmc = C1 * qup + C2 * quc + C3 * qdp + C4;
    float qdc;
    if (Qmc < 0.0f) {
        if (C4 < 0.0f && fabsf(C4) > C1 * qup + C2 * quc + C3 * qdp) qdc = 0.0f;
        else qdc = fmaxf(C1 * qup + C2 * quc + C4, C1 * qup + C3 * qdp + C4);
    } else {
        qdc = Qmc;
    }

    float twl = bw + 2.0f * z * h;
    float Rv = (h * (bw + twl) / 2.0f) /
                (bw + 2.0f * sqrtf(((twl - bw) / 2.0f) * ((twl - bw) / 2.0f) + h * h));
    if (Rv < 0.0f) Rv = 0.0f;
    velc_out = (1.0f / n_ch) * powf(Rv, 2.0f/3.0f) * sqrtf(s0);
    depthc_out = h;
    return qdc;
}

// ==========================================================================
// FP64 CPU reference — ground truth for validation. Uses same algorithm,
// same topology, same schedule as FP32 kernels, but internal math is
// double precision.
// ==========================================================================
static inline void hgeo_inline_d(
    double h, double bfd, double bw, double twcc, double z,
    double& twl, double& R, double& A, double& AC, double& WP, double& WPC)
{
    twl = bw + 2.0 * z * h;
    double hgb = std::fmax(h - bfd, 0.0);
    double hlb = std::fmin(bfd, h);
    if (hgb > 0.0 && twcc <= 0.0) { hgb = 0.0; hlb = h; }
    A = (bw + hlb * z) * hlb;
    WP = bw + 2.0 * hlb * std::sqrt(1.0 + z * z);
    AC = twcc * hgb;
    WPC = (hgb > 0.0) ? (twcc + 2.0 * hgb) : 0.0;
    R = (WP + WPC > 0.0) ? (A + AC) / (WP + WPC) : 0.0;
}

static inline double mc_solve_d(
    double dt, double qup, double quc, double qdp, double ql,
    double dx, double bw, double tw, double twcc,
    double n_ch, double ncc, double cs, double s0,
    double depthp, double& depthc_out, double& velc_out)
{
    double z = (cs == 0.0) ? 1.0 : 1.0 / cs;
    double bfd;
    if (bw > tw) bfd = bw / 0.00001;
    else if (bw == tw) bfd = bw / (2.0 * z);
    else bfd = (tw - bw) / (2.0 * z);

    if (n_ch <= 0.0 || s0 <= 0.0 || z <= 0.0 || bw <= 0.0) {
        depthc_out = 0.0; velc_out = 0.0; return 0.0;
    }
    if (ql <= 0.0 && qup <= 0.0 && quc <= 0.0 && qdp <= 0.0) {
        depthc_out = 0.0; velc_out = 0.0; return 0.0;
    }
    const double mindepth = 0.01;
    double depthc = std::fmax(depthp, 0.0);
    double h = depthc * 1.33 + mindepth;
    double h_0 = depthc * 0.67;

    double C1 = 0.0, C2 = 0.0, C3 = 0.0, C4 = 0.0, X = 0.25;
    double Qj_0 = 0.0, Qj = 0.0;
    int maxiter = 100;

    for (int attempt = 0; attempt < 5; attempt++) {
        double aerror = 0.01, rerror = 1.0;
        int iter = 0;
        double c1_0 = 0.0, c2_0 = 0.0, c3_0 = 0.0, c4_0 = 0.0;
        while (rerror > 0.01 && aerror >= mindepth && iter <= maxiter) {
            double twl0, R0, A0, AC0, WP0, WPC0;
            hgeo_inline_d(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);
            double Ck0 = 0.0;
            if (h_0 > 0.0) {
                double r23 = std::pow(R0, 2.0 / 3.0);
                double r53 = std::pow(R0, 5.0 / 3.0);
                Ck0 = std::fmax(0.0, (std::sqrt(s0) / n_ch) *
                    ((5.0 / 3.0) * r23 - (2.0 / 3.0) * r53 *
                     (2.0 * std::sqrt(1.0 + z * z) / (bw + 2.0 * h_0 * z))));
            }
            double Km0 = (Ck0 > 0.0) ? std::fmax(dt, dx / Ck0) : dt;
            double twu0 = (h_0 > bfd && twcc > 0.0) ? twcc : twl0;
            double X0 = (Ck0 > 0.0 && twu0 * s0 * Ck0 * dx > 0.0)
                ? std::fmin(0.5, std::fmax(0.0, 0.5 * (1.0 - Qj_0 / (2.0 * twu0 * s0 * Ck0 * dx))))
                : 0.5;
            double D0 = Km0 * (1.0 - X0) + dt / 2.0;
            if (D0 == 0.0) D0 = 1.0;
            c1_0 = (Km0 * X0 + dt / 2.0) / D0;
            c2_0 = (dt / 2.0 - Km0 * X0) / D0;
            c3_0 = (Km0 * (1.0 - X0) - dt / 2.0) / D0;
            c4_0 = (ql * dt) / D0;
            double Qmn0 = (WP0 + WPC0 > 0.0)
                ? (1.0 / (((WP0 * n_ch) + (WPC0 * ncc)) / (WP0 + WPC0)))
                  * (A0 + AC0) * std::pow(R0, 2.0 / 3.0) * std::sqrt(s0)
                : 0.0;
            Qj_0 = (c1_0 * qup + c2_0 * quc + c3_0 * qdp + c4_0) - Qmn0;

            double twl1, R1, A1, AC1, WP1, WPC1;
            hgeo_inline_d(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            double Ck1 = 0.0;
            if (h > 0.0) {
                double r23 = std::pow(R1, 2.0 / 3.0);
                double r53 = std::pow(R1, 5.0 / 3.0);
                Ck1 = std::fmax(0.0, (std::sqrt(s0) / n_ch) *
                    ((5.0 / 3.0) * r23 - (2.0 / 3.0) * r53 *
                     (2.0 * std::sqrt(1.0 + z * z) / (bw + 2.0 * h * z))));
            }
            double Km1 = (Ck1 > 0.0) ? std::fmax(dt, dx / Ck1) : dt;
            double twu1 = (h > bfd && twcc > 0.0) ? twcc : twl1;
            X = (Ck1 > 0.0 && twu1 * s0 * Ck1 * dx > 0.0)
                ? std::fmin(0.5, std::fmax(0.25, 0.5 * (1.0 -
                    (c1_0 * qup + c2_0 * quc + c3_0 * qdp + c4_0)
                    / (2.0 * twu1 * s0 * Ck1 * dx))))
                : 0.5;
            double D1 = Km1 * (1.0 - X) + dt / 2.0;
            if (D1 == 0.0) D1 = 1.0;
            C1 = (Km1 * X + dt / 2.0) / D1;
            C2 = (dt / 2.0 - Km1 * X) / D1;
            C3 = (Km1 * (1.0 - X) - dt / 2.0) / D1;
            C4 = (ql * dt) / D1;
            if (C4 < 0.0 && std::fabs(C4) > C1 * qup + C2 * quc + C3 * qdp) {
                C4 = -(C1 * qup + C2 * quc + C3 * qdp);
            }
            double Qmn1 = (WP1 + WPC1 > 0.0)
                ? (1.0 / (((WP1 * n_ch) + (WPC1 * ncc)) / (WP1 + WPC1)))
                  * (A1 + AC1) * std::pow(R1, 2.0 / 3.0) * std::sqrt(s0)
                : 0.0;
            Qj = (C1 * qup + C2 * quc + C3 * qdp + C4) - Qmn1;

            double h_1 = (Qj_0 - Qj != 0.0) ? h - (Qj * (h_0 - h)) / (Qj_0 - Qj) : h;
            if (h_1 < 0.0) h_1 = h;
            if (h > 0.0) { rerror = std::fabs((h_1 - h) / h); aerror = std::fabs(h_1 - h); }
            else { rerror = 0.0; aerror = 0.9; }
            h_0 = std::fmax(0.0, h);
            h = std::fmax(0.0, h_1);
            iter++;
            if (h < mindepth) break;
        }
        if (iter < maxiter) break;
        h *= 1.33;
        h_0 *= 0.67;
        maxiter += 25;
    }

    double Qmc = C1 * qup + C2 * quc + C3 * qdp + C4;
    double qdc;
    if (Qmc < 0.0) {
        if (C4 < 0.0 && std::fabs(C4) > C1 * qup + C2 * quc + C3 * qdp) qdc = 0.0;
        else qdc = std::fmax(C1 * qup + C2 * quc + C4, C1 * qup + C3 * qdp + C4);
    } else {
        qdc = Qmc;
    }

    double twl = bw + 2.0 * z * h;
    double Rv = (h * (bw + twl) / 2.0) /
                (bw + 2.0 * std::sqrt(((twl - bw) / 2.0) * ((twl - bw) / 2.0) + h * h));
    if (Rv < 0.0) Rv = 0.0;
    velc_out = (1.0 / n_ch) * std::pow(Rv, 2.0 / 3.0) * std::sqrt(s0);
    depthc_out = h;
    return qdc;
}

// ==========================================================================
// GPU kernel: run one level of reaches in parallel
// ==========================================================================
__global__ void kernel_level_wavefront(
    const ChParams* __restrict__ params,
    const int*      __restrict__ reach_seg_start,
    const int*      __restrict__ reach_seg_len,
    const int*      __restrict__ reach_up_ptr,
    const int*      __restrict__ reach_up_idx,
    const int*      __restrict__ level_reach_slice,  // reach ids at this level
    const float*    __restrict__ qlat,               // [n_segments]
    const float*    __restrict__ qdp_prev_seg,       // [n_segments]  prev-timestep qdc per segment
    const float*    __restrict__ dp_prev_seg,        // [n_segments]  prev-timestep depth per segment
    const float*    __restrict__ reach_last_prev,    // [n_reaches]   prev-timestep qdc of last seg per reach
    const float*    __restrict__ reach_last_curr,    // [n_reaches]   curr-timestep qdc of upstream reaches
    float*          __restrict__ qdc_seg_out,        // [n_segments]
    float*          __restrict__ velc_seg_out,       // [n_segments]
    float*          __restrict__ depthc_seg_out,     // [n_segments]
    float*          __restrict__ reach_last_out,     // [n_reaches] curr-timestep last-qdc per reach
    float           dt,
    int             level_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int r = level_reach_slice[tid];

    // Aggregate upstream flows at the last segment of every upstream reach
    float qup_reach_prev = 0.0f, quc_reach_curr = 0.0f;
    int up_begin = reach_up_ptr[r];
    int up_end   = reach_up_ptr[r + 1];
    for (int k = up_begin; k < up_end; ++k) {
        int u = reach_up_idx[k];
        qup_reach_prev += reach_last_prev[u];
        quc_reach_curr += reach_last_curr[u];
    }

    int seg_start = reach_seg_start[r];
    int seg_len   = reach_seg_len[r];

    float quc_next = 0.0f;   // just-computed qdc of previous segment in this reach (for quc of next segment)
    float qdc_last = 0.0f;

    for (int i = 0; i < seg_len; ++i) {
        int idx = seg_start + i;
        const ChParams& p = params[idx];

        // qup = (i==0) ? sum(reach_last_prev of upstream reaches) : qdp_prev_seg[idx-1]
        // quc = (i==0) ? sum(reach_last_curr of upstream reaches) : just-computed qdc of segment i-1
        float qup_i = (i == 0) ? qup_reach_prev : qdp_prev_seg[idx - 1];
        float quc_i = (i == 0) ? quc_reach_curr : quc_next;

        float depthc, velc;
        float qdc = mc_solve(
            dt,
            qup_i, quc_i,
            qdp_prev_seg[idx],
            qlat[idx],
            p.dx, p.bw, p.tw, p.twcc, p.n_ch, p.ncc, p.cs, p.s0,
            dp_prev_seg[idx],
            depthc, velc);
        qdc_seg_out[idx]    = qdc;
        velc_seg_out[idx]   = velc;
        depthc_seg_out[idx] = depthc;
        quc_next = qdc;
        qdc_last = qdc;
    }

    reach_last_out[r] = qdc_last;
}

// ==========================================================================
// CPU reference (single-thread, same wavefront order)
// ==========================================================================
static void cpu_wavefront(
    const ChParams* params,
    const int* reach_seg_start, const int* reach_seg_len,
    const int* reach_up_ptr, const int* reach_up_idx,
    const int* level_ptr, const int* level_reach,
    int n_reaches, int n_levels, int n_segments,
    const float* qlat_ts,         // [n_timesteps * n_segments]
    const float* qdp0_seg,        // [n_segments] prev-timestep downstream flow per segment
    const float* dp0_seg,         // [n_segments]
    const float* reach_last0,     // [n_reaches] prev-timestep last-seg qdc per reach
    float* qdc_seg_out,           // [n_segments] final-timestep per-segment qdc
    float* reach_last_out,        // [n_reaches] final-timestep per-reach last-segment qdc
    int n_timesteps, float dt)
{
    std::vector<float> qdp_prev_seg(qdp0_seg, qdp0_seg + n_segments);
    std::vector<float> dp_prev_seg(dp0_seg, dp0_seg + n_segments);
    std::vector<float> reach_last_prev(reach_last0, reach_last0 + n_reaches);
    std::vector<float> reach_last_curr(n_reaches, 0.0f);

    std::vector<float> qdc_seg_curr(n_segments, 0.0f);
    std::vector<float> vel_seg_curr(n_segments, 0.0f);
    std::vector<float> dep_seg_curr(n_segments, 0.0f);

    for (int t = 0; t < n_timesteps; ++t) {
        const float* qlat = qlat_ts + (size_t)t * n_segments;
        // Process levels in order
        for (int L = 0; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            for (int ii = lb; ii < le; ++ii) {
                int r = level_reach[ii];

                float qup_reach_prev = 0.0f, quc_reach_curr = 0.0f;
                int up_begin = reach_up_ptr[r];
                int up_end   = reach_up_ptr[r + 1];
                for (int k = up_begin; k < up_end; ++k) {
                    int u = reach_up_idx[k];
                    qup_reach_prev += reach_last_prev[u];
                    quc_reach_curr += reach_last_curr[u];
                }

                int seg_start = reach_seg_start[r];
                int seg_len   = reach_seg_len[r];
                float quc_next = 0.0f;
                float qdc_last = 0.0f;

                for (int i = 0; i < seg_len; ++i) {
                    int idx = seg_start + i;
                    const ChParams& p = params[idx];
                    float qup_i = (i == 0) ? qup_reach_prev : qdp_prev_seg[idx - 1];
                    float quc_i = (i == 0) ? quc_reach_curr : quc_next;
                    float depthc, velc;
                    float qdc = mc_solve(
                        dt, qup_i, quc_i,
                        qdp_prev_seg[idx],
                        qlat[idx],
                        p.dx, p.bw, p.tw, p.twcc, p.n_ch, p.ncc, p.cs, p.s0,
                        dp_prev_seg[idx],
                        depthc, velc);
                    qdc_seg_curr[idx] = qdc;
                    vel_seg_curr[idx] = velc;
                    dep_seg_curr[idx] = depthc;
                    quc_next = qdc;
                    qdc_last = qdc;
                }

                reach_last_curr[r] = qdc_last;
            }
        }

        // Roll state: prev <- curr
        std::memcpy(qdp_prev_seg.data(), qdc_seg_curr.data(), n_segments * sizeof(float));
        std::memcpy(dp_prev_seg.data(),  dep_seg_curr.data(), n_segments * sizeof(float));
        reach_last_prev.swap(reach_last_curr);
        std::fill(reach_last_curr.begin(), reach_last_curr.end(), 0.0f);
    }

    std::memcpy(qdc_seg_out, qdp_prev_seg.data(), n_segments * sizeof(float));
    std::memcpy(reach_last_out, reach_last_prev.data(), n_reaches * sizeof(float));
}

// CPU reference using FP64 internally — considered ground truth.
// Inputs/outputs are still float32 (matching Fortran REAL interface).
static void cpu_wavefront_fp64(
    const ChParams* params,
    const int* reach_seg_start, const int* reach_seg_len,
    const int* reach_up_ptr, const int* reach_up_idx,
    const int* level_ptr, const int* level_reach,
    int n_reaches, int n_levels, int n_segments,
    const float* qlat_ts,
    const float* qdp0_seg,
    const float* dp0_seg,
    const float* reach_last0,
    double* qdc_seg_out_d,       // keep as double for highest fidelity
    double* reach_last_out_d,
    int n_timesteps, double dt)
{
    std::vector<double> qdp_prev_seg(n_segments);
    std::vector<double> dp_prev_seg(n_segments);
    std::vector<double> reach_last_prev(n_reaches);
    std::vector<double> reach_last_curr(n_reaches, 0.0);
    for (int i = 0; i < n_segments; ++i) { qdp_prev_seg[i] = qdp0_seg[i]; dp_prev_seg[i] = dp0_seg[i]; }
    for (int i = 0; i < n_reaches; ++i) reach_last_prev[i] = reach_last0[i];

    std::vector<double> qdc_seg_curr(n_segments, 0.0);
    std::vector<double> dep_seg_curr(n_segments, 0.0);

    for (int t = 0; t < n_timesteps; ++t) {
        const float* qlat = qlat_ts + (size_t)t * n_segments;
        for (int L = 0; L < n_levels; ++L) {
            int lb = level_ptr[L], le = level_ptr[L + 1];
            for (int ii = lb; ii < le; ++ii) {
                int r = level_reach[ii];
                double qup_reach_prev = 0.0, quc_reach_curr = 0.0;
                int ub = reach_up_ptr[r], ue = reach_up_ptr[r + 1];
                for (int k = ub; k < ue; ++k) {
                    int u = reach_up_idx[k];
                    qup_reach_prev += reach_last_prev[u];
                    quc_reach_curr += reach_last_curr[u];
                }
                int sstart = reach_seg_start[r];
                int slen = reach_seg_len[r];
                double quc_next = 0.0, qdc_last = 0.0;
                for (int i = 0; i < slen; ++i) {
                    int idx = sstart + i;
                    const ChParams& p = params[idx];
                    double qup_i = (i == 0) ? qup_reach_prev : qdp_prev_seg[idx - 1];
                    double quc_i = (i == 0) ? quc_reach_curr : quc_next;
                    double depthc, velc;
                    double qdc = mc_solve_d(dt, qup_i, quc_i,
                        qdp_prev_seg[idx], (double)qlat[idx],
                        (double)p.dx, (double)p.bw, (double)p.tw, (double)p.twcc,
                        (double)p.n_ch, (double)p.ncc, (double)p.cs, (double)p.s0,
                        dp_prev_seg[idx], depthc, velc);
                    qdc_seg_curr[idx] = qdc;
                    dep_seg_curr[idx] = depthc;
                    quc_next = qdc;
                    qdc_last = qdc;
                }
                reach_last_curr[r] = qdc_last;
            }
        }
        qdp_prev_seg.swap(qdc_seg_curr);
        dp_prev_seg.swap(dep_seg_curr);
        reach_last_prev.swap(reach_last_curr);
        std::fill(reach_last_curr.begin(), reach_last_curr.end(), 0.0);
        std::fill(qdc_seg_curr.begin(), qdc_seg_curr.end(), 0.0);
        std::fill(dep_seg_curr.begin(), dep_seg_curr.end(), 0.0);
    }

    for (int i = 0; i < n_segments; ++i) qdc_seg_out_d[i] = qdp_prev_seg[i];
    for (int i = 0; i < n_reaches; ++i) reach_last_out_d[i] = reach_last_prev[i];
}

// Simple percentile helper (nth-element based, mutates input)
static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    size_t k = (size_t)((p / 100.0) * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// ==========================================================================
// I/O helpers
// ==========================================================================
struct Topo {
    int n_reaches = 0, n_segments = 0, n_levels = 0, max_upstream = 0;
    std::vector<int> reach_seg_start;
    std::vector<int> reach_seg_len;
    std::vector<int> reach_level;
    std::vector<int> reach_n_up;
    std::vector<int> reach_up_ptr;
    std::vector<int> reach_up_idx;
    std::vector<int> level_ptr;
    std::vector<int> level_reach;
};

static bool read_all(const std::string& path, void* buf, size_t nbytes)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path.c_str()); return false; }
    size_t got = std::fread(buf, 1, nbytes, f);
    std::fclose(f);
    if (got != nbytes) { fprintf(stderr, "short read %s: %zu vs %zu\n", path.c_str(), got, nbytes); return false; }
    return true;
}

static Topo load_topo(const std::string& dir)
{
    Topo t;
    std::string path = dir + "/topo.bin";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path.c_str()); std::exit(1); }
    std::fread(&t.n_reaches, 4, 1, f);
    std::fread(&t.n_segments, 4, 1, f);
    std::fread(&t.n_levels, 4, 1, f);
    std::fread(&t.max_upstream, 4, 1, f);

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

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <net_dir> [--no-cpu]\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    bool skip_cpu = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-cpu") skip_cpu = true;
    }

    printf("[io] loading topology from %s\n", dir.c_str());
    Topo t = load_topo(dir);
    printf("[io]  n_reaches=%d n_segments=%d n_levels=%d max_upstream=%d\n",
           t.n_reaches, t.n_segments, t.n_levels, t.max_upstream);

    // Load params (double)
    std::vector<ChParams> params(t.n_segments);
    if (!read_all(dir + "/params.bin", params.data(), t.n_segments * sizeof(ChParams))) return 1;

    // Load forcings
    std::string fp = dir + "/forcings.bin";
    FILE* f = std::fopen(fp.c_str(), "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", fp.c_str()); return 1; }
    int n_ts = 0;
    std::fread(&n_ts, 4, 1, f);
    std::vector<float> qlat_ts((size_t)n_ts * t.n_segments);
    std::fread(qlat_ts.data(), 4, qlat_ts.size(), f);
    std::vector<float> qup0_reach(t.n_reaches);
    std::fread(qup0_reach.data(), 4, t.n_reaches, f);
    std::vector<float> qdp0_seg(t.n_segments);
    std::fread(qdp0_seg.data(), 4, t.n_segments, f);
    std::vector<float> dp0_seg(t.n_segments);
    std::fread(dp0_seg.data(), 4, t.n_segments, f);
    std::fclose(f);
    printf("[io]  n_timesteps=%d\n", n_ts);

    // Derive reach_last0 from qdp0_seg (= qdc of last segment of each reach at t=-1)
    std::vector<float> reach_last0(t.n_reaches);
    for (int r = 0; r < t.n_reaches; ++r) {
        int last_idx = t.reach_seg_start[r] + t.reach_seg_len[r] - 1;
        reach_last0[r] = qdp0_seg[last_idx];
    }

    // Allocate GPU arrays
    printf("[gpu] allocating...\n");
    ChParams* d_params = nullptr;
    int *d_seg_start = nullptr, *d_seg_len = nullptr;
    int *d_up_ptr = nullptr, *d_up_idx = nullptr;
    int *d_level_reach = nullptr;
    float *d_qlat = nullptr;
    float *d_qdp_prev = nullptr, *d_dp_prev = nullptr;
    float *d_qdc_curr = nullptr, *d_vel_curr = nullptr, *d_dep_curr = nullptr;
    float *d_reach_last_prev = nullptr, *d_reach_last_curr = nullptr;

    CUDA_CHECK(cudaMalloc(&d_params, t.n_segments * sizeof(ChParams)));
    CUDA_CHECK(cudaMalloc(&d_seg_start, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_seg_len, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_up_ptr, (t.n_reaches + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_up_idx, t.reach_up_idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_level_reach, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_qlat, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qdp_prev, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_dp_prev, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_qdc_curr, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_vel_curr, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_dep_curr, t.n_segments * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_last_prev, t.n_reaches * 4));
    CUDA_CHECK(cudaMalloc(&d_reach_last_curr, t.n_reaches * 4));

    CUDA_CHECK(cudaMemcpy(d_params, params.data(), t.n_segments * sizeof(ChParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_start, t.reach_seg_start.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_len, t.reach_seg_len.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_ptr, t.reach_up_ptr.data(), (t.n_reaches + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_idx, t.reach_up_idx.data(), t.reach_up_idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_reach, t.level_reach.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    // Initial conditions
    CUDA_CHECK(cudaMemcpy(d_qdp_prev, qdp0_seg.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dp_prev, dp0_seg.data(), t.n_segments * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reach_last_prev, reach_last0.data(), t.n_reaches * 4, cudaMemcpyHostToDevice));

    int threads = 256;

    // Warmup: run one level once
    kernel_level_wavefront<<<1, 32>>>(
        d_params, d_seg_start, d_seg_len, d_up_ptr, d_up_idx,
        d_level_reach, d_qlat, d_qdp_prev, d_dp_prev,
        d_reach_last_prev, d_reach_last_curr,
        d_qdc_curr, d_vel_curr, d_dep_curr,
        d_reach_last_curr, 300.0f, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- GPU run ----
    printf("[gpu] running %d timesteps x %d levels\n", n_ts, t.n_levels);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);

    for (int ts = 0; ts < n_ts; ++ts) {
        const float* qlat_host = qlat_ts.data() + (size_t)ts * t.n_segments;
        CUDA_CHECK(cudaMemcpyAsync(d_qlat, qlat_host, t.n_segments * 4, cudaMemcpyHostToDevice));
        // Clear reach_last_curr
        CUDA_CHECK(cudaMemsetAsync(d_reach_last_curr, 0, t.n_reaches * 4));

        for (int L = 0; L < t.n_levels; ++L) {
            int lb = t.level_ptr[L], le = t.level_ptr[L + 1];
            int count = le - lb;
            if (count <= 0) continue;
            int blocks = (count + threads - 1) / threads;
            kernel_level_wavefront<<<blocks, threads>>>(
                d_params, d_seg_start, d_seg_len, d_up_ptr, d_up_idx,
                d_level_reach + lb,
                d_qlat, d_qdp_prev, d_dp_prev,
                d_reach_last_prev, d_reach_last_curr,
                d_qdc_curr, d_vel_curr, d_dep_curr,
                d_reach_last_curr, 300.0f, count);
        }
        // Roll state via pointer swap — no device-to-device memcpy needed.
        // After the swap, the "prev" buffers hold the values just computed,
        // and the "curr" buffers become scratch for the next timestep.
        std::swap(d_qdp_prev, d_qdc_curr);
        std::swap(d_dp_prev,  d_dep_curr);
        std::swap(d_reach_last_prev, d_reach_last_curr);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, e0, e1);
    printf("[gpu] total ms: %.3f  per-timestep: %.3f ms\n", gpu_ms, gpu_ms / n_ts);

    std::vector<float> gpu_qdc_seg(t.n_segments);
    std::vector<float> gpu_reach_last(t.n_reaches);
    CUDA_CHECK(cudaMemcpy(gpu_qdc_seg.data(), d_qdp_prev, t.n_segments * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_reach_last.data(), d_reach_last_prev, t.n_reaches * 4, cudaMemcpyDeviceToHost));

    cudaFree(d_params); cudaFree(d_seg_start); cudaFree(d_seg_len);
    cudaFree(d_up_ptr); cudaFree(d_up_idx); cudaFree(d_level_reach);
    cudaFree(d_qlat); cudaFree(d_qdp_prev); cudaFree(d_dp_prev);
    cudaFree(d_qdc_curr); cudaFree(d_vel_curr); cudaFree(d_dep_curr);
    cudaFree(d_reach_last_prev); cudaFree(d_reach_last_curr);

    if (skip_cpu) {
        printf("[cpu] skipped\n");
        return 0;
    }

    // ---- CPU FP32 reference ----
    printf("[cpu-fp32] running CPU reference (single-thread, same schedule, FP32)\n");
    std::vector<float> cpu_qdc_seg(t.n_segments);
    std::vector<float> cpu_reach_last(t.n_reaches);
    auto c0 = std::chrono::steady_clock::now();
    cpu_wavefront(
        params.data(),
        t.reach_seg_start.data(), t.reach_seg_len.data(),
        t.reach_up_ptr.data(), t.reach_up_idx.data(),
        t.level_ptr.data(), t.level_reach.data(),
        t.n_reaches, t.n_levels, t.n_segments,
        qlat_ts.data(), qdp0_seg.data(), dp0_seg.data(), reach_last0.data(),
        cpu_qdc_seg.data(), cpu_reach_last.data(),
        n_ts, 300.0f);
    auto c1 = std::chrono::steady_clock::now();
    double cpu_fp32_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    printf("[cpu-fp32] total ms: %.3f  per-timestep: %.3f ms\n", cpu_fp32_ms, cpu_fp32_ms / n_ts);

    // ---- CPU FP64 ground truth ----
    printf("[cpu-fp64] running CPU FP64 ground-truth reference\n");
    std::vector<double> truth_qdc_seg(t.n_segments);
    std::vector<double> truth_reach_last(t.n_reaches);
    auto d0 = std::chrono::steady_clock::now();
    cpu_wavefront_fp64(
        params.data(),
        t.reach_seg_start.data(), t.reach_seg_len.data(),
        t.reach_up_ptr.data(), t.reach_up_idx.data(),
        t.level_ptr.data(), t.level_reach.data(),
        t.n_reaches, t.n_levels, t.n_segments,
        qlat_ts.data(), qdp0_seg.data(), dp0_seg.data(), reach_last0.data(),
        truth_qdc_seg.data(), truth_reach_last.data(),
        n_ts, 300.0);
    auto d1 = std::chrono::steady_clock::now();
    double cpu_fp64_ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
    printf("[cpu-fp64] total ms: %.3f\n", cpu_fp64_ms);

    // ---- Accuracy comparison vs FP64 truth ----
    auto compute_stats = [&](const float* pred, const char* name) {
        std::vector<double> rel_errors;
        rel_errors.reserve(t.n_segments);
        double max_abs = 0.0;
        int nans = 0, nonzero = 0;
        for (int i = 0; i < t.n_segments; ++i) {
            double tr = truth_qdc_seg[i];
            double pv = (double)pred[i];
            if (std::isnan(pv) || std::isinf(pv)) { nans++; continue; }
            double ae = std::fabs(pv - tr);
            max_abs = std::fmax(max_abs, ae);
            if (std::fabs(tr) > 1e-3) {
                rel_errors.push_back(ae / std::fabs(tr));
                nonzero++;
            }
        }
        if (rel_errors.empty()) {
            printf("[acc:%s] (no nonzero truth segments)\n", name);
            return;
        }
        std::vector<double> r = rel_errors;
        double p50 = pct(r, 50.0);
        r = rel_errors; double p90 = pct(r, 90.0);
        r = rel_errors; double p99 = pct(r, 99.0);
        r = rel_errors; double p999 = pct(r, 99.9);
        double pmax = *std::max_element(rel_errors.begin(), rel_errors.end());
        int under1pct = 0, under10pct = 0;
        for (double rr : rel_errors) {
            if (rr < 0.01) under1pct++;
            if (rr < 0.1)  under10pct++;
        }
        double frac_1pct  = 100.0 * under1pct  / nonzero;
        double frac_10pct = 100.0 * under10pct / nonzero;
        printf("[acc:%s vs FP64] max_abs=%.3e  p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e max=%.2e  "
               "within-1%%=%.2f%% within-10%%=%.2f%% nans=%d\n",
               name, max_abs, p50, p90, p99, p999, pmax, frac_1pct, frac_10pct, nans);
    };

    compute_stats(gpu_qdc_seg.data(), "GPU-FP32");
    compute_stats(cpu_qdc_seg.data(), "CPU-FP32");

    printf("[sum] Timings: GPU-FP32 %.2f ms | CPU-FP32 %.2f ms | CPU-FP64 %.2f ms\n",
           gpu_ms, cpu_fp32_ms, cpu_fp64_ms);
    printf("[sum] Speedup GPU-FP32 vs CPU-FP32: %.2fx\n", cpu_fp32_ms / gpu_ms);
    printf("[sum] Speedup GPU-FP32 vs CPU-FP64: %.2fx\n", cpu_fp64_ms / gpu_ms);
    return 0;
}
