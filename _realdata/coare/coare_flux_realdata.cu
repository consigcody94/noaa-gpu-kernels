/**
 * COARE 3.6 Bulk Air-Sea Flux Algorithm — GPU Kernel — REAL DATA HARNESS
 *
 * Copy of ocean/coare/coare_flux.cu with ONLY the synthetic input generator
 * (gen_coare + size-sweep main) replaced by a loader for real observations.
 * Kernel math (kernel_coare), CPU reference (cpu_coare) and helper functions
 * are UNCHANGED, byte-for-byte, from the original.
 *
 * Real data: official COARE 3.6 validation dataset test_36_data.txt from
 * https://github.com/NOAA-PSL/COARE-algorithm (Python/COARE3.6), 2165 ship
 * observations, tropical Atlantic (RV Ronald H. Brown, ATOMIC, Jan-Feb 2020).
 * Converted to coare_inputs.bin by prep_realdata.py.
 * Gold-standard expected outputs (official repo file
 * test_36_output_withnowavesinput_withwarmlayer.txt) in coare_expected.bin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define VON 0.4f       // von Karman constant
#define GRAV 9.81f
#define RGAS 287.1f
#define CPA 1004.67f
#define T2K 273.15f
#define BETA 1.2f
#define FDG 1.0f

struct COAREInput {
    float u;     // wind speed (m/s)
    float ta;    // air temperature (C)
    float rh;    // relative humidity (%)
    float P;     // pressure (mb)
    float ts;    // sea surface temperature (C)
    float sw_dn; // downwelling shortwave (W/m2)
    float lw_dn; // downwelling longwave (W/m2)
    float zu;    // wind measurement height (m)
    float zt;    // temperature height (m)
    float zq;    // humidity height (m)
};

struct COAREOutput {
    float tau;   // wind stress (N/m2)
    float hsb;   // sensible heat flux (W/m2)
    float hlb;   // latent heat flux (W/m2)
};

// Stability function for momentum (Paulson 1970 + Beljaars-Holtslag)
__host__ __device__ float psiu_26(float zet) {
    if (zet < 0.0f) {
        float x = (1.0f - 15.0f * zet);
        if (x < 0.0f) x = 0.001f;
        x = sqrtf(sqrtf(x));
        float psik = 2.0f * logf((1.0f + x) / 2.0f) + logf((1.0f + x*x) / 2.0f) - 2.0f * atanf(x) + 1.5707963f;
        float y = (1.0f - 34.15f * zet);
        if (y < 0.0f) y = 0.001f;
        y = cbrtf(y);
        float psic = 1.5f * logf((1.0f + y + y*y) / 3.0f) - 1.7320508f * atanf((1.0f + 2.0f*y) / 1.7320508f) + 1.8137994f;
        float f = zet * zet / (1.0f + zet * zet);
        return (1.0f - f) * psik + f * psic;
    } else {
        float dzet = fminf(50.0f, 0.35f * zet);
        return -((1.0f + zet) + 0.6667f * (zet - 14.28f) * expf(-dzet) + 8.525f);
    }
}

// Stability function for temperature/moisture
__host__ __device__ float psit_26(float zet) {
    if (zet < 0.0f) {
        float x = (1.0f - 15.0f * zet);
        if (x < 0.0f) x = 0.001f;
        x = sqrtf(x);
        float psik = 2.0f * logf((1.0f + x) / 2.0f);
        float y = (1.0f - 34.15f * zet);
        if (y < 0.0f) y = 0.001f;
        y = cbrtf(y);
        float psic = 1.5f * logf((1.0f + y + y*y) / 3.0f) - 1.7320508f * atanf((1.0f + 2.0f*y) / 1.7320508f) + 1.8137994f;
        float f = zet * zet / (1.0f + zet * zet);
        return (1.0f - f) * psik + f * psic;
    } else {
        float dzet = fminf(50.0f, 0.35f * zet);
        return -((1.0f + 0.6667f * zet) * 1.5f + 0.6667f * (zet - 14.28f) * expf(-dzet) + 8.525f);
    }
}

// Saturation specific humidity
__host__ __device__ float qsat_f(float T, float P) {
    float es = 6.1121f * expf(17.502f * T / (T + 240.97f)) * (1.0007f + 3.46e-6f * P);
    return 0.622f * es / (P - 0.378f * es);
}

// CPU reference
void cpu_coare(const COAREInput* in, COAREOutput* out, int n) {
    for (int c = 0; c < n; c++) {
        float u = fmaxf(in[c].u, 0.5f);
        float ta = in[c].ta;
        float ts = in[c].ts;
        float P = in[c].P;
        float zu = in[c].zu;
        float zt = in[c].zt;
        float zq = in[c].zq;

        float rhoa = P * 100.0f / (RGAS * (ta + T2K) * (1.0f + 0.61f * qsat_f(ta, P) * in[c].rh / 100.0f));
        float Le = (2.501f - 0.00237f * ts) * 1e6f;
        float visa = 1.326e-5f * (1.0f + 6.542e-3f * ta + 8.301e-6f * ta*ta);

        float qs = qsat_f(ts, P);
        float qa = qsat_f(ta, P) * in[c].rh / 100.0f;
        float dT = ts - ta - 0.0098f * zt;
        float dq = qs - qa;
        float du = u;

        // Initial guesses
        float usr = 0.035f * u;
        float tsr = -dT * 0.001f;
        float qsr = -dq * 0.001f;
        float tvsr = tsr + 0.61f * (ta + T2K) * qsr;
        float ut = sqrtf(du*du + 0.04f);

        // Iterative stability loop (10 iterations, matching Fortran)
        for (int i = 0; i < 10; i++) {
            float zet = VON * GRAV * zu * tvsr / ((ta + T2K) * usr * usr);
            if (fabsf(zet) > 50.0f) zet = (zet > 0) ? 50.0f : -50.0f;

            float zo = 0.011f * usr*usr / GRAV + 0.11f * visa / usr;
            if (zo < 1e-10f) zo = 1e-10f;

            float rr = zo * usr / visa;
            float zoq = fminf(1.6e-4f, 5.8e-5f / powf(rr, 0.72f));
            float zot = zoq;

            float cdhf = VON / (logf(zu / zo) - psiu_26(zu * zet / zu));
            float cthf = VON * FDG / (logf(zt / zot) - psit_26(zt * zet / zu));
            float cqhf = VON * FDG / (logf(zq / zoq) - psit_26(zq * zet / zu));

            usr = ut * cdhf;
            if (usr < 0.001f) usr = 0.001f;
            tsr = -dT * cthf;
            qsr = -dq * cqhf;
            tvsr = tsr + 0.61f * (ta + T2K) * qsr;

            float Bf = -GRAV / (ta + T2K) * usr * tvsr;
            float gust = 0.2f;
            if (Bf > 0.0f) gust = BETA * cbrtf(Bf * 600.0f);
            ut = sqrtf(du*du + gust*gust);
        }

        out[c].tau = rhoa * usr * usr;
        out[c].hsb = -rhoa * CPA * usr * tsr;
        out[c].hlb = -rhoa * Le * usr * qsr;
    }
}

// GPU kernel
__global__ void kernel_coare(const COAREInput* __restrict__ in,
                              COAREOutput* __restrict__ out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    float u = fmaxf(in[c].u, 0.5f);
    float ta = in[c].ta;
    float ts = in[c].ts;
    float P = in[c].P;
    float zu = in[c].zu;
    float zt = in[c].zt;
    float zq = in[c].zq;

    // Air properties
    float es_a = 6.1121f * __expf(17.502f * ta / (ta + 240.97f)) * (1.0007f + 3.46e-6f * P);
    float qa = 0.622f * es_a / (P - 0.378f * es_a) * in[c].rh / 100.0f;
    float rhoa = P * 100.0f / (RGAS * (ta + T2K) * (1.0f + 0.61f * qa));
    float Le = (2.501f - 0.00237f * ts) * 1e6f;
    float visa = 1.326e-5f * (1.0f + 6.542e-3f * ta + 8.301e-6f * ta*ta);

    float es_s = 6.1121f * __expf(17.502f * ts / (ts + 240.97f)) * (1.0007f + 3.46e-6f * P);
    float qs = 0.622f * es_s / (P - 0.378f * es_s);
    float dT = ts - ta - 0.0098f * zt;
    float dq = qs - qa;
    float du = u;

    float usr = 0.035f * u;
    float tsr = -dT * 0.001f;
    float qsr = -dq * 0.001f;
    float tvsr = tsr + 0.61f * (ta + T2K) * qsr;
    float ut = __fsqrt_rn(du*du + 0.04f);

    for (int i = 0; i < 10; i++) {
        float zet = VON * GRAV * zu * tvsr / ((ta + T2K) * usr * usr);
        zet = fmaxf(-50.0f, fminf(50.0f, zet));

        float zo = 0.011f * usr*usr / GRAV + 0.11f * visa / usr;
        if (zo < 1e-10f) zo = 1e-10f;

        float rr = zo * usr / visa;
        float zoq = fminf(1.6e-4f, 5.8e-5f / __powf(rr, 0.72f));
        float zot = zoq;

        float L_ratio = zet / zu;
        float cdhf = VON / (__logf(zu / zo) - psiu_26(zu * L_ratio));
        float cthf = VON * FDG / (__logf(zt / zot) - psit_26(zt * L_ratio));
        float cqhf = VON * FDG / (__logf(zq / zoq) - psit_26(zq * L_ratio));

        usr = ut * cdhf;
        if (usr < 0.001f) usr = 0.001f;
        tsr = -dT * cthf;
        qsr = -dq * cqhf;
        tvsr = tsr + 0.61f * (ta + T2K) * qsr;

        float Bf = -GRAV / (ta + T2K) * usr * tvsr;
        float gust = 0.2f;
        if (Bf > 0.0f) gust = BETA * cbrtf(Bf * 600.0f);
        ut = __fsqrt_rn(du*du + gust*gust);
    }

    out[c].tau = rhoa * usr * usr;
    out[c].hsb = -rhoa * CPA * usr * tsr;
    out[c].hlb = -rhoa * Le * usr * qsr;
}

// ---------------------------------------------------------------------------
// REAL-DATA LOADER (replaces synthetic gen_coare). No values are generated;
// every record is read from coare_inputs.bin produced by prep_realdata.py
// from the official NOAA-PSL COARE 3.6 test dataset.
// ---------------------------------------------------------------------------
COAREInput* load_coare_inputs(const char* path, int* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fprintf(stderr, "bad header in %s\n", path); exit(1); }
    COAREInput* in = (COAREInput*)malloc((size_t)n * sizeof(COAREInput));
    // Binary record order matches struct order exactly: u,ta,rh,P,ts,sw_dn,lw_dn,zu,zt,zq
    if (fread(in, sizeof(COAREInput), n, f) != (size_t)n) { fprintf(stderr, "short read in %s\n", path); exit(1); }
    fclose(f);
    *n_out = n;
    return in;
}

struct GoldRec { float usr, tau, hsb, hlb; };

GoldRec* load_coare_expected(const char* path, int* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fprintf(stderr, "bad header in %s\n", path); exit(1); }
    GoldRec* g = (GoldRec*)malloc((size_t)n * sizeof(GoldRec));
    if (fread(g, sizeof(GoldRec), n, f) != (size_t)n) { fprintf(stderr, "short read in %s\n", path); exit(1); }
    fclose(f);
    *n_out = n;
    return g;
}

static void report_vs_gold(const char* label, const COAREOutput* o, const GoldRec* g, int n) {
    double b_tau = 0, b_hsb = 0, b_hlb = 0;          // mean bias (model - gold)
    double ma_tau = 0, ma_hsb = 0, ma_hlb = 0;        // mean abs error
    float mx_tau = 0, mx_hsb = 0, mx_hlb = 0;         // max abs error
    for (int c = 0; c < n; c++) {
        float dt = o[c].tau - g[c].tau, dh = o[c].hsb - g[c].hsb, dl = o[c].hlb - g[c].hlb;
        b_tau += dt; b_hsb += dh; b_hlb += dl;
        ma_tau += fabsf(dt); ma_hsb += fabsf(dh); ma_hlb += fabsf(dl);
        mx_tau = fmaxf(mx_tau, fabsf(dt)); mx_hsb = fmaxf(mx_hsb, fabsf(dh)); mx_hlb = fmaxf(mx_hlb, fabsf(dl));
    }
    printf("  %s vs OFFICIAL expected outputs (n=%d):\n", label, n);
    printf("    tau: bias=%+.4f  mean|err|=%.4f  max|err|=%.4f  (N/m2)\n", b_tau/n, ma_tau/n, mx_tau);
    printf("    hsb: bias=%+.3f  mean|err|=%.3f  max|err|=%.3f  (W/m2)\n", b_hsb/n, ma_hsb/n, mx_hsb);
    printf("    hlb: bias=%+.3f  mean|err|=%.3f  max|err|=%.3f  (W/m2)\n", b_hlb/n, ma_hlb/n, mx_hlb);
}

int main() {
    printf("================================================\n");
    printf("  COARE 3.6 Air-Sea Flux GPU Kernel — REAL DATA\n");
    printf("  Input: NOAA-PSL COARE-algorithm test_36_data.txt\n");
    printf("  (2165 ship obs, tropical Atlantic, ATOMIC 2020)\n");
    printf("================================================\n\n");

    int n = 0, ng = 0;
    COAREInput* hi = load_coare_inputs("coare_inputs.bin", &n);
    GoldRec* gold = load_coare_expected("coare_expected.bin", &ng);
    if (ng != n) { fprintf(stderr, "input/gold count mismatch %d vs %d\n", n, ng); return 1; }
    printf("--- %d real observations ---\n", n);

    COAREOutput *ho_cpu = (COAREOutput*)malloc(n * sizeof(COAREOutput));
    COAREOutput *ho_gpu = (COAREOutput*)malloc(n * sizeof(COAREOutput));

    clock_t t0 = clock();
    cpu_coare(hi, ho_cpu, n);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    COAREInput *di; COAREOutput *do_d;
    cudaMalloc(&di, n * sizeof(COAREInput));
    cudaMalloc(&do_d, n * sizeof(COAREOutput));
    cudaMemcpy(di, hi, n * sizeof(COAREInput), cudaMemcpyHostToDevice);

    int thr = 256, blk = (n + thr - 1) / thr;
    kernel_coare<<<blk, thr>>>(di, do_d, n);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 20;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++)
        kernel_coare<<<blk, thr>>>(di, do_d, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    cudaMemcpy(ho_gpu, do_d, n * sizeof(COAREOutput), cudaMemcpyDeviceToHost);

    // CPU-vs-GPU agreement on real data (authoritative accuracy check;
    // same metrics and pass criteria as the original synthetic harness,
    // plus max absolute errors)
    float max_rel_tau = 0, max_rel_hsb = 0, max_rel_hlb = 0;
    float max_abs_tau = 0, max_abs_hsb = 0, max_abs_hlb = 0;
    int nan_c = 0;
    for (int c = 0; c < n; c++) {
        if (isnan(ho_gpu[c].tau) || isnan(ho_gpu[c].hsb) || isnan(ho_gpu[c].hlb)) { nan_c++; continue; }
        max_abs_tau = fmaxf(max_abs_tau, fabsf(ho_gpu[c].tau - ho_cpu[c].tau));
        max_abs_hsb = fmaxf(max_abs_hsb, fabsf(ho_gpu[c].hsb - ho_cpu[c].hsb));
        max_abs_hlb = fmaxf(max_abs_hlb, fabsf(ho_gpu[c].hlb - ho_cpu[c].hlb));
        if (fabsf(ho_cpu[c].tau) > 0.001f)
            max_rel_tau = fmaxf(max_rel_tau, fabsf(ho_gpu[c].tau - ho_cpu[c].tau) / fabsf(ho_cpu[c].tau));
        if (fabsf(ho_cpu[c].hsb) > 0.1f)
            max_rel_hsb = fmaxf(max_rel_hsb, fabsf(ho_gpu[c].hsb - ho_cpu[c].hsb) / fabsf(ho_cpu[c].hsb));
        if (fabsf(ho_cpu[c].hlb) > 0.1f)
            max_rel_hlb = fmaxf(max_rel_hlb, fabsf(ho_gpu[c].hlb - ho_cpu[c].hlb) / fabsf(ho_cpu[c].hlb));
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx (timing indicative; n is small)\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  CPU-vs-GPU max rel: tau=%.2e  hsb=%.2e  hlb=%.2e  NaN=%d\n", max_rel_tau, max_rel_hsb, max_rel_hlb, nan_c);
    printf("  CPU-vs-GPU max abs: tau=%.2e N/m2  hsb=%.2e W/m2  hlb=%.2e W/m2\n", max_abs_tau, max_abs_hsb, max_abs_hlb);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel_tau < 1e-3f && max_rel_hsb < 1e-3f && max_rel_hlb < 1e-3f) ? "PASS" :
           (nan_c == 0 && max_rel_tau < 1e-1f) ? "PASS (fast math)" : "NEEDS REVIEW");

    // Comparison against the official expected outputs (gold standard).
    // NOTE: the official outputs include cool-skin + warm-layer physics,
    // wave-free zo/zot/zoq parameterizations of the full COARE 3.6, and the
    // Webb correction context — the kernel implements a simplified subset
    // (bulk ts, constant zi=600 gustiness, zot=zoq), so systematic physical
    // differences are expected. Reported for reference, not as pass/fail.
    report_vs_gold("CPU reference", ho_cpu, gold, n);
    printf("\n");
    report_vs_gold("GPU kernel   ", ho_gpu, gold, n);
    printf("\n");

    // Indicative throughput on a replicated batch (real records tiled 462x
    // to ~1M points — replication documented, used for TIMING ONLY; all
    // accuracy numbers above come from the 2165 unreplicated real records).
    {
        int rep = 462;
        int nbig = n * rep;
        COAREInput* hbig = (COAREInput*)malloc((size_t)nbig * sizeof(COAREInput));
        for (int r = 0; r < rep; r++)
            memcpy(hbig + (size_t)r * n, hi, (size_t)n * sizeof(COAREInput));
        COAREInput* dbig; COAREOutput* dobig;
        cudaMalloc(&dbig, (size_t)nbig * sizeof(COAREInput));
        cudaMalloc(&dobig, (size_t)nbig * sizeof(COAREOutput));
        cudaMemcpy(dbig, hbig, (size_t)nbig * sizeof(COAREInput), cudaMemcpyHostToDevice);
        int blk2 = (nbig + thr - 1) / thr;
        kernel_coare<<<blk2, thr>>>(dbig, dobig, nbig);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < runs; r++)
            kernel_coare<<<blk2, thr>>>(dbig, dobig, nbig);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float big_ms; cudaEventElapsedTime(&big_ms, e0, e1); big_ms /= runs;
        printf("  Indicative GPU timing on %d points (real data tiled x%d): %.3f ms (%.1f Mpts/s)\n",
               nbig, rep, big_ms, nbig / big_ms / 1000.0f);
        cudaFree(dbig); cudaFree(dobig); free(hbig);
    }

    free(hi); free(ho_cpu); free(ho_gpu); free(gold);
    cudaFree(di); cudaFree(do_d);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    return 0;
}
