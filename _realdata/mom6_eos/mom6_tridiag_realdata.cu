/**
 * MOM6 triDiagTS vertical-mixing kernel — REAL DATA probe.
 *
 * Kernel math (OceanColumn, cpu_tridiag_ts, kernel_tridiag_ts) copied
 * VERBATIM from ocean/mom6/mom6_vortex.cu — DO NOT MODIFY.
 * Only the synthetic generator is replaced by a loader of real Argo columns.
 *
 * Input: data/columns.bin
 *   int32 ncol, then per column: int32 nz, float32 h[nz], T[nz], S[nz]
 *   h from real Argo pressure spacing (1 dbar ~ 1 m), T/S real QC=1/2 values.
 *   Source: Argo GDAC daily geo files 2026-05-15. See extract_argo.py.
 *
 * ea/eb (entrainment, m) are MODEL PARAMETERS, not observables. They are set
 * deterministically from a fixed diapycnal diffusivity K and timestep dt:
 *   ea[k] = K*dt / (0.5*(h[k-1]+h[k]))  for k>0,  ea[0]   = 0
 *   eb[k] = ea[k+1],                              eb[nz-1] = 0
 * Scenarios: K = 1e-5 (abyssal), 1e-4 (thermocline), 1e-3 m^2/s (strong),
 * dt = 3600 s; plus a constant ea=eb=1.0 m case matching the magnitude of the
 * original synthetic generator (which drew ea,eb ~ U[0.1, 2.1] m).
 * No random values are used anywhere.
 *
 * Columns are replicated (tiled) x250 -> ~101k columns for GPU timing;
 * accuracy is identical across replicas, reported on the raw 404 columns.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_OC_LEV 75  // MOM6 typical: 50-75 vertical levels

// ============================================================
// KERNEL: MOM6 triDiagTS (verbatim copy from ocean/mom6/mom6_vortex.cu)
// ============================================================

struct OceanColumn {
    int nz;           // number of vertical levels
    float h[MAX_OC_LEV];    // layer thickness (m)
    float ea[MAX_OC_LEV];   // entrainment from above (m)
    float eb[MAX_OC_LEV];   // entrainment from below (m)
    float T[MAX_OC_LEV];    // temperature (C)
    float S[MAX_OC_LEV];    // salinity (PSU)
};

void cpu_tridiag_ts(OceanColumn* cols, int ncol) {
    for (int c = 0; c < ncol; c++) {
        int nz = cols[c].nz;
        float b1, d1_T, d1_S;
        float c1[MAX_OC_LEV];

        // Forward sweep for T and S simultaneously
        float h_pre = cols[c].h[0] + cols[c].ea[0] + cols[c].eb[0];
        if (h_pre < 1e-10f) h_pre = 1e-10f;
        b1 = 1.0f / h_pre;
        d1_T = b1 * (cols[c].h[0] * cols[c].T[0]);
        d1_S = b1 * (cols[c].h[0] * cols[c].S[0]);
        c1[0] = cols[c].eb[0] * b1;

        for (int k = 1; k < nz; k++) {
            float h_k = cols[c].h[k] + cols[c].ea[k] + cols[c].eb[k];
            if (h_k < 1e-10f) h_k = 1e-10f;
            float a_k = cols[c].ea[k];  // sub-diagonal
            float bet = 1.0f / (h_k - a_k * c1[k-1]);
            c1[k] = cols[c].eb[k] * bet;
            d1_T = bet * (cols[c].h[k] * cols[c].T[k] + a_k * d1_T);
            d1_S = bet * (cols[c].h[k] * cols[c].S[k] + a_k * d1_S);
        }

        // Bottom level
        cols[c].T[nz-1] = d1_T;
        cols[c].S[nz-1] = d1_S;

        // Back substitution
        for (int k = nz - 2; k >= 0; k--) {
            // Need to re-do forward to get per-level d1 values
            // Simplified: just apply the standard Thomas back-sub
        }

        // Actually, let me implement the standard Thomas properly
        // with arrays for the intermediate values
        float fwd_T[MAX_OC_LEV], fwd_S[MAX_OC_LEV];
        float cu[MAX_OC_LEV];

        h_pre = cols[c].h[0] + cols[c].ea[0] + cols[c].eb[0];
        if (h_pre < 1e-10f) h_pre = 1e-10f;
        float bet = h_pre;
        fwd_T[0] = cols[c].h[0] * cols[c].T[0] / bet;
        fwd_S[0] = cols[c].h[0] * cols[c].S[0] / bet;
        cu[0] = cols[c].eb[0] / bet;

        for (int k = 1; k < nz; k++) {
            float h_k = cols[c].h[k] + cols[c].ea[k] + cols[c].eb[k];
            if (h_k < 1e-10f) h_k = 1e-10f;
            float a_k = cols[c].ea[k];
            bet = h_k - a_k * cu[k-1];
            if (fabsf(bet) < 1e-30f) bet = 1e-30f;
            cu[k] = cols[c].eb[k] / bet;
            fwd_T[k] = (cols[c].h[k] * cols[c].T[k] + a_k * fwd_T[k-1]) / bet;
            fwd_S[k] = (cols[c].h[k] * cols[c].S[k] + a_k * fwd_S[k-1]) / bet;
        }

        cols[c].T[nz-1] = fwd_T[nz-1];
        cols[c].S[nz-1] = fwd_S[nz-1];
        for (int k = nz-2; k >= 0; k--) {
            cols[c].T[k] = fwd_T[k] + cu[k] * cols[c].T[k+1];
            cols[c].S[k] = fwd_S[k] + cu[k] * cols[c].S[k+1];
        }
    }
}

__global__ void kernel_tridiag_ts(OceanColumn* __restrict__ cols, int ncol) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    int nz = cols[c].nz;
    float fwd_T[MAX_OC_LEV], fwd_S[MAX_OC_LEV], cu[MAX_OC_LEV];

    float h_pre = cols[c].h[0] + cols[c].ea[0] + cols[c].eb[0];
    if (h_pre < 1e-10f) h_pre = 1e-10f;
    float bet = h_pre;
    fwd_T[0] = cols[c].h[0] * cols[c].T[0] / bet;
    fwd_S[0] = cols[c].h[0] * cols[c].S[0] / bet;
    cu[0] = cols[c].eb[0] / bet;

    for (int k = 1; k < nz; k++) {
        float h_k = cols[c].h[k] + cols[c].ea[k] + cols[c].eb[k];
        if (h_k < 1e-10f) h_k = 1e-10f;
        float a_k = cols[c].ea[k];
        bet = h_k - a_k * cu[k-1];
        if (fabsf(bet) < 1e-30f) bet = 1e-30f;
        cu[k] = cols[c].eb[k] / bet;
        fwd_T[k] = (cols[c].h[k] * cols[c].T[k] + a_k * fwd_T[k-1]) / bet;
        fwd_S[k] = (cols[c].h[k] * cols[c].S[k] + a_k * fwd_S[k-1]) / bet;
    }

    cols[c].T[nz-1] = fwd_T[nz-1];
    cols[c].S[nz-1] = fwd_S[nz-1];
    for (int k = nz-2; k >= 0; k--) {
        cols[c].T[k] = fwd_T[k] + cu[k] * cols[c].T[k+1];
        cols[c].S[k] = fwd_S[k] + cu[k] * cols[c].S[k+1];
    }
}

// ============================================================
// Real-data loader + deterministic entrainment (replaces gen_ocean)
// ============================================================

static int load_columns(const char* path, OceanColumn** out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int ncol = 0;
    if (fread(&ncol, 4, 1, f) != 1 || ncol <= 0) { fprintf(stderr, "bad header\n"); exit(1); }
    OceanColumn* cols = (OceanColumn*)calloc(ncol, sizeof(OceanColumn));
    for (int c = 0; c < ncol; c++) {
        int nz = 0;
        if (fread(&nz, 4, 1, f) != 1 || nz < 2 || nz > MAX_OC_LEV) {
            fprintf(stderr, "bad nz=%d at col %d\n", nz, c); exit(1);
        }
        cols[c].nz = nz;
        if (fread(cols[c].h, 4, nz, f) != (size_t)nz ||
            fread(cols[c].T, 4, nz, f) != (size_t)nz ||
            fread(cols[c].S, 4, nz, f) != (size_t)nz) {
            fprintf(stderr, "short read col %d\n", c); exit(1);
        }
    }
    fclose(f);
    *out = cols;
    return ncol;
}

// ea/eb from fixed diffusivity K (m^2/s), dt (s); kconst<0 means use K,
// kconst>=0 means constant ea=eb=kconst (synthetic-magnitude scenario).
static void set_entrainment(OceanColumn* cols, int ncol, double K, double dt, double kconst) {
    for (int c = 0; c < ncol; c++) {
        int nz = cols[c].nz;
        if (kconst >= 0.0) {
            for (int k = 0; k < nz; k++) {
                cols[c].ea[k] = (float)kconst;
                cols[c].eb[k] = (float)kconst;
            }
            continue;
        }
        float e[MAX_OC_LEV + 1];
        e[0] = 0.0f;                       // no flux through surface
        for (int k = 1; k < nz; k++) {
            double dz = 0.5 * ((double)cols[c].h[k-1] + (double)cols[c].h[k]);
            e[k] = (float)(K * dt / dz);
        }
        e[nz] = 0.0f;                      // no flux through bottom
        for (int k = 0; k < nz; k++) {
            cols[c].ea[k] = e[k];
            cols[c].eb[k] = e[k+1];
        }
    }
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "data/columns.bin";

    printf("================================================\n");
    printf("  MOM6 triDiagTS — REAL ARGO STRATIFICATION\n");
    printf("  GDAC daily geo profiles 2026-05-15, QC 1/2\n");
    printf("================================================\n\n");

    OceanColumn* base;
    int nbase = load_columns(path, &base);
    printf("Loaded %d real Argo columns (nz %d..%d capped at %d)\n\n",
           nbase, 2, MAX_OC_LEV, MAX_OC_LEV);

    const int REP = 250;                  // documented replication for timing
    int n = nbase * REP;

    struct { const char* name; double K; double kconst; } scen[] = {
        {"K=1e-5 m2/s (abyssal), dt=3600s",     1e-5, -1.0},
        {"K=1e-4 m2/s (thermocline), dt=3600s", 1e-4, -1.0},
        {"K=1e-3 m2/s (strong), dt=3600s",      1e-3, -1.0},
        {"const ea=eb=1.0 m (synthetic-scale)",  0.0,  1.0},
    };

    for (int sc = 0; sc < 4; sc++) {
        printf("--- %d columns (%d real x %d) | %s ---\n", n, nbase, REP, scen[sc].name);

        OceanColumn *hc_cpu = (OceanColumn*)malloc((size_t)n * sizeof(OceanColumn));
        OceanColumn *hc_gpu = (OceanColumn*)malloc((size_t)n * sizeof(OceanColumn));
        for (int c = 0; c < n; c++) hc_cpu[c] = base[c % nbase];
        set_entrainment(hc_cpu, n, scen[sc].K, 3600.0, scen[sc].kconst);
        memcpy(hc_gpu, hc_cpu, (size_t)n * sizeof(OceanColumn));

        clock_t t0 = clock();
        cpu_tridiag_ts(hc_cpu, n);
        double cpu_ms = 1000.0 * (clock()-t0) / (double)CLOCKS_PER_SEC;

        OceanColumn *dc;
        cudaMalloc(&dc, (size_t)n * sizeof(OceanColumn));

        int thr = 64, blk = (n+thr-1)/thr;
        cudaMemcpy(dc, hc_gpu, (size_t)n * sizeof(OceanColumn), cudaMemcpyHostToDevice);
        kernel_tridiag_ts<<<blk,thr>>>(dc, n);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        int runs = 5;
        cudaEventRecord(e0);
        for (int r = 0; r < runs; r++) {
            cudaMemcpy(dc, hc_gpu, (size_t)n * sizeof(OceanColumn), cudaMemcpyHostToDevice);
            kernel_tridiag_ts<<<blk,thr>>>(dc, n);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

        cudaMemcpy(dc, hc_gpu, (size_t)n * sizeof(OceanColumn), cudaMemcpyHostToDevice);
        kernel_tridiag_ts<<<blk,thr>>>(dc, n);
        OceanColumn *hc_result = (OceanColumn*)malloc((size_t)nbase * sizeof(OceanColumn));
        cudaMemcpy(hc_result, dc, (size_t)nbase * sizeof(OceanColumn), cudaMemcpyDeviceToHost);

        // accuracy on the raw real columns (first replica) — same metric as original
        float max_rel_T = 0, max_rel_S = 0, max_abs_T = 0, max_abs_S = 0;
        int nan_c = 0, wc = -1, wk = -1;
        for (int c = 0; c < nbase; c++) {
            for (int k = 0; k < hc_cpu[c].nz; k++) {
                if (isnan(hc_result[c].T[k])) { nan_c++; continue; }
                float aT = fabsf(hc_result[c].T[k] - hc_cpu[c].T[k]);
                float aS = fabsf(hc_result[c].S[k] - hc_cpu[c].S[k]);
                if (aT > max_abs_T) max_abs_T = aT;
                if (aS > max_abs_S) max_abs_S = aS;
                if (fabsf(hc_cpu[c].T[k]) > 0.01f) {
                    float re = aT / fabsf(hc_cpu[c].T[k]);
                    if (re > max_rel_T) { max_rel_T = re; wc = c; wk = k; }
                }
                if (fabsf(hc_cpu[c].S[k]) > 0.01f) {
                    float re = aS / fabsf(hc_cpu[c].S[k]);
                    if (re > max_rel_S) max_rel_S = re;
                }
            }
        }

        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms);
        printf("  Max rel: T=%.2e  S=%.2e  NaN=%d | Max abs: T=%.2e C  S=%.2e PSU\n",
               max_rel_T, max_rel_S, nan_c, max_abs_T, max_abs_S);
        if (wc >= 0)
            printf("  worst T: col %d level %d/%d  cpu=%.5f gpu=%.5f\n",
                   wc, wk, hc_cpu[wc].nz, hc_cpu[wc].T[wk], hc_result[wc].T[wk]);
        printf("  Status: %s\n\n",
               (nan_c==0 && max_rel_T<1e-4f && max_rel_S<1e-4f) ? "PASS" :
               (nan_c==0 && max_rel_T<1e-2f) ? "PASS (FP32)" : "NEEDS REVIEW");

        free(hc_cpu); free(hc_gpu); free(hc_result);
        cudaFree(dc);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    free(base);
    return 0;
}
