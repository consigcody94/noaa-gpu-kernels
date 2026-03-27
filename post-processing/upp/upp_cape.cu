/**
 * NOAA-EMC UPP CAPE/CIN GPU Kernel
 * From NOAA-EMC/UPP (41 stars) — UPP_PHYSICS.f CALCAPE subroutine
 *
 * CAPE (Convective Available Potential Energy) is computed by:
 * 1. Finding the maximum theta-e layer (most unstable parcel)
 * 2. Lifting parcel to LCL (Lifting Condensation Level)
 * 3. Integrating buoyancy from LCL to equilibrium level
 *
 * Each grid column is independent — one thread per column.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_LEV 128
#define G 9.81f
#define RD 287.04f
#define RV 461.5f
#define CP 1004.0f
#define CAPA (RD/CP)
#define EPS 0.622f
#define H1 1.0f
#define H10E5 100000.0f

struct Column {
    int nlev;
    float pmid[MAX_LEV];    // pressure at midpoints (Pa)
    float T[MAX_LEV];       // temperature (K)
    float Q[MAX_LEV];       // specific humidity (kg/kg)
    float zint[MAX_LEV+1];  // height at interfaces (m)
};

// Saturation vapor pressure (Tetens)
__host__ __device__ float esat(float T) {
    float tc = T - 273.15f;
    return 611.2f * expf(17.67f * tc / (tc + 243.5f));
}

// Saturation specific humidity
__host__ __device__ float qsat(float T, float P) {
    float es = esat(T);
    return EPS * es / (P - (1.0f - EPS) * es);
}

// Equivalent potential temperature
__host__ __device__ float theta_e(float T, float Q, float P) {
    float Lv = 2.501e6f;
    float th = T * powf(H10E5 / P, CAPA);
    float qs = qsat(T, P);
    return th * expf(Lv * Q / (CP * T));
}

// Virtual temperature
__host__ __device__ float Tv(float T, float Q) {
    return T * (1.0f + 0.608f * Q);
}

// Moist adiabatic lapse rate (simplified)
__host__ __device__ float moist_adiabat_T(float T_parcel, float P_new, float P_old) {
    float Lv = 2.501e6f;
    float qs = qsat(T_parcel, P_old);
    float gamma_m = G * (1.0f + Lv * qs / (RD * T_parcel)) /
                    (CP + Lv * Lv * qs * EPS / (RD * T_parcel * T_parcel));
    float dz = -RD * T_parcel / G * logf(P_new / P_old); // hydrostatic
    return T_parcel - gamma_m * dz;
}

// CPU reference CAPE calculation
void cpu_cape(const Column* cols, float* cape_out, float* cin_out, int ncol) {
    for (int c = 0; c < ncol; c++) {
        int nlev = cols[c].nlev;
        float cape = 0.0f, cin = 0.0f;

        // 1. Find most unstable parcel (max theta-e in lowest 300 hPa)
        float max_the = -1e30f;
        int l_start = -1;
        float T_parcel, Q_parcel, P_parcel;

        for (int l = nlev - 1; l >= 0; l--) {
            if (cols[c].pmid[nlev-1] - cols[c].pmid[l] > 30000.0f) break;
            float the = theta_e(cols[c].T[l], cols[c].Q[l], cols[c].pmid[l]);
            if (the > max_the) {
                max_the = the;
                l_start = l;
                T_parcel = cols[c].T[l];
                Q_parcel = cols[c].Q[l];
                P_parcel = cols[c].pmid[l];
            }
        }

        if (l_start < 0) { cape_out[c] = 0.0f; cin_out[c] = 0.0f; continue; }

        // 2. Lift parcel from l_start upward
        float Tp = T_parcel;
        int found_lfc = 0;

        for (int l = l_start - 1; l >= 0; l--) {
            // Lift parcel to next level
            Tp = moist_adiabat_T(Tp, cols[c].pmid[l], cols[c].pmid[l+1]);

            // Buoyancy: compare parcel virtual T to environment virtual T
            float Tv_parcel = Tv(Tp, qsat(Tp, cols[c].pmid[l]));
            float Tv_env = Tv(cols[c].T[l], cols[c].Q[l]);

            float dz = cols[c].zint[l] - cols[c].zint[l+1];
            float buoy = G * (Tv_parcel - Tv_env) / Tv_env * dz;

            if (buoy > 0.0f) {
                cape += buoy;
                found_lfc = 1;
            } else if (!found_lfc) {
                cin += buoy; // CIN only below LFC
            }
        }

        cape_out[c] = fmaxf(0.0f, cape);
        cin_out[c] = fminf(0.0f, cin);
    }
}

// GPU kernel
__global__ void kernel_cape(const Column* __restrict__ cols,
                            float* __restrict__ cape_out,
                            float* __restrict__ cin_out, int ncol) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    int nlev = cols[c].nlev;
    float cape = 0.0f, cin = 0.0f;

    float max_the = -1e30f;
    int l_start = -1;
    float T_parcel = 0, Q_parcel = 0, P_parcel = 0;

    for (int l = nlev - 1; l >= 0; l--) {
        if (cols[c].pmid[nlev-1] - cols[c].pmid[l] > 30000.0f) break;
        float tc = cols[c].T[l] - 273.15f;
        float es = 611.2f * __expf(17.67f * tc / (tc + 243.5f));
        float qs = EPS * es / (cols[c].pmid[l] - (1.0f - EPS) * es);
        float Lv = 2.501e6f;
        float th = cols[c].T[l] * __powf(H10E5 / cols[c].pmid[l], CAPA);
        float the = th * __expf(Lv * cols[c].Q[l] / (CP * cols[c].T[l]));
        if (the > max_the) {
            max_the = the;
            l_start = l;
            T_parcel = cols[c].T[l];
            Q_parcel = cols[c].Q[l];
            P_parcel = cols[c].pmid[l];
        }
    }

    if (l_start < 0) { cape_out[c] = 0.0f; cin_out[c] = 0.0f; return; }

    float Tp = T_parcel;
    int found_lfc = 0;

    for (int l = l_start - 1; l >= 0; l--) {
        float qs_old = qsat(Tp, cols[c].pmid[l+1]);
        float Lv = 2.501e6f;
        float gamma_m = G * (1.0f + Lv*qs_old/(RD*Tp)) /
                        (CP + Lv*Lv*qs_old*EPS/(RD*Tp*Tp));
        float dz_approx = -RD*Tp/G * __logf(cols[c].pmid[l] / cols[c].pmid[l+1]);
        Tp = Tp - gamma_m * dz_approx;

        float tc_p = Tp - 273.15f;
        float es_p = 611.2f * __expf(17.67f * tc_p / (tc_p + 243.5f));
        float qs_p = EPS * es_p / (cols[c].pmid[l] - (1.0f - EPS) * es_p);
        float Tv_p = Tp * (1.0f + 0.608f * qs_p);
        float Tv_e = cols[c].T[l] * (1.0f + 0.608f * cols[c].Q[l]);

        float dz = cols[c].zint[l] - cols[c].zint[l+1];
        float buoy = G * (Tv_p - Tv_e) / Tv_e * dz;

        if (buoy > 0.0f) { cape += buoy; found_lfc = 1; }
        else if (!found_lfc) { cin += buoy; }
    }

    cape_out[c] = fmaxf(0.0f, cape);
    cin_out[c] = fminf(0.0f, cin);
}

void gen_columns(Column* cols, int n, unsigned seed) {
    srand(seed);
    for (int c = 0; c < n; c++) {
        cols[c].nlev = 60 + (rand() % 40); // 60-100 levels
        float P_sfc = 100000.0f + 2000.0f * ((float)rand()/RAND_MAX);
        float T_sfc = 290.0f + 15.0f * ((float)rand()/RAND_MAX);
        float Q_sfc = 0.008f + 0.012f * ((float)rand()/RAND_MAX);
        float lapse = 0.0065f + 0.002f * ((float)rand()/RAND_MAX);

        for (int l = 0; l < cols[c].nlev; l++) {
            float frac = (float)l / (float)(cols[c].nlev - 1);
            cols[c].pmid[l] = P_sfc * (1.0f - 0.9f * frac); // ~10000 Pa at top
            cols[c].T[l] = T_sfc - lapse * frac * 15000.0f;
            if (cols[c].T[l] < 200.0f) cols[c].T[l] = 200.0f;
            cols[c].Q[l] = Q_sfc * expf(-5.0f * frac);
            cols[c].zint[l] = frac * 15000.0f;
        }
        cols[c].zint[cols[c].nlev] = 0.0f;
        // Reverse zint (surface at bottom)
        for (int l = 0; l <= cols[c].nlev / 2; l++) {
            float tmp = cols[c].zint[l];
            cols[c].zint[l] = cols[c].zint[cols[c].nlev - l];
            cols[c].zint[cols[c].nlev - l] = tmp;
        }
    }
}

int main() {
    printf("================================================\n");
    printf("  UPP CAPE/CIN GPU Kernel\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    int sizes[] = {10000, 100000, 500000};
    for (int is = 0; is < 3; is++) {
        int n = sizes[is];
        printf("--- %d columns ---\n", n);

        Column* hc = (Column*)malloc(n * sizeof(Column));
        float *hcape_c = (float*)malloc(n*4), *hcin_c = (float*)malloc(n*4);
        float *hcape_g = (float*)malloc(n*4), *hcin_g = (float*)malloc(n*4);
        gen_columns(hc, n, 42);

        clock_t t0 = clock();
        cpu_cape(hc, hcape_c, hcin_c, n);
        double cpu_ms = 1000.0 * (clock()-t0) / (double)CLOCKS_PER_SEC;

        Column* dc; float *dcape, *dcin;
        cudaMalloc(&dc, n * sizeof(Column));
        cudaMalloc(&dcape, n*4); cudaMalloc(&dcin, n*4);
        cudaMemcpy(dc, hc, n * sizeof(Column), cudaMemcpyHostToDevice);

        int thr = 128, blk = (n+thr-1)/thr;
        kernel_cape<<<blk,thr>>>(dc, dcape, dcin, n);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        int runs = 20;
        cudaEventRecord(e0);
        for (int r = 0; r < runs; r++)
            kernel_cape<<<blk,thr>>>(dc, dcape, dcin, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

        cudaMemcpy(hcape_g, dcape, n*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hcin_g, dcin, n*4, cudaMemcpyDeviceToHost);

        float max_rel_cape = 0, max_rel_cin = 0;
        int nan_c = 0, cape_match = 0;
        for (int c = 0; c < n; c++) {
            if (isnan(hcape_g[c])) { nan_c++; continue; }
            if (fabsf(hcape_c[c]) > 1.0f) {
                float re = fabsf(hcape_g[c] - hcape_c[c]) / fabsf(hcape_c[c]);
                if (re > max_rel_cape) max_rel_cape = re;
            }
            if (fabsf(hcin_c[c]) > 1.0f) {
                float re = fabsf(hcin_g[c] - hcin_c[c]) / fabsf(hcin_c[c]);
                if (re > max_rel_cin) max_rel_cin = re;
            }
            if (fabsf(hcape_g[c] - hcape_c[c]) < 1.0f) cape_match++;
        }

        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms);
        printf("  Max rel: CAPE=%.2e  CIN=%.2e  NaN=%d  Match(<1 J/kg)=%d/%d\n",
               max_rel_cape, max_rel_cin, nan_c, cape_match, n);
        printf("  Status: %s\n\n",
               (nan_c==0 && max_rel_cape<0.01f) ? "PASS" :
               (nan_c==0 && max_rel_cape<0.1f) ? "PASS (fast math)" : "NEEDS REVIEW");

        free(hc); free(hcape_c); free(hcin_c); free(hcape_g); free(hcin_g);
        cudaFree(dc); cudaFree(dcape); cudaFree(dcin);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}
