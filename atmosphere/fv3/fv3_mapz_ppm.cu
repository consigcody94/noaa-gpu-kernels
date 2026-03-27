/**
 * FV3 Vertical Remapping (map1_ppm) GPU Kernel
 * From NOAA-GFDL/GFDL_atmos_cubed_sphere model/fv_mapz.F90
 *
 * PPM (piecewise parabolic method) vertical remapping:
 * Given a field on old pressure levels, remap to new pressure levels.
 * Each column is independent — perfect for GPU batching.
 *
 * This is NOT the full fv_mapz (which has backward sweep dependencies).
 * This is just the map1_ppm column remapping which is the inner kernel.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_KM 128

// PPM reconstruction for one column
// Given cell averages q[0..km-1] and cell edges dp[0..km-1],
// compute subcell parabolic reconstruction and remap to new grid
void cpu_map1_ppm(const float* q, const float* dp_old, const float* dp_new,
                   float* q_new, int km) {
    // Step 1: Compute interface values using PPM
    float al[MAX_KM+1];  // interface values
    float ar[MAX_KM];     // right edge of parabola
    float a6[MAX_KM];     // curvature coefficient

    // Interface values (4th-order centered)
    al[0] = q[0];
    al[km] = q[km-1];
    for (int k = 1; k < km; k++) {
        float dm_km1 = 0.25f * (q[k] - q[k-1]);  // simplified slope
        float dm_k = (k < km-1) ? 0.25f * (q[k+1] - q[k]) : 0.0f;
        al[k] = 0.5f * (q[k-1] + q[k]) + (dm_km1 - dm_k) / 3.0f;
    }

    // Parabola coefficients
    for (int k = 0; k < km; k++) {
        ar[k] = al[k+1];
        float da = ar[k] - al[k];
        a6[k] = 6.0f * (q[k] - 0.5f * (al[k] + ar[k]));
    }

    // Step 2: Remap — integrate parabola over new grid cells
    // Accumulate old grid edges
    float pe_old[MAX_KM+1], pe_new[MAX_KM+1];
    pe_old[0] = 0.0f;
    pe_new[0] = 0.0f;
    for (int k = 0; k < km; k++) {
        pe_old[k+1] = pe_old[k] + dp_old[k];
        pe_new[k+1] = pe_new[k] + dp_new[k];
    }

    // For each new cell, find overlapping old cells and integrate
    int k0 = 0;
    for (int kn = 0; kn < km; kn++) {
        float pn_lo = pe_new[kn];
        float pn_hi = pe_new[kn+1];
        float dp_n = pn_hi - pn_lo;
        if (dp_n < 1e-20f) { q_new[kn] = q[0]; continue; }

        float sum = 0.0f;

        // Find first overlapping old cell
        while (k0 < km - 1 && pe_old[k0+1] <= pn_lo) k0++;

        for (int ko = k0; ko < km; ko++) {
            float po_lo = pe_old[ko];
            float po_hi = pe_old[ko+1];
            float dp_o = po_hi - po_lo;

            // Overlap region
            float lo = fmaxf(pn_lo, po_lo);
            float hi = fminf(pn_hi, po_hi);
            if (hi <= lo) break;

            // Integrate parabola over [lo, hi] within old cell ko
            if (dp_o > 1e-20f) {
                float x0 = (lo - po_lo) / dp_o;
                float x1 = (hi - po_lo) / dp_o;
                float dx = x1 - x0;
                // Integral of al + (ar-al)*x + a6*x*(1-x) from x0 to x1
                float avg = al[ko] + 0.5f * (ar[ko] - al[ko]) * (x0 + x1)
                           + a6[ko] * (0.5f * (x0 + x1) - (x0*x0 + x0*x1 + x1*x1) / 3.0f);
                sum += avg * (hi - lo);
            }
        }

        q_new[kn] = sum / dp_n;
    }
}

__global__ void kernel_map1_ppm(const float* __restrict__ q,
                                 const float* __restrict__ dp_old,
                                 const float* __restrict__ dp_new,
                                 float* __restrict__ q_new,
                                 int ncol, int km) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    const float* my_q = q + col * km;
    const float* my_dp_old = dp_old + col * km;
    const float* my_dp_new = dp_new + col * km;
    float* my_q_new = q_new + col * km;

    // Step 1: Interface values
    float al[MAX_KM+1], ar_v[MAX_KM], a6_v[MAX_KM];

    al[0] = my_q[0];
    al[km] = my_q[km-1];
    for (int k = 1; k < km; k++) {
        float dm_km1 = 0.25f * (my_q[k] - my_q[k-1]);
        float dm_k = (k < km-1) ? 0.25f * (my_q[k+1] - my_q[k]) : 0.0f;
        al[k] = 0.5f * (my_q[k-1] + my_q[k]) + (dm_km1 - dm_k) / 3.0f;
    }

    for (int k = 0; k < km; k++) {
        ar_v[k] = al[k+1];
        a6_v[k] = 6.0f * (my_q[k] - 0.5f * (al[k] + ar_v[k]));
    }

    // Step 2: Accumulate edges
    float pe_old[MAX_KM+1], pe_new[MAX_KM+1];
    pe_old[0] = 0.0f; pe_new[0] = 0.0f;
    for (int k = 0; k < km; k++) {
        pe_old[k+1] = pe_old[k] + my_dp_old[k];
        pe_new[k+1] = pe_new[k] + my_dp_new[k];
    }

    // Step 3: Remap
    int k0 = 0;
    for (int kn = 0; kn < km; kn++) {
        float pn_lo = pe_new[kn];
        float pn_hi = pe_new[kn+1];
        float dp_n = pn_hi - pn_lo;
        if (dp_n < 1e-20f) { my_q_new[kn] = my_q[0]; continue; }

        float sum = 0.0f;
        while (k0 < km - 1 && pe_old[k0+1] <= pn_lo) k0++;

        for (int ko = k0; ko < km; ko++) {
            float po_lo = pe_old[ko];
            float po_hi = pe_old[ko+1];
            float dp_o = po_hi - po_lo;
            float lo = fmaxf(pn_lo, po_lo);
            float hi = fminf(pn_hi, po_hi);
            if (hi <= lo) break;

            if (dp_o > 1e-20f) {
                float x0 = (lo - po_lo) / dp_o;
                float x1 = (hi - po_lo) / dp_o;
                float avg = al[ko] + 0.5f * (ar_v[ko] - al[ko]) * (x0 + x1)
                           + a6_v[ko] * (0.5f * (x0 + x1) - (x0*x0 + x0*x1 + x1*x1) / 3.0f);
                sum += avg * (hi - lo);
            }
        }

        my_q_new[kn] = sum / dp_n;
    }
}

int main() {
    printf("================================================\n");
    printf("  FV3 Vertical Remapping (map1_ppm) GPU Kernel\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    int km_vals[] = {64, 91, 127};
    int ncol_vals[] = {100000, 500000, 1000000};

    for (int ik = 0; ik < 3; ik++) {
        int km = km_vals[ik];
        for (int in = 0; in < 3; in++) {
            int ncol = ncol_vals[in];
            size_t sz = (size_t)ncol * km;

            // Check memory
            if (sz * 4 * 4 > 10ULL * 1024 * 1024 * 1024) {
                printf("--- km=%d, ncol=%d: SKIP (>10GB) ---\n\n", km, ncol);
                continue;
            }

            printf("--- km=%d, ncol=%d ---\n", km, ncol);

            float *hq = (float*)malloc(sz*4);
            float *hdp_old = (float*)malloc(sz*4);
            float *hdp_new = (float*)malloc(sz*4);
            float *hq_cpu = (float*)malloc(sz*4);
            float *hq_gpu = (float*)malloc(sz*4);

            srand(42);
            for (size_t i = 0; i < sz; i++) {
                hq[i] = 200.0f + 100.0f * ((float)rand()/RAND_MAX);
                hdp_old[i] = 50.0f + 50.0f * ((float)rand()/RAND_MAX);
                hdp_new[i] = hdp_old[i] * (0.95f + 0.1f * ((float)rand()/RAND_MAX));
            }

            // CPU
            clock_t t0 = clock();
            for (int c = 0; c < ncol; c++)
                cpu_map1_ppm(hq + c*km, hdp_old + c*km, hdp_new + c*km, hq_cpu + c*km, km);
            double cpu_ms = 1000.0 * (clock()-t0) / (double)CLOCKS_PER_SEC;

            // GPU
            float *dq, *ddp_old, *ddp_new, *dq_new;
            cudaMalloc(&dq, sz*4); cudaMalloc(&ddp_old, sz*4);
            cudaMalloc(&ddp_new, sz*4); cudaMalloc(&dq_new, sz*4);
            cudaMemcpy(dq, hq, sz*4, cudaMemcpyHostToDevice);
            cudaMemcpy(ddp_old, hdp_old, sz*4, cudaMemcpyHostToDevice);
            cudaMemcpy(ddp_new, hdp_new, sz*4, cudaMemcpyHostToDevice);

            int thr = 64, blk = (ncol+thr-1)/thr;
            kernel_map1_ppm<<<blk,thr>>>(dq, ddp_old, ddp_new, dq_new, ncol, km);
            cudaDeviceSynchronize();

            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            int runs = 5;
            cudaEventRecord(e0);
            for (int r = 0; r < runs; r++)
                kernel_map1_ppm<<<blk,thr>>>(dq, ddp_old, ddp_new, dq_new, ncol, km);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

            cudaMemcpy(hq_gpu, dq_new, sz*4, cudaMemcpyDeviceToHost);

            float max_rel = 0; int nan_c = 0;
            for (size_t i = 0; i < sz; i++) {
                if (isnan(hq_gpu[i])) { nan_c++; continue; }
                if (fabsf(hq_cpu[i]) > 0.01f) {
                    float re = fabsf(hq_gpu[i] - hq_cpu[i]) / fabsf(hq_cpu[i]);
                    if (re > max_rel) max_rel = re;
                }
            }

            printf("  CPU: %.0f ms | GPU: %.2f ms | Speedup: %.1fx\n",
                   cpu_ms, gpu_ms, cpu_ms/gpu_ms);
            printf("  Max rel: %.2e  NaN: %d\n", max_rel, nan_c);
            printf("  Status: %s\n\n",
                   (nan_c == 0 && max_rel < 1e-5f) ? "PASS" :
                   (nan_c == 0 && max_rel < 1e-3f) ? "PASS (FP32)" : "NEEDS REVIEW");

            free(hq); free(hdp_old); free(hdp_new); free(hq_cpu); free(hq_gpu);
            cudaFree(dq); cudaFree(ddp_old); cudaFree(ddp_new); cudaFree(dq_new);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    }

    return 0;
}
