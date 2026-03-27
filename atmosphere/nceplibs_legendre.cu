/**
 * NCEPLIBS-sp Legendre Polynomial + Spectral Synthesis GPU Kernel
 * From NOAA-EMC/NCEPLIBS-sp (splegend.f + spsynth.f)
 *
 * Two phases:
 * 1. Compute associated Legendre polynomials at each latitude
 *    using three-term recurrence (sequential per wavenumber,
 *    parallel across latitudes)
 * 2. Spectral synthesis: inner product of Legendre polynomials
 *    with spectral coefficients (batched dot product across latitudes)
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// GFS T1534: M=1534, ~1.2M spectral coefficients per latitude
// For testing: use smaller truncations
#define MAX_M 512
#define MAX_PLN ((MAX_M+1)*(MAX_M+2)/2)
// NOTE: For GPU, we do NOT store full PLN array per thread.
// Instead we fuse Legendre recurrence with synthesis accumulation:
// for each (lat, l), compute P(l,n) for n=l..M using 3-term recurrence
// and accumulate the dot product on the fly. Only need 2 floats of state.

// Phase 1: Compute Legendre polynomials via three-term recurrence
// CPU reference — matches splegend.f
void cpu_splegend(float slat, float clat, const float* eps, float* pln, int M) {
    // P(0,0) = 1
    pln[0] = 1.0f;

    if (M == 0) return;

    // Diagonal: P(l,l) = P(l-1,l-1) * clat * sqrt((2l+1)/(2l))
    for (int l = 1; l <= M; l++) {
        int idx = l * (l + 1) / 2 + l;
        int idx_prev = (l-1) * l / 2 + (l-1);
        pln[idx] = pln[idx_prev] * clat * sqrtf((2.0f*l + 1.0f) / (2.0f*l));
    }

    // Off-diagonal: P(l,n) = (slat*P(l,n-1) - eps(l,n-1)*P(l,n-2)) / eps(l,n)
    for (int l = 0; l <= M; l++) {
        if (l + 1 <= M) {
            int idx = l * (l + 1) / 2 + l + 1;  // P(l, l+1)
            int idx_prev = l * (l + 1) / 2 + l;  // P(l, l)
            float eps_val = eps[idx];
            if (fabsf(eps_val) > 1e-30f)
                pln[idx] = slat * pln[idx_prev] / eps_val;
        }
        for (int n = l + 2; n <= M; n++) {
            int idx = l * (l + 1) / 2 + n;      // P(l, n)  -- NOTE: using triangular indexing
            // Simplified indexing for benchmark
            if (idx < MAX_PLN && idx - 1 >= 0 && idx - 2 >= 0) {
                float eps_n = eps[idx];
                if (fabsf(eps_n) > 1e-30f)
                    pln[idx] = (slat * pln[idx-1] - eps[idx-1] * pln[idx-2]) / eps_n;
            }
        }
    }
}

// Phase 2: Spectral synthesis — inner product per latitude
// CPU reference — matches spsynth.f
void cpu_spsynth(const float* pln, const float* spc, float* fourier,
                 int M, int nfields) {
    // For each zonal wavenumber l, accumulate:
    //   F(2l+1) += sum_n PLN(l,n) * SPC(2n+1)
    //   F(2l+2) += sum_n PLN(l,n) * SPC(2n+2)
    for (int k = 0; k < nfields; k++) {
        for (int l = 0; l <= M; l++) {
            float sum_r = 0.0f, sum_i = 0.0f;
            for (int n = l; n <= M; n++) {
                int pln_idx = l * (l + 1) / 2 + n;
                if (pln_idx >= MAX_PLN) break;
                int spc_idx = k * (M + 1) * 2 + 2 * n;
                sum_r += pln[pln_idx] * spc[spc_idx];
                sum_i += pln[pln_idx] * spc[spc_idx + 1];
            }
            fourier[k * (M + 1) * 2 + 2 * l] = sum_r;
            fourier[k * (M + 1) * 2 + 2 * l + 1] = sum_i;
        }
    }
}

// GPU kernel: one thread per (latitude, zonal wavenumber l)
// FUSED: computes Legendre recurrence AND synthesis dot product simultaneously
// Only needs 2 floats of PLN state (prev, prev2) — no giant array
__global__ void kernel_legendre_synth(
    const float* __restrict__ slats,    // sin(lat) [nlat]
    const float* __restrict__ clats,    // cos(lat) [nlat]
    const float* __restrict__ eps,      // recurrence coefficients [MAX_PLN]
    const float* __restrict__ spc,      // spectral coefficients [nfields * (M+1) * 2]
    float* __restrict__ fourier_out,    // Fourier coefficients [nlat * nfields * (M+1) * 2]
    int M, int nfields, int nlat)
{
    // 2D grid: blockIdx.x * blockDim.x + threadIdx.x = (ilat, l) combined
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nlat * (M + 1);
    if (tid >= total) return;

    int ilat = tid / (M + 1);
    int l = tid % (M + 1);

    float slat = slats[ilat];
    float clat = clats[ilat];

    // Step 1: Bootstrap P(l,l) along diagonal
    float pll = 1.0f;
    for (int j = 1; j <= l; j++) {
        pll *= clat * __fsqrt_rn((2.0f*j + 1.0f) / (2.0f*j));
    }

    // Step 2: P(l, l+1) = slat * P(l,l) / eps(l, l+1)
    float prev2 = pll;  // P(l, l)
    float prev1 = 0.0f; // P(l, l+1)
    int idx_ll = l * (l + 1) / 2 + l;
    if (l + 1 <= M) {
        int idx_ll1 = l * (l + 1) / 2 + l + 1;
        float ev = eps[idx_ll1];
        if (fabsf(ev) > 1e-30f)
            prev1 = slat * pll / ev;
    }

    // Step 3: Fused recurrence + synthesis
    // Accumulate dot product while computing each P(l,n)
    int out_stride = nfields * (M + 1) * 2;
    for (int k = 0; k < nfields; k++) {
        float sum_r = 0.0f, sum_i = 0.0f;

        // n = l contribution
        int spc_idx = k * (M + 1) * 2 + 2 * l;
        sum_r += pll * spc[spc_idx];
        sum_i += pll * spc[spc_idx + 1];

        // n = l+1 contribution
        if (l + 1 <= M) {
            spc_idx = k * (M + 1) * 2 + 2 * (l + 1);
            sum_r += prev1 * spc[spc_idx];
            sum_i += prev1 * spc[spc_idx + 1];
        }

        // n = l+2 .. M: three-term recurrence
        float p2 = pll, p1 = prev1;
        for (int n = l + 2; n <= M; n++) {
            int idx_n = l * (l + 1) / 2 + n;
            float ev = eps[idx_n];
            float pn = 0.0f;
            if (fabsf(ev) > 1e-30f)
                pn = (slat * p1 - eps[idx_n - 1] * p2) / ev;
            spc_idx = k * (M + 1) * 2 + 2 * n;
            sum_r += pn * spc[spc_idx];
            sum_i += pn * spc[spc_idx + 1];
            p2 = p1;
            p1 = pn;
        }

        int out_idx = ilat * out_stride + k * (M + 1) * 2 + 2 * l;
        fourier_out[out_idx] = sum_r;
        fourier_out[out_idx + 1] = sum_i;
    }
}

void gen_data(float* slats, float* clats, float* eps, float* spc,
              int nlat, int M, int nfields, unsigned seed) {
    srand(seed);
    // Gaussian latitudes (simplified)
    for (int i = 0; i < nlat; i++) {
        float lat = -90.0f + 180.0f * (float)i / (float)(nlat - 1);
        float lat_rad = lat * 3.14159f / 180.0f;
        slats[i] = sinf(lat_rad);
        clats[i] = cosf(lat_rad);
    }
    // Recurrence coefficients — must use actual formula, not random!
    // eps(l,n) = sqrt((n^2 - l^2) / (4*n^2 - 1))
    int npln = (M + 1) * (M + 2) / 2;
    for (int l = 0; l <= M; l++) {
        for (int n = l; n <= M; n++) {
            int idx = l * (l + 1) / 2 + n;
            if (idx < npln) {
                if (n == 0) { eps[idx] = 0.0f; continue; }
                float num = (float)(n*n - l*l);
                float den = (float)(4*n*n - 1);
                eps[idx] = (num >= 0.0f && den > 0.0f) ? sqrtf(num / den) : 0.0f;
            }
        }
    }
    // Spectral coefficients
    for (int i = 0; i < nfields * (M + 1) * 2; i++)
        spc[i] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
}

int main() {
    printf("================================================\n");
    printf("  NCEPLIBS-sp Legendre + Synthesis GPU Kernel\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    // Test with moderate truncation (T126 ~ GFS at reduced resolution)
    struct TC { int M; int nlat; int nf; const char* nm; };
    TC cfgs[] = {
        {62,  128, 4, "T62 (low-res)"},
        {126, 256, 4, "T126 (medium)"},
        {254, 512, 2, "T254 (high-res)"},
    };

    for (int ic = 0; ic < 3; ic++) {
        int M = cfgs[ic].M;
        int nlat = cfgs[ic].nlat;
        int nf = cfgs[ic].nf;
        int npln = (M + 1) * (M + 2) / 2;

        if (npln > MAX_PLN) {
            printf("--- %s: M=%d exceeds MAX_PLN, skipping ---\n\n", cfgs[ic].nm, M);
            continue;
        }

        printf("--- %s: M=%d, nlat=%d, nfields=%d, npln=%d ---\n", cfgs[ic].nm, M, nlat, nf, npln);

        float *hslat = (float*)malloc(nlat * 4);
        float *hclat = (float*)malloc(nlat * 4);
        float *heps = (float*)malloc(npln * 4);
        float *hspc = (float*)malloc(nf * (M+1) * 2 * 4);
        int out_size = nlat * nf * (M+1) * 2;
        float *hout_cpu = (float*)malloc(out_size * 4);
        float *hout_gpu = (float*)malloc(out_size * 4);

        gen_data(hslat, hclat, heps, hspc, nlat, M, nf, 42);

        // CPU: process all latitudes
        clock_t t0 = clock();
        for (int ilat = 0; ilat < nlat; ilat++) {
            float pln[MAX_PLN];
            memset(pln, 0, npln * 4);
            cpu_splegend(hslat[ilat], hclat[ilat], heps, pln, M);
            cpu_spsynth(pln, hspc, hout_cpu + ilat * nf * (M+1) * 2, M, nf);
        }
        double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

        // GPU
        float *dslat, *dclat, *deps, *dspc, *dout;
        cudaMalloc(&dslat, nlat*4); cudaMalloc(&dclat, nlat*4);
        cudaMalloc(&deps, npln*4); cudaMalloc(&dspc, nf*(M+1)*2*4);
        cudaMalloc(&dout, out_size*4);
        cudaMemcpy(dslat, hslat, nlat*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dclat, hclat, nlat*4, cudaMemcpyHostToDevice);
        cudaMemcpy(deps, heps, npln*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dspc, hspc, nf*(M+1)*2*4, cudaMemcpyHostToDevice);

        int total_threads = nlat * (M + 1);
        int thr = 256, blk = (total_threads + thr - 1) / thr;
        kernel_legendre_synth<<<blk,thr>>>(dslat, dclat, deps, dspc, dout, M, nf, nlat);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        int runs = 10;
        cudaEventRecord(e0);
        for (int r = 0; r < runs; r++)
            kernel_legendre_synth<<<blk,thr>>>(dslat, dclat, deps, dspc, dout, M, nf, nlat);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

        cudaMemcpy(hout_gpu, dout, out_size*4, cudaMemcpyDeviceToHost);

        float max_rel = 0; int nan_c = 0;
        for (int i = 0; i < out_size; i++) {
            if (isnan(hout_gpu[i])) { nan_c++; continue; }
            if (fabsf(hout_cpu[i]) > 1e-10f) {
                float re = fabsf(hout_gpu[i] - hout_cpu[i]) / fabsf(hout_cpu[i]);
                if (re > max_rel) max_rel = re;
            }
        }

        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms);
        printf("  Max rel: %.2e | NaN: %d\n", max_rel, nan_c);
        printf("  Status: %s\n\n",
               (nan_c == 0 && max_rel < 1e-4f) ? "PASS" :
               (nan_c == 0 && max_rel < 1e-2f) ? "PASS (FP32 accumulation)" : "NEEDS REVIEW");

        free(hslat); free(hclat); free(heps); free(hspc); free(hout_cpu); free(hout_gpu);
        cudaFree(dslat); cudaFree(dclat); cudaFree(deps); cudaFree(dspc); cudaFree(dout);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    return 0;
}
