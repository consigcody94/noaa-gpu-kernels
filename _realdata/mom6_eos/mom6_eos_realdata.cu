/**
 * MOM6 Equation-of-State kernel — REAL DATA harness.
 *
 * Kernel math (eos_density, cpu_eos, kernel_eos) copied VERBATIM from
 * ocean/mom6/mom6_vortex.cu — DO NOT MODIFY.
 * Only the synthetic data generator is replaced by a file loader.
 *
 * Input: data/eos_tsp.bin  (int32 n, float32 T[n], S[n], P[n])
 *        data/eos_region.u8 (uint8 region[n]: 0 open, 1 polar, 2 Med)
 * Real Argo profiles, GDAC daily geo files 2026-05-15 (atlantic/pacific/
 * indian), QC flags 1/2 only. See extract_argo.py for provenance.
 *
 * Scaling: real triplets are tiled (replicated) to reach the original
 * benchmark sizes (1M/5M/10M). Accuracy is reported on the raw, un-replicated
 * records (authoritative) and on the full tiled array (identical by
 * construction, sanity check only).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ============================================================
// KERNEL: Equation of State — density from T/S/P
// (verbatim copy from ocean/mom6/mom6_vortex.cu)
// ============================================================

__host__ __device__ float eos_density(float T, float S, float P_dbar) {
    // Simplified UNESCO equation of state
    float T2 = T * T, T3 = T2 * T;
    float S32 = S * sqrtf(fabsf(S));

    float rho0 = 999.842594f + 6.793952e-2f*T - 9.095290e-3f*T2
                + 1.001685e-4f*T3 - 1.120083e-6f*T2*T2 + 6.536332e-9f*T2*T3;

    float A = 8.24493e-1f - 4.0899e-3f*T + 7.6438e-5f*T2
             - 8.2467e-7f*T3 + 5.3875e-9f*T2*T2;
    float B = -5.72466e-3f + 1.0227e-4f*T - 1.6546e-6f*T2;

    return rho0 + A*S + B*S32 + 4.8314e-4f*S*S;
}

void cpu_eos(const float* T, const float* S, const float* P, float* rho, int n) {
    for (int i = 0; i < n; i++)
        rho[i] = eos_density(T[i], S[i], P[i]);
}

__global__ void kernel_eos(const float* __restrict__ T, const float* __restrict__ S,
                            const float* __restrict__ P, float* __restrict__ rho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = T[i], s = S[i];
    float t2=t*t, t3=t2*t;
    float s32 = s * __fsqrt_rn(fabsf(s));
    float rho0 = 999.842594f+6.793952e-2f*t-9.095290e-3f*t2+1.001685e-4f*t3-1.120083e-6f*t2*t2+6.536332e-9f*t2*t3;
    float A = 8.24493e-1f-4.0899e-3f*t+7.6438e-5f*t2-8.2467e-7f*t3+5.3875e-9f*t2*t2;
    float B = -5.72466e-3f+1.0227e-4f*t-1.6546e-6f*t2;
    rho[i] = rho0 + A*s + B*s32 + 4.8314e-4f*s*s;
}

// ============================================================
// Real-data loader (replaces synthetic generator)
// ============================================================

static int load_tsp(const char* path, float** T, float** S, float** P) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fprintf(stderr, "bad header\n"); exit(1); }
    *T = (float*)malloc((size_t)n * 4);
    *S = (float*)malloc((size_t)n * 4);
    *P = (float*)malloc((size_t)n * 4);
    if (fread(*T, 4, n, f) != (size_t)n || fread(*S, 4, n, f) != (size_t)n ||
        fread(*P, 4, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return n;
}

static unsigned char* load_regions(const char* path, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    unsigned char* r = (unsigned char*)malloc(n);
    if (fread(r, 1, n, f) != (size_t)n) { fprintf(stderr, "region short read\n"); exit(1); }
    fclose(f);
    return r;
}

int main(int argc, char** argv) {
    const char* tsp_path = (argc > 1) ? argv[1] : "data/eos_tsp.bin";
    const char* reg_path = (argc > 2) ? argv[2] : "data/eos_region.u8";

    printf("================================================\n");
    printf("  MOM6 Equation of State — REAL ARGO DATA\n");
    printf("  GDAC daily geo profiles 2026-05-15, QC 1/2\n");
    printf("================================================\n\n");

    float *rT, *rS, *rP;
    int nreal = load_tsp(tsp_path, &rT, &rS, &rP);
    unsigned char* reg = load_regions(reg_path, nreal);

    float tmin = 1e9f, tmax = -1e9f, smin = 1e9f, smax = -1e9f, pmin = 1e9f, pmax = -1e9f;
    for (int i = 0; i < nreal; i++) {
        if (rT[i] < tmin) tmin = rT[i]; if (rT[i] > tmax) tmax = rT[i];
        if (rS[i] < smin) smin = rS[i]; if (rS[i] > smax) smax = rS[i];
        if (rP[i] < pmin) pmin = rP[i]; if (rP[i] > pmax) pmax = rP[i];
    }
    printf("Loaded %d real (T,S,P) records\n", nreal);
    printf("  T: [%.3f, %.3f] degC | S: [%.3f, %.3f] PSU | P: [%.1f, %.1f] dbar\n\n",
           tmin, tmax, smin, smax, pmin, pmax);

    // Benchmark sizes from the original harness; real data tiled to fill
    int sizes_eos[] = {1000000, 5000000, 10000000};
    for (int is = 0; is < 3; is++) {
        int n = sizes_eos[is];
        int reps = (n + nreal - 1) / nreal;
        printf("--- %d points (real %d x %d tiles, last tile truncated) ---\n", n, nreal, reps);

        float *hT=(float*)malloc((size_t)n*4), *hS=(float*)malloc((size_t)n*4), *hP=(float*)malloc((size_t)n*4);
        float *hrho_c=(float*)malloc((size_t)n*4), *hrho_g=(float*)malloc((size_t)n*4);
        for (int i = 0; i < n; i++) {
            int j = i % nreal;                 // tiling (documented replication)
            hT[i] = rT[j]; hS[i] = rS[j]; hP[i] = rP[j];
        }

        clock_t t0 = clock();
        cpu_eos(hT, hS, hP, hrho_c, n);
        double cpu_ms = 1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

        float *dT,*dS,*dP,*drho;
        cudaMalloc(&dT,(size_t)n*4);cudaMalloc(&dS,(size_t)n*4);cudaMalloc(&dP,(size_t)n*4);cudaMalloc(&drho,(size_t)n*4);
        cudaMemcpy(dT,hT,(size_t)n*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dS,hS,(size_t)n*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dP,hP,(size_t)n*4,cudaMemcpyHostToDevice);

        int thr=256,blk=(n+thr-1)/thr;
        kernel_eos<<<blk,thr>>>(dT,dS,dP,drho,n);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
        int runs=20;
        cudaEventRecord(e0);
        for(int r=0;r<runs;r++) kernel_eos<<<blk,thr>>>(dT,dS,dP,drho,n);
        cudaEventRecord(e1);cudaEventSynchronize(e1);
        float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

        cudaMemcpy(hrho_g,drho,(size_t)n*4,cudaMemcpyDeviceToHost);

        // Accuracy over full tiled array (original criteria)
        float max_rel=0; int nan_c=0;
        for(int i=0;i<n;i++){
            if(isnan(hrho_g[i])){nan_c++;continue;}
            float re=fabsf(hrho_g[i]-hrho_c[i])/fabsf(hrho_c[i]);
            if(re>max_rel)max_rel=re;
        }

        // Authoritative accuracy on raw (un-replicated) records, by region
        int nauth = (n < nreal) ? n : nreal;
        float max_abs_r=0, max_rel_r=0;
        float max_rel_reg[3] = {0,0,0};
        int   argmax = -1;
        for (int i = 0; i < nauth; i++) {
            if (isnan(hrho_g[i])) continue;
            float ae = fabsf(hrho_g[i]-hrho_c[i]);
            float re = ae / fabsf(hrho_c[i]);
            if (ae > max_abs_r) max_abs_r = ae;
            if (re > max_rel_r) { max_rel_r = re; argmax = i; }
            int rg = reg[i] < 3 ? reg[i] : 0;
            if (re > max_rel_reg[rg]) max_rel_reg[rg] = re;
        }

        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
        printf("  Max rel (tiled %d pts): %.2e  NaN=%d\n",n,max_rel,nan_c);
        printf("  Real-record accuracy (n=%d): max abs %.3e kg/m3 | max rel %.3e\n",
               nauth, max_abs_r, max_rel_r);
        printf("    by region: open %.3e | polar %.3e | Med %.3e\n",
               max_rel_reg[0], max_rel_reg[1], max_rel_reg[2]);
        if (argmax >= 0)
            printf("    worst point: T=%.3f S=%.3f P=%.1f  rho_cpu=%.4f rho_gpu=%.4f\n",
                   hT[argmax], hS[argmax], hP[argmax], hrho_c[argmax], hrho_g[argmax]);
        printf("  Status: %s\n\n",(nan_c==0&&max_rel<1e-5f)?"PASS":"NEEDS REVIEW");

        free(hT);free(hS);free(hP);free(hrho_c);free(hrho_g);
        cudaFree(dT);cudaFree(dS);cudaFree(dP);cudaFree(drho);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }

    free(rT); free(rS); free(rP); free(reg);
    return 0;
}
