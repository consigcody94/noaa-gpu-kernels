/**
 * ccpp_tridi1_realdata.cu — CCPP tridi1 PBL tridiagonal solver, REAL-DATA harness.
 *
 * cpu_tridi1, kernel_tridi1 and compare() are VERBATIM copies of the
 * corresponding code in ../../noaa_multi_kernel.cu (kernel math untouched).
 * The ONLY change vs the original harness: the synthetic generator
 * gen_tridi() is replaced by a loader for 'tridi_real.bin', which holds
 * implicit vertical-diffusion tridiagonal systems built from REAL IGRA2
 * radiosonde profiles (see prep_igra2_tridi.py for provenance and the
 * bulk-Richardson K(z) preprocessing). Real columns are tiled (replicated
 * modulo ncol_real) to reach the original benchmark sizes 10k/100k/500k;
 * the replication factor is printed per size.
 *
 * Build: nvcc -O3 -arch=sm_120 -o ccpp_tridi1_realdata.exe ccpp_tridi1_realdata.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_LEV 128   // max vertical levels (PBL)

// ============================================================
// KERNEL 1: CCPP tridi1 — Thomas algorithm for PBL diffusion
// From NCAR/ccpp-physics/physics/PBL/tridi.f
// Solves: cl*x(k-1) + cm*x(k) + cu*x(k+1) = r1
// (verbatim from noaa_multi_kernel.cu)
// ============================================================

void cpu_tridi1(int l, int n, const float* cl, const float* cm,
                const float* cu, const float* r1, float* a1, int stride) {
    float au[MAX_LEV];
    for (int i = 0; i < l; i++) {
        // Forward sweep
        float fk = 1.0f / cm[i * stride + 0];
        au[0] = fk * cu[i * stride + 0];
        a1[i * stride + 0] = fk * r1[i * stride + 0];
        for (int k = 1; k < n - 1; k++) {
            fk = 1.0f / (cm[i*stride+k] - cl[i*stride+k] * au[k-1]);
            au[k] = fk * cu[i*stride+k];
            a1[i*stride+k] = fk * (r1[i*stride+k] - cl[i*stride+k] * a1[i*stride+k-1]);
        }
        fk = 1.0f / (cm[i*stride+n-1] - cl[i*stride+n-1] * au[n-2]);
        a1[i*stride+n-1] = fk * (r1[i*stride+n-1] - cl[i*stride+n-1] * a1[i*stride+n-2]);
        // Back substitution
        for (int k = n - 2; k >= 0; k--)
            a1[i*stride+k] -= au[k] * a1[i*stride+k+1];
    }
}

__global__ void kernel_tridi1(int n, const float* __restrict__ cl,
    const float* __restrict__ cm, const float* __restrict__ cu,
    const float* __restrict__ r1, float* __restrict__ a1, int stride, int ncol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;

    float au[MAX_LEV];
    float fk = 1.0f / cm[i*stride+0];
    au[0] = fk * cu[i*stride+0];
    a1[i*stride+0] = fk * r1[i*stride+0];
    for (int k = 1; k < n-1; k++) {
        fk = 1.0f / (cm[i*stride+k] - cl[i*stride+k]*au[k-1]);
        au[k] = fk * cu[i*stride+k];
        a1[i*stride+k] = fk*(r1[i*stride+k] - cl[i*stride+k]*a1[i*stride+k-1]);
    }
    fk = 1.0f / (cm[i*stride+n-1] - cl[i*stride+n-1]*au[n-2]);
    a1[i*stride+n-1] = fk*(r1[i*stride+n-1] - cl[i*stride+n-1]*a1[i*stride+n-2]);
    for (int k = n-2; k >= 0; k--)
        a1[i*stride+k] -= au[k]*a1[i*stride+k+1];
}

// ============================================================
// REAL-DATA loader (replaces gen_tridi)
// tridi_real.bin: int32 ncol, int32 nlev,
//                 float32 cl[ncol*nlev], cm[...], cu[...], r1[...]
// ============================================================

static float *g_cl = NULL, *g_cm = NULL, *g_cu = NULL, *g_r1 = NULL;
static int g_ncol_real = 0, g_nlev = 0;

int load_real(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 0; }
    int hdr[2];
    if (fread(hdr, sizeof(int), 2, f) != 2) { fclose(f); return 0; }
    g_ncol_real = hdr[0]; g_nlev = hdr[1];
    size_t nv = (size_t)g_ncol_real * g_nlev;
    g_cl = (float*)malloc(nv*4); g_cm = (float*)malloc(nv*4);
    g_cu = (float*)malloc(nv*4); g_r1 = (float*)malloc(nv*4);
    if (fread(g_cl,4,nv,f)!=nv || fread(g_cm,4,nv,f)!=nv ||
        fread(g_cu,4,nv,f)!=nv || fread(g_r1,4,nv,f)!=nv) {
        fprintf(stderr, "ERROR: short read in %s\n", path); fclose(f); return 0;
    }
    fclose(f);
    printf("Loaded %d REAL columns x %d levels from %s\n", g_ncol_real, g_nlev, path);
    return 1;
}

// Tile real columns to l columns (column i <- real column i %% ncol_real)
void fill_tiled(float* cl, float* cm, float* cu, float* r1, int l, int n, int stride) {
    for (int i = 0; i < l; i++) {
        int s = (i % g_ncol_real) * g_nlev;
        memcpy(cl + (size_t)i*stride, g_cl + s, n*4);
        memcpy(cm + (size_t)i*stride, g_cm + s, n*4);
        memcpy(cu + (size_t)i*stride, g_cu + s, n*4);
        memcpy(r1 + (size_t)i*stride, g_r1 + s, n*4);
    }
}

// ============================================================
// compare() — verbatim from noaa_multi_kernel.cu
// ============================================================

template<typename T>
void compare(const T* cpu, const T* gpu, int n, const char* name) {
    float max_abs=0,max_rel=0; int nan_c=0,fail_c=0;
    for (int i = 0; i < n; i++) {
        float cv = ((const float*)cpu)[i];
        float gv = ((const float*)gpu)[i];
        if (isnan(gv)||isinf(gv)){nan_c++;continue;}
        float ae = fabsf(gv-cv);
        if (ae>max_abs) max_abs=ae;
        if (fabsf(cv)>1e-10f) {
            float re=ae/fabsf(cv);
            if(re>max_rel)max_rel=re;
            if(re>1e-4f)fail_c++;
        }
    }
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d | >0.01%%: %d/%d\n",
           max_abs,max_rel,nan_c,fail_c,n);
    printf("  Status: %s\n\n",
           (nan_c==0&&max_rel<1e-4f)?"PASS":
           (nan_c==0&&max_rel<1e-2f)?"PASS (fast math)":"NEEDS REVIEW");
}

int main(int argc, char** argv) {
    const char* binpath = (argc > 1) ? argv[1] : "tridi_real.bin";
    printf("================================================\n");
    printf("  CCPP tridi1 (PBL solver) — REAL DATA\n");
    printf("  IGRA2 radiosonde profiles (NOAA NCEI)\n");
    printf("================================================\n\n");
    if (!load_real(binpath)) return 1;

    int sizes[] = {10000, 100000, 500000};
    int n = g_nlev; // 64 vertical levels, same as original harness

    for (int is = 0; is < 3; is++) {
        int l = sizes[is];
        double repl = (double)l / g_ncol_real;
        printf("--- %d columns x %d levels (replication %.2fx of %d real cols) ---\n",
               l, n, repl, g_ncol_real);
        size_t sz = (size_t)l*n*sizeof(float);
        float *hcl=(float*)malloc(sz),*hcm=(float*)malloc(sz),*hcu=(float*)malloc(sz);
        float *hr1=(float*)malloc(sz),*ha_cpu=(float*)malloc(sz),*ha_gpu=(float*)malloc(sz);
        fill_tiled(hcl,hcm,hcu,hr1,l,n,n);

        clock_t t0=clock();
        cpu_tridi1(l,n,hcl,hcm,hcu,hr1,ha_cpu,n);
        double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

        float *dcl,*dcm,*dcu,*dr1,*da1;
        cudaMalloc(&dcl,sz);cudaMalloc(&dcm,sz);cudaMalloc(&dcu,sz);
        cudaMalloc(&dr1,sz);cudaMalloc(&da1,sz);
        cudaMemcpy(dcl,hcl,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dcm,hcm,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dcu,hcu,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dr1,hr1,sz,cudaMemcpyHostToDevice);

        int thr=256,blk=(l+thr-1)/thr;
        kernel_tridi1<<<blk,thr>>>(n,dcl,dcm,dcu,dr1,da1,n,l);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
        int runs=20;
        cudaEventRecord(e0);
        for(int r=0;r<runs;r++)
            kernel_tridi1<<<blk,thr>>>(n,dcl,dcm,dcu,dr1,da1,n,l);
        cudaEventRecord(e1);cudaEventSynchronize(e1);
        float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

        cudaMemcpy(ha_gpu,da1,sz,cudaMemcpyDeviceToHost);
        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
        compare<float>(ha_cpu,ha_gpu,l*n,"tridi1");

        // --- diagnostic only (does not affect pass/fail above): locate worst rel-error point ---
        {
            const char* fields[4] = {"T","q","u","v"};
            float worst=0; int wi=-1, wk=-1;
            for (int i = 0; i < l; i++) for (int k = 0; k < n; k++) {
                float cv=ha_cpu[(size_t)i*n+k], gv=ha_gpu[(size_t)i*n+k];
                if (isnan(gv)||isinf(gv)||fabsf(cv)<=1e-10f) continue;
                float re=fabsf(gv-cv)/fabsf(cv);
                if (re>worst){worst=re;wi=i;wk=k;}
            }
            if (wi>=0) {
                int rc = wi % g_ncol_real;
                printf("  [diag] worst rel %.2e at col %d (real col %d, field %s, sounding %d), level k=%d, cpu=%.6e gpu=%.6e\n\n",
                       worst, wi, rc, fields[rc%4], rc/4, wk,
                       ha_cpu[(size_t)wi*n+wk], ha_gpu[(size_t)wi*n+wk]);
            }
        }

        free(hcl);free(hcm);free(hcu);free(hr1);free(ha_cpu);free(ha_gpu);
        cudaFree(dcl);cudaFree(dcm);cudaFree(dcu);cudaFree(dr1);cudaFree(da1);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    return 0;
}
