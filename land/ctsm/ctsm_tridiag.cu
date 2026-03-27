/**
 * CTSM (Community Terrestrial Systems Model) Tridiagonal Solver
 * From ESCOMP/CTSM (344 stars) — TridiagonalMod.F90
 *
 * Solves vertical soil/snow temperature diffusion at each land column.
 * Same Thomas algorithm as NOAH-MP but for the CESM land model.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_LEV 25  // up to 15 soil + 10 snow layers

void cpu_ctsm_tridiag(const float* a, const float* b, const float* c,
                       const float* r, float* u, const int* nlev,
                       int ncol) {
    for (int ci = 0; ci < ncol; ci++) {
        int nl = nlev[ci];
        int base = ci * MAX_LEV;
        float gam[MAX_LEV], bet;

        bet = b[base + 0];
        u[base + 0] = r[base + 0] / bet;

        for (int j = 1; j < nl; j++) {
            gam[j] = c[base + j - 1] / bet;
            bet = b[base + j] - a[base + j] * gam[j];
            if (fabsf(bet) < 1e-30f) bet = 1e-30f;
            u[base + j] = (r[base + j] - a[base + j] * u[base + j - 1]) / bet;
        }

        for (int j = nl - 2; j >= 0; j--) {
            u[base + j] = u[base + j] - gam[j + 1] * u[base + j + 1];
        }
    }
}

__global__ void kernel_ctsm_tridiag(const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ c,
    const float* __restrict__ r, float* __restrict__ u,
    const int* __restrict__ nlev, int ncol) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= ncol) return;

    int nl = nlev[ci];
    int base = ci * MAX_LEV;
    float gam[MAX_LEV], bet;

    bet = b[base + 0];
    u[base + 0] = r[base + 0] / bet;

    for (int j = 1; j < nl; j++) {
        gam[j] = c[base + j - 1] / bet;
        bet = b[base + j] - a[base + j] * gam[j];
        if (fabsf(bet) < 1e-30f) bet = 1e-30f;
        u[base + j] = (r[base + j] - a[base + j] * u[base + j - 1]) / bet;
    }

    for (int j = nl - 2; j >= 0; j--) {
        u[base + j] = u[base + j] - gam[j + 1] * u[base + j + 1];
    }
}

void gen(float* a, float* b, float* c, float* r, int* nl, int ncol, unsigned seed) {
    srand(seed);
    for (int ci = 0; ci < ncol; ci++) {
        nl[ci] = 10 + (rand() % 16); // 10-25 levels
        int base = ci * MAX_LEV;
        for (int j = 0; j < nl[ci]; j++) {
            a[base+j] = (j > 0) ? -(0.1f+0.5f*((float)rand()/RAND_MAX)) : 0.0f;
            c[base+j] = (j < nl[ci]-1) ? -(0.1f+0.5f*((float)rand()/RAND_MAX)) : 0.0f;
            b[base+j] = fabsf(a[base+j])+fabsf(c[base+j])+0.5f+((float)rand()/RAND_MAX);
            r[base+j] = -1.0f+2.0f*((float)rand()/RAND_MAX);
        }
    }
}

int main() {
    printf("================================================\n");
    printf("  CTSM Tridiagonal Solver GPU Kernel\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    int sizes[] = {100000, 500000, 1000000, 2000000};
    for (int is = 0; is < 4; is++) {
        int n = sizes[is];
        printf("--- %d land columns ---\n", n);

        size_t sz = n * MAX_LEV * sizeof(float);
        float *ha=(float*)malloc(sz), *hb=(float*)malloc(sz);
        float *hc=(float*)malloc(sz), *hr=(float*)malloc(sz);
        float *hu_cpu=(float*)malloc(sz), *hu_gpu=(float*)malloc(sz);
        int *hnl=(int*)malloc(n*sizeof(int));

        gen(ha, hb, hc, hr, hnl, n, 42);

        clock_t t0 = clock();
        cpu_ctsm_tridiag(ha, hb, hc, hr, hu_cpu, hnl, n);
        double cpu_ms = 1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

        float *da,*db,*dc,*dr_d,*du; int *dnl;
        cudaMalloc(&da,sz);cudaMalloc(&db,sz);cudaMalloc(&dc,sz);
        cudaMalloc(&dr_d,sz);cudaMalloc(&du,sz);cudaMalloc(&dnl,n*4);
        cudaMemcpy(da,ha,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(db,hb,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dc,hc,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dr_d,hr,sz,cudaMemcpyHostToDevice);
        cudaMemcpy(dnl,hnl,n*4,cudaMemcpyHostToDevice);

        int thr=256,blk=(n+thr-1)/thr;
        kernel_ctsm_tridiag<<<blk,thr>>>(da,db,dc,dr_d,du,dnl,n);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
        int runs=50;
        cudaEventRecord(e0);
        for(int r=0;r<runs;r++)
            kernel_ctsm_tridiag<<<blk,thr>>>(da,db,dc,dr_d,du,dnl,n);
        cudaEventRecord(e1);cudaEventSynchronize(e1);
        float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

        cudaMemcpy(hu_gpu,du,sz,cudaMemcpyDeviceToHost);

        float max_rel=0,max_res=0; int nan_c=0;
        for(int ci=0;ci<n;ci++){
            int base=ci*MAX_LEV;
            for(int j=0;j<hnl[ci];j++){
                if(isnan(hu_gpu[base+j])){nan_c++;continue;}
                if(fabsf(hu_cpu[base+j])>1e-10f){
                    float re=fabsf(hu_gpu[base+j]-hu_cpu[base+j])/fabsf(hu_cpu[base+j]);
                    if(re>max_rel)max_rel=re;
                }
                // Residual
                float res = hb[base+j]*hu_gpu[base+j]
                    + ((j>0)?ha[base+j]*hu_gpu[base+j-1]:0.0f)
                    + ((j<hnl[ci]-1)?hc[base+j]*hu_gpu[base+j+1]:0.0f)
                    - hr[base+j];
                if(fabsf(res)>max_res)max_res=fabsf(res);
            }
        }

        printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
        printf("  Max rel: %.2e | Max residual: %.2e | NaN: %d\n",max_rel,max_res,nan_c);
        printf("  Status: %s\n\n",
               (nan_c==0&&max_rel<1e-5f)?"PASS":
               (nan_c==0&&max_rel<1e-3f)?"PASS (FP32)":"NEEDS REVIEW");

        free(ha);free(hb);free(hc);free(hr);free(hu_cpu);free(hu_gpu);free(hnl);
        cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dr_d);cudaFree(du);cudaFree(dnl);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    return 0;
}
