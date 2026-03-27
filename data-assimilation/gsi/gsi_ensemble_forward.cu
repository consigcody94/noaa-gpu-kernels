/*
 * GPU Kernel: GSI Ensemble Forward Model
 *
 * Parallelizes the ensemble_forward_model weighted sum from
 * hybrid_ensemble_isotropic.F90 (lines 1977-2000):
 *
 *   cvec(i,j,k) = sum_n( a_en(i,j,k,n) * en_perts(i,j,k,n) )
 *
 * This is the single most expensive operation in hybrid EnVar mode,
 * estimated at 30-50% of total GSI wall-clock time.
 *
 * It's perfectly parallel across all grid points — zero dependencies.
 *
 * Compile: nvcc -O3 -arch=sm_86 gsi_ensemble_forward.cu -o gsi_ens
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } }

#define NITERS 100

// ============================================================
// GPU Kernel: Ensemble weighted sum
// Each thread handles one (i, j, k) grid point
// ============================================================
__global__ void kernel_ensemble_forward(
    const float* __restrict__ a_en,      // [n_ens, km, jm, im] — ensemble weights
    const float* __restrict__ en_perts,  // [n_ens, km, jm, im] — ensemble perturbations
    float* __restrict__ cvec,            // [km, jm, im] — output control vector
    int im, int jm, int km, int n_ens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = im * jm * km;
    if (idx >= total) return;

    float sum = 0.0f;
    for (int n = 0; n < n_ens; n++) {
        sum += a_en[n * total + idx] * en_perts[n * total + idx];
    }
    cvec[idx] = sum;
}

// Optimized version: use shared memory for ensemble reduction
__global__ void kernel_ensemble_forward_v2(
    const float* __restrict__ a_en,
    const float* __restrict__ en_perts,
    float* __restrict__ cvec,
    int im, int jm, int km, int n_ens)
{
    // Each block handles one grid point, threads cooperate on ensemble reduction
    int grid_idx = blockIdx.x;
    int total = im * jm * km;
    if (grid_idx >= total) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread sums a subset of ensemble members
    float partial = 0.0f;
    for (int n = tid; n < n_ens; n += nthreads) {
        partial += a_en[n * total + grid_idx] * en_perts[n * total + grid_idx];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }

    if (tid == 0) cvec[grid_idx] = partial;
}

// Simple version: one thread per grid point, loop over ensemble
// This is the most memory-efficient for large grids
__global__ void kernel_ensemble_forward_simple(
    const float* __restrict__ a_en,
    const float* __restrict__ en_perts,
    float* __restrict__ cvec,
    int total_points, int n_ens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_points) return;

    float sum = 0.0f;
    for (int n = 0; n < n_ens; n++) {
        sum += a_en[n * total_points + idx] * en_perts[n * total_points + idx];
    }
    cvec[idx] = sum;
}

// ============================================================
// CPU Reference
// ============================================================
void cpu_ensemble_forward(
    const float* a_en, const float* en_perts, float* cvec,
    int total_points, int n_ens)
{
    for (int idx = 0; idx < total_points; idx++) {
        float sum = 0.0f;
        for (int n = 0; n < n_ens; n++) {
            sum += a_en[n * total_points + idx] * en_perts[n * total_points + idx];
        }
        cvec[idx] = sum;
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  GSI Ensemble Forward Model: GPU Benchmark\n");
    printf("  hybrid_ensemble_isotropic.F90 — weighted ensemble sum\n");
    printf("  RTX 3060 12GB | CUDA %d.%d\n", CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    printf("================================================================\n\n");

    // Test configurations matching realistic GSI setups
    struct TestCase {
        int im, jm, km, n_ens;
        const char* name;
    };

    TestCase tests[] = {
        {192, 192, 64,  30,  "RAP/HRRR (small, 30 ens)"},
        {384, 192, 64,  80,  "GFS C384 (medium, 80 ens)"},
        {384, 384, 127, 80,  "GFS C768 (large, 80 ens)"},
        {768, 384, 127, 80,  "GFS C1152 (very large, 80 ens)"},
        {192, 192, 64,  160, "Experimental (160 ens)"},
    };
    int ntests = 5;

    int total_pass = 0;

    for (int t = 0; t < ntests; t++) {
        int im = tests[t].im, jm = tests[t].jm, km = tests[t].km, n_ens = tests[t].n_ens;
        int total_points = im * jm * km;
        long long total_elem = (long long)n_ens * total_points;
        float data_mb = total_elem * 4.0f / (1024*1024) * 2;  // a_en + en_perts

        printf("--- %s ---\n", tests[t].name);
        printf("  Grid: %d x %d x %d = %d points, %d ensemble members\n",
               im, jm, km, total_points, n_ens);
        printf("  Data: %.0f MB (a_en + en_perts)\n", data_mb);

        // Check if fits in GPU memory (12 GB limit)
        if (data_mb > 10000) {
            printf("  SKIP: Too large for 12GB GPU\n\n");
            continue;
        }

        // Allocate
        float *h_a_en    = (float*)malloc(total_elem * sizeof(float));
        float *h_perts   = (float*)malloc(total_elem * sizeof(float));
        float *h_cvec_ref = (float*)malloc(total_points * sizeof(float));
        float *h_cvec_gpu = (float*)malloc(total_points * sizeof(float));

        srand(42 + t);
        for (long long i = 0; i < total_elem; i++) {
            h_a_en[i]  = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
            h_perts[i] = -2.0f + 4.0f * (rand() / (float)RAND_MAX);
        }

        // CPU reference
        cpu_ensemble_forward(h_a_en, h_perts, h_cvec_ref, total_points, n_ens);

        // GPU
        float *d_a_en, *d_perts, *d_cvec;
        CUDA_CHECK(cudaMalloc(&d_a_en,  total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perts, total_elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cvec,  total_points * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_a_en,  h_a_en,  total_elem*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_perts, h_perts, total_elem*sizeof(float), cudaMemcpyHostToDevice));

        int blockSize = 256;
        int gridSize = (total_points + blockSize - 1) / blockSize;

        // Warmup
        kernel_ensemble_forward_simple<<<gridSize, blockSize>>>(
            d_a_en, d_perts, d_cvec, total_points, n_ens);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < NITERS; it++) {
            kernel_ensemble_forward_simple<<<gridSize, blockSize>>>(
                d_a_en, d_perts, d_cvec, total_points, n_ens);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_gpu;
        CUDA_CHECK(cudaEventElapsedTime(&t_gpu, start, stop));
        t_gpu /= NITERS;

        // Check accuracy
        CUDA_CHECK(cudaMemcpy(h_cvec_gpu, d_cvec, total_points*sizeof(float), cudaMemcpyDeviceToHost));

        float max_rel = 0, max_abs = 0;
        int nan_count = 0;
        for (int i = 0; i < total_points; i++) {
            if (h_cvec_gpu[i] != h_cvec_gpu[i]) { nan_count++; continue; }
            float ae = fabsf(h_cvec_gpu[i] - h_cvec_ref[i]);
            if (ae > max_abs) max_abs = ae;
            if (fabsf(h_cvec_ref[i]) > 1e-6f) {
                float re = ae / fabsf(h_cvec_ref[i]);
                if (re > max_rel) max_rel = re;
            }
        }

        // Compute effective bandwidth and FLOPS
        float bandwidth_gb = 2.0f * total_elem * 4.0f / (t_gpu * 1e6f);  // GB/s
        float gflops = 2.0f * total_elem / (t_gpu * 1e6f);  // GFLOP/s (multiply + add per element)

        // FP32 summation of n_ens terms: expected abs error ~ sqrt(n_ens) * eps * max_val
        // For n_ens=80, max_val~2: expected ~ 9 * 1.2e-7 * 2 = 2.2e-6
        // Use absolute error threshold scaled by n_ens
        float expected_abs = sqrtf((float)n_ens) * 1.2e-7f * 4.0f;  // 4.0 = 2x max value range
        int pass = (nan_count == 0 && max_abs < expected_abs * 10.0f);
        if (pass) total_pass++;

        printf("  GPU time:     %.4f ms\n", t_gpu);
        printf("  Bandwidth:    %.1f GB/s (GPU HBM peak: ~360 GB/s)\n", bandwidth_gb);
        printf("  Throughput:   %.1f GFLOP/s\n", gflops);
        printf("  Max rel err:  %.1e\n", max_rel);
        printf("  Max abs err:  %.1e\n", max_abs);
        printf("  NaN count:    %d\n", nan_count);
        printf("  Status:       %s\n\n", pass ? "PASS" : "FAIL");

        free(h_a_en); free(h_perts); free(h_cvec_ref); free(h_cvec_gpu);
        cudaFree(d_a_en); cudaFree(d_perts); cudaFree(d_cvec);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("================================================================\n");
    printf("  RESULTS: %d / %d tests passed\n", total_pass, ntests);
    if (total_pass == ntests)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  %d TESTS FAILED\n", ntests - total_pass);
    printf("================================================================\n");

    return 0;
}
