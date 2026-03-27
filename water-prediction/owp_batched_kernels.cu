/**
 * NOAA-OWP Batched GPU Kernels
 *
 * Two kernels for the National Water Model's 2.7M catchment processing:
 *
 * 1. CFE Nash Cascade (github.com/NOAA-OWP/cfe)
 *    - Routes lateral flow through N_nash linear reservoirs
 *    - Sequential within each cascade (N=2-10), parallel across catchments
 *
 * 2. NOAH-MP Tridiagonal Soil Water Solver (github.com/NOAA-OWP/noah-owp-modular)
 *    - Solves Richards equation implicitly with Thomas algorithm (ROSR12)
 *    - Sequential forward/backward sweep per column, parallel across columns
 *
 * Both kernels are validated against CPU reference implementations.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Maximum sizes
#define MAX_NASH 10
#define MAX_SOIL 10

// ============================================================
// KERNEL 1: CFE Nash Cascade
// ============================================================

// CPU reference — matches nash_cascade.c from NOAA-OWP/cfe
void cpu_nash_cascade(
    const float* flux_lat,     // lateral inflow per catchment [ncatch]
    const float* K_nash,       // Nash reservoir coefficient [ncatch]
    const int* N_nash,         // number of Nash reservoirs [ncatch]
    float* storage,            // Nash storage [ncatch * MAX_NASH] (in/out)
    float* Q_out,              // outflow from last reservoir [ncatch]
    int ncatch)
{
    for (int c = 0; c < ncatch; c++) {
        int N = N_nash[c];
        float K = K_nash[c];
        int base = c * MAX_NASH;
        float Q_prev = 0.0f;

        for (int i = 0; i < N; i++) {
            float Q_i = K * storage[base + i];
            storage[base + i] -= Q_i;

            if (i == 0)
                storage[base + i] += flux_lat[c];
            else
                storage[base + i] += Q_prev;

            Q_prev = Q_i;
        }
        Q_out[c] = Q_prev;
    }
}

// GPU kernel — one thread per catchment
__global__ void kernel_nash_cascade(
    const float* __restrict__ flux_lat,
    const float* __restrict__ K_nash,
    const int* __restrict__ N_nash,
    float* __restrict__ storage,
    float* __restrict__ Q_out,
    int ncatch)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncatch) return;

    int N = N_nash[c];
    float K = K_nash[c];
    int base = c * MAX_NASH;
    float Q_prev = 0.0f;

    for (int i = 0; i < N; i++) {
        float Q_i = K * storage[base + i];
        storage[base + i] -= Q_i;

        if (i == 0)
            storage[base + i] += flux_lat[c];
        else
            storage[base + i] += Q_prev;

        Q_prev = Q_i;
    }
    Q_out[c] = Q_prev;
}

// ============================================================
// KERNEL 2: NOAH-MP Tridiagonal Solver (ROSR12)
// ============================================================

// CPU reference — matches ROSR12 from noah-owp-modular
void cpu_tridiag_solve(
    const float* A,    // lower diagonal [ncol * MAX_SOIL]
    const float* B,    // main diagonal  [ncol * MAX_SOIL]
    const float* C,    // upper diagonal [ncol * MAX_SOIL]
    const float* D,    // right-hand side [ncol * MAX_SOIL]
    float* X,          // solution [ncol * MAX_SOIL]
    const int* nsoil,  // soil layers per column [ncol]
    int ncol)
{
    for (int col = 0; col < ncol; col++) {
        int ns = nsoil[col];
        int base = col * MAX_SOIL;

        // Work arrays
        float P[MAX_SOIL], Delta[MAX_SOIL];

        // Forward sweep (Thomas algorithm)
        P[0] = -C[base + 0] / B[base + 0];
        Delta[0] = D[base + 0] / B[base + 0];

        for (int k = 1; k < ns; k++) {
            float denom = B[base + k] + A[base + k] * P[k - 1];
            if (fabsf(denom) < 1e-30f) denom = 1e-30f;
            P[k] = -C[base + k] / denom;
            Delta[k] = (D[base + k] - A[base + k] * Delta[k - 1]) / denom;
        }

        // Back substitution
        X[base + ns - 1] = Delta[ns - 1];
        for (int k = ns - 2; k >= 0; k--) {
            X[base + k] = P[k] * X[base + k + 1] + Delta[k];
        }
    }
}

// GPU kernel — one thread per column
__global__ void kernel_tridiag_solve(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    float* __restrict__ X,
    const int* __restrict__ nsoil,
    int ncol)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    int ns = nsoil[col];
    int base = col * MAX_SOIL;

    // Local work arrays (registers for small MAX_SOIL)
    float P[MAX_SOIL], Delta[MAX_SOIL];

    // Forward sweep
    P[0] = -C[base + 0] / B[base + 0];
    Delta[0] = D[base + 0] / B[base + 0];

    for (int k = 1; k < ns; k++) {
        float denom = B[base + k] + A[base + k] * P[k - 1];
        if (fabsf(denom) < 1e-30f) denom = 1e-30f;
        P[k] = -C[base + k] / denom;
        Delta[k] = (D[base + k] - A[base + k] * Delta[k - 1]) / denom;
    }

    // Back substitution
    X[base + ns - 1] = Delta[ns - 1];
    for (int k = ns - 2; k >= 0; k--) {
        X[base + k] = P[k] * X[base + k + 1] + Delta[k];
    }
}

// ============================================================
// Test data generation
// ============================================================

void gen_nash_data(float* flux, float* K, int* N, float* stor, int nc, unsigned seed) {
    srand(seed);
    for (int c = 0; c < nc; c++) {
        flux[c] = 0.001f + 0.01f * ((float)rand() / RAND_MAX); // lateral inflow (m)
        K[c] = 0.01f + 0.09f * ((float)rand() / RAND_MAX);     // K in [0.01, 0.10]
        N[c] = 2 + (rand() % 9);                                 // 2-10 reservoirs
        for (int i = 0; i < MAX_NASH; i++) {
            stor[c * MAX_NASH + i] = 0.001f + 0.05f * ((float)rand() / RAND_MAX);
        }
    }
}

void gen_tridiag_data(float* A, float* B, float* C, float* D, int* ns, int nc, unsigned seed) {
    srand(seed);
    for (int col = 0; col < nc; col++) {
        ns[col] = 4 + (rand() % 7); // 4-10 soil layers
        int base = col * MAX_SOIL;
        for (int k = 0; k < ns[col]; k++) {
            // Diagonally dominant system (ensures stability)
            A[base + k] = (k > 0) ? -(0.1f + 0.5f * ((float)rand() / RAND_MAX)) : 0.0f;
            C[base + k] = (k < ns[col] - 1) ? -(0.1f + 0.5f * ((float)rand() / RAND_MAX)) : 0.0f;
            B[base + k] = fabsf(A[base + k]) + fabsf(C[base + k]) + 0.5f + ((float)rand() / RAND_MAX);
            D[base + k] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
        }
    }
}

// ============================================================
// Benchmark runner
// ============================================================

void benchmark_nash(int ncatch) {
    printf("--- Nash Cascade: %d catchments ---\n", ncatch);

    size_t sz_f = ncatch * sizeof(float);
    size_t sz_i = ncatch * sizeof(int);
    size_t sz_s = ncatch * MAX_NASH * sizeof(float);

    float *h_flux = (float*)malloc(sz_f);
    float *h_K = (float*)malloc(sz_f);
    int *h_N = (int*)malloc(sz_i);
    float *h_stor_cpu = (float*)malloc(sz_s);
    float *h_stor_gpu = (float*)malloc(sz_s);
    float *h_Q_cpu = (float*)malloc(sz_f);
    float *h_Q_gpu = (float*)malloc(sz_f);

    gen_nash_data(h_flux, h_K, h_N, h_stor_cpu, ncatch, 42);
    memcpy(h_stor_gpu, h_stor_cpu, sz_s); // same initial conditions

    // CPU
    clock_t t0 = clock();
    cpu_nash_cascade(h_flux, h_K, h_N, h_stor_cpu, h_Q_cpu, ncatch);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    // GPU
    float *d_flux, *d_K, *d_stor, *d_Q;
    int *d_N;
    cudaMalloc(&d_flux, sz_f); cudaMalloc(&d_K, sz_f);
    cudaMalloc(&d_N, sz_i); cudaMalloc(&d_stor, sz_s); cudaMalloc(&d_Q, sz_f);
    cudaMemcpy(d_flux, h_flux, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sz_i, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stor, h_stor_gpu, sz_s, cudaMemcpyHostToDevice);

    int thr = 256, blk = (ncatch + thr - 1) / thr;

    // Warmup
    kernel_nash_cascade<<<blk, thr>>>(d_flux, d_K, d_N, d_stor, d_Q, ncatch);
    cudaDeviceSynchronize();

    // Reset storage for fair benchmark
    cudaMemcpy(d_stor, h_stor_gpu, sz_s, cudaMemcpyHostToDevice);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 50;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++) {
        // Reset storage each run for consistency
        cudaMemcpy(d_stor, h_stor_gpu, sz_s, cudaMemcpyHostToDevice);
        kernel_nash_cascade<<<blk, thr>>>(d_flux, d_K, d_N, d_stor, d_Q, ncatch);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    // Get results (run once more with original storage)
    cudaMemcpy(d_stor, h_stor_gpu, sz_s, cudaMemcpyHostToDevice);
    kernel_nash_cascade<<<blk, thr>>>(d_flux, d_K, d_N, d_stor, d_Q, ncatch);
    cudaMemcpy(h_Q_gpu, d_Q, sz_f, cudaMemcpyDeviceToHost);

    // Accuracy
    float max_abs = 0, max_rel = 0;
    int nan_c = 0, fail_c = 0;
    for (int i = 0; i < ncatch; i++) {
        if (isnan(h_Q_gpu[i]) || isinf(h_Q_gpu[i])) { nan_c++; continue; }
        float ae = fabsf(h_Q_gpu[i] - h_Q_cpu[i]);
        if (ae > max_abs) max_abs = ae;
        if (fabsf(h_Q_cpu[i]) > 1e-10f) {
            float re = ae / fabsf(h_Q_cpu[i]);
            if (re > max_rel) max_rel = re;
            if (re > 1e-5f) fail_c++;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d | >1e-5 err: %d/%d\n",
           max_abs, max_rel, nan_c, fail_c, ncatch);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-5f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-3f) ? "PASS (FP32 rounding)" : "FAIL");

    free(h_flux); free(h_K); free(h_N); free(h_stor_cpu); free(h_stor_gpu);
    free(h_Q_cpu); free(h_Q_gpu);
    cudaFree(d_flux); cudaFree(d_K); cudaFree(d_N); cudaFree(d_stor); cudaFree(d_Q);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

void benchmark_tridiag(int ncol) {
    printf("--- NOAH-MP Tridiag Solver: %d columns ---\n", ncol);

    size_t sz_f = ncol * MAX_SOIL * sizeof(float);
    size_t sz_i = ncol * sizeof(int);

    float *h_A = (float*)malloc(sz_f), *h_B = (float*)malloc(sz_f);
    float *h_C = (float*)malloc(sz_f), *h_D = (float*)malloc(sz_f);
    float *h_X_cpu = (float*)malloc(sz_f), *h_X_gpu = (float*)malloc(sz_f);
    int *h_ns = (int*)malloc(sz_i);

    gen_tridiag_data(h_A, h_B, h_C, h_D, h_ns, ncol, 99);

    // CPU
    clock_t t0 = clock();
    cpu_tridiag_solve(h_A, h_B, h_C, h_D, h_X_cpu, h_ns, ncol);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    // GPU
    float *d_A, *d_B, *d_C, *d_D, *d_X;
    int *d_ns;
    cudaMalloc(&d_A, sz_f); cudaMalloc(&d_B, sz_f);
    cudaMalloc(&d_C, sz_f); cudaMalloc(&d_D, sz_f);
    cudaMalloc(&d_X, sz_f); cudaMalloc(&d_ns, sz_i);
    cudaMemcpy(d_A, h_A, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ns, h_ns, sz_i, cudaMemcpyHostToDevice);

    int thr = 256, blk = (ncol + thr - 1) / thr;

    // Warmup
    kernel_tridiag_solve<<<blk, thr>>>(d_A, d_B, d_C, d_D, d_X, d_ns, ncol);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 50;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++)
        kernel_tridiag_solve<<<blk, thr>>>(d_A, d_B, d_C, d_D, d_X, d_ns, ncol);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    cudaMemcpy(h_X_gpu, d_X, sz_f, cudaMemcpyDeviceToHost);

    // Accuracy — verify Ax=D (residual check, not just comparison)
    float max_abs = 0, max_rel = 0, max_residual = 0;
    int nan_c = 0, fail_c = 0;
    for (int col = 0; col < ncol; col++) {
        int ns = h_ns[col];
        int base = col * MAX_SOIL;
        for (int k = 0; k < ns; k++) {
            // Check GPU vs CPU
            float ae = fabsf(h_X_gpu[base + k] - h_X_cpu[base + k]);
            if (ae > max_abs) max_abs = ae;
            if (fabsf(h_X_cpu[base + k]) > 1e-10f) {
                float re = ae / fabsf(h_X_cpu[base + k]);
                if (re > max_rel) max_rel = re;
                if (re > 1e-4f) fail_c++;
            }
            if (isnan(h_X_gpu[base + k])) nan_c++;

            // Residual check: A*x + B*x + C*x should equal D
            float res = h_B[base + k] * h_X_gpu[base + k]
                      + ((k > 0) ? h_A[base + k] * h_X_gpu[base + k - 1] : 0.0f)
                      + ((k < ns - 1) ? h_C[base + k] * h_X_gpu[base + k + 1] : 0.0f)
                      - h_D[base + k];
            if (fabsf(res) > max_residual) max_residual = fabsf(res);
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max abs: %.2e | Max rel: %.2e | Max residual: %.2e\n", max_abs, max_rel, max_residual);
    printf("  NaN: %d | >0.01%% err: %d/%d\n", nan_c, fail_c, ncol * MAX_SOIL);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-5f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-3f) ? "PASS (FP32 rounding)" : "FAIL");

    free(h_A); free(h_B); free(h_C); free(h_D); free(h_X_cpu); free(h_X_gpu); free(h_ns);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_X); cudaFree(d_ns);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    printf("================================================\n");
    printf("  NOAA-OWP Batched GPU Kernels\n");
    printf("  CFE Nash Cascade + NOAH-MP Tridiag Solver\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    // Nash Cascade benchmarks
    printf("========== CFE NASH CASCADE ==========\n\n");
    benchmark_nash(10000);
    benchmark_nash(100000);
    benchmark_nash(1000000);
    benchmark_nash(2700000);  // Full CONUS NWM

    // Tridiag solver benchmarks
    printf("========== NOAH-MP TRIDIAG SOLVER ==========\n\n");
    benchmark_tridiag(10000);
    benchmark_tridiag(100000);
    benchmark_tridiag(1000000);
    benchmark_tridiag(2700000);  // Full CONUS NWM

    printf("================================================\n");
    printf("  Notes:\n");
    printf("  - Both kernels run identical algorithms to CPU\n");
    printf("  - Speedup comes from batching millions of\n");
    printf("    independent catchments/columns on GPU\n");
    printf("  - No algorithmic changes, no approximations\n");
    printf("================================================\n");

    return 0;
}
