/*
 * GPU Benchmark: Fast exp() approximation for radiative transfer
 *
 * Compares our range-reduced Horner polynomial exp() against CUDA's
 * built-in __expf() (single precision) and exp() (double precision)
 * for the specific argument range used in atmospheric radiative transfer.
 *
 * The transmissivity computation trans = exp(-tau * D) is the single
 * most expensive operation in the RTE solver, called ~32 million times
 * per GFS radiation timestep.
 *
 * Compile: nvcc -O3 -arch=sm_86 fast_exp_benchmark.cu -o fast_exp_bench
 * Run:     ./fast_exp_bench
 *
 * Copyright 2026, NOAA/EPIC Optimization Project. BSD-3-Clause License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

// Problem dimensions (GFS-like)
#define NCOL  1024
#define NLAY  128
#define NGPT  256
#define NMUS  1
#define TOTAL (NCOL * NLAY * NGPT * NMUS)

// Number of benchmark iterations
#define NITERS 100

// RT cutoff: exp(-50) ~ 2e-22, negligible for radiation
#define RT_CUTOFF (-50.0)
#define RT_CUTOFF_F (-50.0f)

// ============================================================================
// DEVICE FUNCTIONS: Fast exp() approximations
// ============================================================================

// --- Double precision: 12th order Horner polynomial with range reduction ---
__device__ __forceinline__ double fast_exp_dp(double x) {
    if (x < RT_CUTOFF) return 0.0;

    // Range reduction: x = n * ln(2) + r
    const double LOG2E = 1.4426950408889634;
    const double LN2HI = 6.93147180369123816490e-01;
    const double LN2LO = 1.90821492927058500170e-10;

    double n_real = rint(x * LOG2E);
    int n = (int)n_real;
    double r = x - n_real * LN2HI - n_real * LN2LO;

    // 12th order Horner polynomial: exp(r) on [-ln2/2, ln2/2]
    double res = 1.0 + r * (1.0 + r * (0.5 + r * (
        1.6666666666666666e-01 + r * (4.1666666666666666e-02 + r * (
        8.3333333333333333e-03 + r * (1.3888888888888889e-03 + r * (
        1.9841269841269841e-04 + r * (2.4801587301587302e-05 + r * (
        2.7557319223985891e-06 + r * (2.7557319223985891e-07 + r * (
        2.5052108385441719e-08 + r * 2.0876756987868099e-09)))))))))));

    return scalbn(res, n);
}

// --- Single precision: 5th order Horner polynomial with range reduction ---
__device__ __forceinline__ float fast_exp_sp(float x) {
    if (x < RT_CUTOFF_F) return 0.0f;

    const float LOG2E = 1.44269504f;
    const float LN2   = 0.6931472f;

    float n_real = rintf(x * LOG2E);
    int n = (int)n_real;
    float r = x - n_real * LN2;

    // 5th order Horner: exp(r) on [-ln2/2, ln2/2]
    float res = 1.0f + r * (1.0f + r * (0.5f + r * (
        0.16666667f + r * (0.041666217f + r * 0.008333169f))));

    return scalbnf(res, n);
}

// ============================================================================
// KERNELS: Transmissivity computation (THE hot path)
// ============================================================================

// Baseline: standard CUDA exp()
__global__ void kernel_trans_baseline_dp(const double* __restrict__ tau,
                                          const double D,
                                          double* __restrict__ trans,
                                          int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        trans[idx] = exp(-tau[idx] * D);
    }
}

// Optimized: fast_exp_dp
__global__ void kernel_trans_fast_dp(const double* __restrict__ tau,
                                      const double D,
                                      double* __restrict__ trans,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        trans[idx] = fast_exp_dp(-tau[idx] * D);
    }
}

// Baseline: standard CUDA expf()
__global__ void kernel_trans_baseline_sp(const float* __restrict__ tau,
                                          const float D,
                                          float* __restrict__ trans,
                                          int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        trans[idx] = expf(-tau[idx] * D);
    }
}

// Optimized: fast_exp_sp
__global__ void kernel_trans_fast_sp(const float* __restrict__ tau,
                                      const float D,
                                      float* __restrict__ trans,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        trans[idx] = fast_exp_sp(-tau[idx] * D);
    }
}

// Ultra-fast: CUDA intrinsic __expf (uses hardware SFU, ~2.3 ULP)
__global__ void kernel_trans_intrinsic_sp(const float* __restrict__ tau,
                                           const float D,
                                           float* __restrict__ trans,
                                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        trans[idx] = __expf(-tau[idx] * D);
    }
}

// ============================================================================
// PARALLEL PREFIX SCAN KERNEL for adding method (NOVEL)
// ============================================================================

// 2x2 Mobius matrix: represents a layer's transformation on albedo
// The recurrence albedo[i] = R + T^2*x/(1-R*x) is the Mobius transformation
// f(x) = (ax+b)/(cx+d) with: a = T^2-R^2, b = R, c = -R, d = 1
// Composition = matrix multiplication (ASSOCIATIVE) => parallel scan works
struct Mat2x2 {
    float a, b, c, d;
};

// 2x2 matrix multiply: THE associative operator for the parallel scan
__device__ __forceinline__ Mat2x2 mat2_mul(Mat2x2 A, Mat2x2 B) {
    Mat2x2 C;
    C.a = A.a * B.a + A.b * B.c;
    C.b = A.a * B.b + A.b * B.d;
    C.c = A.c * B.a + A.d * B.c;
    C.d = A.c * B.b + A.d * B.d;
    return C;
}

// Apply Mobius transformation: f(x) = (a*x + b) / (c*x + d)
__device__ __forceinline__ float mat2_apply(Mat2x2 M, float x) {
    return (M.a * x + M.b) / (M.c * x + M.d);
}

// RIGHT-TO-LEFT suffix scan using 2x2 Mobius matrices
// Computes suffix[k] = M_k * M_{k+1} * ... * M_{n-1}
// Then albedo[k] = apply(suffix[k], albedo_sfc)
__global__ void kernel_adding_parallel_scan(
    const float* __restrict__ rdif,
    const float* __restrict__ tdif,
    const float* __restrict__ src_up_in,
    const float* __restrict__ src_dn_in,
    float* __restrict__ albedo_out,
    float albedo_sfc,
    int ncol, int nlay)
{
    int icol = blockIdx.x;
    if (icol >= ncol) return;

    extern __shared__ Mat2x2 shared[];
    int tid = threadIdx.x;

    // Load: convert (R, T) into 2x2 Mobius matrix
    // tid=0 is TOP layer, tid=nlay-1 is BOTTOM layer
    if (tid < nlay) {
        int idx = icol * nlay + tid;
        float R = rdif[idx];
        float T = tdif[idx];
        shared[tid].a = T * T - R * R;
        shared[tid].b = R;
        shared[tid].c = -R;
        shared[tid].d = 1.0f;
    }
    __syncthreads();

    // RIGHT-TO-LEFT Hillis-Steele inclusive suffix scan
    // shared[tid] = M_tid * M_{tid+1} * ... * M_{nlay-1}
    // Uses double-buffering via a second shared array to avoid race conditions
    // We allocate 2*nlay entries: [0..nlay-1] = read, [nlay..2*nlay-1] = write
    for (int stride = 1; stride < nlay; stride *= 2) {
        Mat2x2 val;
        if (tid < nlay) {
            int partner = tid + stride;
            if (partner < nlay) {
                val = mat2_mul(shared[tid], shared[partner]);
            } else {
                val = shared[tid];
            }
        }
        __syncthreads();
        if (tid < nlay) {
            shared[tid] = val;
        }
        __syncthreads();
    }

    // shared[tid] now holds the suffix product M_tid * ... * M_{nlay-1}
    // Apply to surface albedo to get albedo at level tid
    if (tid < nlay) {
        albedo_out[icol * (nlay + 1) + tid] = mat2_apply(shared[tid], albedo_sfc);
    }
    if (tid == 0) {
        albedo_out[icol * (nlay + 1) + nlay] = albedo_sfc;
    }
}

// Sequential reference: standard adding method for one column
__global__ void kernel_adding_sequential(
    const float* __restrict__ rdif,
    const float* __restrict__ tdif,
    const float* __restrict__ src_up_in,
    const float* __restrict__ src_dn_in,
    float* __restrict__ albedo_out,
    float albedo_sfc,
    int ncol, int nlay)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    if (icol >= ncol) return;

    // Bottom-up sequential scan
    albedo_out[icol * (nlay + 1) + nlay] = albedo_sfc;
    for (int ilay = nlay - 1; ilay >= 0; ilay--) {
        int idx = icol * nlay + ilay;
        float R = rdif[idx];
        float T = tdif[idx];
        float alb_below = albedo_out[icol * (nlay + 1) + ilay + 1];
        float denom = 1.0f / (1.0f - R * alb_below);
        albedo_out[icol * (nlay + 1) + ilay] = R + T * T * alb_below * denom;
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

float benchmark_kernel_dp(void (*kernel)(const double*, double, double*, int),
                           double* d_tau, double D, double* d_trans, int N, int niters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_tau, D, d_trans, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < niters; i++) {
        kernel<<<gridSize, blockSize>>>(d_tau, D, d_trans, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / niters;
}

float benchmark_kernel_sp(void (*kernel)(const float*, float, float*, int),
                           float* d_tau, float D, float* d_trans, int N, int niters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_tau, D, d_trans, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < niters; i++) {
        kernel<<<gridSize, blockSize>>>(d_tau, D, d_trans, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / niters;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("=============================================================\n");
    printf("  GPU Benchmark: Radiative Transfer exp() Optimization\n");
    printf("  RTX 3060 12GB | CUDA %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("=============================================================\n\n");

    int N = NCOL * NLAY * NGPT;
    printf("Problem size: %d x %d x %d = %d elements (%.1f MB fp64, %.1f MB fp32)\n",
           NCOL, NLAY, NGPT, N, N * 8.0 / 1e6, N * 4.0 / 1e6);
    printf("Benchmark iterations: %d\n\n", NITERS);

    // ---- Allocate and initialize ----
    double *h_tau_dp = (double*)malloc(N * sizeof(double));
    double *h_trans_ref_dp = (double*)malloc(N * sizeof(double));
    double *h_trans_fast_dp = (double*)malloc(N * sizeof(double));
    float  *h_tau_sp = (float*)malloc(N * sizeof(float));
    float  *h_trans_ref_sp = (float*)malloc(N * sizeof(float));
    float  *h_trans_fast_sp = (float*)malloc(N * sizeof(float));

    // Generate realistic optical depth distribution
    srand(42);
    for (int i = 0; i < N; i++) {
        // Mix of thin layers (90%) and thick layers (10%)
        if (i % 10 == 0)
            h_tau_dp[i] = 1.0 + 30.0 * (rand() / (double)RAND_MAX);
        else
            h_tau_dp[i] = 0.001 + 3.0 * (rand() / (double)RAND_MAX);
        h_tau_sp[i] = (float)h_tau_dp[i];
    }
    double D_dp = 1.66;  // Diffusivity factor
    float  D_sp = 1.66f;

    // ---- Device memory ----
    double *d_tau_dp, *d_trans_dp;
    float  *d_tau_sp, *d_trans_sp;
    CUDA_CHECK(cudaMalloc(&d_tau_dp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trans_dp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau_sp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trans_sp, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tau_dp, h_tau_dp, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau_sp, h_tau_sp, N * sizeof(float), cudaMemcpyHostToDevice));

    // ==================================================================
    // BENCHMARK 1: Double precision exp()
    // ==================================================================
    printf("--- Double Precision (fp64) ---\n");

    float t_base_dp = benchmark_kernel_dp(kernel_trans_baseline_dp, d_tau_dp, D_dp, d_trans_dp, N, NITERS);
    CUDA_CHECK(cudaMemcpy(h_trans_ref_dp, d_trans_dp, N * sizeof(double), cudaMemcpyDeviceToHost));

    float t_fast_dp = benchmark_kernel_dp(kernel_trans_fast_dp, d_tau_dp, D_dp, d_trans_dp, N, NITERS);
    CUDA_CHECK(cudaMemcpy(h_trans_fast_dp, d_trans_dp, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Check accuracy
    double max_rel_err_dp = 0.0;
    for (int i = 0; i < N; i++) {
        if (h_trans_ref_dp[i] > 1e-300) {
            double err = fabs(h_trans_fast_dp[i] - h_trans_ref_dp[i]) / h_trans_ref_dp[i];
            if (err > max_rel_err_dp) max_rel_err_dp = err;
        }
    }

    printf("  Baseline exp():     %.4f ms  (%.2f GB/s)\n", t_base_dp, 2.0*N*8.0/(t_base_dp*1e6));
    printf("  Fast exp():         %.4f ms  (%.2f GB/s)\n", t_fast_dp, 2.0*N*8.0/(t_fast_dp*1e6));
    printf("  Speedup:            %.2fx\n", t_base_dp / t_fast_dp);
    printf("  Max relative error: %.3e\n\n", max_rel_err_dp);

    // ==================================================================
    // BENCHMARK 2: Single precision exp()
    // ==================================================================
    printf("--- Single Precision (fp32) ---\n");

    float t_base_sp = benchmark_kernel_sp(kernel_trans_baseline_sp, d_tau_sp, D_sp, d_trans_sp, N, NITERS);
    CUDA_CHECK(cudaMemcpy(h_trans_ref_sp, d_trans_sp, N * sizeof(float), cudaMemcpyDeviceToHost));

    float t_fast_sp = benchmark_kernel_sp(kernel_trans_fast_sp, d_tau_sp, D_sp, d_trans_sp, N, NITERS);
    CUDA_CHECK(cudaMemcpy(h_trans_fast_sp, d_trans_sp, N * sizeof(float), cudaMemcpyDeviceToHost));

    float t_intr_sp = benchmark_kernel_sp(kernel_trans_intrinsic_sp, d_tau_sp, D_sp, d_trans_sp, N, NITERS);

    float max_rel_err_sp = 0.0f;
    for (int i = 0; i < N; i++) {
        if (h_trans_ref_sp[i] > 1e-30f) {
            float err = fabsf(h_trans_fast_sp[i] - h_trans_ref_sp[i]) / h_trans_ref_sp[i];
            if (err > max_rel_err_sp) max_rel_err_sp = err;
        }
    }

    printf("  Baseline expf():    %.4f ms  (%.2f GB/s)\n", t_base_sp, 2.0*N*4.0/(t_base_sp*1e6));
    printf("  Fast exp():         %.4f ms  (%.2f GB/s)\n", t_fast_sp, 2.0*N*4.0/(t_fast_sp*1e6));
    printf("  __expf() intrinsic: %.4f ms  (%.2f GB/s)\n", t_intr_sp, 2.0*N*4.0/(t_intr_sp*1e6));
    printf("  Speedup (fast/base):  %.2fx\n", t_base_sp / t_fast_sp);
    printf("  Speedup (intr/base):  %.2fx\n", t_base_sp / t_intr_sp);
    printf("  Max relative error:   %.3e\n\n", max_rel_err_sp);

    // ==================================================================
    // BENCHMARK 3: Parallel prefix scan for adding method
    // ==================================================================
    printf("--- Parallel Prefix Scan (adding method) ---\n");
    printf("  ncol=%d, nlay=%d\n", NCOL, NLAY);

    float *d_rdif, *d_tdif, *d_src_up, *d_src_dn, *d_albedo_par, *d_albedo_seq;
    int nlay_elem = NCOL * NLAY;
    int nlev_elem = NCOL * (NLAY + 1);

    CUDA_CHECK(cudaMalloc(&d_rdif,       nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tdif,       nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_up,     nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_dn,     nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_albedo_par, nlev_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_albedo_seq, nlev_elem * sizeof(float)));

    // Initialize with realistic atmospheric values
    float *h_rdif = (float*)malloc(nlay_elem * sizeof(float));
    float *h_tdif = (float*)malloc(nlay_elem * sizeof(float));
    float *h_src_up = (float*)malloc(nlay_elem * sizeof(float));
    float *h_src_dn = (float*)malloc(nlay_elem * sizeof(float));
    for (int i = 0; i < nlay_elem; i++) {
        h_rdif[i] = 0.01f + 0.1f * (rand() / (float)RAND_MAX);   // Small reflection
        h_tdif[i] = 0.8f + 0.19f * (rand() / (float)RAND_MAX);   // High transmission
        h_src_up[i] = 10.0f * (rand() / (float)RAND_MAX);
        h_src_dn[i] = 10.0f * (rand() / (float)RAND_MAX);
    }
    float albedo_sfc = 0.3f;

    CUDA_CHECK(cudaMemcpy(d_rdif,   h_rdif,   nlay_elem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tdif,   h_tdif,   nlay_elem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_up, h_src_up, nlay_elem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_dn, h_src_dn, nlay_elem * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark sequential adding
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Sequential: one thread per column
    int seq_block = 256;
    int seq_grid = (NCOL + seq_block - 1) / seq_block;

    kernel_adding_sequential<<<seq_grid, seq_block>>>(d_rdif, d_tdif, d_src_up, d_src_dn,
                                                       d_albedo_seq, albedo_sfc, NCOL, NLAY);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NITERS; i++) {
        kernel_adding_sequential<<<seq_grid, seq_block>>>(d_rdif, d_tdif, d_src_up, d_src_dn,
                                                           d_albedo_seq, albedo_sfc, NCOL, NLAY);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_seq;
    CUDA_CHECK(cudaEventElapsedTime(&t_seq, start, stop));
    t_seq /= NITERS;

    // Parallel prefix: one block per column, nlay threads per block
    int par_shared = NLAY * sizeof(Mat2x2);
    kernel_adding_parallel_scan<<<NCOL, NLAY, par_shared>>>(d_rdif, d_tdif, d_src_up, d_src_dn,
                                                             d_albedo_par, albedo_sfc, NCOL, NLAY);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NITERS; i++) {
        kernel_adding_parallel_scan<<<NCOL, NLAY, par_shared>>>(d_rdif, d_tdif, d_src_up, d_src_dn,
                                                                 d_albedo_par, albedo_sfc, NCOL, NLAY);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_par;
    CUDA_CHECK(cudaEventElapsedTime(&t_par, start, stop));
    t_par /= NITERS;

    // Check accuracy: compare parallel vs sequential
    float *h_albedo_seq = (float*)malloc(nlev_elem * sizeof(float));
    float *h_albedo_par = (float*)malloc(nlev_elem * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_albedo_seq, d_albedo_seq, nlev_elem * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_albedo_par, d_albedo_par, nlev_elem * sizeof(float), cudaMemcpyDeviceToHost));

    float max_scan_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < nlev_elem; i++) {
        if (fabsf(h_albedo_seq[i]) > 1e-10f) {
            float err = fabsf(h_albedo_par[i] - h_albedo_seq[i]) / fabsf(h_albedo_seq[i]);
            if (err > max_scan_err) max_scan_err = err;
            if (err > 1e-4f) err_count++;
        }
    }

    printf("  Sequential:         %.4f ms\n", t_seq);
    printf("  Parallel scan:      %.4f ms\n", t_par);
    printf("  Speedup:            %.2fx\n", t_seq / t_par);
    printf("  Max relative error: %.3e\n", max_scan_err);
    printf("  Error count (>1e-4): %d / %d\n\n", err_count, nlev_elem);

    // ==================================================================
    // SUMMARY
    // ==================================================================
    printf("=============================================================\n");
    printf("  SUMMARY\n");
    printf("=============================================================\n");
    printf("  Fast exp (fp64):        %.2fx speedup, %.1e max error\n", t_base_dp/t_fast_dp, max_rel_err_dp);
    printf("  Fast exp (fp32):        %.2fx speedup, %.1e max error\n", t_base_sp/t_fast_sp, max_rel_err_sp);
    printf("  __expf intrinsic:       %.2fx speedup (vs expf)\n", t_base_sp/t_intr_sp);
    printf("  Parallel prefix scan:   %.2fx speedup, %.1e max error\n", t_seq/t_par, max_scan_err);
    printf("  fp64 -> fp32 alone:     %.2fx speedup (memory bandwidth)\n", t_base_dp/t_base_sp);
    printf("=============================================================\n");

    // Cleanup
    free(h_tau_dp); free(h_trans_ref_dp); free(h_trans_fast_dp);
    free(h_tau_sp); free(h_trans_ref_sp); free(h_trans_fast_sp);
    free(h_rdif); free(h_tdif); free(h_src_up); free(h_src_dn);
    free(h_albedo_seq); free(h_albedo_par);
    cudaFree(d_tau_dp); cudaFree(d_trans_dp);
    cudaFree(d_tau_sp); cudaFree(d_trans_sp);
    cudaFree(d_rdif); cudaFree(d_tdif); cudaFree(d_src_up); cudaFree(d_src_dn);
    cudaFree(d_albedo_par); cudaFree(d_albedo_seq);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
