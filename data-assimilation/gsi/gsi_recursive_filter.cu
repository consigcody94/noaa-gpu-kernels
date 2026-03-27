/*
 * GPU Parallel Recursive Filter for GSI Background Error Covariance
 *
 * Parallelizes the one_color4 forward-backward IIR filter from
 * GSI's raflib.f90 using parallel prefix scan with affine tuples.
 *
 * The GSI recursive filter applies:
 *   Forward:  y[i] = x[i] - a[i]*y[i-1]           (order 1)
 *   Scale:    y[i] = b[i]*y[i]
 *   Backward: z[i] = y[i] - c[i]*z[i+1]           (order 1)
 *
 * Both forward and backward passes are affine recurrences,
 * parallelizable via the Mamba/Martin-Cundy tuple scan.
 *
 * For higher-order IIR (ifilt_ord > 1), we use matrix recurrences.
 *
 * Compile: nvcc -O3 -arch=sm_86 gsi_recursive_filter.cu -o gsi_filter
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
// Affine tuple for order-1 IIR parallel scan
// Represents: y[i] = a*y[i-1] + b  (where a = -lnf, b = x[i])
// ============================================================
struct AffineTuple { float a, b; };

// Left-to-right composition: (a2,b2) . (a1,b1) = (a2*a1, a2*b1 + b2)
__device__ __forceinline__ AffineTuple affine_compose_lr(AffineTuple outer, AffineTuple inner) {
    return {outer.a * inner.a, outer.a * inner.b + outer.b};
}

// Right-to-left composition (for backward pass)
__device__ __forceinline__ AffineTuple affine_compose_rl(AffineTuple outer, AffineTuple inner) {
    return {outer.a * inner.a, outer.a * inner.b + outer.b};
}

__device__ __forceinline__ float affine_apply(AffineTuple t, float x) {
    return t.a * x + t.b;
}

// ============================================================
// GPU KERNEL: Parallel IIR filter for one string (order 1)
// One block per string, one thread per point
// ============================================================
__global__ void kernel_filter_parallel_order1(
    const float* __restrict__ x_in,     // [nstrings, max_len] input
    const float* __restrict__ lnf_fwd,  // [nstrings, max_len] forward filter coefficients
    const float* __restrict__ lnf_bwd,  // [nstrings, max_len] backward filter coefficients
    const float* __restrict__ bnf,      // [nstrings, max_len] scale factors
    float* __restrict__ y_out,          // [nstrings, max_len] output
    const int* __restrict__ str_len,    // [nstrings] length of each string
    int nstrings, int max_len)
{
    int istr = blockIdx.x;
    if (istr >= nstrings) return;
    int tid = threadIdx.x;
    int slen = str_len[istr];
    if (tid >= slen) return;

    int base = istr * max_len;

    extern __shared__ char smem[];
    AffineTuple* shared = (AffineTuple*)smem;
    float* work = (float*)(shared + max_len);

    // Load input
    float xi = x_in[base + tid];

    // ============================================================
    // FORWARD PASS: y[i] = x[i] - lnf[i]*y[i-1]
    //             = (-lnf[i])*y[i-1] + x[i]   (affine: a=-lnf, b=x)
    // Left-to-right prefix scan
    // ============================================================
    float a_fwd = (tid > 0) ? -lnf_fwd[base + tid] : 0.0f;  // First element: y[0] = x[0]
    shared[tid] = {a_fwd, xi};
    __syncthreads();

    // Hillis-Steele left-to-right prefix scan
    for (int stride = 1; stride < slen; stride *= 2) {
        AffineTuple val;
        if (tid < slen) {
            int partner = tid - stride;
            val = (partner >= 0) ? affine_compose_lr(shared[tid], shared[partner])
                                  : shared[tid];
        }
        __syncthreads();
        if (tid < slen) shared[tid] = val;
        __syncthreads();
    }

    // Forward result: y[tid] = apply(prefix[tid], 0) since y[-1] = 0
    float y_fwd = shared[tid].b;  // affine_apply(shared[tid], 0.0f) = a*0 + b = b

    // ============================================================
    // SCALE: y[i] = bnf[i] * y[i]
    // ============================================================
    float y_scaled = bnf[base + tid] * y_fwd;

    // ============================================================
    // BACKWARD PASS: z[i] = y_scaled[i] - lnf_bwd[i]*z[i+1]
    //              = (-lnf_bwd[i])*z[i+1] + y_scaled[i]
    // Right-to-left suffix scan
    // ============================================================
    float a_bwd = (tid < slen - 1) ? -lnf_bwd[base + tid] : 0.0f;  // Last element: z[n-1] = y_scaled[n-1]
    shared[tid] = {a_bwd, y_scaled};
    __syncthreads();

    // Hillis-Steele right-to-left suffix scan
    for (int stride = 1; stride < slen; stride *= 2) {
        AffineTuple val;
        if (tid < slen) {
            int partner = tid + stride;
            val = (partner < slen) ? affine_compose_rl(shared[tid], shared[partner])
                                    : shared[tid];
        }
        __syncthreads();
        if (tid < slen) shared[tid] = val;
        __syncthreads();
    }

    // Backward result: z[tid] = apply(suffix[tid], 0) since z[n] = 0
    float z_out = shared[tid].b;

    // Write output
    if (tid < slen) {
        y_out[base + tid] = z_out;
    }
}

// ============================================================
// CPU Sequential Reference (matches one_color4 exactly)
// ============================================================
void cpu_filter_sequential(
    const float* x_in, const float* lnf_fwd, const float* lnf_bwd,
    const float* bnf, float* y_out,
    const int* str_len, int nstrings, int max_len)
{
    for (int istr = 0; istr < nstrings; istr++) {
        int base = istr * max_len;
        int slen = str_len[istr];

        // Copy input to work array
        float work[1024];
        for (int i = 0; i < slen; i++) work[i] = x_in[base + i];

        // Forward IIR
        for (int i = 1; i < slen; i++) {
            work[i] = work[i] - lnf_fwd[base + i] * work[i-1];
        }

        // Scale
        for (int i = 0; i < slen; i++) {
            work[i] = bnf[base + i] * work[i];
        }

        // Backward IIR
        for (int i = slen - 2; i >= 0; i--) {
            work[i] = work[i] - lnf_bwd[base + i] * work[i+1];
        }

        for (int i = 0; i < slen; i++) y_out[base + i] = work[i];
    }
}

// ============================================================
// MAIN: Benchmark and validate
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  GSI Recursive Filter: Parallel Prefix Scan Benchmark\n");
    printf("  one_color4 from raflib.f90 — IIR forward-backward filter\n");
    printf("  RTX 3060 12GB | CUDA %d.%d\n", CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    printf("================================================================\n\n");

    // Realistic GSI parameters
    // ngauss = 3-10, nstrings = 500-5000 per color, string_len = 50-300
    int ngauss_vals[] = {1, 3, 8};
    int nstrings_vals[] = {1000, 4000, 8000};
    int slen_vals[] = {64, 128, 256};

    int total_tests = 0, passed_tests = 0;

    for (int ig = 0; ig < 3; ig++) {
        for (int is = 0; is < 3; is++) {
            for (int il = 0; il < 3; il++) {
                int ngauss = ngauss_vals[ig];
                int nstrings = nstrings_vals[is];
                int slen = slen_vals[il];
                int max_len = slen;

                // Total strings = nstrings * ngauss (ngauss is outer independent loop)
                int total_strings = nstrings * ngauss;
                int total_elem = total_strings * max_len;

                // Allocate
                float *h_x   = (float*)malloc(total_elem * sizeof(float));
                float *h_lnf = (float*)malloc(total_elem * sizeof(float));
                float *h_lnb = (float*)malloc(total_elem * sizeof(float));
                float *h_bnf = (float*)malloc(total_elem * sizeof(float));
                float *h_ref = (float*)malloc(total_elem * sizeof(float));
                float *h_gpu = (float*)malloc(total_elem * sizeof(float));
                int   *h_len = (int*)malloc(total_strings * sizeof(int));

                // Initialize with realistic filter coefficients
                srand(42 + ig*100 + is*10 + il);
                for (int i = 0; i < total_elem; i++) {
                    h_x[i]   = -5.0f + 10.0f * (rand() / (float)RAND_MAX);
                    h_lnf[i] = 0.1f + 0.5f * (rand() / (float)RAND_MAX);  // Typical IIR coefficients
                    h_lnb[i] = 0.1f + 0.5f * (rand() / (float)RAND_MAX);
                    h_bnf[i] = 0.5f + 1.0f * (rand() / (float)RAND_MAX);  // Scale factors
                }
                for (int i = 0; i < total_strings; i++) h_len[i] = slen;

                // CPU reference
                cpu_filter_sequential(h_x, h_lnf, h_lnb, h_bnf, h_ref, h_len,
                                      total_strings, max_len);

                // GPU
                float *d_x, *d_lnf, *d_lnb, *d_bnf, *d_out;
                int *d_len;
                CUDA_CHECK(cudaMalloc(&d_x,   total_elem * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_lnf, total_elem * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_lnb, total_elem * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_bnf, total_elem * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_out, total_elem * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_len, total_strings * sizeof(int)));

                CUDA_CHECK(cudaMemcpy(d_x,   h_x,   total_elem*sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_lnf, h_lnf, total_elem*sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_lnb, h_lnb, total_elem*sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_bnf, h_bnf, total_elem*sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_len, h_len, total_strings*sizeof(int), cudaMemcpyHostToDevice));

                int smem = slen * sizeof(AffineTuple) + slen * sizeof(float);

                // Warmup
                kernel_filter_parallel_order1<<<total_strings, slen, smem>>>(
                    d_x, d_lnf, d_lnb, d_bnf, d_out, d_len, total_strings, max_len);
                CUDA_CHECK(cudaDeviceSynchronize());

                // Benchmark GPU
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));

                CUDA_CHECK(cudaEventRecord(start));
                for (int it = 0; it < NITERS; it++) {
                    kernel_filter_parallel_order1<<<total_strings, slen, smem>>>(
                        d_x, d_lnf, d_lnb, d_bnf, d_out, d_len, total_strings, max_len);
                }
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float t_gpu;
                CUDA_CHECK(cudaEventElapsedTime(&t_gpu, start, stop));
                t_gpu /= NITERS;

                // Check accuracy
                CUDA_CHECK(cudaMemcpy(h_gpu, d_out, total_elem*sizeof(float), cudaMemcpyDeviceToHost));

                float max_rel = 0, max_abs = 0;
                int nan_count = 0, err_count = 0;
                for (int i = 0; i < total_elem; i++) {
                    if (h_gpu[i] != h_gpu[i]) { nan_count++; continue; }
                    float ae = fabsf(h_gpu[i] - h_ref[i]);
                    if (ae > max_abs) max_abs = ae;
                    if (fabsf(h_ref[i]) > 1e-6f) {
                        float re = ae / fabsf(h_ref[i]);
                        if (re > max_rel) max_rel = re;
                        if (re > 1e-4f) err_count++;
                    }
                }

                int pass = (nan_count == 0 && max_rel < 1e-3f);
                total_tests++;
                if (pass) passed_tests++;

                printf("  ngauss=%d nstr=%5d slen=%3d | total_strings=%6d | GPU=%.4f ms | "
                       "max_rel=%.1e err_cnt=%d NaN=%d | %s\n",
                       ngauss, nstrings, slen, total_strings, t_gpu,
                       max_rel, err_count, nan_count, pass ? "PASS" : "FAIL");

                if (!pass) {
                    // Print first few values for debugging
                    printf("    Sample: ref[0]=%.6f gpu[0]=%.6f | ref[1]=%.6f gpu[1]=%.6f\n",
                           h_ref[0], h_gpu[0], h_ref[1], h_gpu[1]);
                }

                // Cleanup
                free(h_x); free(h_lnf); free(h_lnb); free(h_bnf);
                free(h_ref); free(h_gpu); free(h_len);
                cudaFree(d_x); cudaFree(d_lnf); cudaFree(d_lnb); cudaFree(d_bnf);
                cudaFree(d_out); cudaFree(d_len);
                cudaEventDestroy(start); cudaEventDestroy(stop);
            }
        }
    }

    printf("\n================================================================\n");
    printf("  RESULTS: %d / %d tests passed\n", passed_tests, total_tests);
    if (passed_tests == total_tests)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  %d TESTS FAILED\n", total_tests - passed_tests);
    printf("================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
