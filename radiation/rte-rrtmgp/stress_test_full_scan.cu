/*
 * Stress test: Full two-stream parallel scan solver
 * Tests edge cases, extreme regimes, and physical consistency
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_NLAY 256
#define MAX_NLEV (MAX_NLAY + 1)

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } }

// ---- Scan data structures and operators (same as main benchmark) ----
struct Mat2x2 { float a, b, c, d; };
struct AffineTuple { float a, b; };

__device__ __forceinline__ Mat2x2 mat2_mul(Mat2x2 A, Mat2x2 B) {
    return {A.a*B.a + A.b*B.c, A.a*B.b + A.b*B.d,
            A.c*B.a + A.d*B.c, A.c*B.b + A.d*B.d};
}
__device__ __forceinline__ float mat2_apply(Mat2x2 M, float x) {
    return (M.a*x + M.b) / (M.c*x + M.d);
}
__device__ __forceinline__ AffineTuple affine_compose(AffineTuple outer, AffineTuple inner) {
    return {outer.a * inner.a, outer.a * inner.b + outer.b};
}
__device__ __forceinline__ float affine_apply(AffineTuple t, float x) {
    return t.a * x + t.b;
}

// ---- CPU sequential reference (golden truth) ----
void cpu_adding_sequential(
    const float* rdif, const float* tdif,
    const float* src_up, const float* src_dn,
    float sfc_alb, float sfc_src, float inc_flux,
    float* flux_up, float* flux_dn,
    int nlay)
{
    int nlev = nlay + 1;
    float albedo[MAX_NLEV], src[MAX_NLEV], denom[MAX_NLAY];

    // Boundary (bottom)
    albedo[nlay] = sfc_alb;
    src[nlay] = sfc_src;

    // Pass 1: bottom-up
    for (int i = nlay - 1; i >= 0; i--) {
        denom[i] = 1.0f / (1.0f - rdif[i] * albedo[i+1]);
        albedo[i] = rdif[i] + tdif[i]*tdif[i] * albedo[i+1] * denom[i];
        src[i] = src_up[i] + tdif[i] * denom[i] * (src[i+1] + albedo[i+1]*src_dn[i]);
    }

    // Pass 2: top-down
    flux_dn[0] = inc_flux;
    flux_up[0] = albedo[0] * flux_dn[0] + src[0];
    for (int i = 0; i < nlay; i++) {
        flux_dn[i+1] = (tdif[i]*flux_dn[i] + rdif[i]*src[i+1] + src_dn[i]) * denom[i];
        flux_up[i+1] = albedo[i+1] * flux_dn[i+1] + src[i+1];
    }
}

// ---- GPU parallel solver (single column for stress testing) ----
__global__ void kernel_parallel_single_col(
    const float* __restrict__ rdif, const float* __restrict__ tdif,
    const float* __restrict__ src_up, const float* __restrict__ src_dn,
    float sfc_alb, float sfc_src, float inc_flux,
    float* __restrict__ flux_up, float* __restrict__ flux_dn,
    int nlay)
{
    int tid = threadIdx.x;
    int nlev = nlay + 1;

    extern __shared__ char smem[];
    Mat2x2* shared_mat = (Mat2x2*)smem;

    float R_local = 0, T_local = 0;
    if (tid < nlay) {
        R_local = rdif[tid];
        T_local = tdif[tid];
        shared_mat[tid] = {T_local*T_local - R_local*R_local, R_local,
                           -R_local, 1.0f};
    }
    __syncthreads();

    // SCAN 1: Albedo (right-to-left suffix)
    for (int stride = 1; stride < nlay; stride *= 2) {
        Mat2x2 val;
        if (tid < nlay) {
            int p = tid + stride;
            val = (p < nlay) ? mat2_mul(shared_mat[tid], shared_mat[p]) : shared_mat[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_mat[tid] = val;
        __syncthreads();
    }

    __shared__ float albedo[MAX_NLEV];
    if (tid < nlay) albedo[tid] = mat2_apply(shared_mat[tid], sfc_alb);
    if (tid == 0) albedo[nlay] = sfc_alb;
    __syncthreads();

    // SCAN 2: Source (right-to-left affine suffix)
    AffineTuple* shared_aff = (AffineTuple*)smem;
    float denom_local = 0;
    if (tid < nlay) {
        denom_local = 1.0f / (1.0f - R_local * albedo[tid + 1]);
        float A = T_local * denom_local;
        float B = src_up[tid] + T_local * denom_local * albedo[tid+1] * src_dn[tid];
        shared_aff[tid] = {A, B};
    }
    __syncthreads();

    for (int stride = 1; stride < nlay; stride *= 2) {
        AffineTuple val;
        if (tid < nlay) {
            int p = tid + stride;
            val = (p < nlay) ? affine_compose(shared_aff[tid], shared_aff[p]) : shared_aff[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_aff[tid] = val;
        __syncthreads();
    }

    __shared__ float src[MAX_NLEV];
    if (tid < nlay) src[tid] = affine_apply(shared_aff[tid], sfc_src);
    if (tid == 0) src[nlay] = sfc_src;
    __syncthreads();

    // SCAN 3: Flux_dn (left-to-right affine prefix)
    AffineTuple* shared_flux = (AffineTuple*)smem;
    if (tid < nlay) {
        float C = T_local * denom_local;
        float D = (R_local * src[tid + 1] + src_dn[tid]) * denom_local;
        shared_flux[tid] = {C, D};
    }
    __syncthreads();

    for (int stride = 1; stride < nlay; stride *= 2) {
        AffineTuple val;
        if (tid < nlay) {
            int p = tid - stride;
            val = (p >= 0) ? affine_compose(shared_flux[tid], shared_flux[p]) : shared_flux[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_flux[tid] = val;
        __syncthreads();
    }

    // Write flux_dn
    if (tid == 0) flux_dn[0] = inc_flux;
    if (tid < nlay) flux_dn[tid + 1] = affine_apply(shared_flux[tid], inc_flux);
    __syncthreads();

    // Pointwise: flux_up
    if (tid < nlay) {
        flux_up[tid] = albedo[tid] * flux_dn[tid] + src[tid];
    }
    if (tid == 0) {
        flux_up[nlay] = albedo[nlay] * flux_dn[nlay] + src[nlay];
    }
}

// ---- Test runner ----
int run_test(const char* name, int nlay,
             float* h_rdif, float* h_tdif, float* h_src_up, float* h_src_dn,
             float sfc_alb, float sfc_src, float inc_flux)
{
    int nlev = nlay + 1;

    // CPU reference
    float cpu_up[MAX_NLEV], cpu_dn[MAX_NLEV];
    cpu_adding_sequential(h_rdif, h_tdif, h_src_up, h_src_dn,
                          sfc_alb, sfc_src, inc_flux, cpu_up, cpu_dn, nlay);

    // GPU parallel
    float *d_rdif, *d_tdif, *d_src_up, *d_src_dn, *d_flux_up, *d_flux_dn;
    CUDA_CHECK(cudaMalloc(&d_rdif, nlay*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tdif, nlay*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_up, nlay*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_dn, nlay*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_up, nlev*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_dn, nlev*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rdif, h_rdif, nlay*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tdif, h_tdif, nlay*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_up, h_src_up, nlay*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_dn, h_src_dn, nlay*sizeof(float), cudaMemcpyHostToDevice));

    int smem = nlay * sizeof(Mat2x2) + 2 * nlev * sizeof(float);
    kernel_parallel_single_col<<<1, nlay, smem>>>(
        d_rdif, d_tdif, d_src_up, d_src_dn,
        sfc_alb, sfc_src, inc_flux, d_flux_up, d_flux_dn, nlay);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_up[MAX_NLEV], gpu_dn[MAX_NLEV];
    CUDA_CHECK(cudaMemcpy(gpu_up, d_flux_up, nlev*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_dn, d_flux_dn, nlev*sizeof(float), cudaMemcpyDeviceToHost));

    // Check accuracy
    float max_rel_up = 0, max_rel_dn = 0;
    float max_abs_up = 0, max_abs_dn = 0;
    int nan_count = 0, inf_count = 0, neg_count = 0;

    for (int i = 0; i < nlev; i++) {
        if (gpu_up[i] != gpu_up[i] || gpu_dn[i] != gpu_dn[i]) nan_count++;
        if (isinf(gpu_up[i]) || isinf(gpu_dn[i])) inf_count++;
        if (gpu_up[i] < -1e-6f || gpu_dn[i] < -1e-6f) neg_count++;

        float au = fabsf(gpu_up[i] - cpu_up[i]);
        float ad = fabsf(gpu_dn[i] - cpu_dn[i]);
        if (au > max_abs_up) max_abs_up = au;
        if (ad > max_abs_dn) max_abs_dn = ad;

        if (fabsf(cpu_up[i]) > 1e-6f) {
            float r = au / fabsf(cpu_up[i]);
            if (r > max_rel_up) max_rel_up = r;
        }
        if (fabsf(cpu_dn[i]) > 1e-6f) {
            float r = ad / fabsf(cpu_dn[i]);
            if (r > max_rel_dn) max_rel_dn = r;
        }
    }

    int passed = (nan_count == 0 && inf_count == 0 && neg_count == 0 &&
                  max_rel_up < 1e-3f && max_rel_dn < 1e-3f);

    printf("  %-35s nlay=%3d | up_err=%.1e dn_err=%.1e | NaN=%d Inf=%d Neg=%d | %s\n",
           name, nlay, max_rel_up, max_rel_dn, nan_count, inf_count, neg_count,
           passed ? "PASS" : "FAIL");

    if (!passed) {
        // Print first few levels for debugging
        printf("    Level 0: cpu_up=%.6f gpu_up=%.6f | cpu_dn=%.6f gpu_dn=%.6f\n",
               cpu_up[0], gpu_up[0], cpu_dn[0], gpu_dn[0]);
        printf("    Level 1: cpu_up=%.6f gpu_up=%.6f | cpu_dn=%.6f gpu_dn=%.6f\n",
               cpu_up[1], gpu_up[1], cpu_dn[1], gpu_dn[1]);
        int mid = nlay/2;
        printf("    Level %d: cpu_up=%.6f gpu_up=%.6f | cpu_dn=%.6f gpu_dn=%.6f\n",
               mid, cpu_up[mid], gpu_up[mid], cpu_dn[mid], gpu_dn[mid]);
        printf("    Level %d: cpu_up=%.6f gpu_up=%.6f | cpu_dn=%.6f gpu_dn=%.6f\n",
               nlay, cpu_up[nlay], gpu_up[nlay], cpu_dn[nlay], gpu_dn[nlay]);
    }

    cudaFree(d_rdif); cudaFree(d_tdif); cudaFree(d_src_up); cudaFree(d_src_dn);
    cudaFree(d_flux_up); cudaFree(d_flux_dn);
    return passed;
}

int main() {
    printf("================================================================\n");
    printf("  STRESS TEST: Full Two-Stream Parallel Scan\n");
    printf("================================================================\n\n");

    int total = 0, passed = 0;
    float rdif[MAX_NLAY], tdif[MAX_NLAY], src_up[MAX_NLAY], src_dn[MAX_NLAY];

    // ---- TEST 1: Standard clear-sky atmosphere ----
    srand(42);
    for (int i = 0; i < 128; i++) {
        rdif[i] = 0.01f + 0.1f * (rand()/(float)RAND_MAX);
        tdif[i] = 0.8f + 0.19f * (rand()/(float)RAND_MAX);
        src_up[i] = 10.0f + 20.0f * (rand()/(float)RAND_MAX);
        src_dn[i] = 10.0f + 20.0f * (rand()/(float)RAND_MAX);
    }
    total++; passed += run_test("Clear-sky (R<0.1, T>0.8)", 128,
                                 rdif, tdif, src_up, src_dn, 0.3f, 100.0f, 0.0f);

    // ---- TEST 2: Thick cloud layers ----
    for (int i = 0; i < 128; i++) {
        rdif[i] = 0.3f + 0.4f * (rand()/(float)RAND_MAX);  // R up to 0.7
        tdif[i] = 0.1f + 0.3f * (rand()/(float)RAND_MAX);  // T as low as 0.1
        src_up[i] = 50.0f * (rand()/(float)RAND_MAX);
        src_dn[i] = 50.0f * (rand()/(float)RAND_MAX);
    }
    total++; passed += run_test("Thick clouds (R<0.7, T>0.1)", 128,
                                 rdif, tdif, src_up, src_dn, 0.3f, 100.0f, 0.0f);

    // ---- TEST 3: Near-conservative scattering ----
    for (int i = 0; i < 64; i++) {
        rdif[i] = 0.45f + 0.04f * (rand()/(float)RAND_MAX);  // R near 0.5
        tdif[i] = 0.45f + 0.04f * (rand()/(float)RAND_MAX);  // T near 0.5
        src_up[i] = 10.0f;
        src_dn[i] = 10.0f;
    }
    total++; passed += run_test("Near-conservative (R~0.47, T~0.47)", 64,
                                 rdif, tdif, src_up, src_dn, 0.5f, 50.0f, 0.0f);

    // ---- TEST 4: Transparent atmosphere (R~0, T~1) ----
    for (int i = 0; i < 128; i++) {
        rdif[i] = 0.001f * (rand()/(float)RAND_MAX);
        tdif[i] = 0.99f + 0.01f * (rand()/(float)RAND_MAX);
        src_up[i] = 5.0f + 10.0f * (rand()/(float)RAND_MAX);
        src_dn[i] = 5.0f + 10.0f * (rand()/(float)RAND_MAX);
    }
    total++; passed += run_test("Transparent (R~0, T~1)", 128,
                                 rdif, tdif, src_up, src_dn, 0.1f, 20.0f, 0.0f);

    // ---- TEST 5: Non-zero incident flux (shortwave) ----
    for (int i = 0; i < 128; i++) {
        rdif[i] = 0.02f + 0.15f * (rand()/(float)RAND_MAX);
        tdif[i] = 0.7f + 0.28f * (rand()/(float)RAND_MAX);
        src_up[i] = 0.0f;  // No thermal emission in SW
        src_dn[i] = 0.0f;
    }
    total++; passed += run_test("SW with inc_flux=340 W/m2", 128,
                                 rdif, tdif, src_up, src_dn, 0.3f, 0.0f, 340.0f);

    // ---- TEST 6: Zero surface albedo ----
    for (int i = 0; i < 128; i++) {
        rdif[i] = 0.05f;
        tdif[i] = 0.9f;
        src_up[i] = 15.0f;
        src_dn[i] = 15.0f;
    }
    total++; passed += run_test("Zero surface albedo", 128,
                                 rdif, tdif, src_up, src_dn, 0.0f, 100.0f, 0.0f);

    // ---- TEST 7: High surface albedo (ice) ----
    total++; passed += run_test("High surface albedo=0.9", 128,
                                 rdif, tdif, src_up, src_dn, 0.9f, 100.0f, 0.0f);

    // ---- TEST 8: Surface albedo = 1 (perfect reflector) ----
    total++; passed += run_test("Perfect reflector sfc_alb=1", 128,
                                 rdif, tdif, src_up, src_dn, 1.0f, 0.0f, 100.0f);

    // ---- TEST 9: Very few layers (nlay=4) ----
    total++; passed += run_test("Minimal layers (nlay=4)", 4,
                                 rdif, tdif, src_up, src_dn, 0.3f, 100.0f, 0.0f);

    // ---- TEST 10: nlay=16 ----
    total++; passed += run_test("Small (nlay=16)", 16,
                                 rdif, tdif, src_up, src_dn, 0.3f, 50.0f, 10.0f);

    // ---- TEST 11: nlay=32 ----
    total++; passed += run_test("Medium (nlay=32)", 32,
                                 rdif, tdif, src_up, src_dn, 0.3f, 50.0f, 10.0f);

    // ---- TEST 12: nlay=64 ----
    total++; passed += run_test("Standard (nlay=64)", 64,
                                 rdif, tdif, src_up, src_dn, 0.3f, 50.0f, 10.0f);

    // ---- TEST 13: nlay=256 (maximum) ----
    for (int i = 0; i < 256; i++) {
        rdif[i] = 0.02f + 0.1f * (rand()/(float)RAND_MAX);
        tdif[i] = 0.8f + 0.19f * (rand()/(float)RAND_MAX);
        src_up[i] = 10.0f + 15.0f * (rand()/(float)RAND_MAX);
        src_dn[i] = 10.0f + 15.0f * (rand()/(float)RAND_MAX);
    }
    total++; passed += run_test("Large (nlay=256)", 256,
                                 rdif, tdif, src_up, src_dn, 0.3f, 80.0f, 0.0f);

    // ---- TEST 14: Mixed cloud/clear with SW incident flux ----
    for (int i = 0; i < 128; i++) {
        if (i >= 30 && i <= 50) {
            // Cloud layer
            rdif[i] = 0.4f + 0.2f * (rand()/(float)RAND_MAX);
            tdif[i] = 0.2f + 0.2f * (rand()/(float)RAND_MAX);
        } else {
            // Clear layer
            rdif[i] = 0.01f + 0.03f * (rand()/(float)RAND_MAX);
            tdif[i] = 0.9f + 0.09f * (rand()/(float)RAND_MAX);
        }
        src_up[i] = 5.0f + 15.0f * (rand()/(float)RAND_MAX);
        src_dn[i] = 5.0f + 15.0f * (rand()/(float)RAND_MAX);
    }
    total++; passed += run_test("Mixed cloud/clear + SW flux", 128,
                                 rdif, tdif, src_up, src_dn, 0.25f, 80.0f, 200.0f);

    // ---- TEST 15: Extreme - very thick single layer embedded ----
    for (int i = 0; i < 64; i++) {
        rdif[i] = 0.02f;
        tdif[i] = 0.95f;
        src_up[i] = 10.0f;
        src_dn[i] = 10.0f;
    }
    rdif[32] = 0.8f;  // Extremely reflective single layer
    tdif[32] = 0.05f; // Nearly opaque
    total++; passed += run_test("Single opaque layer (R=0.8, T=0.05)", 64,
                                 rdif, tdif, src_up, src_dn, 0.3f, 100.0f, 50.0f);

    // ---- SUMMARY ----
    printf("\n================================================================\n");
    printf("  RESULTS: %d / %d tests passed\n", passed, total);
    if (passed == total)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  %d TESTS FAILED — DO NOT PUBLISH\n", total - passed);
    printf("================================================================\n");

    return (passed == total) ? 0 : 1;
}
