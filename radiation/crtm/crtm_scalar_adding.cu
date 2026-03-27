/*
 * CRTM Clear-Sky Adding Method: Parallel Prefix Scan
 *
 * The CRTM ADA_Module.f90 non-scattering path (lines 228-246) uses
 * scalar adding identical to rte-rrtmgp. For each discrete ordinate
 * angle, the recurrence is:
 *
 *   Refl_UP(i,i,k-1) = Trans(i,k)^2 * Refl_UP(i,i,k)
 *   Rad_UP(i,k-1) = Source_UP(i,k) + Trans(i,k) * (Refl_UP(i,:,k) . Source_DOWN(:,k) + Rad_UP(i,k))
 *
 * For the diagonal (non-scattering) case, each angle is independent
 * and the reflectance update is: R[k-1] = T[k]^2 * R[k]
 * This is simpler than rte-rrtmgp (no denominator), making it even
 * MORE suitable for parallel scan.
 *
 * Compile: nvcc -O3 -arch=sm_86 crtm_scalar_adding.cu -o crtm_scan
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

// Affine tuple for radiance scan
struct AffineTuple { float a, b; };

__device__ __forceinline__ AffineTuple affine_compose(AffineTuple outer, AffineTuple inner) {
    return {outer.a * inner.a, outer.a * inner.b + outer.b};
}

// ============================================================
// GPU: Parallel clear-sky CRTM adding
// Each block = one profile (column), each thread = one layer
// Process n_angles streams sequentially per block (small, typically 6)
// ============================================================
__global__ void kernel_crtm_clearsky_parallel(
    const float* __restrict__ tau,        // [nprofiles, nlayers] optical depth
    const float* __restrict__ planck_atm, // [nprofiles, nlayers] Planck function per layer
    const float* __restrict__ planck_sfc, // [nprofiles] surface Planck
    const float* __restrict__ cos_angle,  // [n_angles] cosines of discrete ordinate angles
    const float* __restrict__ sfc_emis,   // [nprofiles, n_angles] surface emissivity
    float* __restrict__ rad_toa,          // [nprofiles, n_angles] output TOA radiance
    int nprofiles, int nlayers, int n_angles)
{
    int iprof = blockIdx.x;
    if (iprof >= nprofiles) return;
    int tid = threadIdx.x;
    if (tid >= nlayers) return;

    int base = iprof * nlayers;
    extern __shared__ char smem[];
    AffineTuple* shared = (AffineTuple*)smem;

    // Process each angle independently
    for (int iang = 0; iang < n_angles; iang++) {
        float mu = cos_angle[iang];
        float od = tau[base + tid];

        // Layer transmittance for this angle
        float trans = expf(-od / mu);

        // Layer source (Planck emission)
        float source_up = planck_atm[base + tid] * (1.0f - trans);

        // The radiance adding recurrence (non-scattering, clear-sky):
        // Rad_UP[k-1] = source_up[k] + trans[k] * Rad_UP[k]
        // This is affine: y = a*x + b where a = trans, b = source_up
        shared[tid] = {trans, source_up};
        __syncthreads();

        // Right-to-left suffix scan (bottom-up, same as rte-rrtmgp)
        for (int stride = 1; stride < nlayers; stride *= 2) {
            AffineTuple val;
            int partner = tid + stride;
            if (tid < nlayers) {
                val = (partner < nlayers) ? affine_compose(shared[tid], shared[partner])
                                           : shared[tid];
            }
            __syncthreads();
            if (tid < nlayers) shared[tid] = val;
            __syncthreads();
        }

        // Boundary condition: Rad_UP at surface = emis * Planck_sfc
        float rad_sfc = sfc_emis[iprof * n_angles + iang] * planck_sfc[iprof];

        // TOA radiance = apply(suffix[0], rad_sfc)
        if (tid == 0) {
            float toa = shared[0].a * rad_sfc + shared[0].b;
            rad_toa[iprof * n_angles + iang] = toa;
        }
        __syncthreads();
    }
}

// ============================================================
// CPU Reference
// ============================================================
void cpu_crtm_clearsky(
    const float* tau, const float* planck_atm, const float* planck_sfc,
    const float* cos_angle, const float* sfc_emis, float* rad_toa,
    int nprofiles, int nlayers, int n_angles)
{
    for (int ip = 0; ip < nprofiles; ip++) {
        for (int ia = 0; ia < n_angles; ia++) {
            float mu = cos_angle[ia];
            float rad = sfc_emis[ip * n_angles + ia] * planck_sfc[ip];

            // Bottom-up adding
            for (int k = nlayers - 1; k >= 0; k--) {
                float trans = expf(-tau[ip * nlayers + k] / mu);
                float src = planck_atm[ip * nlayers + k] * (1.0f - trans);
                rad = src + trans * rad;
            }
            rad_toa[ip * n_angles + ia] = rad;
        }
    }
}

int main() {
    printf("================================================================\n");
    printf("  CRTM Clear-Sky Adding: Parallel Prefix Scan Benchmark\n");
    printf("  ADA_Module.f90 non-scattering path (lines 228-246)\n");
    printf("  RTX 3060 12GB | CUDA %d.%d\n", CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    printf("================================================================\n\n");

    // Test configs matching real CRTM usage
    struct TestCase {
        int nprofiles, nlayers, n_angles;
        const char* name;
    };
    TestCase tests[] = {
        {10000,  60, 6,  "IR sounder (10K profiles, 60 layers, 6 angles)"},
        {50000,  60, 6,  "IR sounder (50K profiles, 60 layers, 6 angles)"},
        {10000, 100, 6,  "High-res (10K profiles, 100 layers, 6 angles)"},
        {100000, 60, 2,  "MW sounder (100K profiles, 60 layers, 2 angles)"},
        {10000,  60, 16, "Multi-stream (10K profiles, 60 layers, 16 angles)"},
        {1000,  200, 6,  "Deep atmosphere (1K profiles, 200 layers, 6 angles)"},
    };
    int ntests = 6;
    int total_pass = 0;

    for (int t = 0; t < ntests; t++) {
        int np = tests[t].nprofiles, nl = tests[t].nlayers, na = tests[t].n_angles;

        printf("--- %s ---\n", tests[t].name);

        float *h_tau   = (float*)malloc(np * nl * sizeof(float));
        float *h_plnk  = (float*)malloc(np * nl * sizeof(float));
        float *h_psfc  = (float*)malloc(np * sizeof(float));
        float *h_cos   = (float*)malloc(na * sizeof(float));
        float *h_emis  = (float*)malloc(np * na * sizeof(float));
        float *h_ref   = (float*)malloc(np * na * sizeof(float));
        float *h_gpu   = (float*)malloc(np * na * sizeof(float));

        srand(42 + t);
        for (int i = 0; i < np * nl; i++) {
            h_tau[i]  = 0.001f + 0.5f * (rand()/(float)RAND_MAX);  // Optical depth
            h_plnk[i] = 50.0f + 200.0f * (rand()/(float)RAND_MAX); // Planck radiance
        }
        for (int i = 0; i < np; i++) {
            h_psfc[i] = 200.0f + 100.0f * (rand()/(float)RAND_MAX);
        }
        // Typical Gaussian quadrature angles
        float angles[] = {0.2113, 0.3887, 0.5774, 0.7071, 0.8165, 0.9258,
                          0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
                          0.7000, 0.8000, 0.9000, 0.9500};
        for (int i = 0; i < na; i++) h_cos[i] = angles[i % 16];
        for (int i = 0; i < np * na; i++) h_emis[i] = 0.95f + 0.05f * (rand()/(float)RAND_MAX);

        // CPU reference
        cpu_crtm_clearsky(h_tau, h_plnk, h_psfc, h_cos, h_emis, h_ref, np, nl, na);

        // GPU
        float *d_tau, *d_plnk, *d_psfc, *d_cos, *d_emis, *d_rad;
        CUDA_CHECK(cudaMalloc(&d_tau,  np*nl*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_plnk, np*nl*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_psfc, np*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos,  na*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_emis, np*na*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rad,  np*na*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_tau,  h_tau,  np*nl*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_plnk, h_plnk, np*nl*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_psfc, h_psfc, np*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos,  h_cos,  na*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_emis, h_emis, np*na*sizeof(float), cudaMemcpyHostToDevice));

        int smem = nl * sizeof(AffineTuple);

        // Warmup + Benchmark
        kernel_crtm_clearsky_parallel<<<np, nl, smem>>>(
            d_tau, d_plnk, d_psfc, d_cos, d_emis, d_rad, np, nl, na);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < NITERS; it++) {
            kernel_crtm_clearsky_parallel<<<np, nl, smem>>>(
                d_tau, d_plnk, d_psfc, d_cos, d_emis, d_rad, np, nl, na);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_gpu;
        CUDA_CHECK(cudaEventElapsedTime(&t_gpu, start, stop));
        t_gpu /= NITERS;

        CUDA_CHECK(cudaMemcpy(h_gpu, d_rad, np*na*sizeof(float), cudaMemcpyDeviceToHost));

        float max_rel = 0, max_abs = 0;
        int nan_count = 0;
        for (int i = 0; i < np * na; i++) {
            if (h_gpu[i] != h_gpu[i]) { nan_count++; continue; }
            float ae = fabsf(h_gpu[i] - h_ref[i]);
            if (ae > max_abs) max_abs = ae;
            if (fabsf(h_ref[i]) > 1e-6f) {
                float re = ae / fabsf(h_ref[i]);
                if (re > max_rel) max_rel = re;
            }
        }

        int pass = (nan_count == 0 && max_rel < 1e-4f);
        if (pass) total_pass++;

        printf("  GPU time: %.4f ms | max_rel=%.1e max_abs=%.1e NaN=%d | %s\n",
               t_gpu, max_rel, max_abs, nan_count, pass ? "PASS" : "FAIL");

        free(h_tau); free(h_plnk); free(h_psfc); free(h_cos); free(h_emis);
        free(h_ref); free(h_gpu);
        cudaFree(d_tau); cudaFree(d_plnk); cudaFree(d_psfc);
        cudaFree(d_cos); cudaFree(d_emis); cudaFree(d_rad);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n================================================================\n");
    printf("  RESULTS: %d / %d tests passed\n", total_pass, ntests);
    if (total_pass == ntests) printf("  ALL TESTS PASSED\n");
    else printf("  %d TESTS FAILED\n", ntests - total_pass);
    printf("================================================================\n");

    return 0;
}
