/*
 * Full Two-Stream Parallel Flux Solver
 *
 * Parallelizes the COMPLETE adding method (albedo + source + fluxes)
 * using 3 parallel prefix scans + 1 pointwise step.
 *
 * Mathematical basis:
 *   Scan 1: Albedo via 2x2 Möbius matrices (Grant & Hunt 1969)
 *   Scan 2: Source via affine tuple scan (Martin & Cundy 2018 / Mamba)
 *   Scan 3: Flux_dn via affine tuple scan
 *   Pointwise: flux_up = albedo * flux_dn + src
 *
 * Compile: nvcc -O3 -arch=sm_86 full_flux_parallel_scan.cu -o full_flux_scan
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NCOL 1024
#define NLAY 128
#define NLEV (NLAY + 1)
#define NITERS 100

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } }

// ============================================================
// Data structures for parallel scans
// ============================================================

// 2x2 Möbius matrix for albedo scan
struct Mat2x2 { float a, b, c, d; };

// Affine tuple (a, b) representing map x -> a*x + b
struct AffineTuple { float a, b; };

// ============================================================
// Associative operators
// ============================================================

// 2x2 matrix multiply (Möbius composition)
__device__ __forceinline__ Mat2x2 mat2_mul(Mat2x2 A, Mat2x2 B) {
    return {A.a*B.a + A.b*B.c, A.a*B.b + A.b*B.d,
            A.c*B.a + A.d*B.c, A.c*B.b + A.d*B.d};
}

// Apply Möbius: f(x) = (a*x+b)/(c*x+d)
__device__ __forceinline__ float mat2_apply(Mat2x2 M, float x) {
    return (M.a*x + M.b) / (M.c*x + M.d);
}

// Affine tuple composition: (a2,b2) . (a1,b1) = (a2*a1, a2*b1 + b2)
// Represents: f2(f1(x)) = a2*(a1*x + b1) + b2 = (a2*a1)*x + (a2*b1 + b2)
__device__ __forceinline__ AffineTuple affine_compose(AffineTuple outer, AffineTuple inner) {
    return {outer.a * inner.a, outer.a * inner.b + outer.b};
}

// Apply affine: f(x) = a*x + b
__device__ __forceinline__ float affine_apply(AffineTuple t, float x) {
    return t.a * x + t.b;
}

// ============================================================
// KERNEL: Full parallel flux solver (3 scans + pointwise)
// One block per column, nlay threads per block
// ============================================================
__global__ void kernel_full_flux_parallel(
    const float* __restrict__ rdif,     // [ncol, nlay] diffuse reflectivity
    const float* __restrict__ tdif,     // [ncol, nlay] diffuse transmissivity
    const float* __restrict__ src_up,   // [ncol, nlay] upward source per layer
    const float* __restrict__ src_dn,   // [ncol, nlay] downward source per layer
    const float* __restrict__ sfc_albedo, // [ncol] surface albedo
    const float* __restrict__ sfc_src,    // [ncol] surface emission source
    const float* __restrict__ inc_flux_dn, // [ncol] incident flux at TOA
    float* __restrict__ flux_up_out,    // [ncol, nlev]
    float* __restrict__ flux_dn_out,    // [ncol, nlev]
    int ncol, int nlay)
{
    int icol = blockIdx.x;
    if (icol >= ncol) return;
    int tid = threadIdx.x;
    int nlev = nlay + 1;
    int base = icol * nlay;

    // top_at_1 = true convention: layer 0 is TOP, layer nlay-1 is BOTTOM
    // Albedo scan goes bottom-up (nlay-1 -> 0)
    // Flux scan goes top-down (0 -> nlay)

    // ============================================================
    // SCAN 1: Albedo (bottom-up Möbius scan)
    // albedo[i] = R[i] + T[i]^2 * albedo[i+1] / (1 - R[i]*albedo[i+1])
    // ============================================================
    extern __shared__ char smem[];
    Mat2x2* shared_mat = (Mat2x2*)smem;

    float R_local, T_local;
    if (tid < nlay) {
        R_local = rdif[base + tid];
        T_local = tdif[base + tid];
        shared_mat[tid] = {T_local*T_local - R_local*R_local, R_local,
                           -R_local, 1.0f};
    }
    __syncthreads();

    // Right-to-left Hillis-Steele suffix scan for albedo
    for (int stride = 1; stride < nlay; stride *= 2) {
        Mat2x2 val;
        if (tid < nlay) {
            int partner = tid + stride;
            val = (partner < nlay) ? mat2_mul(shared_mat[tid], shared_mat[partner])
                                    : shared_mat[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_mat[tid] = val;
        __syncthreads();
    }

    // Compute albedo at all levels
    // shared_mat[tid] = suffix product M_tid * ... * M_{nlay-1}
    // albedo[tid] = apply(suffix[tid], albedo_sfc)
    // albedo[nlay] = albedo_sfc (surface)
    __shared__ float albedo[NLAY + 1];
    float alb_sfc = sfc_albedo[icol];

    if (tid < nlay) {
        albedo[tid] = mat2_apply(shared_mat[tid], alb_sfc);
    }
    if (tid == 0) albedo[nlay] = alb_sfc;
    __syncthreads();

    // ============================================================
    // SCAN 2: Source (bottom-up affine scan)
    // src[i] = src_up[i] + T[i]*denom[i] * (src[i+1] + albedo[i+1]*src_dn[i])
    // where denom[i] = 1/(1 - R[i]*albedo[i+1])
    // This is: src[i] = A[i]*src[i+1] + B[i]
    //   A[i] = T[i] * denom[i]
    //   B[i] = src_up[i] + T[i]*denom[i]*albedo[i+1]*src_dn[i]
    // ============================================================
    AffineTuple* shared_aff = (AffineTuple*)smem;

    float denom_local;
    if (tid < nlay) {
        denom_local = 1.0f / (1.0f - R_local * albedo[tid + 1]);
        float A = T_local * denom_local;
        float B = src_up[base + tid] + T_local * denom_local * albedo[tid + 1] * src_dn[base + tid];
        shared_aff[tid] = {A, B};
    }
    __syncthreads();

    // Right-to-left suffix scan for source
    for (int stride = 1; stride < nlay; stride *= 2) {
        AffineTuple val;
        if (tid < nlay) {
            int partner = tid + stride;
            val = (partner < nlay) ? affine_compose(shared_aff[tid], shared_aff[partner])
                                    : shared_aff[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_aff[tid] = val;
        __syncthreads();
    }

    // Compute source at all levels
    // src[tid] = apply(suffix[tid], src_sfc)
    __shared__ float src[NLAY + 1];
    float src_sfc_val = sfc_src[icol];

    if (tid < nlay) {
        src[tid] = affine_apply(shared_aff[tid], src_sfc_val);
    }
    if (tid == 0) src[nlay] = src_sfc_val;
    __syncthreads();

    // ============================================================
    // SCAN 3: Flux_dn (top-down affine scan)
    // flux_dn[i+1] = T[i]*denom[i]*flux_dn[i] + (R[i]*src[i+1] + src_dn[i])*denom[i]
    // This is: flux_dn[i+1] = C[i]*flux_dn[i] + D[i]
    //   C[i] = T[i]*denom[i]
    //   D[i] = (R[i]*src[i+1] + src_dn[i])*denom[i]
    //
    // TOP-DOWN: level 0 = TOA (known), scan from 0 to nlay
    // We use a LEFT-TO-RIGHT prefix scan (standard Hillis-Steele)
    // ============================================================
    // Reuse shared memory
    AffineTuple* shared_flux = (AffineTuple*)smem;

    if (tid < nlay) {
        float C = T_local * denom_local;
        float D = (R_local * src[tid + 1] + src_dn[base + tid]) * denom_local;
        shared_flux[tid] = {C, D};
    }
    __syncthreads();

    // LEFT-TO-RIGHT Hillis-Steele prefix scan (top-down)
    // After scan: shared_flux[tid] = compose(f_0, f_1, ..., f_tid)
    // flux_dn[tid+1] = apply(shared_flux[tid], flux_dn[0])
    for (int stride = 1; stride < nlay; stride *= 2) {
        AffineTuple val;
        if (tid < nlay) {
            int partner = tid - stride;
            // Outer . Inner: compose(shared[tid], shared[partner])
            // f_tid(...(f_partner(x))) — tid is the outermost
            val = (partner >= 0) ? affine_compose(shared_flux[tid], shared_flux[partner])
                                  : shared_flux[tid];
        }
        __syncthreads();
        if (tid < nlay) shared_flux[tid] = val;
        __syncthreads();
    }

    // Compute flux_dn at all levels
    float inc_flux = inc_flux_dn[icol];
    flux_dn_out[icol * nlev + 0] = inc_flux;  // TOA boundary condition

    if (tid < nlay) {
        flux_dn_out[icol * nlev + tid + 1] = affine_apply(shared_flux[tid], inc_flux);
    }
    __syncthreads();

    // ============================================================
    // POINTWISE: flux_up = albedo * flux_dn + src (Eq. 12)
    // We have nlay threads (0..nlay-1) but need nlev values (0..nlay)
    // ============================================================
    if (tid < nlay) {
        float fd = flux_dn_out[icol * nlev + tid];
        flux_up_out[icol * nlev + tid] = albedo[tid] * fd + src[tid];
    }
    // Thread 0 also handles the surface level (nlay)
    if (tid == 0) {
        float fd = flux_dn_out[icol * nlev + nlay];
        flux_up_out[icol * nlev + nlay] = albedo[nlay] * fd + src[nlay];
    }
}

// ============================================================
// SEQUENTIAL REFERENCE: Standard adding method
// ============================================================
__global__ void kernel_full_flux_sequential(
    const float* __restrict__ rdif,
    const float* __restrict__ tdif,
    const float* __restrict__ src_up,
    const float* __restrict__ src_dn,
    const float* __restrict__ sfc_albedo,
    const float* __restrict__ sfc_src,
    const float* __restrict__ inc_flux_dn,
    float* __restrict__ flux_up_out,
    float* __restrict__ flux_dn_out,
    int ncol, int nlay)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    if (icol >= ncol) return;
    int nlev = nlay + 1;
    int base = icol * nlay;

    // Local arrays for albedo, source, denom
    float l_albedo[NLAY + 1], l_src[NLAY + 1], l_denom[NLAY];

    // Boundary conditions (bottom)
    l_albedo[nlay] = sfc_albedo[icol];
    l_src[nlay] = sfc_src[icol];

    // PASS 1: bottom-up (albedo + source)
    for (int i = nlay - 1; i >= 0; i--) {
        float R = rdif[base + i];
        float T = tdif[base + i];
        l_denom[i] = 1.0f / (1.0f - R * l_albedo[i + 1]);
        l_albedo[i] = R + T * T * l_albedo[i + 1] * l_denom[i];
        l_src[i] = src_up[base + i] +
                   T * l_denom[i] * (l_src[i + 1] + l_albedo[i + 1] * src_dn[base + i]);
    }

    // PASS 2: top-down (flux_dn)
    flux_dn_out[icol * nlev + 0] = inc_flux_dn[icol];
    flux_up_out[icol * nlev + 0] = l_albedo[0] * inc_flux_dn[icol] + l_src[0];

    for (int i = 0; i < nlay; i++) {
        float R = rdif[base + i];
        float T = tdif[base + i];
        float fd_prev = flux_dn_out[icol * nlev + i];
        flux_dn_out[icol * nlev + i + 1] =
            (T * fd_prev + R * l_src[i + 1] + src_dn[base + i]) * l_denom[i];
        flux_up_out[icol * nlev + i + 1] =
            l_albedo[i + 1] * flux_dn_out[icol * nlev + i + 1] + l_src[i + 1];
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  Full Two-Stream Parallel Flux Solver Benchmark\n");
    printf("  3 Parallel Scans + 1 Pointwise vs Sequential Adding\n");
    printf("  RTX 3060 12GB | CUDA %d.%d\n", CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    printf("================================================================\n\n");

    int nlay = NLAY, nlev = NLEV, ncol = NCOL;
    printf("Problem: %d columns x %d layers = %d levels\n\n", ncol, nlay, nlev);

    // Allocate and initialize with realistic atmospheric values
    int nlay_elem = ncol * nlay;
    int nlev_elem = ncol * nlev;

    float *h_rdif     = (float*)malloc(nlay_elem * sizeof(float));
    float *h_tdif     = (float*)malloc(nlay_elem * sizeof(float));
    float *h_src_up   = (float*)malloc(nlay_elem * sizeof(float));
    float *h_src_dn   = (float*)malloc(nlay_elem * sizeof(float));
    float *h_sfc_alb  = (float*)malloc(ncol * sizeof(float));
    float *h_sfc_src  = (float*)malloc(ncol * sizeof(float));
    float *h_inc_flux = (float*)malloc(ncol * sizeof(float));

    srand(42);
    for (int i = 0; i < nlay_elem; i++) {
        h_rdif[i]   = 0.01f + 0.15f * (rand() / (float)RAND_MAX);
        h_tdif[i]   = 0.7f + 0.29f * (rand() / (float)RAND_MAX);
        h_src_up[i] = 5.0f + 20.0f * (rand() / (float)RAND_MAX);
        h_src_dn[i] = 5.0f + 20.0f * (rand() / (float)RAND_MAX);
    }
    for (int i = 0; i < ncol; i++) {
        h_sfc_alb[i]  = 0.1f + 0.3f * (rand() / (float)RAND_MAX);
        h_sfc_src[i]  = 50.0f + 100.0f * (rand() / (float)RAND_MAX);
        h_inc_flux[i] = 0.0f;  // LW: no incident flux at TOA
    }

    // Device memory
    float *d_rdif, *d_tdif, *d_src_up, *d_src_dn;
    float *d_sfc_alb, *d_sfc_src, *d_inc_flux;
    float *d_flux_up_seq, *d_flux_dn_seq, *d_flux_up_par, *d_flux_dn_par;

    CUDA_CHECK(cudaMalloc(&d_rdif,     nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tdif,     nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_up,   nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_dn,   nlay_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sfc_alb,  ncol * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sfc_src,  ncol * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inc_flux, ncol * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_up_seq, nlev_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_dn_seq, nlev_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_up_par, nlev_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flux_dn_par, nlev_elem * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rdif,     h_rdif,     nlay_elem*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tdif,     h_tdif,     nlay_elem*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_up,   h_src_up,   nlay_elem*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_dn,   h_src_dn,   nlay_elem*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfc_alb,  h_sfc_alb,  ncol*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sfc_src,  h_sfc_src,  ncol*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inc_flux, h_inc_flux, ncol*sizeof(float), cudaMemcpyHostToDevice));

    // ============================================================
    // BENCHMARK: Sequential
    // ============================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int seq_block = 64;
    int seq_grid = (ncol + seq_block - 1) / seq_block;

    // Warmup
    kernel_full_flux_sequential<<<seq_grid, seq_block>>>(
        d_rdif, d_tdif, d_src_up, d_src_dn, d_sfc_alb, d_sfc_src, d_inc_flux,
        d_flux_up_seq, d_flux_dn_seq, ncol, nlay);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NITERS; i++) {
        kernel_full_flux_sequential<<<seq_grid, seq_block>>>(
            d_rdif, d_tdif, d_src_up, d_src_dn, d_sfc_alb, d_sfc_src, d_inc_flux,
            d_flux_up_seq, d_flux_dn_seq, ncol, nlay);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_seq;
    CUDA_CHECK(cudaEventElapsedTime(&t_seq, start, stop));
    t_seq /= NITERS;

    // ============================================================
    // BENCHMARK: Parallel (3 scans)
    // ============================================================
    // Shared memory: max of Mat2x2[NLAY] or AffineTuple[NLAY] + float[NLAY+1]*2
    int smem_size = NLAY * sizeof(Mat2x2) + 2 * NLEV * sizeof(float);

    // Warmup
    kernel_full_flux_parallel<<<ncol, nlay, smem_size>>>(
        d_rdif, d_tdif, d_src_up, d_src_dn, d_sfc_alb, d_sfc_src, d_inc_flux,
        d_flux_up_par, d_flux_dn_par, ncol, nlay);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NITERS; i++) {
        kernel_full_flux_parallel<<<ncol, nlay, smem_size>>>(
            d_rdif, d_tdif, d_src_up, d_src_dn, d_sfc_alb, d_sfc_src, d_inc_flux,
            d_flux_up_par, d_flux_dn_par, ncol, nlay);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_par;
    CUDA_CHECK(cudaEventElapsedTime(&t_par, start, stop));
    t_par /= NITERS;

    // ============================================================
    // ACCURACY CHECK
    // ============================================================
    float *h_flux_up_seq = (float*)malloc(nlev_elem * sizeof(float));
    float *h_flux_dn_seq = (float*)malloc(nlev_elem * sizeof(float));
    float *h_flux_up_par = (float*)malloc(nlev_elem * sizeof(float));
    float *h_flux_dn_par = (float*)malloc(nlev_elem * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_flux_up_seq, d_flux_up_seq, nlev_elem*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux_dn_seq, d_flux_dn_seq, nlev_elem*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux_up_par, d_flux_up_par, nlev_elem*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux_dn_par, d_flux_dn_par, nlev_elem*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err_up = 0, max_err_dn = 0;
    float max_rel_up = 0, max_rel_dn = 0;
    int err_count_up = 0, err_count_dn = 0;
    float sum_err_up = 0, sum_err_dn = 0;

    for (int i = 0; i < nlev_elem; i++) {
        float eu = fabsf(h_flux_up_par[i] - h_flux_up_seq[i]);
        float ed = fabsf(h_flux_dn_par[i] - h_flux_dn_seq[i]);
        if (eu > max_err_up) max_err_up = eu;
        if (ed > max_err_dn) max_err_dn = ed;
        sum_err_up += eu;
        sum_err_dn += ed;

        if (fabsf(h_flux_up_seq[i]) > 1e-6f) {
            float re = eu / fabsf(h_flux_up_seq[i]);
            if (re > max_rel_up) max_rel_up = re;
            if (re > 1e-4f) err_count_up++;
        }
        if (fabsf(h_flux_dn_seq[i]) > 1e-6f) {
            float re = ed / fabsf(h_flux_dn_seq[i]);
            if (re > max_rel_dn) max_rel_dn = re;
            if (re > 1e-4f) err_count_dn++;
        }
    }

    // Print sample values for verification
    printf("--- Sample Flux Values (Column 0) ---\n");
    printf("Level | Seq flux_up | Par flux_up | Seq flux_dn | Par flux_dn\n");
    for (int i = 0; i < nlev; i += nlev/8) {
        int idx = i;
        printf("  %3d | %11.4f | %11.4f | %11.4f | %11.4f\n",
               i, h_flux_up_seq[idx], h_flux_up_par[idx],
               h_flux_dn_seq[idx], h_flux_dn_par[idx]);
    }
    printf("  %3d | %11.4f | %11.4f | %11.4f | %11.4f\n",
           nlay, h_flux_up_seq[nlay], h_flux_up_par[nlay],
           h_flux_dn_seq[nlay], h_flux_dn_par[nlay]);

    printf("\n--- Accuracy ---\n");
    printf("  Flux_up: max_abs_err=%.3e  max_rel_err=%.3e  mean_abs_err=%.3e  errors(>1e-4)=%d/%d\n",
           max_err_up, max_rel_up, sum_err_up/nlev_elem, err_count_up, nlev_elem);
    printf("  Flux_dn: max_abs_err=%.3e  max_rel_err=%.3e  mean_abs_err=%.3e  errors(>1e-4)=%d/%d\n",
           max_err_dn, max_rel_dn, sum_err_dn/nlev_elem, err_count_dn, nlev_elem);

    // Physical sanity checks
    int nan_count = 0, neg_count = 0;
    for (int i = 0; i < nlev_elem; i++) {
        if (h_flux_up_par[i] != h_flux_up_par[i] || h_flux_dn_par[i] != h_flux_dn_par[i]) nan_count++;
        if (h_flux_up_par[i] < -1e-6f || h_flux_dn_par[i] < -1e-6f) neg_count++;
    }
    printf("  NaN count: %d  Negative flux count: %d\n", nan_count, neg_count);

    // Energy conservation: net flux should vary smoothly
    float max_net_jump = 0;
    for (int icol = 0; icol < 10; icol++) {
        for (int i = 1; i < nlev; i++) {
            float net_prev = h_flux_up_par[icol*nlev + i-1] - h_flux_dn_par[icol*nlev + i-1];
            float net_curr = h_flux_up_par[icol*nlev + i]   - h_flux_dn_par[icol*nlev + i];
            float jump = fabsf(net_curr - net_prev);
            if (jump > max_net_jump) max_net_jump = jump;
        }
    }
    printf("  Max net flux jump (energy check): %.3f W/m2\n", max_net_jump);

    printf("\n--- Performance ---\n");
    printf("  Sequential:     %.4f ms\n", t_seq);
    printf("  Parallel (3 scans): %.4f ms\n", t_par);
    printf("  Speedup:        %.2fx\n", t_seq / t_par);

    printf("\n================================================================\n");
    printf("  RESULT: Full flux solver speedup = %.2fx\n", t_seq / t_par);
    printf("  (Albedo scan + Source scan + Flux scan + Pointwise)\n");
    printf("================================================================\n");

    // Cleanup
    free(h_rdif); free(h_tdif); free(h_src_up); free(h_src_dn);
    free(h_sfc_alb); free(h_sfc_src); free(h_inc_flux);
    free(h_flux_up_seq); free(h_flux_dn_seq); free(h_flux_up_par); free(h_flux_dn_par);
    cudaFree(d_rdif); cudaFree(d_tdif); cudaFree(d_src_up); cudaFree(d_src_dn);
    cudaFree(d_sfc_alb); cudaFree(d_sfc_src); cudaFree(d_inc_flux);
    cudaFree(d_flux_up_seq); cudaFree(d_flux_dn_seq);
    cudaFree(d_flux_up_par); cudaFree(d_flux_dn_par);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
