/**
 * NOAA-OWP Batched GPU Kernels — REAL DATA harness
 *
 * Copy of water-prediction/owp_batched_kernels.cu with the synthetic data
 * generators (gen_nash_data / gen_tridiag_data) replaced by loaders for real
 * NOAA-OWP datasets. Kernel math and CPU reference implementations are
 * copied VERBATIM from the original and are unchanged.
 *
 * Real data:
 *  1. CFE Nash Cascade  — calibrated (K, N, initial storage) from
 *     NOAA-OWP/cfe configs/cfe_config_cat_87.txt (subsurface + surface Nash
 *     cascades) driven by 720 h of real AORC precipitation
 *     (forcings/cat87_01Dec2015.csv). A library of 1440 (state, flux) pairs is
 *     produced by running the UNTOUCHED CPU reference sequentially over the
 *     real series; batches tile this library (documented replication).
 *  2. NOAH-MP ROSR12 tridiagonal solver — 17,521 Richards-equation systems
 *     assembled by prep_noahmp_tridiag.py with the model's own SRT equations
 *     from real Bondville, IL 1998 forcing (NOAA-OWP/noah-owp-modular
 *     data/bondville.dat) + SOILPARM.TBL/namelist parameters.
 *
 * Prep: prep_cfe_nash.py, prep_noahmp_tridiag.py  (see headers for provenance)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Maximum sizes
#define MAX_NASH 10
#define MAX_SOIL 10

// ============================================================
// KERNEL 1: CFE Nash Cascade   (verbatim from owp_batched_kernels.cu)
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
// KERNEL 2: NOAH-MP Tridiagonal Solver (ROSR12)  (verbatim)
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
// REAL DATA loading (replaces synthetic generators)
// ============================================================

static void die(const char* msg) { fprintf(stderr, "FATAL: %s\n", msg); exit(1); }

// ---- Nash: library of real (flux, K, N, storage) tuples ----
typedef struct {
    int npairs;
    float* flux;                 // [npairs]
    float* K;                    // [npairs]
    int*   N;                    // [npairs]
    float* storage;              // [npairs * MAX_NASH] pre-step states
} NashLibrary;

static NashLibrary g_nash;

// Load real config/forcing and build the (state, flux) pair library by running
// the untouched CPU reference sequentially through the real precipitation series.
void load_nash_library(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open nash_realdata.bin (run prep_cfe_nash.py)");
    int magic, nps, nhours;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4E415348) die("nash magic");
    fread(&nps, 4, 1, f);
    fread(&nhours, 4, 1, f);

    float* Kp = (float*)malloc(nps * sizeof(float));
    int* Np = (int*)malloc(nps * sizeof(int));
    float* stor0 = (float*)malloc(nps * MAX_NASH * sizeof(float));
    for (int p = 0; p < nps; p++) {
        fread(&Kp[p], 4, 1, f);
        fread(&Np[p], 4, 1, f);
        fread(&stor0[p * MAX_NASH], 4, MAX_NASH, f);
    }
    float* precip = (float*)malloc(nhours * sizeof(float));
    fread(precip, 4, nhours, f);
    fclose(f);

    g_nash.npairs = nps * nhours;
    g_nash.flux = (float*)malloc(g_nash.npairs * sizeof(float));
    g_nash.K = (float*)malloc(g_nash.npairs * sizeof(float));
    g_nash.N = (int*)malloc(g_nash.npairs * sizeof(int));
    g_nash.storage = (float*)malloc((size_t)g_nash.npairs * MAX_NASH * sizeof(float));

    // sequential drive with the untouched CPU reference (ncatch = 1)
    for (int p = 0; p < nps; p++) {
        float stor[MAX_NASH];
        memcpy(stor, &stor0[p * MAX_NASH], MAX_NASH * sizeof(float));
        for (int t = 0; t < nhours; t++) {
            int idx = p * nhours + t;
            g_nash.flux[idx] = precip[t];
            g_nash.K[idx] = Kp[p];
            g_nash.N[idx] = Np[p];
            memcpy(&g_nash.storage[(size_t)idx * MAX_NASH], stor, MAX_NASH * sizeof(float));
            float q;
            cpu_nash_cascade(&precip[t], &Kp[p], &Np[p], stor, &q, 1);
        }
    }
    printf("[data] Nash library: %d real (state,flux) pairs (%d paramsets x %d h, cat-87 AORC Dec 2015)\n",
           g_nash.npairs, nps, nhours);
    free(Kp); free(Np); free(stor0); free(precip);
}

// Fill a batch by tiling the real pair library (replication documented in RESULTS.md)
void fill_nash_batch(float* flux, float* K, int* N, float* stor, int nc) {
    for (int c = 0; c < nc; c++) {
        int s = c % g_nash.npairs;
        flux[c] = g_nash.flux[s];
        K[c] = g_nash.K[s];
        N[c] = g_nash.N[s];
        memcpy(&stor[(size_t)c * MAX_NASH], &g_nash.storage[(size_t)s * MAX_NASH],
               MAX_NASH * sizeof(float));
    }
}

// ---- Tridiag: real Richards systems from Bondville forcing ----
typedef struct {
    int ncol;
    int nsoil;
    float *A, *B, *C, *D;        // [ncol * MAX_SOIL]
} TridiagData;

static TridiagData g_tri;

void load_tridiag_data(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open noahmp_tridiag_realdata.bin (run prep_noahmp_tridiag.py)");
    int magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4E4F4148) die("tridiag magic");
    fread(&g_tri.ncol, 4, 1, f);
    fread(&g_tri.nsoil, 4, 1, f);
    size_t n = (size_t)g_tri.ncol * MAX_SOIL;
    g_tri.A = (float*)malloc(n * 4); g_tri.B = (float*)malloc(n * 4);
    g_tri.C = (float*)malloc(n * 4); g_tri.D = (float*)malloc(n * 4);
    if (fread(g_tri.A, 4, n, f) != n || fread(g_tri.B, 4, n, f) != n ||
        fread(g_tri.C, 4, n, f) != n || fread(g_tri.D, 4, n, f) != n) die("tridiag payload");
    fclose(f);
    printf("[data] Tridiag: %d real Richards systems (nsoil=%d, Bondville 1998, 30-min steps)\n",
           g_tri.ncol, g_tri.nsoil);
}

void fill_tridiag_batch(float* A, float* B, float* C, float* D, int* ns, int nc) {
    for (int col = 0; col < nc; col++) {
        int s = col % g_tri.ncol;
        memcpy(&A[(size_t)col * MAX_SOIL], &g_tri.A[(size_t)s * MAX_SOIL], MAX_SOIL * 4);
        memcpy(&B[(size_t)col * MAX_SOIL], &g_tri.B[(size_t)s * MAX_SOIL], MAX_SOIL * 4);
        memcpy(&C[(size_t)col * MAX_SOIL], &g_tri.C[(size_t)s * MAX_SOIL], MAX_SOIL * 4);
        memcpy(&D[(size_t)col * MAX_SOIL], &g_tri.D[(size_t)s * MAX_SOIL], MAX_SOIL * 4);
        ns[col] = g_tri.nsoil;
    }
}

// ============================================================
// Benchmark runner   (structure as original; data source swapped)
// ============================================================

void benchmark_nash(int ncatch) {
    printf("--- Nash Cascade (REAL cat-87 data): %d catchments ---\n", ncatch);

    size_t sz_f = ncatch * sizeof(float);
    size_t sz_i = ncatch * sizeof(int);
    size_t sz_s = (size_t)ncatch * MAX_NASH * sizeof(float);

    float *h_flux = (float*)malloc(sz_f);
    float *h_K = (float*)malloc(sz_f);
    int *h_N = (int*)malloc(sz_i);
    float *h_stor_cpu = (float*)malloc(sz_s);
    float *h_stor_gpu = (float*)malloc(sz_s);
    float *h_Q_cpu = (float*)malloc(sz_f);
    float *h_Q_gpu = (float*)malloc(sz_f);

    fill_nash_batch(h_flux, h_K, h_N, h_stor_cpu, ncatch);   // REAL DATA (was gen_nash_data)
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
    printf("--- NOAH-MP Tridiag Solver (REAL Bondville data): %d columns ---\n", ncol);

    size_t sz_f = (size_t)ncol * MAX_SOIL * sizeof(float);
    size_t sz_i = ncol * sizeof(int);

    float *h_A = (float*)malloc(sz_f), *h_B = (float*)malloc(sz_f);
    float *h_C = (float*)malloc(sz_f), *h_D = (float*)malloc(sz_f);
    float *h_X_cpu = (float*)malloc(sz_f), *h_X_gpu = (float*)malloc(sz_f);
    int *h_ns = (int*)malloc(sz_i);

    fill_tridiag_batch(h_A, h_B, h_C, h_D, h_ns, ncol);   // REAL DATA (was gen_tridiag_data)

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
        size_t base = (size_t)col * MAX_SOIL;
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

int main(int argc, char** argv) {
    const char* nash_path = (argc > 1) ? argv[1] : "data/nash_realdata.bin";
    const char* tri_path  = (argc > 2) ? argv[2] : "data/noahmp_tridiag_realdata.bin";

    printf("================================================\n");
    printf("  NOAA-OWP Batched GPU Kernels — REAL DATA\n");
    printf("  CFE Nash Cascade (cat-87 AORC) +\n");
    printf("  NOAH-MP Tridiag (Bondville 1998)\n");
    printf("  RTX 5070 12GB (sm_120)\n");
    printf("================================================\n\n");

    load_nash_library(nash_path);
    load_tridiag_data(tri_path);
    printf("\n");

    // Nash Cascade benchmarks
    printf("========== CFE NASH CASCADE ==========\n\n");
    benchmark_nash(1440);     // pure real library, no replication
    benchmark_nash(100000);
    benchmark_nash(1000000);
    benchmark_nash(2700000);  // Full CONUS NWM

    // Tridiag solver benchmarks
    printf("========== NOAH-MP TRIDIAG SOLVER ==========\n\n");
    benchmark_tridiag(17521); // pure real Bondville year, no replication
    benchmark_tridiag(100000);
    benchmark_tridiag(1000000);
    benchmark_tridiag(2700000);  // Full CONUS NWM

    printf("================================================\n");
    printf("  Notes:\n");
    printf("  - Kernel math and CPU references unchanged\n");
    printf("  - Inputs are real NOAA-OWP datasets; batches\n");
    printf("    larger than the real library tile it (c %% n)\n");
    printf("================================================\n");

    return 0;
}
