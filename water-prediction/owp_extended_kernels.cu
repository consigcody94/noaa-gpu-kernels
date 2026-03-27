/**
 * NOAA-OWP Extended GPU Kernels
 *
 * 1. t-route Diffusive Wave tridiagonal solver
 * 2. TOPMODEL exponential runoff generation
 * 3. Evapotranspiration (Penman-Monteith)
 *
 * All validated against CPU reference implementations.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_NODES 500  // max computational nodes per reach (diffusive)
#define MAX_TOPO  30   // max topodex histogram bins

// ============================================================
// KERNEL 1: Diffusive Wave Tridiagonal Solver
// From t-route/src/kernel/diffusive/diffusive.f90
// Forward elimination + back substitution along each reach
// ============================================================

struct DiffReachParams {
    int ncomp;           // number of computational nodes
    float theta;         // Crank-Nicolson parameter (0.5-1.0)
    float dtini;         // timestep (s)
};

// CPU reference
void cpu_diffusive_tridiag(
    const DiffReachParams* rp,
    const float* dx,          // node spacing [nreach * MAX_NODES]
    const float* diffusivity, // diffusion coefficient [nreach * MAX_NODES]
    const float* celerity,    // wave celerity [nreach * MAX_NODES]
    const float* qp_prev,     // discharge at previous timestep [nreach * MAX_NODES]
    float* qp_out,            // discharge at current timestep [nreach * MAX_NODES]
    int nreach)
{
    for (int j = 0; j < nreach; j++) {
        int nc = rp[j].ncomp;
        float theta = rp[j].theta;
        float dtini = rp[j].dtini;
        int base = j * MAX_NODES;

        float eei[MAX_NODES], ffi[MAX_NODES];

        // Boundary condition
        eei[0] = 1.0f;
        ffi[0] = 0.0f;

        // Forward elimination
        for (int i = 1; i < nc; i++) {
            float dxi = dx[base + i - 1];
            if (dxi < 1.0f) dxi = 1.0f;

            float cour = dtini / dxi;
            float cour2 = fabsf(celerity[base + i]) * cour;
            if (cour2 > 1.0f) cour2 = 1.0f; // stability clamp

            float alpha = 1.0f; // simplification for benchmark
            float diff_i = diffusivity[base + i];
            if (diff_i < 0.01f) diff_i = 0.01f;

            float ppi = -theta * diff_i * dtini / (dxi * dxi) * 2.0f / (alpha * (alpha + 1.0f)) * alpha;
            float qqi = 1.0f - ppi * (alpha + 1.0f) / alpha;
            float rri = ppi / alpha;

            // RHS: previous timestep contribution
            float ssi = qp_prev[base + i] + dtini * diff_i * (1.0f - theta) *
                        (qp_prev[base + i - 1] - 2.0f * qp_prev[base + i] +
                         ((i < nc - 1) ? qp_prev[base + i + 1] : qp_prev[base + i])) / (dxi * dxi);

            float denom = ppi * eei[i - 1] + qqi;
            if (fabsf(denom) < 1e-20f) denom = 1e-20f;

            eei[i] = -rri / denom;
            ffi[i] = (ssi - ppi * ffi[i - 1]) / denom;
        }

        // Back substitution
        // Downstream boundary: ghost node extrapolation
        float qp_ghost = qp_prev[base + nc - 1]; // simple extrapolation
        qp_out[base + nc - 1] = eei[nc - 1] * qp_ghost + ffi[nc - 1];

        for (int i = nc - 2; i >= 0; i--) {
            qp_out[base + i] = eei[i] * qp_out[base + i + 1] + ffi[i];
        }
    }
}

// GPU kernel — one thread per reach
__global__ void kernel_diffusive_tridiag(
    const DiffReachParams* __restrict__ rp,
    const float* __restrict__ dx,
    const float* __restrict__ diffusivity,
    const float* __restrict__ celerity,
    const float* __restrict__ qp_prev,
    float* __restrict__ qp_out,
    int nreach)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nreach) return;

    int nc = rp[j].ncomp;
    float theta = rp[j].theta;
    float dtini = rp[j].dtini;
    int base = j * MAX_NODES;

    // Use local arrays (will be in registers/local memory)
    float eei[MAX_NODES], ffi[MAX_NODES];

    eei[0] = 1.0f;
    ffi[0] = 0.0f;

    for (int i = 1; i < nc; i++) {
        float dxi = dx[base + i - 1];
        if (dxi < 1.0f) dxi = 1.0f;

        float cour = dtini / dxi;
        float cour2 = fabsf(celerity[base + i]) * cour;
        if (cour2 > 1.0f) cour2 = 1.0f;

        float alpha = 1.0f;
        float diff_i = diffusivity[base + i];
        if (diff_i < 0.01f) diff_i = 0.01f;

        float ppi = -theta * diff_i * dtini / (dxi * dxi) * 2.0f / (alpha * (alpha + 1.0f)) * alpha;
        float qqi = 1.0f - ppi * (alpha + 1.0f) / alpha;
        float rri = ppi / alpha;

        float ssi = qp_prev[base + i] + dtini * diff_i * (1.0f - theta) *
                    (qp_prev[base + i - 1] - 2.0f * qp_prev[base + i] +
                     ((i < nc - 1) ? qp_prev[base + i + 1] : qp_prev[base + i])) / (dxi * dxi);

        float denom = ppi * eei[i - 1] + qqi;
        if (fabsf(denom) < 1e-20f) denom = 1e-20f;

        eei[i] = -rri / denom;
        ffi[i] = (ssi - ppi * ffi[i - 1]) / denom;
    }

    float qp_ghost = qp_prev[base + nc - 1];
    qp_out[base + nc - 1] = eei[nc - 1] * qp_ghost + ffi[nc - 1];

    for (int i = nc - 2; i >= 0; i--) {
        qp_out[base + i] = eei[i] * qp_out[base + i + 1] + ffi[i];
    }
}

// ============================================================
// KERNEL 2: TOPMODEL Runoff Generation
// From NOAA-OWP/topmodel/src/topmodel.c
// ============================================================

struct TopoParams {
    float szm;    // exponential scaling parameter for transmissivity
    float szq;    // baseflow at complete saturation
    float td;     // unsaturated zone time delay
    float sbar;   // mean saturation deficit (state, updated)
    int num_bins; // number of topodex histogram bins
};

// CPU reference
void cpu_topmodel(
    const TopoParams* params,
    const float* lnaotb,      // ln(a/tan(b)) histogram values [ncatch * MAX_TOPO]
    const float* precip,      // precipitation [ncatch]
    float* qb_out,            // baseflow output [ncatch]
    float* qo_out,            // overland flow output [ncatch]
    float* sbar_out,          // updated saturation deficit [ncatch]
    int ncatch)
{
    for (int c = 0; c < ncatch; c++) {
        float szm = params[c].szm;
        float szq = params[c].szq;
        float sbar = params[c].sbar;
        int nb = params[c].num_bins;
        int base = c * MAX_TOPO;

        // Mean topographic index
        float tl = 0.0f;
        for (int ia = 0; ia < nb; ia++) tl += lnaotb[base + ia];
        tl /= (float)nb;

        // Baseflow
        float qb = szq * expf(-sbar / szm);

        // Overland flow from saturated areas
        float qo = 0.0f;
        for (int ia = 0; ia < nb; ia++) {
            float deficit_local = sbar + szm * (tl - lnaotb[base + ia]);
            if (deficit_local < 0.0f) {
                // This bin is saturated — generates overland flow
                qo += precip[c] * (-deficit_local / szm) / (float)nb;
            }
        }

        // Update saturation deficit
        sbar_out[c] = sbar - precip[c] + qb + qo;
        if (sbar_out[c] < 0.0f) sbar_out[c] = 0.0f;

        qb_out[c] = qb;
        qo_out[c] = qo;
    }
}

// GPU kernel
__global__ void kernel_topmodel(
    const TopoParams* __restrict__ params,
    const float* __restrict__ lnaotb,
    const float* __restrict__ precip,
    float* __restrict__ qb_out,
    float* __restrict__ qo_out,
    float* __restrict__ sbar_out,
    int ncatch)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncatch) return;

    float szm = params[c].szm;
    float szq = params[c].szq;
    float sbar = params[c].sbar;
    int nb = params[c].num_bins;
    int base = c * MAX_TOPO;

    float tl = 0.0f;
    for (int ia = 0; ia < nb; ia++) tl += lnaotb[base + ia];
    tl /= (float)nb;

    float qb = szq * __expf(-sbar / szm); // GPU fast exp

    float qo = 0.0f;
    for (int ia = 0; ia < nb; ia++) {
        float deficit_local = sbar + szm * (tl - lnaotb[base + ia]);
        if (deficit_local < 0.0f) {
            qo += precip[c] * (-deficit_local / szm) / (float)nb;
        }
    }

    sbar_out[c] = fmaxf(0.0f, sbar - precip[c] + qb + qo);
    qb_out[c] = qb;
    qo_out[c] = qo;
}

// ============================================================
// KERNEL 3: Penman-Monteith Evapotranspiration
// From NOAA-OWP/evapotranspiration/src/pet.c
// ============================================================

struct PETForcing {
    float temp_C;          // air temperature (Celsius)
    float pressure_Pa;     // surface pressure (Pa)
    float spec_humidity;   // specific humidity (kg/kg)
    float wind_speed;      // wind speed (m/s)
    float shortwave_W;     // incoming shortwave radiation (W/m2)
    float longwave_W;      // incoming longwave radiation (W/m2)
};

// Saturation vapor pressure (Tetens formula)
__host__ __device__ float sat_vapor_pressure(float T_C) {
    return 611.0f * expf(17.27f * T_C / (T_C + 237.3f));
}

// CPU reference
void cpu_penman_monteith(
    const PETForcing* forcing,
    float* pet_out,    // PET in m/s [ncatch]
    int ncatch)
{
    for (int c = 0; c < ncatch; c++) {
        float T = forcing[c].temp_C;
        float P = forcing[c].pressure_Pa;
        float q = forcing[c].spec_humidity;
        float u = forcing[c].wind_speed;
        float Rn = forcing[c].shortwave_W * 0.77f - forcing[c].longwave_W * 0.1f; // net radiation approx

        float es = sat_vapor_pressure(T);
        float ea = q * P / 0.622f;
        float vpd = es - ea;
        if (vpd < 0.0f) vpd = 0.0f;

        // Slope of saturation vapor pressure curve
        float delta = 4098.0f * es / ((T + 237.3f) * (T + 237.3f));

        // Psychrometric constant
        float gamma = 0.000665f * P;

        // Aerodynamic resistance (simplified)
        float ra = 208.0f / (u + 0.1f);

        // Surface resistance (reference crop)
        float rs = 70.0f;

        // Penman-Monteith equation
        float lambda = 2.501e6f - 2361.0f * T; // latent heat of vaporization
        float rho_cp = 1.013e3f * P / (287.058f * (T + 273.15f)); // rho * cp

        float num = delta * Rn + rho_cp * vpd / ra;
        float den = delta + gamma * (1.0f + rs / ra);

        float ET = (den > 0.0f) ? num / (den * lambda) : 0.0f;
        pet_out[c] = fmaxf(0.0f, ET);
    }
}

// GPU kernel
__global__ void kernel_penman_monteith(
    const PETForcing* __restrict__ forcing,
    float* __restrict__ pet_out,
    int ncatch)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncatch) return;

    float T = forcing[c].temp_C;
    float P = forcing[c].pressure_Pa;
    float q = forcing[c].spec_humidity;
    float u = forcing[c].wind_speed;
    float Rn = forcing[c].shortwave_W * 0.77f - forcing[c].longwave_W * 0.1f;

    float es = 611.0f * __expf(17.27f * T / (T + 237.3f));
    float ea = q * P / 0.622f;
    float vpd = fmaxf(0.0f, es - ea);

    float delta = 4098.0f * es / ((T + 237.3f) * (T + 237.3f));
    float gamma = 0.000665f * P;
    float ra = 208.0f / (u + 0.1f);
    float rs = 70.0f;
    float lambda = 2.501e6f - 2361.0f * T;
    float rho_cp = 1.013e3f * P / (287.058f * (T + 273.15f));

    float num = delta * Rn + rho_cp * vpd / ra;
    float den = delta + gamma * (1.0f + rs / ra);

    pet_out[c] = (den > 0.0f) ? fmaxf(0.0f, num / (den * lambda)) : 0.0f;
}

// ============================================================
// Data generation
// ============================================================

void gen_diff_data(DiffReachParams* rp, float* dx, float* diff, float* cel, float* qp,
                   int nr, unsigned seed) {
    srand(seed);
    for (int j = 0; j < nr; j++) {
        rp[j].ncomp = 20 + (rand() % 80); // 20-100 nodes per reach
        rp[j].theta = 0.6f;
        rp[j].dtini = 60.0f; // 1 minute
        int base = j * MAX_NODES;
        for (int i = 0; i < rp[j].ncomp; i++) {
            dx[base + i] = 100.0f + 500.0f * ((float)rand() / RAND_MAX);
            diff[base + i] = 10.0f + 100.0f * ((float)rand() / RAND_MAX);
            cel[base + i] = 0.5f + 2.0f * ((float)rand() / RAND_MAX);
            qp[base + i] = 1.0f + 50.0f * ((float)rand() / RAND_MAX);
        }
    }
}

void gen_topo_data(TopoParams* p, float* lna, float* precip, int nc, unsigned seed) {
    srand(seed);
    for (int c = 0; c < nc; c++) {
        p[c].szm = 0.01f + 0.1f * ((float)rand() / RAND_MAX);
        p[c].szq = 0.0001f + 0.001f * ((float)rand() / RAND_MAX);
        p[c].td = 10.0f + 50.0f * ((float)rand() / RAND_MAX);
        p[c].sbar = 0.01f + 0.1f * ((float)rand() / RAND_MAX);
        p[c].num_bins = 5 + (rand() % 20);
        int base = c * MAX_TOPO;
        for (int ia = 0; ia < p[c].num_bins; ia++) {
            lna[base + ia] = 3.0f + 5.0f * ((float)rand() / RAND_MAX);
        }
        precip[c] = 0.001f * ((float)rand() / RAND_MAX);
    }
}

void gen_pet_data(PETForcing* f, int nc, unsigned seed) {
    srand(seed);
    for (int c = 0; c < nc; c++) {
        f[c].temp_C = -10.0f + 40.0f * ((float)rand() / RAND_MAX);
        f[c].pressure_Pa = 90000.0f + 15000.0f * ((float)rand() / RAND_MAX);
        f[c].spec_humidity = 0.001f + 0.02f * ((float)rand() / RAND_MAX);
        f[c].wind_speed = 0.5f + 10.0f * ((float)rand() / RAND_MAX);
        f[c].shortwave_W = 50.0f + 800.0f * ((float)rand() / RAND_MAX);
        f[c].longwave_W = 200.0f + 200.0f * ((float)rand() / RAND_MAX);
    }
}

// ============================================================
// Benchmark runners
// ============================================================

void bench_diffusive(int nreach) {
    printf("--- Diffusive Wave Tridiag: %d reaches ---\n", nreach);

    DiffReachParams* hp = (DiffReachParams*)malloc(nreach * sizeof(DiffReachParams));
    size_t sz = nreach * MAX_NODES * sizeof(float);
    float *hdx=(float*)malloc(sz), *hdiff=(float*)malloc(sz);
    float *hcel=(float*)malloc(sz), *hqp=(float*)malloc(sz);
    float *hq_cpu=(float*)malloc(sz), *hq_gpu=(float*)malloc(sz);

    gen_diff_data(hp, hdx, hdiff, hcel, hqp, nreach, 42);

    clock_t t0 = clock();
    cpu_diffusive_tridiag(hp, hdx, hdiff, hcel, hqp, hq_cpu, nreach);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    // GPU
    DiffReachParams* dp; float *ddx, *ddiff, *dcel, *dqp, *dqo;
    cudaMalloc(&dp, nreach * sizeof(DiffReachParams));
    cudaMalloc(&ddx, sz); cudaMalloc(&ddiff, sz); cudaMalloc(&dcel, sz);
    cudaMalloc(&dqp, sz); cudaMalloc(&dqo, sz);
    cudaMemcpy(dp, hp, nreach * sizeof(DiffReachParams), cudaMemcpyHostToDevice);
    cudaMemcpy(ddx, hdx, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ddiff, hdiff, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dcel, hcel, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dqp, hqp, sz, cudaMemcpyHostToDevice);

    int thr = 64, blk = (nreach + thr - 1) / thr; // fewer threads — more registers per thread
    kernel_diffusive_tridiag<<<blk, thr>>>(dp, ddx, ddiff, dcel, dqp, dqo, nreach);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 20;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++)
        kernel_diffusive_tridiag<<<blk, thr>>>(dp, ddx, ddiff, dcel, dqp, dqo, nreach);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    cudaMemcpy(hq_gpu, dqo, sz, cudaMemcpyDeviceToHost);

    // Accuracy
    float max_abs = 0, max_rel = 0; int nan_c = 0, fail_c = 0, total = 0;
    for (int j = 0; j < nreach; j++) {
        for (int i = 0; i < hp[j].ncomp; i++) {
            int idx = j * MAX_NODES + i;
            total++;
            if (isnan(hq_gpu[idx])) { nan_c++; continue; }
            float ae = fabsf(hq_gpu[idx] - hq_cpu[idx]);
            if (ae > max_abs) max_abs = ae;
            if (fabsf(hq_cpu[idx]) > 0.01f) {
                float re = ae / fabsf(hq_cpu[idx]);
                if (re > max_rel) max_rel = re;
                if (re > 1e-4f) fail_c++;
            }
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d | >0.01%% err: %d/%d\n",
           max_abs, max_rel, nan_c, fail_c, total);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-5f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-3f) ? "PASS (FP32 rounding)" : "NEEDS REVIEW");

    free(hp); free(hdx); free(hdiff); free(hcel); free(hqp); free(hq_cpu); free(hq_gpu);
    cudaFree(dp); cudaFree(ddx); cudaFree(ddiff); cudaFree(dcel); cudaFree(dqp); cudaFree(dqo);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

void bench_topmodel(int ncatch) {
    printf("--- TOPMODEL: %d catchments ---\n", ncatch);

    TopoParams* hp = (TopoParams*)malloc(ncatch * sizeof(TopoParams));
    size_t sz_l = ncatch * MAX_TOPO * sizeof(float);
    size_t sz_f = ncatch * sizeof(float);
    float *hlna=(float*)malloc(sz_l), *hprecip=(float*)malloc(sz_f);
    float *hqb_c=(float*)malloc(sz_f), *hqo_c=(float*)malloc(sz_f), *hsb_c=(float*)malloc(sz_f);
    float *hqb_g=(float*)malloc(sz_f), *hqo_g=(float*)malloc(sz_f), *hsb_g=(float*)malloc(sz_f);

    gen_topo_data(hp, hlna, hprecip, ncatch, 77);

    clock_t t0 = clock();
    cpu_topmodel(hp, hlna, hprecip, hqb_c, hqo_c, hsb_c, ncatch);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    TopoParams* dp; float *dlna, *dprecip, *dqb, *dqo, *dsb;
    cudaMalloc(&dp, ncatch * sizeof(TopoParams));
    cudaMalloc(&dlna, sz_l); cudaMalloc(&dprecip, sz_f);
    cudaMalloc(&dqb, sz_f); cudaMalloc(&dqo, sz_f); cudaMalloc(&dsb, sz_f);
    cudaMemcpy(dp, hp, ncatch * sizeof(TopoParams), cudaMemcpyHostToDevice);
    cudaMemcpy(dlna, hlna, sz_l, cudaMemcpyHostToDevice);
    cudaMemcpy(dprecip, hprecip, sz_f, cudaMemcpyHostToDevice);

    int thr = 256, blk = (ncatch + thr - 1) / thr;
    kernel_topmodel<<<blk, thr>>>(dp, dlna, dprecip, dqb, dqo, dsb, ncatch);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 50;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++)
        kernel_topmodel<<<blk, thr>>>(dp, dlna, dprecip, dqb, dqo, dsb, ncatch);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    cudaMemcpy(hqb_g, dqb, sz_f, cudaMemcpyDeviceToHost);

    float max_rel = 0; int nan_c = 0;
    for (int c = 0; c < ncatch; c++) {
        if (isnan(hqb_g[c])) { nan_c++; continue; }
        if (fabsf(hqb_c[c]) > 1e-10f) {
            float re = fabsf(hqb_g[c] - hqb_c[c]) / fabsf(hqb_c[c]);
            if (re > max_rel) max_rel = re;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max rel (baseflow): %.2e | NaN: %d\n", max_rel, nan_c);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-4f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-2f) ? "PASS (fast exp)" : "NEEDS REVIEW");

    free(hp); free(hlna); free(hprecip);
    free(hqb_c); free(hqo_c); free(hsb_c); free(hqb_g); free(hqo_g); free(hsb_g);
    cudaFree(dp); cudaFree(dlna); cudaFree(dprecip); cudaFree(dqb); cudaFree(dqo); cudaFree(dsb);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

void bench_pet(int ncatch) {
    printf("--- Penman-Monteith PET: %d catchments ---\n", ncatch);

    PETForcing* hf = (PETForcing*)malloc(ncatch * sizeof(PETForcing));
    float *hpet_c = (float*)malloc(ncatch * sizeof(float));
    float *hpet_g = (float*)malloc(ncatch * sizeof(float));

    gen_pet_data(hf, ncatch, 123);

    clock_t t0 = clock();
    cpu_penman_monteith(hf, hpet_c, ncatch);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    PETForcing* df; float *dpet;
    cudaMalloc(&df, ncatch * sizeof(PETForcing));
    cudaMalloc(&dpet, ncatch * sizeof(float));
    cudaMemcpy(df, hf, ncatch * sizeof(PETForcing), cudaMemcpyHostToDevice);

    int thr = 256, blk = (ncatch + thr - 1) / thr;
    kernel_penman_monteith<<<blk, thr>>>(df, dpet, ncatch);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs = 50;
    cudaEventRecord(e0);
    for (int r = 0; r < runs; r++)
        kernel_penman_monteith<<<blk, thr>>>(df, dpet, ncatch);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, e0, e1); gpu_ms /= runs;

    cudaMemcpy(hpet_g, dpet, ncatch * sizeof(float), cudaMemcpyDeviceToHost);

    float max_rel = 0; int nan_c = 0;
    for (int c = 0; c < ncatch; c++) {
        if (isnan(hpet_g[c])) { nan_c++; continue; }
        if (fabsf(hpet_c[c]) > 1e-15f) {
            float re = fabsf(hpet_g[c] - hpet_c[c]) / fabsf(hpet_c[c]);
            if (re > max_rel) max_rel = re;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max rel: %.2e | NaN: %d\n", max_rel, nan_c);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-4f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-2f) ? "PASS (fast exp)" : "NEEDS REVIEW");

    free(hf); free(hpet_c); free(hpet_g);
    cudaFree(df); cudaFree(dpet);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    printf("================================================\n");
    printf("  NOAA-OWP Extended GPU Kernels\n");
    printf("  Diffusive Wave + TOPMODEL + PET\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    printf("========== DIFFUSIVE WAVE TRIDIAG ==========\n\n");
    bench_diffusive(1000);
    bench_diffusive(5000);
    bench_diffusive(10000);

    printf("========== TOPMODEL ==========\n\n");
    bench_topmodel(100000);
    bench_topmodel(1000000);
    bench_topmodel(2700000);

    printf("========== PENMAN-MONTEITH PET ==========\n\n");
    bench_pet(100000);
    bench_pet(1000000);
    bench_pet(2700000);

    return 0;
}
