/**
 * NOAA-OWP Extended GPU Kernels — REAL DATA harness
 *
 * Copy of water-prediction/owp_extended_kernels.cu with the synthetic data
 * generators (gen_topo_data / gen_pet_data) replaced by loaders for real
 * NOAA-OWP datasets. Kernel math and CPU reference implementations are copied
 * VERBATIM from the original and unchanged.
 *
 * The diffusive-wave kernel from the original file is NOT included here: it has
 * no real dataset in the upstream repos that maps to its simplified inputs
 * (constant-alpha benchmark form), so only TOPMODEL and PET are validated on
 * real data. (Originals untouched.)
 *
 * Real data:
 *  2. TOPMODEL — the real catchment dataset shipped in NOAA-OWP/topmodel data/
 *     ("Extracted study basin: Taegu Pyungkwang River"): 30-ordinate ln(a/tanB)
 *     areal distribution (subcat.dat), calibrated parameters (params.dat),
 *     950 hourly rain/pe/Qobs records (inputs.dat). szq and initial sbar are
 *     derived with the model's own init equations (see prep_topmodel.py).
 *     A 950-entry (precip, sbar) state library is built by running the
 *     UNTOUCHED CPU reference sequentially through the real rain series.
 *  3. Penman-Monteith PET — 25,582 real AORC hourly forcing records from
 *     NOAA-OWP/cfe (cat-87 Dec 2015 + Laramie River Jun 2009 - Apr 2012).
 *
 * Prep: prep_topmodel.py, prep_pet.py  (see headers for provenance)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_TOPO  30   // max topodex histogram bins

// ============================================================
// KERNEL 2: TOPMODEL Runoff Generation        (verbatim)
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
// KERNEL 3: Penman-Monteith Evapotranspiration   (verbatim)
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
// REAL DATA loading (replaces synthetic generators)
// ============================================================

static void die(const char* msg) { fprintf(stderr, "FATAL: %s\n", msg); exit(1); }

// ---- TOPMODEL: real Pyungkwang River catchment + state trajectory ----
typedef struct {
    float szm, szq, td, sbar0;
    int num_bins;
    float lnaotb[MAX_TOPO];
    int nstep;
    float* rain;     // [nstep] m per timestep
    float* sbar_t;   // [nstep] sbar state BEFORE step t (from CPU-reference trajectory)
} TopoLibrary;

static TopoLibrary g_topo;

void load_topo_library(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open topmodel_realdata.bin (run prep_topmodel.py)");
    int magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x544F504D) die("topo magic");
    fread(&g_topo.num_bins, 4, 1, f);
    if (g_topo.num_bins > MAX_TOPO) die("num_bins > MAX_TOPO");
    fread(&g_topo.szm, 4, 1, f);
    fread(&g_topo.szq, 4, 1, f);
    fread(&g_topo.td, 4, 1, f);
    fread(&g_topo.sbar0, 4, 1, f);
    memset(g_topo.lnaotb, 0, sizeof(g_topo.lnaotb));
    fread(g_topo.lnaotb, 4, g_topo.num_bins, f);
    fread(&g_topo.nstep, 4, 1, f);
    g_topo.rain = (float*)malloc(g_topo.nstep * 4);
    fread(g_topo.rain, 4, g_topo.nstep, f);
    fclose(f);

    // Build the sbar state trajectory by running the UNTOUCHED CPU reference
    // sequentially through the real rain series (ncatch = 1 each step).
    g_topo.sbar_t = (float*)malloc(g_topo.nstep * 4);
    TopoParams p;
    p.szm = g_topo.szm; p.szq = g_topo.szq; p.td = g_topo.td;
    p.sbar = g_topo.sbar0; p.num_bins = g_topo.num_bins;
    float qb, qo, sbar_next;
    for (int t = 0; t < g_topo.nstep; t++) {
        g_topo.sbar_t[t] = p.sbar;
        cpu_topmodel(&p, g_topo.lnaotb, &g_topo.rain[t], &qb, &qo, &sbar_next, 1);
        p.sbar = sbar_next;
    }
    printf("[data] TOPMODEL: Pyungkwang River, %d bins, szm=%.4f szq=%.5f sbar0=%.5f, "
           "%d real hourly steps\n",
           g_topo.num_bins, g_topo.szm, g_topo.szq, g_topo.sbar0, g_topo.nstep);
}

// Fill a batch by tiling the real (precip, sbar) trajectory (replication documented)
void fill_topo_batch(TopoParams* p, float* lna, float* precip, int nc) {
    for (int c = 0; c < nc; c++) {
        int t = c % g_topo.nstep;
        p[c].szm = g_topo.szm;
        p[c].szq = g_topo.szq;
        p[c].td = g_topo.td;
        p[c].sbar = g_topo.sbar_t[t];
        p[c].num_bins = g_topo.num_bins;
        memcpy(&lna[(size_t)c * MAX_TOPO], g_topo.lnaotb, MAX_TOPO * 4);
        precip[c] = g_topo.rain[t];
    }
}

// ---- PET: real AORC forcing records ----
static PETForcing* g_pet_rec = NULL;
static int g_pet_n = 0;

void load_pet_records(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open pet_realdata.bin (run prep_pet.py)");
    int magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x50455446) die("pet magic");
    fread(&g_pet_n, 4, 1, f);
    g_pet_rec = (PETForcing*)malloc((size_t)g_pet_n * sizeof(PETForcing));
    if (fread(g_pet_rec, sizeof(PETForcing), g_pet_n, f) != (size_t)g_pet_n) die("pet payload");
    fclose(f);
    printf("[data] PET: %d real AORC hourly records (cat-87 + Laramie)\n", g_pet_n);
}

void fill_pet_batch(PETForcing* fz, int nc) {
    for (int c = 0; c < nc; c++) fz[c] = g_pet_rec[c % g_pet_n];
}

// ============================================================
// Benchmark runners  (structure as original; data source swapped)
// ============================================================

void bench_topmodel(int ncatch) {
    printf("--- TOPMODEL (REAL Pyungkwang data): %d catchments ---\n", ncatch);

    TopoParams* hp = (TopoParams*)malloc(ncatch * sizeof(TopoParams));
    size_t sz_l = (size_t)ncatch * MAX_TOPO * sizeof(float);
    size_t sz_f = ncatch * sizeof(float);
    float *hlna=(float*)malloc(sz_l), *hprecip=(float*)malloc(sz_f);
    float *hqb_c=(float*)malloc(sz_f), *hqo_c=(float*)malloc(sz_f), *hsb_c=(float*)malloc(sz_f);
    float *hqb_g=(float*)malloc(sz_f), *hqo_g=(float*)malloc(sz_f), *hsb_g=(float*)malloc(sz_f);

    fill_topo_batch(hp, hlna, hprecip, ncatch);    // REAL DATA (was gen_topo_data)

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
    cudaMemcpy(hqo_g, dqo, sz_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(hsb_g, dsb, sz_f, cudaMemcpyDeviceToHost);

    // qb (baseflow) accuracy — as original; plus qo and sbar on real data
    float max_rel = 0; int nan_c = 0;
    for (int c = 0; c < ncatch; c++) {
        if (isnan(hqb_g[c])) { nan_c++; continue; }
        if (fabsf(hqb_c[c]) > 1e-10f) {
            float re = fabsf(hqb_g[c] - hqb_c[c]) / fabsf(hqb_c[c]);
            if (re > max_rel) max_rel = re;
        }
    }
    float max_rel_qo = 0, max_rel_sb = 0; int nan_qo = 0, nan_sb = 0;
    for (int c = 0; c < ncatch; c++) {
        if (isnan(hqo_g[c])) nan_qo++;
        else if (fabsf(hqo_c[c]) > 1e-10f) {
            float re = fabsf(hqo_g[c] - hqo_c[c]) / fabsf(hqo_c[c]);
            if (re > max_rel_qo) max_rel_qo = re;
        }
        if (isnan(hsb_g[c])) nan_sb++;
        else if (fabsf(hsb_c[c]) > 1e-10f) {
            float re = fabsf(hsb_g[c] - hsb_c[c]) / fabsf(hsb_c[c]);
            if (re > max_rel_sb) max_rel_sb = re;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Max rel (baseflow): %.2e | NaN: %d\n", max_rel, nan_c);
    printf("  Max rel (overland): %.2e (NaN %d) | Max rel (sbar): %.2e (NaN %d)\n",
           max_rel_qo, nan_qo, max_rel_sb, nan_sb);
    printf("  Status: %s\n\n",
           (nan_c == 0 && max_rel < 1e-4f) ? "PASS" :
           (nan_c == 0 && max_rel < 1e-2f) ? "PASS (fast exp)" : "NEEDS REVIEW");

    free(hp); free(hlna); free(hprecip);
    free(hqb_c); free(hqo_c); free(hsb_c); free(hqb_g); free(hqo_g); free(hsb_g);
    cudaFree(dp); cudaFree(dlna); cudaFree(dprecip); cudaFree(dqb); cudaFree(dqo); cudaFree(dsb);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

void bench_pet(int ncatch) {
    printf("--- Penman-Monteith PET (REAL AORC data): %d catchments ---\n", ncatch);

    PETForcing* hf = (PETForcing*)malloc((size_t)ncatch * sizeof(PETForcing));
    float *hpet_c = (float*)malloc(ncatch * sizeof(float));
    float *hpet_g = (float*)malloc(ncatch * sizeof(float));

    fill_pet_batch(hf, ncatch);    // REAL DATA (was gen_pet_data)

    clock_t t0 = clock();
    cpu_penman_monteith(hf, hpet_c, ncatch);
    double cpu_ms = 1000.0 * (clock() - t0) / (double)CLOCKS_PER_SEC;

    PETForcing* df; float *dpet;
    cudaMalloc(&df, (size_t)ncatch * sizeof(PETForcing));
    cudaMalloc(&dpet, ncatch * sizeof(float));
    cudaMemcpy(df, hf, (size_t)ncatch * sizeof(PETForcing), cudaMemcpyHostToDevice);

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

int main(int argc, char** argv) {
    const char* topo_path = (argc > 1) ? argv[1] : "data/topmodel_realdata.bin";
    const char* pet_path  = (argc > 2) ? argv[2] : "data/pet_realdata.bin";

    printf("================================================\n");
    printf("  NOAA-OWP Extended GPU Kernels — REAL DATA\n");
    printf("  TOPMODEL (Pyungkwang River) + PET (AORC)\n");
    printf("  RTX 5070 12GB (sm_120)\n");
    printf("================================================\n\n");

    load_topo_library(topo_path);
    load_pet_records(pet_path);
    printf("\n");

    printf("========== TOPMODEL ==========\n\n");
    bench_topmodel(950);      // pure real trajectory, no replication
    bench_topmodel(100000);
    bench_topmodel(1000000);
    bench_topmodel(2700000);

    printf("========== PENMAN-MONTEITH PET ==========\n\n");
    bench_pet(25582);         // pure real records, no replication
    bench_pet(100000);
    bench_pet(1000000);
    bench_pet(2700000);

    return 0;
}
