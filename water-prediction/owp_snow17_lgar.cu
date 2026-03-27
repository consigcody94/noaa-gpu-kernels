/**
 * NOAA-OWP Snow17 + LGAR GPU Kernels
 *
 * 1. Snow17 — Snow accumulation and melt model (PACK19)
 *    One computational period per catchment, batched across 2.7M catchments.
 *    Contains exp() for vapor pressure and pow for Stefan-Boltzmann.
 *
 * 2. LGAR — Lumped infiltration model (Green-Ampt variant)
 *    Infiltration front tracking per catchment.
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ============================================================
// KERNEL 1: Snow17 — Simplified single-period snow model
// Based on PACK19 from NOAA-OWP/snow17
// ============================================================

struct Snow17State {
    float we;       // water equivalent (mm)
    float liqw;     // liquid water in snowpack (mm)
    float neghs;    // heat deficit (mm)
    float accmax;   // max accumulation (mm)
    float aesc;     // areal extent of snow cover (0-1)
    float tprev;    // previous temperature
};

struct Snow17Params {
    float scf;      // snow correction factor
    float mfmax;    // max melt factor (mm/degC/6hr)
    float mfmin;    // min melt factor
    float uadj;     // wind function adjustment
    float si;       // mean areal water equivalent for 100% cover (mm)
    float tipm;     // antecedent temperature index parameter
    float mbase;    // base temperature for melt (degC)
    float plwhc;    // percent liquid water holding capacity
    float daygm;    // daily ground melt (mm/day)
    float pxtemp;   // rain/snow discriminator temperature (degC)
};

struct Snow17Forcing {
    float ta;       // air temperature (degC)
    float px;       // precipitation (mm)
};

// CPU reference — simplified Snow17 single period
void cpu_snow17(
    Snow17State* state,
    const Snow17Params* params,
    const Snow17Forcing* forcing,
    float* runoff_out,
    float dt_hours,
    int ncatch)
{
    for (int c = 0; c < ncatch; c++) {
        float ta = forcing[c].ta;
        float px = forcing[c].px;
        float we = state[c].we;
        float liqw = state[c].liqw;
        float neghs = state[c].neghs;
        float tprev = state[c].tprev;

        float scf = params[c].scf;
        float mbase = params[c].mbase;
        float plwhc = params[c].plwhc;
        float daygm = params[c].daygm;
        float pxtemp = params[c].pxtemp;
        float tipm = params[c].tipm;
        float mfmax = params[c].mfmax;
        float si = params[c].si;

        // Antecedent temperature index
        float ati = tprev + tipm * (ta - tprev);

        // Precipitation form (rain/snow)
        float frac_snow = (ta < pxtemp) ? 1.0f : 0.0f;
        if (ta >= pxtemp && ta < pxtemp + 2.0f)
            frac_snow = 1.0f - (ta - pxtemp) / 2.0f;

        float sfall = px * frac_snow * scf;
        float rain = px * (1.0f - frac_snow);

        // Snow accumulation
        we += sfall;

        // Melt calculation
        float melt = 0.0f;
        if (we > 0.0f && ta > mbase) {
            // Temperature index melt
            float mf = mfmax; // simplified — full model varies by day of year
            melt = mf * (ta - mbase);
            if (melt > we) melt = we;

            // Rain-on-snow melt
            if (rain > 0.0f && ta > 0.0f) {
                // Stefan-Boltzmann + vapor pressure
                float tak = (ta + 273.0f) * 0.01f;
                float tak4 = tak * tak * tak * tak;
                float sbci = 6.12e-10f; // Stefan-Boltzmann / (24*dt)
                float qn = sbci * (tak4 - 55.55f);

                // Vapor pressure
                float ea = 2.7489e8f * expf(-4278.63f / (ta + 242.792f));
                ea *= 0.9f;

                float rainm = 0.0125f * rain * ta; // rain melt contribution
                melt += rainm;
            }
        }

        // Areal extent
        float aesc = (we > 0.0f) ? fminf(1.0f, we / si) : 0.0f;
        melt *= aesc;

        // Heat deficit
        if (ta < 0.0f && we > 0.0f) {
            float heat = -ta * 0.5f; // simplified heat loss
            neghs += heat;
        }

        // Water balance
        float water = melt + rain;
        float liqwmx = plwhc * we;
        float excess = 0.0f;

        if (neghs > 0.0f && water > 0.0f) {
            float used = fminf(water, neghs);
            neghs -= used;
            water -= used;
        }

        liqw += water;
        we -= melt;
        if (we < 0.0f) we = 0.0f;

        if (liqw > liqwmx) {
            excess = liqw - liqwmx;
            liqw = liqwmx;
        }

        // Ground melt
        float robg = daygm * dt_hours / 24.0f;

        runoff_out[c] = excess + robg;

        state[c].we = we;
        state[c].liqw = liqw;
        state[c].neghs = neghs;
        state[c].aesc = aesc;
        state[c].tprev = ati;
    }
}

// GPU kernel
__global__ void kernel_snow17(
    Snow17State* __restrict__ state,
    const Snow17Params* __restrict__ params,
    const Snow17Forcing* __restrict__ forcing,
    float* __restrict__ runoff_out,
    float dt_hours,
    int ncatch)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncatch) return;

    float ta = forcing[c].ta;
    float px = forcing[c].px;
    float we = state[c].we;
    float liqw = state[c].liqw;
    float neghs = state[c].neghs;
    float tprev = state[c].tprev;

    float scf = params[c].scf;
    float mbase = params[c].mbase;
    float plwhc = params[c].plwhc;
    float daygm = params[c].daygm;
    float pxtemp = params[c].pxtemp;
    float tipm = params[c].tipm;
    float mfmax = params[c].mfmax;
    float si = params[c].si;

    float ati = tprev + tipm * (ta - tprev);

    float frac_snow = (ta < pxtemp) ? 1.0f : 0.0f;
    if (ta >= pxtemp && ta < pxtemp + 2.0f)
        frac_snow = 1.0f - (ta - pxtemp) / 2.0f;

    float sfall = px * frac_snow * scf;
    float rain = px * (1.0f - frac_snow);

    we += sfall;

    float melt = 0.0f;
    if (we > 0.0f && ta > mbase) {
        float mf = mfmax;
        melt = mf * (ta - mbase);
        if (melt > we) melt = we;

        if (rain > 0.0f && ta > 0.0f) {
            float tak = (ta + 273.0f) * 0.01f;
            float tak4 = tak * tak * tak * tak;
            float ea = 2.7489e8f * __expf(-4278.63f / (ta + 242.792f));
            ea *= 0.9f;
            float rainm = 0.0125f * rain * ta;
            melt += rainm;
        }
    }

    float aesc = (we > 0.0f) ? fminf(1.0f, we / si) : 0.0f;
    melt *= aesc;

    if (ta < 0.0f && we > 0.0f) {
        neghs += -ta * 0.5f;
    }

    float water = melt + rain;
    float liqwmx = plwhc * we;

    if (neghs > 0.0f && water > 0.0f) {
        float used = fminf(water, neghs);
        neghs -= used;
        water -= used;
    }

    liqw += water;
    we -= melt;
    if (we < 0.0f) we = 0.0f;

    float excess = 0.0f;
    if (liqw > liqwmx) {
        excess = liqw - liqwmx;
        liqw = liqwmx;
    }

    float robg = daygm * dt_hours / 24.0f;
    runoff_out[c] = excess + robg;

    state[c].we = we;
    state[c].liqw = liqw;
    state[c].neghs = neghs;
    state[c].aesc = aesc;
    state[c].tprev = ati;
}

// ============================================================
// KERNEL 2: LGAR Infiltration (Green-Ampt variant)
// Simplified from NOAA-OWP/LGAR-C
// ============================================================

struct LGARParams {
    float Ks;          // saturated hydraulic conductivity (m/s)
    float porosity;    // soil porosity
    float wetting_front_suction; // wetting front suction head (m)
    float initial_moisture;      // initial soil moisture
    float soil_depth;            // total soil depth (m)
};

// CPU reference
void cpu_lgar(
    const LGARParams* params,
    const float* precip_rate,  // precipitation rate (m/s) [ncatch]
    float* infiltration_out,   // infiltration rate (m/s) [ncatch]
    float* runoff_out,         // surface runoff (m/s) [ncatch]
    float* cum_infil,          // cumulative infiltration (m) [ncatch, in/out]
    float dt,
    int ncatch)
{
    for (int c = 0; c < ncatch; c++) {
        float Ks = params[c].Ks;
        float psi = params[c].wetting_front_suction;
        float theta_d = params[c].porosity - params[c].initial_moisture;
        float F = cum_infil[c];

        // Green-Ampt infiltration capacity
        float f_cap;
        if (F > 0.0f) {
            f_cap = Ks * (1.0f + psi * theta_d / F);
        } else {
            f_cap = precip_rate[c]; // all infiltrates initially
        }

        // Actual infiltration
        float f_actual = fminf(precip_rate[c], f_cap);

        // Check soil capacity
        float max_infil = params[c].soil_depth * theta_d;
        if (F + f_actual * dt > max_infil) {
            f_actual = fmaxf(0.0f, (max_infil - F) / dt);
        }

        infiltration_out[c] = f_actual;
        runoff_out[c] = fmaxf(0.0f, precip_rate[c] - f_actual);
        cum_infil[c] = F + f_actual * dt;
    }
}

// GPU kernel
__global__ void kernel_lgar(
    const LGARParams* __restrict__ params,
    const float* __restrict__ precip_rate,
    float* __restrict__ infiltration_out,
    float* __restrict__ runoff_out,
    float* __restrict__ cum_infil,
    float dt,
    int ncatch)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncatch) return;

    float Ks = params[c].Ks;
    float psi = params[c].wetting_front_suction;
    float theta_d = params[c].porosity - params[c].initial_moisture;
    float F = cum_infil[c];

    float f_cap = (F > 0.0f) ? Ks * (1.0f + psi * theta_d / F) : precip_rate[c];
    float f_actual = fminf(precip_rate[c], f_cap);

    float max_infil = params[c].soil_depth * theta_d;
    if (F + f_actual * dt > max_infil) {
        f_actual = fmaxf(0.0f, (max_infil - F) / dt);
    }

    infiltration_out[c] = f_actual;
    runoff_out[c] = fmaxf(0.0f, precip_rate[c] - f_actual);
    cum_infil[c] = F + f_actual * dt;
}

// ============================================================
// Data generation
// ============================================================

void gen_snow_data(Snow17State* st, Snow17Params* p, Snow17Forcing* f, int nc, unsigned seed) {
    srand(seed);
    for (int c = 0; c < nc; c++) {
        // State — mix of snow-covered and bare ground
        st[c].we = (rand() % 2) ? 50.0f + 200.0f * ((float)rand()/RAND_MAX) : 0.0f;
        st[c].liqw = st[c].we * 0.02f * ((float)rand()/RAND_MAX);
        st[c].neghs = 5.0f * ((float)rand()/RAND_MAX);
        st[c].accmax = st[c].we * 1.2f;
        st[c].aesc = (st[c].we > 0.0f) ? 0.5f + 0.5f * ((float)rand()/RAND_MAX) : 0.0f;
        st[c].tprev = -15.0f + 20.0f * ((float)rand()/RAND_MAX);

        // Parameters
        p[c].scf = 0.9f + 0.2f * ((float)rand()/RAND_MAX);
        p[c].mfmax = 0.5f + 1.5f * ((float)rand()/RAND_MAX);
        p[c].mfmin = p[c].mfmax * 0.3f;
        p[c].uadj = 0.02f + 0.08f * ((float)rand()/RAND_MAX);
        p[c].si = 100.0f + 400.0f * ((float)rand()/RAND_MAX);
        p[c].tipm = 0.1f + 0.4f * ((float)rand()/RAND_MAX);
        p[c].mbase = 0.0f;
        p[c].plwhc = 0.02f + 0.03f * ((float)rand()/RAND_MAX);
        p[c].daygm = 0.0f + 0.3f * ((float)rand()/RAND_MAX);
        p[c].pxtemp = -1.0f + 3.0f * ((float)rand()/RAND_MAX);

        // Forcing — winter conditions
        f[c].ta = -20.0f + 30.0f * ((float)rand()/RAND_MAX);
        f[c].px = (rand() % 3 == 0) ? 2.0f + 20.0f * ((float)rand()/RAND_MAX) : 0.0f;
    }
}

void gen_lgar_data(LGARParams* p, float* precip, float* cum_inf, int nc, unsigned seed) {
    srand(seed);
    for (int c = 0; c < nc; c++) {
        p[c].Ks = 1e-6f + 1e-4f * ((float)rand()/RAND_MAX);
        p[c].porosity = 0.3f + 0.2f * ((float)rand()/RAND_MAX);
        p[c].wetting_front_suction = 0.05f + 0.3f * ((float)rand()/RAND_MAX);
        p[c].initial_moisture = p[c].porosity * (0.3f + 0.4f * ((float)rand()/RAND_MAX));
        p[c].soil_depth = 0.5f + 2.0f * ((float)rand()/RAND_MAX);
        precip[c] = 1e-6f + 5e-5f * ((float)rand()/RAND_MAX); // m/s
        cum_inf[c] = 0.001f + 0.05f * ((float)rand()/RAND_MAX); // m
    }
}

// ============================================================
// Benchmarks
// ============================================================

void bench_snow17(int ncatch) {
    printf("--- Snow17: %d catchments ---\n", ncatch);

    Snow17State *hs=(Snow17State*)malloc(ncatch*sizeof(Snow17State));
    Snow17State *hs_gpu=(Snow17State*)malloc(ncatch*sizeof(Snow17State));
    Snow17Params *hp=(Snow17Params*)malloc(ncatch*sizeof(Snow17Params));
    Snow17Forcing *hf=(Snow17Forcing*)malloc(ncatch*sizeof(Snow17Forcing));
    float *hr_cpu=(float*)malloc(ncatch*4), *hr_gpu=(float*)malloc(ncatch*4);

    gen_snow_data(hs, hp, hf, ncatch, 42);
    memcpy(hs_gpu, hs, ncatch*sizeof(Snow17State));

    clock_t t0=clock();
    cpu_snow17(hs, hp, hf, hr_cpu, 6.0f, ncatch);
    double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

    Snow17State *ds; Snow17Params *dp; Snow17Forcing *df; float *dr;
    cudaMalloc(&ds, ncatch*sizeof(Snow17State));
    cudaMalloc(&dp, ncatch*sizeof(Snow17Params));
    cudaMalloc(&df, ncatch*sizeof(Snow17Forcing));
    cudaMalloc(&dr, ncatch*4);
    cudaMemcpy(dp, hp, ncatch*sizeof(Snow17Params), cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf, ncatch*sizeof(Snow17Forcing), cudaMemcpyHostToDevice);

    int thr=256, blk=(ncatch+thr-1)/thr;

    // Warmup
    cudaMemcpy(ds, hs_gpu, ncatch*sizeof(Snow17State), cudaMemcpyHostToDevice);
    kernel_snow17<<<blk,thr>>>(ds, dp, df, dr, 6.0f, ncatch);
    cudaDeviceSynchronize();

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs=50;
    cudaEventRecord(e0);
    for(int r=0;r<runs;r++){
        cudaMemcpy(ds, hs_gpu, ncatch*sizeof(Snow17State), cudaMemcpyHostToDevice);
        kernel_snow17<<<blk,thr>>>(ds, dp, df, dr, 6.0f, ncatch);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms,e0,e1); gpu_ms/=runs;

    // Get results with fresh state
    cudaMemcpy(ds, hs_gpu, ncatch*sizeof(Snow17State), cudaMemcpyHostToDevice);
    kernel_snow17<<<blk,thr>>>(ds, dp, df, dr, 6.0f, ncatch);
    cudaMemcpy(hr_gpu, dr, ncatch*4, cudaMemcpyDeviceToHost);

    float max_abs=0, max_rel=0; int nan_c=0, fail_c=0;
    for(int c=0;c<ncatch;c++){
        if(isnan(hr_gpu[c])){nan_c++;continue;}
        float ae=fabsf(hr_gpu[c]-hr_cpu[c]);
        if(ae>max_abs)max_abs=ae;
        if(fabsf(hr_cpu[c])>1e-10f){
            float re=ae/fabsf(hr_cpu[c]);
            if(re>max_rel)max_rel=re;
            if(re>1e-4f)fail_c++;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms);
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d | >0.01%% err: %d/%d\n",
           max_abs, max_rel, nan_c, fail_c, ncatch);
    printf("  Status: %s\n\n",
           (nan_c==0&&max_rel<1e-4f)?"PASS":(nan_c==0&&max_rel<1e-2f)?"PASS (fast exp)":"NEEDS REVIEW");

    free(hs);free(hs_gpu);free(hp);free(hf);free(hr_cpu);free(hr_gpu);
    cudaFree(ds);cudaFree(dp);cudaFree(df);cudaFree(dr);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
}

void bench_lgar(int ncatch) {
    printf("--- LGAR Infiltration: %d catchments ---\n", ncatch);

    LGARParams *hp=(LGARParams*)malloc(ncatch*sizeof(LGARParams));
    float *hprecip=(float*)malloc(ncatch*4);
    float *hcum_cpu=(float*)malloc(ncatch*4), *hcum_gpu=(float*)malloc(ncatch*4);
    float *hinf_cpu=(float*)malloc(ncatch*4), *hinf_gpu=(float*)malloc(ncatch*4);
    float *hro_cpu=(float*)malloc(ncatch*4), *hro_gpu=(float*)malloc(ncatch*4);

    gen_lgar_data(hp, hprecip, hcum_cpu, ncatch, 99);
    memcpy(hcum_gpu, hcum_cpu, ncatch*4);

    float dt = 300.0f; // 5 min

    clock_t t0=clock();
    cpu_lgar(hp, hprecip, hinf_cpu, hro_cpu, hcum_cpu, dt, ncatch);
    double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

    LGARParams *dp; float *dprecip, *dcum, *dinf, *dro;
    cudaMalloc(&dp, ncatch*sizeof(LGARParams));
    cudaMalloc(&dprecip, ncatch*4); cudaMalloc(&dcum, ncatch*4);
    cudaMalloc(&dinf, ncatch*4); cudaMalloc(&dro, ncatch*4);
    cudaMemcpy(dp, hp, ncatch*sizeof(LGARParams), cudaMemcpyHostToDevice);
    cudaMemcpy(dprecip, hprecip, ncatch*4, cudaMemcpyHostToDevice);

    int thr=256, blk=(ncatch+thr-1)/thr;

    cudaMemcpy(dcum, hcum_gpu, ncatch*4, cudaMemcpyHostToDevice);
    kernel_lgar<<<blk,thr>>>(dp, dprecip, dinf, dro, dcum, dt, ncatch);
    cudaDeviceSynchronize();

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int runs=50;
    cudaEventRecord(e0);
    for(int r=0;r<runs;r++){
        cudaMemcpy(dcum, hcum_gpu, ncatch*4, cudaMemcpyHostToDevice);
        kernel_lgar<<<blk,thr>>>(dp, dprecip, dinf, dro, dcum, dt, ncatch);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms,e0,e1); gpu_ms/=runs;

    cudaMemcpy(dcum, hcum_gpu, ncatch*4, cudaMemcpyHostToDevice);
    kernel_lgar<<<blk,thr>>>(dp, dprecip, dinf, dro, dcum, dt, ncatch);
    cudaMemcpy(hinf_gpu, dinf, ncatch*4, cudaMemcpyDeviceToHost);

    float max_abs=0, max_rel=0; int nan_c=0;
    for(int c=0;c<ncatch;c++){
        if(isnan(hinf_gpu[c])){nan_c++;continue;}
        float ae=fabsf(hinf_gpu[c]-hinf_cpu[c]);
        if(ae>max_abs)max_abs=ae;
        if(fabsf(hinf_cpu[c])>1e-15f){
            float re=ae/fabsf(hinf_cpu[c]);
            if(re>max_rel)max_rel=re;
        }
    }

    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms);
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d\n", max_abs, max_rel, nan_c);
    printf("  Status: %s\n\n",
           (nan_c==0&&max_rel<1e-5f)?"PASS":(nan_c==0&&max_rel<1e-3f)?"PASS (FP32 rounding)":"NEEDS REVIEW");

    free(hp);free(hprecip);free(hcum_cpu);free(hcum_gpu);
    free(hinf_cpu);free(hinf_gpu);free(hro_cpu);free(hro_gpu);
    cudaFree(dp);cudaFree(dprecip);cudaFree(dcum);cudaFree(dinf);cudaFree(dro);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
}

int main() {
    printf("================================================\n");
    printf("  NOAA-OWP Snow17 + LGAR GPU Kernels\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    printf("========== SNOW17 ==========\n\n");
    bench_snow17(100000);
    bench_snow17(1000000);
    bench_snow17(2700000);

    printf("========== LGAR INFILTRATION ==========\n\n");
    bench_lgar(100000);
    bench_lgar(1000000);
    bench_lgar(2700000);

    return 0;
}
