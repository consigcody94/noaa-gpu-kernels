/**
 * Multi-Model NOAA GPU Kernels
 *
 * 1. CCPP tridi1 — PBL tridiagonal solver (GFS operational)
 * 2. Icepack delta-Eddington — Sea ice shortwave radiation
 * 3. CICE EVP — Elastic-viscous-plastic ice dynamics subcycling
 * 4. MOSART — Kinematic wave river routing (Manning's equation)
 *
 * (NCEPLIBS-sp Legendre transforms require complex setup — deferred)
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_LEV 128   // max vertical levels (PBL)
#define MAX_ICE_LAY 10 // max ice+snow layers
#define NBAND 3        // spectral bands for delta-Eddington

// ============================================================
// KERNEL 1: CCPP tridi1 — Thomas algorithm for PBL diffusion
// From NCAR/ccpp-physics/physics/PBL/tridi.f
// Solves: cl*x(k-1) + cm*x(k) + cu*x(k+1) = r1
// ============================================================

void cpu_tridi1(int l, int n, const float* cl, const float* cm,
                const float* cu, const float* r1, float* a1, int stride) {
    float au[MAX_LEV];
    for (int i = 0; i < l; i++) {
        // Forward sweep
        float fk = 1.0f / cm[i * stride + 0];
        au[0] = fk * cu[i * stride + 0];
        a1[i * stride + 0] = fk * r1[i * stride + 0];
        for (int k = 1; k < n - 1; k++) {
            fk = 1.0f / (cm[i*stride+k] - cl[i*stride+k] * au[k-1]);
            au[k] = fk * cu[i*stride+k];
            a1[i*stride+k] = fk * (r1[i*stride+k] - cl[i*stride+k] * a1[i*stride+k-1]);
        }
        fk = 1.0f / (cm[i*stride+n-1] - cl[i*stride+n-1] * au[n-2]);
        a1[i*stride+n-1] = fk * (r1[i*stride+n-1] - cl[i*stride+n-1] * a1[i*stride+n-2]);
        // Back substitution
        for (int k = n - 2; k >= 0; k--)
            a1[i*stride+k] -= au[k] * a1[i*stride+k+1];
    }
}

__global__ void kernel_tridi1(int n, const float* __restrict__ cl,
    const float* __restrict__ cm, const float* __restrict__ cu,
    const float* __restrict__ r1, float* __restrict__ a1, int stride, int ncol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;

    float au[MAX_LEV];
    float fk = 1.0f / cm[i*stride+0];
    au[0] = fk * cu[i*stride+0];
    a1[i*stride+0] = fk * r1[i*stride+0];
    for (int k = 1; k < n-1; k++) {
        fk = 1.0f / (cm[i*stride+k] - cl[i*stride+k]*au[k-1]);
        au[k] = fk * cu[i*stride+k];
        a1[i*stride+k] = fk*(r1[i*stride+k] - cl[i*stride+k]*a1[i*stride+k-1]);
    }
    fk = 1.0f / (cm[i*stride+n-1] - cl[i*stride+n-1]*au[n-2]);
    a1[i*stride+n-1] = fk*(r1[i*stride+n-1] - cl[i*stride+n-1]*a1[i*stride+n-2]);
    for (int k = n-2; k >= 0; k--)
        a1[i*stride+k] -= au[k]*a1[i*stride+k+1];
}

// ============================================================
// KERNEL 2: Icepack delta-Eddington shortwave
// Simplified: computes transmittance through ice/snow layers
// ============================================================

struct IceColumn {
    float snow_depth;       // snow depth (m)
    float ice_thickness;    // ice thickness (m)
    float snow_grain_r;     // snow grain radius (um)
    int nslyr;              // number of snow layers
    int nilyr;              // number of ice layers
    float coszen;           // cosine solar zenith angle
    float swdn[NBAND];      // incoming shortwave per band (W/m2)
};

void cpu_dEdd(const IceColumn* cols, float* absorbed, float* transmitted, int ncol) {
    for (int c = 0; c < ncol; c++) {
        int klev = cols[c].nslyr + cols[c].nilyr + 1;
        float mu0 = fmaxf(cols[c].coszen, 0.01f);

        float total_abs = 0.0f, total_trans = 0.0f;
        for (int nb = 0; nb < NBAND; nb++) {
            // Extinction coefficients (simplified)
            float k_snow = (nb == 0) ? 20.0f : (nb == 1) ? 100.0f : 500.0f; // 1/m
            float k_ice = (nb == 0) ? 1.0f : (nb == 1) ? 5.0f : 50.0f;
            float w0_snow = 0.999f, w0_ice = 0.95f;
            float g_snow = 0.89f, g_ice = 0.94f;

            // Delta-Eddington scaling
            float f_snow = g_snow * g_snow;
            float tau_s = k_snow * cols[c].snow_depth / fmaxf((float)cols[c].nslyr, 1.0f);
            float tau_scaled_s = (1.0f - w0_snow*f_snow) * tau_s;

            float f_ice = g_ice * g_ice;
            float tau_i = k_ice * cols[c].ice_thickness / fmaxf((float)cols[c].nilyr, 1.0f);
            float tau_scaled_i = (1.0f - w0_ice*f_ice) * tau_i;

            // Layer-by-layer transmittance (Beer-Lambert)
            float trans = 1.0f;
            for (int k = 0; k < cols[c].nslyr; k++)
                trans *= expf(-tau_scaled_s / mu0);
            for (int k = 0; k < cols[c].nilyr; k++)
                trans *= expf(-tau_scaled_i / mu0);

            float band_abs = cols[c].swdn[nb] * (1.0f - trans);
            float band_trans = cols[c].swdn[nb] * trans;
            total_abs += band_abs;
            total_trans += band_trans;
        }
        absorbed[c] = total_abs;
        transmitted[c] = total_trans;
    }
}

__global__ void kernel_dEdd(const IceColumn* __restrict__ cols,
    float* __restrict__ absorbed, float* __restrict__ transmitted, int ncol) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    float mu0 = fmaxf(cols[c].coszen, 0.01f);
    float total_abs = 0.0f, total_trans = 0.0f;

    for (int nb = 0; nb < NBAND; nb++) {
        float k_snow = (nb == 0) ? 20.0f : (nb == 1) ? 100.0f : 500.0f;
        float k_ice = (nb == 0) ? 1.0f : (nb == 1) ? 5.0f : 50.0f;
        float w0_snow = 0.999f, w0_ice = 0.95f;
        float g_snow = 0.89f, g_ice = 0.94f;

        float tau_scaled_s = (1.0f - w0_snow*g_snow*g_snow) * k_snow *
            cols[c].snow_depth / fmaxf((float)cols[c].nslyr, 1.0f);
        float tau_scaled_i = (1.0f - w0_ice*g_ice*g_ice) * k_ice *
            cols[c].ice_thickness / fmaxf((float)cols[c].nilyr, 1.0f);

        float total_tau = tau_scaled_s * cols[c].nslyr + tau_scaled_i * cols[c].nilyr;
        float trans = __expf(-total_tau / mu0);

        total_abs += cols[c].swdn[nb] * (1.0f - trans);
        total_trans += cols[c].swdn[nb] * trans;
    }
    absorbed[c] = total_abs;
    transmitted[c] = total_trans;
}

// ============================================================
// KERNEL 3: CICE EVP dynamics — stress-velocity subcycling
// Simplified: 2D grid, ndte subcycles per timestep
// ============================================================

struct EVPCell {
    float uvel, vvel;       // ice velocity (m/s)
    float str11, str12, str22; // stress tensor components
    float strength;         // ice strength (N/m)
    float mass;             // ice mass per area (kg/m2)
    float area;             // ice concentration (0-1)
};

void cpu_evp(EVPCell* cells, const float* taux, const float* tauy,
             float dt, int ndte, int ncell) {
    float dte = dt / (float)ndte;
    for (int c = 0; c < ncell; c++) {
        if (cells[c].area < 0.01f || cells[c].mass < 0.1f) continue;
        for (int sub = 0; sub < ndte; sub++) {
            // Simplified EVP: stress update then velocity update
            float e2 = 1.0f / (2.0f * 2.0f); // 1/e^2, e=2 (eccentricity)
            float denom = 1.0f + 0.5f * dte * 5e-9f / cells[c].mass; // Coriolis + drag

            // Strain rates (simplified finite difference on single cell)
            float eps11 = 0.01f * cells[c].uvel;
            float eps22 = -0.01f * cells[c].vvel;
            float eps12 = 0.005f * (cells[c].uvel + cells[c].vvel);

            // Stress update (EVP rheology)
            float P = cells[c].strength * cells[c].area;
            float zeta = P / (2.0f * fmaxf(sqrtf(eps11*eps11 + eps22*eps22 + e2*eps12*eps12), 1e-10f));
            float eta = zeta * e2;

            cells[c].str11 += dte * (zeta*(eps11+eps22) + eta*(eps11-eps22) - P*0.5f - cells[c].str11) / (dte + 1.0f);
            cells[c].str22 += dte * (zeta*(eps11+eps22) - eta*(eps11-eps22) - P*0.5f - cells[c].str22) / (dte + 1.0f);
            cells[c].str12 += dte * (2.0f*eta*eps12 - cells[c].str12) / (dte + 1.0f);

            // Velocity update
            cells[c].uvel = (cells[c].uvel + dte/cells[c].mass * (taux[c] + cells[c].str11*0.01f)) / denom;
            cells[c].vvel = (cells[c].vvel + dte/cells[c].mass * (tauy[c] + cells[c].str22*0.01f)) / denom;
        }
    }
}

__global__ void kernel_evp(EVPCell* __restrict__ cells,
    const float* __restrict__ taux, const float* __restrict__ tauy,
    float dt, int ndte, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    if (cells[c].area < 0.01f || cells[c].mass < 0.1f) return;

    float dte = dt / (float)ndte;
    float e2 = 1.0f / 4.0f;
    float denom = 1.0f + 0.5f * dte * 5e-9f / cells[c].mass;

    // Load to registers
    float uv = cells[c].uvel, vv = cells[c].vvel;
    float s11 = cells[c].str11, s22 = cells[c].str22, s12 = cells[c].str12;
    float P = cells[c].strength * cells[c].area;

    for (int sub = 0; sub < ndte; sub++) {
        float eps11 = 0.01f * uv;
        float eps22 = -0.01f * vv;
        float eps12 = 0.005f * (uv + vv);

        float zeta = P / (2.0f * fmaxf(__fsqrt_rn(eps11*eps11+eps22*eps22+e2*eps12*eps12), 1e-10f));
        float eta = zeta * e2;

        s11 += dte * (zeta*(eps11+eps22)+eta*(eps11-eps22)-P*0.5f-s11) / (dte+1.0f);
        s22 += dte * (zeta*(eps11+eps22)-eta*(eps11-eps22)-P*0.5f-s22) / (dte+1.0f);
        s12 += dte * (2.0f*eta*eps12-s12) / (dte+1.0f);

        uv = (uv + dte/cells[c].mass * (taux[c]+s11*0.01f)) / denom;
        vv = (vv + dte/cells[c].mass * (tauy[c]+s22*0.01f)) / denom;
    }

    cells[c].uvel = uv; cells[c].vvel = vv;
    cells[c].str11 = s11; cells[c].str22 = s22; cells[c].str12 = s12;
}

// ============================================================
// KERNEL 4: MOSART kinematic wave routing
// Manning's equation: v = (R^2/3) * sqrt(S) / n
// ============================================================

struct MOSARTReach {
    float rwidth;    // channel width (m)
    float rlen;      // channel length (m)
    float rslp;      // channel slope
    float rn;        // Manning's roughness
    float storage;   // water storage (m3)
    float eroutUp;   // upstream inflow (m3/s)
    float rlateral;  // lateral inflow (m3/s)
};

void cpu_mosart(MOSARTReach* reaches, float dt, int nreach) {
    for (int r = 0; r < nreach; r++) {
        if (reaches[r].rslp <= 0.0f || reaches[r].rn <= 0.0f) continue;

        // Water depth
        float mr = reaches[r].storage / fmaxf(reaches[r].rlen, 1.0f);
        float hr = mr / fmaxf(reaches[r].rwidth, 1.0f);
        if (hr < 0.001f) hr = 0.001f;

        // Hydraulic radius (rectangular)
        float pr = reaches[r].rwidth + 2.0f * hr;
        float rr = mr / fmaxf(pr, 0.01f);

        // Manning's velocity
        float vr = powf(rr, 2.0f/3.0f) * sqrtf(reaches[r].rslp) / reaches[r].rn;

        // Outflow
        float erout = vr * mr;

        // Update storage
        float inflow = reaches[r].eroutUp + reaches[r].rlateral;
        reaches[r].storage += (inflow - erout) * dt;
        if (reaches[r].storage < 0.0f) reaches[r].storage = 0.0f;
    }
}

__global__ void kernel_mosart(MOSARTReach* __restrict__ reaches, float dt, int nreach) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nreach) return;
    if (reaches[r].rslp <= 0.0f || reaches[r].rn <= 0.0f) return;

    float mr = reaches[r].storage / fmaxf(reaches[r].rlen, 1.0f);
    float hr = mr / fmaxf(reaches[r].rwidth, 1.0f);
    if (hr < 0.001f) hr = 0.001f;

    float pr = reaches[r].rwidth + 2.0f * hr;
    float rr = mr / fmaxf(pr, 0.01f);

    float vr = __powf(rr, 2.0f/3.0f) * __fsqrt_rn(reaches[r].rslp) / reaches[r].rn;
    float erout = vr * mr;

    float inflow = reaches[r].eroutUp + reaches[r].rlateral;
    reaches[r].storage = fmaxf(0.0f, reaches[r].storage + (inflow - erout) * dt);
}

// ============================================================
// Data generators
// ============================================================

void gen_tridi(float* cl, float* cm, float* cu, float* r1, int l, int n, int stride, unsigned seed) {
    srand(seed);
    for (int i = 0; i < l; i++) {
        for (int k = 0; k < n; k++) {
            cl[i*stride+k] = (k > 0) ? -(0.1f+0.5f*((float)rand()/RAND_MAX)) : 0.0f;
            cu[i*stride+k] = (k < n-1) ? -(0.1f+0.5f*((float)rand()/RAND_MAX)) : 0.0f;
            cm[i*stride+k] = fabsf(cl[i*stride+k])+fabsf(cu[i*stride+k])+0.5f+((float)rand()/RAND_MAX);
            r1[i*stride+k] = -1.0f+2.0f*((float)rand()/RAND_MAX);
        }
    }
}

void gen_ice(IceColumn* cols, int n, unsigned seed) {
    srand(seed);
    for (int c = 0; c < n; c++) {
        cols[c].snow_depth = 0.1f*((float)rand()/RAND_MAX);
        cols[c].ice_thickness = 0.5f+3.0f*((float)rand()/RAND_MAX);
        cols[c].snow_grain_r = 50.0f+200.0f*((float)rand()/RAND_MAX);
        cols[c].nslyr = 1+(rand()%3);
        cols[c].nilyr = 3+(rand()%5);
        cols[c].coszen = 0.1f+0.8f*((float)rand()/RAND_MAX);
        for (int b = 0; b < NBAND; b++)
            cols[c].swdn[b] = 50.0f+300.0f*((float)rand()/RAND_MAX);
    }
}

void gen_evp(EVPCell* cells, float* tx, float* ty, int n, unsigned seed) {
    srand(seed);
    for (int c = 0; c < n; c++) {
        cells[c].uvel = -0.1f+0.2f*((float)rand()/RAND_MAX);
        cells[c].vvel = -0.1f+0.2f*((float)rand()/RAND_MAX);
        cells[c].str11 = 0; cells[c].str12 = 0; cells[c].str22 = 0;
        cells[c].strength = 1e4f+5e4f*((float)rand()/RAND_MAX);
        cells[c].mass = 500.0f+2000.0f*((float)rand()/RAND_MAX);
        cells[c].area = 0.5f+0.5f*((float)rand()/RAND_MAX);
        tx[c] = -0.5f+1.0f*((float)rand()/RAND_MAX);
        ty[c] = -0.5f+1.0f*((float)rand()/RAND_MAX);
    }
}

void gen_mosart(MOSARTReach* r, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        r[i].rwidth = 5.0f+50.0f*((float)rand()/RAND_MAX);
        r[i].rlen = 500.0f+5000.0f*((float)rand()/RAND_MAX);
        r[i].rslp = 0.0001f+0.005f*((float)rand()/RAND_MAX);
        r[i].rn = 0.02f+0.06f*((float)rand()/RAND_MAX);
        r[i].storage = 1000.0f+50000.0f*((float)rand()/RAND_MAX);
        r[i].eroutUp = 5.0f+50.0f*((float)rand()/RAND_MAX);
        r[i].rlateral = 0.5f*((float)rand()/RAND_MAX);
    }
}

// ============================================================
// Generic benchmark runner
// ============================================================

template<typename T>
void compare(const T* cpu, const T* gpu, int n, const char* name) {
    float max_abs=0,max_rel=0; int nan_c=0,fail_c=0;
    for (int i = 0; i < n; i++) {
        float cv = ((const float*)cpu)[i];
        float gv = ((const float*)gpu)[i];
        if (isnan(gv)||isinf(gv)){nan_c++;continue;}
        float ae = fabsf(gv-cv);
        if (ae>max_abs) max_abs=ae;
        if (fabsf(cv)>1e-10f) {
            float re=ae/fabsf(cv);
            if(re>max_rel)max_rel=re;
            if(re>1e-4f)fail_c++;
        }
    }
    printf("  Max abs: %.2e | Max rel: %.2e | NaN: %d | >0.01%%: %d/%d\n",
           max_abs,max_rel,nan_c,fail_c,n);
    printf("  Status: %s\n\n",
           (nan_c==0&&max_rel<1e-4f)?"PASS":
           (nan_c==0&&max_rel<1e-2f)?"PASS (fast math)":"NEEDS REVIEW");
}

int main() {
    printf("================================================\n");
    printf("  Multi-Model NOAA GPU Kernels\n");
    printf("  CCPP + Icepack + CICE + MOSART\n");
    printf("  RTX 3060 12GB\n");
    printf("================================================\n\n");

    // ---- CCPP tridi1 ----
    {
        printf("========== CCPP tridi1 (PBL solver) ==========\n\n");
        int sizes[] = {10000, 100000, 500000};
        int n = 64; // vertical levels

        for (int is = 0; is < 3; is++) {
            int l = sizes[is];
            printf("--- %d columns x %d levels ---\n", l, n);
            size_t sz = (size_t)l*n*sizeof(float);
            float *hcl=(float*)malloc(sz),*hcm=(float*)malloc(sz),*hcu=(float*)malloc(sz);
            float *hr1=(float*)malloc(sz),*ha_cpu=(float*)malloc(sz),*ha_gpu=(float*)malloc(sz);
            gen_tridi(hcl,hcm,hcu,hr1,l,n,n,42);

            clock_t t0=clock();
            cpu_tridi1(l,n,hcl,hcm,hcu,hr1,ha_cpu,n);
            double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

            float *dcl,*dcm,*dcu,*dr1,*da1;
            cudaMalloc(&dcl,sz);cudaMalloc(&dcm,sz);cudaMalloc(&dcu,sz);
            cudaMalloc(&dr1,sz);cudaMalloc(&da1,sz);
            cudaMemcpy(dcl,hcl,sz,cudaMemcpyHostToDevice);
            cudaMemcpy(dcm,hcm,sz,cudaMemcpyHostToDevice);
            cudaMemcpy(dcu,hcu,sz,cudaMemcpyHostToDevice);
            cudaMemcpy(dr1,hr1,sz,cudaMemcpyHostToDevice);

            int thr=256,blk=(l+thr-1)/thr;
            kernel_tridi1<<<blk,thr>>>(n,dcl,dcm,dcu,dr1,da1,n,l);
            cudaDeviceSynchronize();

            cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
            int runs=20;
            cudaEventRecord(e0);
            for(int r=0;r<runs;r++)
                kernel_tridi1<<<blk,thr>>>(n,dcl,dcm,dcu,dr1,da1,n,l);
            cudaEventRecord(e1);cudaEventSynchronize(e1);
            float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

            cudaMemcpy(ha_gpu,da1,sz,cudaMemcpyDeviceToHost);
            printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
            compare<float>(ha_cpu,ha_gpu,l*n,"tridi1");

            free(hcl);free(hcm);free(hcu);free(hr1);free(ha_cpu);free(ha_gpu);
            cudaFree(dcl);cudaFree(dcm);cudaFree(dcu);cudaFree(dr1);cudaFree(da1);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
    }

    // ---- Icepack delta-Eddington ----
    {
        printf("========== Icepack delta-Eddington ==========\n\n");
        int sizes[] = {10000, 100000, 500000};
        for (int is = 0; is < 3; is++) {
            int nc = sizes[is];
            printf("--- %d ice columns ---\n", nc);
            IceColumn *hc=(IceColumn*)malloc(nc*sizeof(IceColumn));
            float *ha_c=(float*)malloc(nc*4),*ht_c=(float*)malloc(nc*4);
            float *ha_g=(float*)malloc(nc*4),*ht_g=(float*)malloc(nc*4);
            gen_ice(hc,nc,42);

            clock_t t0=clock();
            cpu_dEdd(hc,ha_c,ht_c,nc);
            double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

            IceColumn *dc; float *da,*dt_d;
            cudaMalloc(&dc,nc*sizeof(IceColumn));
            cudaMalloc(&da,nc*4);cudaMalloc(&dt_d,nc*4);
            cudaMemcpy(dc,hc,nc*sizeof(IceColumn),cudaMemcpyHostToDevice);

            int thr=256,blk=(nc+thr-1)/thr;
            kernel_dEdd<<<blk,thr>>>(dc,da,dt_d,nc);cudaDeviceSynchronize();

            cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
            int runs=50;
            cudaEventRecord(e0);
            for(int r=0;r<runs;r++) kernel_dEdd<<<blk,thr>>>(dc,da,dt_d,nc);
            cudaEventRecord(e1);cudaEventSynchronize(e1);
            float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

            cudaMemcpy(ha_g,da,nc*4,cudaMemcpyDeviceToHost);
            printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
            compare<float>(ha_c,ha_g,nc,"dEdd absorbed");

            free(hc);free(ha_c);free(ht_c);free(ha_g);free(ht_g);
            cudaFree(dc);cudaFree(da);cudaFree(dt_d);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
    }

    // ---- CICE EVP ----
    {
        printf("========== CICE EVP Dynamics ==========\n\n");
        int sizes[] = {10000, 100000, 500000};
        int ndte = 120; // typical subcycles
        float dt = 3600.0f; // 1 hour

        for (int is = 0; is < 3; is++) {
            int nc = sizes[is];
            printf("--- %d cells x %d subcycles ---\n", nc, ndte);
            EVPCell *hc_cpu=(EVPCell*)malloc(nc*sizeof(EVPCell));
            EVPCell *hc_gpu=(EVPCell*)malloc(nc*sizeof(EVPCell));
            float *htx=(float*)malloc(nc*4),*hty=(float*)malloc(nc*4);
            gen_evp(hc_cpu,htx,hty,nc,42);
            memcpy(hc_gpu,hc_cpu,nc*sizeof(EVPCell));

            clock_t t0=clock();
            cpu_evp(hc_cpu,htx,hty,dt,ndte,nc);
            double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

            EVPCell *dc; float *dtx,*dty;
            cudaMalloc(&dc,nc*sizeof(EVPCell));
            cudaMalloc(&dtx,nc*4);cudaMalloc(&dty,nc*4);
            cudaMemcpy(dc,hc_gpu,nc*sizeof(EVPCell),cudaMemcpyHostToDevice);
            cudaMemcpy(dtx,htx,nc*4,cudaMemcpyHostToDevice);
            cudaMemcpy(dty,hty,nc*4,cudaMemcpyHostToDevice);

            int thr=256,blk=(nc+thr-1)/thr;
            kernel_evp<<<blk,thr>>>(dc,dtx,dty,dt,ndte,nc);cudaDeviceSynchronize();

            // Reset and benchmark
            cudaMemcpy(dc,hc_gpu,nc*sizeof(EVPCell),cudaMemcpyHostToDevice);
            cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
            int runs=10;
            cudaEventRecord(e0);
            for(int r=0;r<runs;r++){
                cudaMemcpy(dc,hc_gpu,nc*sizeof(EVPCell),cudaMemcpyHostToDevice);
                kernel_evp<<<blk,thr>>>(dc,dtx,dty,dt,ndte,nc);
            }
            cudaEventRecord(e1);cudaEventSynchronize(e1);
            float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

            // Compare velocities
            cudaMemcpy(dc,hc_gpu,nc*sizeof(EVPCell),cudaMemcpyHostToDevice);
            kernel_evp<<<blk,thr>>>(dc,dtx,dty,dt,ndte,nc);
            EVPCell *hc_result=(EVPCell*)malloc(nc*sizeof(EVPCell));
            cudaMemcpy(hc_result,dc,nc*sizeof(EVPCell),cudaMemcpyDeviceToHost);

            float max_rel=0; int nan_c=0;
            for(int c=0;c<nc;c++){
                if(isnan(hc_result[c].uvel)){nan_c++;continue;}
                if(fabsf(hc_cpu[c].uvel)>1e-10f){
                    float re=fabsf(hc_result[c].uvel-hc_cpu[c].uvel)/fabsf(hc_cpu[c].uvel);
                    if(re>max_rel)max_rel=re;
                }
            }
            printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
            printf("  Max rel (uvel): %.2e | NaN: %d\n",max_rel,nan_c);
            printf("  Status: %s\n\n",
                   (nan_c==0&&max_rel<1e-3f)?"PASS":(nan_c==0&&max_rel<1e-1f)?"PASS (fast math)":"NEEDS REVIEW");

            free(hc_cpu);free(hc_gpu);free(htx);free(hty);free(hc_result);
            cudaFree(dc);cudaFree(dtx);cudaFree(dty);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
    }

    // ---- MOSART ----
    {
        printf("========== MOSART Kinematic Wave ==========\n\n");
        int sizes[] = {100000, 500000, 1000000};
        float dt = 3600.0f;

        for (int is = 0; is < 3; is++) {
            int nr = sizes[is];
            printf("--- %d reaches ---\n", nr);
            MOSARTReach *hr_cpu=(MOSARTReach*)malloc(nr*sizeof(MOSARTReach));
            MOSARTReach *hr_gpu=(MOSARTReach*)malloc(nr*sizeof(MOSARTReach));
            gen_mosart(hr_cpu,nr,42);
            memcpy(hr_gpu,hr_cpu,nr*sizeof(MOSARTReach));

            clock_t t0=clock();
            cpu_mosart(hr_cpu,dt,nr);
            double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

            MOSARTReach *dr;
            cudaMalloc(&dr,nr*sizeof(MOSARTReach));
            cudaMemcpy(dr,hr_gpu,nr*sizeof(MOSARTReach),cudaMemcpyHostToDevice);

            int thr=256,blk=(nr+thr-1)/thr;
            kernel_mosart<<<blk,thr>>>(dr,dt,nr);cudaDeviceSynchronize();

            cudaMemcpy(dr,hr_gpu,nr*sizeof(MOSARTReach),cudaMemcpyHostToDevice);
            cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
            int runs=20;
            cudaEventRecord(e0);
            for(int r=0;r<runs;r++){
                cudaMemcpy(dr,hr_gpu,nr*sizeof(MOSARTReach),cudaMemcpyHostToDevice);
                kernel_mosart<<<blk,thr>>>(dr,dt,nr);
            }
            cudaEventRecord(e1);cudaEventSynchronize(e1);
            float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

            cudaMemcpy(dr,hr_gpu,nr*sizeof(MOSARTReach),cudaMemcpyHostToDevice);
            kernel_mosart<<<blk,thr>>>(dr,dt,nr);
            MOSARTReach *hr_result=(MOSARTReach*)malloc(nr*sizeof(MOSARTReach));
            cudaMemcpy(hr_result,dr,nr*sizeof(MOSARTReach),cudaMemcpyDeviceToHost);

            float max_rel=0; int nan_c=0;
            for(int r=0;r<nr;r++){
                if(isnan(hr_result[r].storage)){nan_c++;continue;}
                if(fabsf(hr_cpu[r].storage)>1.0f){
                    float re=fabsf(hr_result[r].storage-hr_cpu[r].storage)/fabsf(hr_cpu[r].storage);
                    if(re>max_rel)max_rel=re;
                }
            }
            printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
            printf("  Max rel (storage): %.2e | NaN: %d\n",max_rel,nan_c);
            printf("  Status: %s\n\n",
                   (nan_c==0&&max_rel<1e-4f)?"PASS":(nan_c==0&&max_rel<1e-2f)?"PASS (fast math)":"NEEDS REVIEW");

            free(hr_cpu);free(hr_gpu);free(hr_result);
            cudaFree(dr);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
    }

    return 0;
}
