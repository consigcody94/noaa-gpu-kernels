/**
 * Icepack delta-Eddington shortwave kernel — REAL DATA harness
 *
 * Kernel (kernel_dEdd), CPU reference (cpu_dEdd), IceColumn struct, and the
 * compare() criteria are copied VERBATIM from ../../noaa_multi_kernel.cu
 * (consigcody94/noaa-gpu-kernels). The ONLY change: the synthetic generator
 * gen_ice() is replaced by load_ice(), which reads real MOSAiC 2019-2020
 * ice columns prepared by prep_icepack_realdata.py:
 *
 *   - snow depth + co-located ice thickness: Magnaprobe/GEM-2 transects,
 *     Itkin et al. (2021), PANGAEA, doi:10.1594/PANGAEA.937781
 *   - downwelling shortwave: MOSAiC Merged Data Files (rsds; SPN1 after
 *     2020-07-31), NSF Arctic Data Center, doi:10.18739/A2WD3Q35Z
 *     (the dataset the Icepack consortium uses for its MOSAiC forcing)
 *   - coszen: NOAA GMD solar position from each record's UTC time + lat/lon
 *   - band split / layer counts / grain radius: Icepack constants
 *     (frcvdr+frcvdf=0.52, frcidr=0.31, frcidf=0.17; nslyr=1, nilyr=7;
 *      rsnw_nonmelt=500 um)
 *
 * Binary layout: int32 n, then n x struct IceColumn (36 bytes, no padding).
 *
 * Usage: icepack_de_realdata.exe [ice_columns_real.bin] [replicate_factor]
 *   replicate_factor (default 1): tile the real columns K times for GPU
 *   timing at larger sizes. Accuracy is ALWAYS reported for the unreplicated
 *   real set first.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define NBAND 3        // spectral bands for delta-Eddington

// ============================================================
// KERNEL 2: Icepack delta-Eddington shortwave
// Simplified: computes transmittance through ice/snow layers
// (copied verbatim from noaa_multi_kernel.cu)
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
// Real-data loader (replaces gen_ice; this is the ONLY functional change)
// ============================================================

IceColumn* load_ice(const char* path, int* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) {
        fprintf(stderr, "ERROR: bad record count in %s\n", path);
        fclose(f); return NULL;
    }
    if (sizeof(IceColumn) != 36) {
        fprintf(stderr, "ERROR: IceColumn struct size %zu != 36 (padding?)\n",
                sizeof(IceColumn));
        fclose(f); return NULL;
    }
    IceColumn* cols = (IceColumn*)malloc((size_t)n * sizeof(IceColumn));
    if (fread(cols, sizeof(IceColumn), n, f) != (size_t)n) {
        fprintf(stderr, "ERROR: short read in %s\n", path);
        free(cols); fclose(f); return NULL;
    }
    fclose(f);
    *n_out = n;
    return cols;
}

// ============================================================
// Comparison (copied verbatim from noaa_multi_kernel.cu)
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

// ============================================================
// Main: run CPU reference + GPU kernel on the real columns
// ============================================================

static void run_case(const IceColumn* hc, int nc, const char* label) {
    printf("--- %d ice columns (%s) ---\n", nc, label);
    float *ha_c=(float*)malloc(nc*4),*ht_c=(float*)malloc(nc*4);
    float *ha_g=(float*)malloc(nc*4),*ht_g=(float*)malloc(nc*4);

    clock_t t0=clock();
    cpu_dEdd(hc,ha_c,ht_c,nc);
    double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

    IceColumn *dc; float *da,*dt_d;
    cudaMalloc(&dc,(size_t)nc*sizeof(IceColumn));
    cudaMalloc(&da,nc*4);cudaMalloc(&dt_d,nc*4);
    cudaMemcpy(dc,hc,(size_t)nc*sizeof(IceColumn),cudaMemcpyHostToDevice);

    int thr=256,blk=(nc+thr-1)/thr;
    kernel_dEdd<<<blk,thr>>>(dc,da,dt_d,nc);cudaDeviceSynchronize();

    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    int runs=50;
    cudaEventRecord(e0);
    for(int r=0;r<runs;r++) kernel_dEdd<<<blk,thr>>>(dc,da,dt_d,nc);
    cudaEventRecord(e1);cudaEventSynchronize(e1);
    float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

    cudaMemcpy(ha_g,da,nc*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(ht_g,dt_d,nc*4,cudaMemcpyDeviceToHost);
    printf("  CPU: %.1f ms | GPU: %.3f ms | Speedup: %.1fx\n",cpu_ms,gpu_ms,cpu_ms/gpu_ms);
    printf("  [absorbed]\n");
    compare<float>(ha_c,ha_g,nc,"dEdd absorbed");
    printf("  [transmitted]\n");
    compare<float>(ht_c,ht_g,nc,"dEdd transmitted");

    free(ha_c);free(ht_c);free(ha_g);free(ht_g);
    cudaFree(dc);cudaFree(da);cudaFree(dt_d);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "ice_columns_real.bin";
    int rep = (argc > 2) ? atoi(argv[2]) : 1;
    if (rep < 1) rep = 1;

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("================================================\n");
    printf("  Icepack delta-Eddington — REAL DATA\n");
    printf("  MOSAiC 2019-2020 transects + MDF shortwave\n");
    printf("  GPU: %s\n", prop.name);
    printf("================================================\n\n");

    int nc = 0;
    IceColumn* hc = load_ice(path, &nc);
    if (!hc) return 1;
    printf("Loaded %d real ice columns from %s\n\n", nc, path);

    run_case(hc, nc, "real, unreplicated");

    if (rep > 1) {
        size_t nbig = (size_t)nc * rep;
        IceColumn* big = (IceColumn*)malloc(nbig * sizeof(IceColumn));
        for (int r = 0; r < rep; r++)
            memcpy(big + (size_t)r * nc, hc, (size_t)nc * sizeof(IceColumn));
        char label[64];
        snprintf(label, sizeof(label), "real x%d replication, timing only", rep);
        run_case(big, (int)nbig, label);
        free(big);
    }

    free(hc);
    return 0;
}
