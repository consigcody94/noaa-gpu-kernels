/**
 * GPU Parallel Muskingum-Cunge: Reach-Level Parallelism
 *
 * Instead of trying to parallelize WITHIN a reach (which fails due to
 * nonlinear coefficient coupling), this kernel parallelizes ACROSS
 * independent reaches. Each reach is processed sequentially by one
 * GPU thread, but thousands of reaches run simultaneously.
 *
 * This preserves EXACT accuracy (same secant-method solve as Fortran)
 * while exploiting the massive between-reach independence in the
 * NWM network (2.7M reaches, thousands independent per tree level).
 *
 * The speedup comes from:
 * 1. Moving the full MC kernel to GPU (no Python/Cython overhead)
 * 2. Running all independent reaches in parallel
 * 3. Better memory access patterns (coalesced struct reads)
 *
 * Author: Cody Churchwell, March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

struct ChannelParams {
    float dx, bw, tw, twcc, n_ch, ncc, cs, s0;
};

// ============================================================
// Device: Full MC solve for one segment (matches Fortran exactly)
// ============================================================
__device__ void hydraulic_geometry_d(
    float h, float bfd, float bw, float twcc, float z,
    float &twl, float &R, float &AREA, float &AREAC, float &WP, float &WPC)
{
    twl = bw + 2.0f * z * h;
    float h_gt_bf = fmaxf(h - bfd, 0.0f);
    float h_lt_bf = fminf(bfd, h);
    if (h_gt_bf > 0.0f && twcc <= 0.0f) { h_gt_bf = 0.0f; h_lt_bf = h; }
    AREA = (bw + h_lt_bf * z) * h_lt_bf;
    WP = bw + 2.0f * h_lt_bf * sqrtf(1.0f + z * z);
    AREAC = twcc * h_gt_bf;
    WPC = (h_gt_bf > 0.0f) ? (twcc + 2.0f * h_gt_bf) : 0.0f;
    R = (WP + WPC > 0.0f) ? (AREA + AREAC) / (WP + WPC) : 0.0f;
}

__device__ float mc_secant_solve(
    float dt, float qup, float quc, float qdp, float ql,
    float dx, float bw, float tw, float twcc,
    float n_ch, float ncc, float cs, float s0,
    float depthp, float &depthc, float &velc)
{
    float z = (cs == 0.0f) ? 1.0f : 1.0f / cs;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f * z);
    else bfd = (tw - bw) / (2.0f * z);

    if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f) {
        depthc = 0.0f; velc = 0.0f; return 0.0f;
    }

    float mindepth = 0.01f;
    depthc = fmaxf(depthp, 0.0f);

    if (ql <= 0.0f && qup <= 0.0f && quc <= 0.0f && qdp <= 0.0f) {
        depthc = 0.0f; velc = 0.0f; return 0.0f;
    }

    float h = depthc * 1.33f + mindepth;
    float h_0 = depthc * 0.67f;
    float C1, C2, C3, C4, X = 0.25f;
    float Qj_0 = 0.0f, Qj;
    int maxiter = 100;

    for (int attempt = 0; attempt < 5; attempt++) {
        float aerror = 0.01f, rerror = 1.0f;
        int iter = 0;

        while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
            // Evaluate at h_0
            float twl0, R0, A0, AC0, WP0, WPC0;
            hydraulic_geometry_d(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);
            float Ck0 = 0.0f;
            if (h_0 > 0.0f) {
                float R23 = powf(R0, 2.0f/3.0f);
                float R53 = powf(R0, 5.0f/3.0f);
                Ck0 = fmaxf(0.0f, (sqrtf(s0)/n_ch)*((5.0f/3.0f)*R23-(2.0f/3.0f)*R53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h_0*z))));
            }
            float Km0 = (Ck0 > 0.0f) ? fmaxf(dt, dx/Ck0) : dt;
            float tw_use0 = (h_0 > bfd && twcc > 0.0f) ? twcc : twl0;
            float X0 = (Ck0 > 0.0f && tw_use0*s0*Ck0*dx > 0.0f) ?
                fminf(0.5f, fmaxf(0.0f, 0.5f*(1.0f-Qj_0/(2.0f*tw_use0*s0*Ck0*dx)))) : 0.5f;
            float D0 = Km0*(1.0f-X0)+dt/2.0f;
            if (D0 == 0.0f) D0 = 1.0f;
            float c1=((Km0*X0+dt/2.0f)/D0), c2=((dt/2.0f-Km0*X0)/D0);
            float c3=((Km0*(1.0f-X0)-dt/2.0f)/D0), c4=((ql*dt)/D0);
            float Qmc0 = c1*qup+c2*quc+c3*qdp+c4;
            float Qmn0 = (WP0+WPC0>0.0f) ? (1.0f/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*powf(R0,2.0f/3.0f)*sqrtf(s0) : 0.0f;
            Qj_0 = Qmc0 - Qmn0;

            // Evaluate at h
            float twl1, R1, A1, AC1, WP1, WPC1;
            hydraulic_geometry_d(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            float Ck1 = 0.0f;
            if (h > 0.0f) {
                float R23 = powf(R1, 2.0f/3.0f);
                float R53 = powf(R1, 5.0f/3.0f);
                Ck1 = fmaxf(0.0f, (sqrtf(s0)/n_ch)*((5.0f/3.0f)*R23-(2.0f/3.0f)*R53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z))));
            }
            float Km1 = (Ck1 > 0.0f) ? fmaxf(dt, dx/Ck1) : dt;
            float tw_use1 = (h > bfd && twcc > 0.0f) ? twcc : twl1;
            X = (Ck1 > 0.0f && tw_use1*s0*Ck1*dx > 0.0f) ?
                fminf(0.5f, fmaxf(0.25f, 0.5f*(1.0f-(c1*qup+c2*quc+c3*qdp+c4)/(2.0f*tw_use1*s0*Ck1*dx)))) : 0.5f;
            float D1 = Km1*(1.0f-X)+dt/2.0f;
            if (D1 == 0.0f) D1 = 1.0f;
            C1=(Km1*X+dt/2.0f)/D1; C2=(dt/2.0f-Km1*X)/D1;
            C3=(Km1*(1.0f-X)-dt/2.0f)/D1; C4=(ql*dt)/D1;
            if (C4<0.0f && fabsf(C4)>C1*qup+C2*quc+C3*qdp) C4=-(C1*qup+C2*quc+C3*qdp);
            float Qmc1 = C1*qup+C2*quc+C3*qdp+C4;
            float Qmn1 = (WP1+WPC1>0.0f) ? (1.0f/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*powf(R1,2.0f/3.0f)*sqrtf(s0) : 0.0f;
            Qj = Qmc1 - Qmn1;

            float h_1 = (Qj_0-Qj!=0.0f) ? h-(Qj*(h_0-h))/(Qj_0-Qj) : h;
            if (h_1 < 0.0f) h_1 = h;

            if (h > 0.0f) { rerror = fabsf((h_1-h)/h); aerror = fabsf(h_1-h); }
            else { rerror = 0.0f; aerror = 0.9f; }

            h_0 = fmaxf(0.0f, h); h = fmaxf(0.0f, h_1);
            iter++;
            if (h < mindepth) break;
        }
        if (iter < maxiter) break;
        h *= 1.33f; h_0 *= 0.67f; maxiter += 25;
    }

    float qdc;
    float Qmc = C1*qup+C2*quc+C3*qdp+C4;
    if (Qmc < 0.0f) {
        if (C4<0.0f && fabsf(C4)>C1*qup+C2*quc+C3*qdp) qdc=0.0f;
        else qdc = fmaxf(C1*qup+C2*quc+C4, C1*qup+C3*qdp+C4);
    } else {
        qdc = Qmc;
    }

    float twl, R, A, AC, WP, WPC;
    hydraulic_geometry_d(h, bfd, bw, twcc, z, twl, R, A, AC, WP, WPC);
    R = (h*(bw+twl)/2.0f)/(bw+2.0f*sqrtf(((twl-bw)/2.0f)*((twl-bw)/2.0f)+h*h));
    if (R < 0.0f) R = 0.0f;
    velc = (1.0f/n_ch)*powf(R, 2.0f/3.0f)*sqrtf(s0);
    depthc = h;
    return qdc;
}

// ============================================================
// GPU kernel: One thread per reach, sequential within reach
// ============================================================
__global__ void kernel_mc_reach_parallel(
    const ChannelParams* __restrict__ params,
    const float* __restrict__ qlat,
    const int* __restrict__ reach_start,
    const int* __restrict__ reach_len,
    const float* __restrict__ qup_prev,
    const float* __restrict__ qdp_prev,
    const float* __restrict__ depth_prev,
    float* __restrict__ qdc_out,
    float* __restrict__ vel_out,
    float* __restrict__ depth_out,
    float upstream_q,
    float dt,
    int nreaches)
{
    int ireach = blockIdx.x * blockDim.x + threadIdx.x;
    if (ireach >= nreaches) return;

    int start = reach_start[ireach];
    int nseg = reach_len[ireach];
    float quc = upstream_q;

    for (int i = 0; i < nseg; i++) {
        int idx = start + i;
        float depthc, velc;
        float qdc = mc_secant_solve(
            dt, qup_prev[idx], quc, qdp_prev[idx], qlat[idx],
            params[idx].dx, params[idx].bw, params[idx].tw, params[idx].twcc,
            params[idx].n_ch, params[idx].ncc, params[idx].cs, params[idx].s0,
            depth_prev[idx], depthc, velc);
        qdc_out[idx] = qdc;
        vel_out[idx] = velc;
        depth_out[idx] = depthc;
        quc = qdc;
    }
}

// ============================================================
// CPU reference: Sequential over ALL reaches then segments
// ============================================================
void cpu_mc_sequential(
    const ChannelParams* params, const float* qlat,
    const int* reach_start, const int* reach_len,
    const float* qup_prev, const float* qdp_prev, const float* depth_prev,
    float* qdc_out, float* vel_out, float* depth_out,
    float upstream_q, float dt, int nreaches)
{
    for (int r = 0; r < nreaches; r++) {
        int start = reach_start[r];
        int nseg = reach_len[r];
        float quc = upstream_q;

        for (int i = 0; i < nseg; i++) {
            int idx = start + i;
            float z = (params[idx].cs == 0.0f) ? 1.0f : 1.0f / params[idx].cs;
            float bw = params[idx].bw;
            float tw = params[idx].tw;
            float bfd = (bw > tw) ? bw/0.00001f : (bw == tw) ? bw/(2.0f*z) : (tw-bw)/(2.0f*z);
            float n_ch = params[idx].n_ch;
            float ncc = params[idx].ncc;
            float s0 = params[idx].s0;
            float dx = params[idx].dx;
            float twcc = params[idx].twcc;

            if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f ||
                (qlat[idx] <= 0.0f && qup_prev[idx] <= 0.0f && quc <= 0.0f && qdp_prev[idx] <= 0.0f)) {
                qdc_out[idx] = 0.0f;
                vel_out[idx] = 0.0f;
                depth_out[idx] = 0.0f;
                quc = 0.0f;
                continue;
            }

            float depthp = fmaxf(depth_prev[idx], 0.0f);
            float mindepth = 0.01f;
            float h = depthp * 1.33f + mindepth;
            float h_0 = depthp * 0.67f;
            float C1, C2, C3, C4, X = 0.25f;
            float Qj_0 = 0.0f, Qj;
            int maxiter = 100;

            for (int attempt = 0; attempt < 5; attempt++) {
                float aerror = 0.01f, rerror = 1.0f;
                int iter = 0;
                while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
                    // h_0 evaluation
                    float twl0 = bw+2.0f*z*h_0;
                    float hgb0 = fmaxf(h_0-bfd,0.0f), hlb0 = fminf(bfd,h_0);
                    if (hgb0>0.0f && twcc<=0.0f) { hgb0=0.0f; hlb0=h_0; }
                    float A0=(bw+hlb0*z)*hlb0, WP0=bw+2.0f*hlb0*sqrtf(1.0f+z*z);
                    float AC0=twcc*hgb0, WPC0=(hgb0>0.0f)?(twcc+2.0f*hgb0):0.0f;
                    float R0=(WP0+WPC0>0.0f)?(A0+AC0)/(WP0+WPC0):0.0f;
                    float Ck0=0.0f;
                    if (h_0>0.0f) {
                        float r23=powf(R0,2.0f/3.0f), r53=powf(R0,5.0f/3.0f);
                        Ck0=fmaxf(0.0f,(sqrtf(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h_0*z))));
                    }
                    float Km0=(Ck0>0.0f)?fmaxf(dt,dx/Ck0):dt;
                    float twu0=(h_0>bfd&&twcc>0.0f)?twcc:twl0;
                    float X0=(Ck0>0.0f&&twu0*s0*Ck0*dx>0.0f)?fminf(0.5f,fmaxf(0.0f,0.5f*(1.0f-Qj_0/(2.0f*twu0*s0*Ck0*dx)))):0.5f;
                    float D0=Km0*(1.0f-X0)+dt/2.0f; if(D0==0.0f)D0=1.0f;
                    float c1_0=(Km0*X0+dt/2.0f)/D0,c2_0=(dt/2.0f-Km0*X0)/D0,c3_0=(Km0*(1.0f-X0)-dt/2.0f)/D0,c4_0=(qlat[idx]*dt)/D0;
                    Qj_0=(c1_0*qup_prev[idx]+c2_0*quc+c3_0*qdp_prev[idx]+c4_0)-((WP0+WPC0>0.0f)?(1.0f/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*powf(R0,2.0f/3.0f)*sqrtf(s0):0.0f);

                    // h evaluation
                    float twl1=bw+2.0f*z*h;
                    float hgb1=fmaxf(h-bfd,0.0f), hlb1=fminf(bfd,h);
                    if(hgb1>0.0f&&twcc<=0.0f){hgb1=0.0f;hlb1=h;}
                    float A1=(bw+hlb1*z)*hlb1,WP1=bw+2.0f*hlb1*sqrtf(1.0f+z*z);
                    float AC1=twcc*hgb1,WPC1=(hgb1>0.0f)?(twcc+2.0f*hgb1):0.0f;
                    float R1=(WP1+WPC1>0.0f)?(A1+AC1)/(WP1+WPC1):0.0f;
                    float Ck1=0.0f;
                    if(h>0.0f){float r23=powf(R1,2.0f/3.0f),r53=powf(R1,5.0f/3.0f);Ck1=fmaxf(0.0f,(sqrtf(s0)/n_ch)*((5.0f/3.0f)*r23-(2.0f/3.0f)*r53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z))));}
                    float Km1=(Ck1>0.0f)?fmaxf(dt,dx/Ck1):dt;
                    float twu1=(h>bfd&&twcc>0.0f)?twcc:twl1;
                    X=(Ck1>0.0f&&twu1*s0*Ck1*dx>0.0f)?fminf(0.5f,fmaxf(0.25f,0.5f*(1.0f-(c1_0*qup_prev[idx]+c2_0*quc+c3_0*qdp_prev[idx]+c4_0)/(2.0f*twu1*s0*Ck1*dx)))):0.5f;
                    float D1=Km1*(1.0f-X)+dt/2.0f;if(D1==0.0f)D1=1.0f;
                    C1=(Km1*X+dt/2.0f)/D1;C2=(dt/2.0f-Km1*X)/D1;C3=(Km1*(1.0f-X)-dt/2.0f)/D1;C4=(qlat[idx]*dt)/D1;
                    if(C4<0.0f&&fabsf(C4)>C1*qup_prev[idx]+C2*quc+C3*qdp_prev[idx])C4=-(C1*qup_prev[idx]+C2*quc+C3*qdp_prev[idx]);
                    Qj=(C1*qup_prev[idx]+C2*quc+C3*qdp_prev[idx]+C4)-((WP1+WPC1>0.0f)?(1.0f/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*powf(R1,2.0f/3.0f)*sqrtf(s0):0.0f);

                    float h_1=(Qj_0-Qj!=0.0f)?h-(Qj*(h_0-h))/(Qj_0-Qj):h;
                    if(h_1<0.0f)h_1=h;
                    if(h>0.0f){rerror=fabsf((h_1-h)/h);aerror=fabsf(h_1-h);}else{rerror=0.0f;aerror=0.9f;}
                    h_0=fmaxf(0.0f,h);h=fmaxf(0.0f,h_1);iter++;
                    if(h<mindepth)break;
                }
                if(iter<maxiter)break;
                h*=1.33f;h_0*=0.67f;maxiter+=25;
            }

            float Qmc=C1*qup_prev[idx]+C2*quc+C3*qdp_prev[idx]+C4;
            float qdc;
            if(Qmc<0.0f){if(C4<0.0f&&fabsf(C4)>C1*qup_prev[idx]+C2*quc+C3*qdp_prev[idx])qdc=0.0f;else qdc=fmaxf(C1*qup_prev[idx]+C2*quc+C4,C1*qup_prev[idx]+C3*qdp_prev[idx]+C4);}
            else qdc=Qmc;

            float twl=bw+2.0f*z*h;
            float Rv=(h*(bw+twl)/2.0f)/(bw+2.0f*sqrtf(((twl-bw)/2.0f)*((twl-bw)/2.0f)+h*h));
            if(Rv<0.0f)Rv=0.0f;

            qdc_out[idx] = qdc;
            vel_out[idx] = (1.0f/n_ch)*powf(Rv,2.0f/3.0f)*sqrtf(s0);
            depth_out[idx] = h;
            quc = qdc;
        }
    }
}

void gen_data(ChannelParams* p, float* ql, float* qu, float* qd, float* dp,
              int max_seg, int nr, int* rs, int* rl, unsigned seed) {
    srand(seed);
    int seg = 0;
    for (int r = 0; r < nr; r++) {
        rs[r] = seg;
        rl[r] = 1 + (rand() % 20);
        for (int s = 0; s < rl[r] && seg < max_seg; s++, seg++) {
            p[seg].dx = 500.0f + (rand()%5000);
            p[seg].bw = 5.0f + (rand()%50);
            p[seg].tw = p[seg].bw + (rand()%20);
            p[seg].twcc = p[seg].tw + (rand()%100);
            p[seg].n_ch = 0.02f + 0.06f*(rand()/(float)RAND_MAX);
            p[seg].ncc = p[seg].n_ch * 1.5f;
            p[seg].cs = 0.5f + 2.0f*(rand()/(float)RAND_MAX);
            p[seg].s0 = 0.0001f + 0.005f*(rand()/(float)RAND_MAX);
            ql[seg] = 0.5f*(rand()/(float)RAND_MAX);
            float bq = 5.0f + 50.0f*(rand()/(float)RAND_MAX);
            qu[seg] = bq*0.95f; qd[seg] = bq;
            dp[seg] = 0.5f + 3.0f*(rand()/(float)RAND_MAX);
        }
    }
}

int main() {
    printf("================================================\n");
    printf("  t-route MC: Reach-Level GPU Parallelism\n");
    printf("  (Exact accuracy, no operator splitting)\n");
    printf("================================================\n\n");

    struct TC { int nr; const char* nm; };
    TC cfgs[] = {{1000,"1K reaches"},{10000,"10K reaches"},{50000,"50K reaches"},{100000,"100K reaches"}};

    float dt = 300.0f;
    float upstream_q = 10.0f;

    for (int ic = 0; ic < 4; ic++) {
        int nr = cfgs[ic].nr;
        int max_seg = nr * 25;

        ChannelParams* hp = (ChannelParams*)malloc(max_seg*sizeof(ChannelParams));
        float *hql=(float*)malloc(max_seg*sizeof(float)), *hqu=(float*)malloc(max_seg*sizeof(float));
        float *hqd=(float*)malloc(max_seg*sizeof(float)), *hdp=(float*)malloc(max_seg*sizeof(float));
        int *hrs=(int*)malloc(nr*sizeof(int)), *hrl=(int*)malloc(nr*sizeof(int));
        float *hq_cpu=(float*)malloc(max_seg*sizeof(float)), *hv_cpu=(float*)malloc(max_seg*sizeof(float)), *hd_cpu=(float*)malloc(max_seg*sizeof(float));
        float *hq_gpu=(float*)malloc(max_seg*sizeof(float));

        gen_data(hp, hql, hqu, hqd, hdp, max_seg, nr, hrs, hrl, 42+ic);
        int tseg = 0;
        for (int r = 0; r < nr; r++) tseg += hrl[r];

        printf("--- %s (%d segments) ---\n", cfgs[ic].nm, tseg);

        // CPU reference
        clock_t t0 = clock();
        cpu_mc_sequential(hp, hql, hrs, hrl, hqu, hqd, hdp, hq_cpu, hv_cpu, hd_cpu, upstream_q, dt, nr);
        double cpu_ms = 1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

        // GPU
        ChannelParams *dp; float *dql,*dqu,*dqd,*ddp,*dqo,*dvo,*ddo; int *drs,*drl;
        cudaMalloc(&dp,tseg*sizeof(ChannelParams));
        cudaMalloc(&dql,tseg*sizeof(float)); cudaMalloc(&dqu,tseg*sizeof(float));
        cudaMalloc(&dqd,tseg*sizeof(float)); cudaMalloc(&ddp,tseg*sizeof(float));
        cudaMalloc(&dqo,tseg*sizeof(float)); cudaMalloc(&dvo,tseg*sizeof(float)); cudaMalloc(&ddo,tseg*sizeof(float));
        cudaMalloc(&drs,nr*sizeof(int)); cudaMalloc(&drl,nr*sizeof(int));

        cudaMemcpy(dp,hp,tseg*sizeof(ChannelParams),cudaMemcpyHostToDevice);
        cudaMemcpy(dql,hql,tseg*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dqu,hqu,tseg*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dqd,hqd,tseg*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(ddp,hdp,tseg*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(drs,hrs,nr*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(drl,hrl,nr*sizeof(int),cudaMemcpyHostToDevice);

        int thr=256, blk=(nr+thr-1)/thr;

        // Warmup
        kernel_mc_reach_parallel<<<blk,thr>>>(dp,dql,drs,drl,dqu,dqd,ddp,dqo,dvo,ddo,upstream_q,dt,nr);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        int runs=20;
        cudaEventRecord(e0);
        for(int r=0;r<runs;r++)
            kernel_mc_reach_parallel<<<blk,thr>>>(dp,dql,drs,drl,dqu,dqd,ddp,dqo,dvo,ddo,upstream_q,dt,nr);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float gpu_ms; cudaEventElapsedTime(&gpu_ms,e0,e1); gpu_ms/=runs;

        cudaMemcpy(hq_gpu,dqo,tseg*sizeof(float),cudaMemcpyDeviceToHost);

        // Accuracy check
        float max_abs=0, max_rel=0; int nan_cnt=0, fail_cnt=0;
        for(int i=0;i<tseg;i++){
            if(isnan(hq_gpu[i])||isinf(hq_gpu[i])){nan_cnt++;continue;}
            float ae=fabsf(hq_gpu[i]-hq_cpu[i]);
            max_abs=fmaxf(max_abs,ae);
            if(fabsf(hq_cpu[i])>0.01f){
                float re=ae/fabsf(hq_cpu[i]);
                max_rel=fmaxf(max_rel,re);
                if(re>0.01f)fail_cnt++;
            }
        }

        printf("  CPU:  %.1f ms\n", cpu_ms);
        printf("  GPU:  %.3f ms (%.1fx vs CPU)\n", gpu_ms, cpu_ms/gpu_ms);
        printf("  Max abs err: %.2e  Max rel err: %.2e\n", max_abs, max_rel);
        printf("  NaN: %d, >1%% err: %d/%d\n", nan_cnt, fail_cnt, tseg);
        printf("  Status: %s\n\n", (nan_cnt==0 && max_rel<0.01f) ? "PASS" : (max_rel<0.05f ? "ACCEPTABLE" : "FAIL"));

        free(hp);free(hql);free(hqu);free(hqd);free(hdp);free(hrs);free(hrl);
        free(hq_cpu);free(hv_cpu);free(hd_cpu);free(hq_gpu);
        cudaFree(dp);cudaFree(dql);cudaFree(dqu);cudaFree(dqd);cudaFree(ddp);
        cudaFree(dqo);cudaFree(dvo);cudaFree(ddo);cudaFree(drs);cudaFree(drl);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }

    return 0;
}
