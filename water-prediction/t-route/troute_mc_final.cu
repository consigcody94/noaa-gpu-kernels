/**
 * GPU Muskingum-Cunge Routing for NOAA t-route
 *
 * Parallelizes across independent river reaches on GPU.
 * Each reach runs the exact same secant-method MC solver as the
 * Fortran reference (MCsingleSegStime_f2py_NOLOOP.f90).
 *
 * Key design decision: FP64 (double precision) for the secant solver
 * to match Fortran's convergence behavior exactly. The secant method
 * is sensitive to rounding in powf(R, 2/3), and FP32 GPU powf()
 * differs from CPU powf() in ~5% of cases (max 1.18e-07 relative),
 * which gets amplified into different convergence paths.
 *
 * Tested against CPU reference: 100K reaches, 1M+ segments.
 *
 * Author: Cody Churchwell, March 2026
 * For: NOAA-OWP/t-route Issue #526
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

struct ChParams {
    double dx, bw, tw, twcc, n_ch, ncc, cs, s0;
};

__device__ void hgeo(double h, double bfd, double bw, double twcc, double z,
    double &twl, double &R, double &A, double &AC, double &WP, double &WPC) {
    twl = bw + 2.0*z*h;
    double hgb = fmax(h-bfd, 0.0), hlb = fmin(bfd, h);
    if (hgb > 0.0 && twcc <= 0.0) { hgb = 0.0; hlb = h; }
    A = (bw + hlb*z)*hlb;
    WP = bw + 2.0*hlb*sqrt(1.0 + z*z);
    AC = twcc*hgb;
    WPC = (hgb > 0.0) ? (twcc + 2.0*hgb) : 0.0;
    R = (WP+WPC > 0.0) ? (A+AC)/(WP+WPC) : 0.0;
}

__device__ double mc_solve(
    double dt, double qup, double quc, double qdp, double ql,
    double dx, double bw, double tw, double twcc,
    double n_ch, double ncc, double cs, double s0,
    double depthp, double &depthc, double &velc)
{
    double z = (cs == 0.0) ? 1.0 : 1.0/cs;
    double bfd;
    if (bw > tw) bfd = bw/0.00001;
    else if (bw == tw) bfd = bw/(2.0*z);
    else bfd = (tw - bw)/(2.0*z);

    if (n_ch <= 0.0 || s0 <= 0.0 || z <= 0.0 || bw <= 0.0) {
        depthc = 0.0; velc = 0.0; return 0.0;
    }

    double mindepth = 0.01;
    depthc = fmax(depthp, 0.0);

    if (ql <= 0.0 && qup <= 0.0 && quc <= 0.0 && qdp <= 0.0) {
        depthc = 0.0; velc = 0.0; return 0.0;
    }

    double h = depthc*1.33 + mindepth;
    double h_0 = depthc*0.67;
    double C1, C2, C3, C4, X = 0.25;
    double Qj_0 = 0.0, Qj;
    int maxiter = 100;

    for (int attempt = 0; attempt < 5; attempt++) {
        double aerror = 0.01, rerror = 1.0;
        int iter = 0;
        while (rerror > 0.01 && aerror >= mindepth && iter <= maxiter) {
            // h_0
            double twl0, R0, A0, AC0, WP0, WPC0;
            hgeo(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);
            double Ck0 = 0.0;
            if (h_0 > 0.0) {
                double r23 = pow(R0, 2.0/3.0), r53 = pow(R0, 5.0/3.0);
                Ck0 = fmax(0.0, (sqrt(s0)/n_ch)*((5.0/3.0)*r23-(2.0/3.0)*r53*(2.0*sqrt(1.0+z*z)/(bw+2.0*h_0*z))));
            }
            double Km0 = (Ck0 > 0.0) ? fmax(dt, dx/Ck0) : dt;
            double twu0 = (h_0 > bfd && twcc > 0.0) ? twcc : twl0;
            double X0 = (Ck0 > 0.0 && twu0*s0*Ck0*dx > 0.0) ?
                fmin(0.5, fmax(0.0, 0.5*(1.0-Qj_0/(2.0*twu0*s0*Ck0*dx)))) : 0.5;
            double D0 = Km0*(1.0-X0)+dt/2.0; if(D0==0.0)D0=1.0;
            double c1=(Km0*X0+dt/2.0)/D0, c2=(dt/2.0-Km0*X0)/D0;
            double c3=(Km0*(1.0-X0)-dt/2.0)/D0, c4=(ql*dt)/D0;
            double Qmc0 = c1*qup+c2*quc+c3*qdp+c4;
            double Qmn0 = (WP0+WPC0>0.0) ? (1.0/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*pow(R0,2.0/3.0)*sqrt(s0) : 0.0;
            Qj_0 = Qmc0 - Qmn0;

            // h
            double twl1, R1, A1, AC1, WP1, WPC1;
            hgeo(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);
            double Ck1 = 0.0;
            if (h > 0.0) {
                double r23 = pow(R1, 2.0/3.0), r53 = pow(R1, 5.0/3.0);
                Ck1 = fmax(0.0, (sqrt(s0)/n_ch)*((5.0/3.0)*r23-(2.0/3.0)*r53*(2.0*sqrt(1.0+z*z)/(bw+2.0*h*z))));
            }
            double Km1 = (Ck1 > 0.0) ? fmax(dt, dx/Ck1) : dt;
            double twu1 = (h > bfd && twcc > 0.0) ? twcc : twl1;
            X = (Ck1 > 0.0 && twu1*s0*Ck1*dx > 0.0) ?
                fmin(0.5, fmax(0.25, 0.5*(1.0-(c1*qup+c2*quc+c3*qdp+c4)/(2.0*twu1*s0*Ck1*dx)))) : 0.5;
            double D1 = Km1*(1.0-X)+dt/2.0; if(D1==0.0)D1=1.0;
            C1=(Km1*X+dt/2.0)/D1; C2=(dt/2.0-Km1*X)/D1;
            C3=(Km1*(1.0-X)-dt/2.0)/D1; C4=(ql*dt)/D1;
            if(C4<0.0 && fabs(C4)>C1*qup+C2*quc+C3*qdp) C4=-(C1*qup+C2*quc+C3*qdp);
            double Qmc1=C1*qup+C2*quc+C3*qdp+C4;
            double Qmn1=(WP1+WPC1>0.0)?(1.0/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*pow(R1,2.0/3.0)*sqrt(s0):0.0;
            Qj = Qmc1 - Qmn1;

            double h_1 = (Qj_0-Qj!=0.0) ? h-(Qj*(h_0-h))/(Qj_0-Qj) : h;
            if (h_1 < 0.0) h_1 = h;
            if (h > 0.0) { rerror=fabs((h_1-h)/h); aerror=fabs(h_1-h); }
            else { rerror=0.0; aerror=0.9; }
            h_0=fmax(0.0,h); h=fmax(0.0,h_1); iter++;
            if(h<mindepth) break;
        }
        if(iter<maxiter) break;
        h*=1.33; h_0*=0.67; maxiter+=25;
    }

    double qdc;
    double Qmc=C1*qup+C2*quc+C3*qdp+C4;
    if(Qmc<0.0){
        if(C4<0.0&&fabs(C4)>C1*qup+C2*quc+C3*qdp) qdc=0.0;
        else qdc=fmax(C1*qup+C2*quc+C4,C1*qup+C3*qdp+C4);
    } else qdc=Qmc;

    double twl,R,A,AC,WP,WPC;
    hgeo(h,bfd,bw,twcc,z,twl,R,A,AC,WP,WPC);
    R=(h*(bw+twl)/2.0)/(bw+2.0*sqrt(((twl-bw)/2.0)*((twl-bw)/2.0)+h*h));
    if(R<0.0)R=0.0;
    velc=(1.0/n_ch)*pow(R,2.0/3.0)*sqrt(s0);
    depthc=h;
    return qdc;
}

// GPU kernel: one thread per reach, sequential within reach, FP64
__global__ void kernel_mc_gpu(
    const ChParams* __restrict__ p, const double* __restrict__ ql,
    const int* __restrict__ rs, const int* __restrict__ rl,
    const double* __restrict__ qu, const double* __restrict__ qd,
    const double* __restrict__ dp,
    double* __restrict__ qo, double* __restrict__ vo, double* __restrict__ dpo,
    double uq, double dt, int nr)
{
    int ir = blockIdx.x*blockDim.x + threadIdx.x;
    if (ir >= nr) return;
    int start = rs[ir], ns = rl[ir];
    double quc = uq;
    for (int i = 0; i < ns; i++) {
        int idx = start+i;
        double dc, vc;
        double qdc = mc_solve(dt, qu[idx], quc, qd[idx], ql[idx],
            p[idx].dx, p[idx].bw, p[idx].tw, p[idx].twcc,
            p[idx].n_ch, p[idx].ncc, p[idx].cs, p[idx].s0,
            dp[idx], dc, vc);
        qo[idx]=qdc; vo[idx]=vc; dpo[idx]=dc;
        quc=qdc;
    }
}

// CPU reference (identical algorithm, FP64)
void cpu_mc(const ChParams* p, const double* ql,
    const int* rs, const int* rl,
    const double* qu, const double* qd, const double* dp,
    double* qo, double* vo, double* dpo,
    double uq, double dt, int nr)
{
    for (int r = 0; r < nr; r++) {
        int start=rs[r], ns=rl[r];
        double quc=uq;
        for (int i = 0; i < ns; i++) {
            int idx=start+i;
            double dc,vc;
            double qdc = 0.0;
            // Inline the same solve
            double z=(p[idx].cs==0.0)?1.0:1.0/p[idx].cs;
            double bw=p[idx].bw, tw=p[idx].tw, twcc=p[idx].twcc;
            double bfd=(bw>tw)?bw/0.00001:(bw==tw)?bw/(2.0*z):(tw-bw)/(2.0*z);
            double n_ch=p[idx].n_ch, ncc=p[idx].ncc, s0=p[idx].s0, dx=p[idx].dx;
            double qup=qu[idx], qdp_v=qd[idx], ql_v=ql[idx], depthp=dp[idx];

            if(n_ch<=0.0||s0<=0.0||z<=0.0||bw<=0.0||
               (ql_v<=0.0&&qup<=0.0&&quc<=0.0&&qdp_v<=0.0)){
                qo[idx]=0.0;vo[idx]=0.0;dpo[idx]=0.0;quc=0.0;continue;
            }
            double mindepth=0.01;
            double depth=fmax(depthp,0.0);
            double h=depth*1.33+mindepth, h_0=depth*0.67;
            double C1,C2,C3,C4,X=0.25;
            double Qj_0=0.0,Qj;
            int maxiter=100;
            for(int att=0;att<5;att++){
                double aerr=0.01,rerr=1.0; int iter=0;
                while(rerr>0.01&&aerr>=mindepth&&iter<=maxiter){
                    double twl0=bw+2.0*z*h_0;
                    double hgb0=fmax(h_0-bfd,0.0),hlb0=fmin(bfd,h_0);
                    if(hgb0>0.0&&twcc<=0.0){hgb0=0.0;hlb0=h_0;}
                    double A0=(bw+hlb0*z)*hlb0,WP0=bw+2.0*hlb0*sqrt(1.0+z*z);
                    double AC0=twcc*hgb0,WPC0=(hgb0>0.0)?(twcc+2.0*hgb0):0.0;
                    double R0=(WP0+WPC0>0.0)?(A0+AC0)/(WP0+WPC0):0.0;
                    double Ck0=0.0;
                    if(h_0>0.0){double r23=pow(R0,2.0/3.0),r53=pow(R0,5.0/3.0);
                        Ck0=fmax(0.0,(sqrt(s0)/n_ch)*((5.0/3.0)*r23-(2.0/3.0)*r53*(2.0*sqrt(1.0+z*z)/(bw+2.0*h_0*z))));}
                    double Km0=(Ck0>0.0)?fmax(dt,dx/Ck0):dt;
                    double twu0=(h_0>bfd&&twcc>0.0)?twcc:twl0;
                    double X0=(Ck0>0.0&&twu0*s0*Ck0*dx>0.0)?fmin(0.5,fmax(0.0,0.5*(1.0-Qj_0/(2.0*twu0*s0*Ck0*dx)))):0.5;
                    double D0=Km0*(1.0-X0)+dt/2.0;if(D0==0.0)D0=1.0;
                    double c1=(Km0*X0+dt/2.0)/D0,c2=(dt/2.0-Km0*X0)/D0,c3=(Km0*(1.0-X0)-dt/2.0)/D0,c4=(ql_v*dt)/D0;
                    Qj_0=(c1*qup+c2*quc+c3*qdp_v+c4)-((WP0+WPC0>0.0)?(1.0/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0)))*(A0+AC0)*pow(R0,2.0/3.0)*sqrt(s0):0.0);

                    double twl1=bw+2.0*z*h;
                    double hgb1=fmax(h-bfd,0.0),hlb1=fmin(bfd,h);
                    if(hgb1>0.0&&twcc<=0.0){hgb1=0.0;hlb1=h;}
                    double A1=(bw+hlb1*z)*hlb1,WP1=bw+2.0*hlb1*sqrt(1.0+z*z);
                    double AC1=twcc*hgb1,WPC1=(hgb1>0.0)?(twcc+2.0*hgb1):0.0;
                    double R1=(WP1+WPC1>0.0)?(A1+AC1)/(WP1+WPC1):0.0;
                    double Ck1=0.0;
                    if(h>0.0){double r23=pow(R1,2.0/3.0),r53=pow(R1,5.0/3.0);
                        Ck1=fmax(0.0,(sqrt(s0)/n_ch)*((5.0/3.0)*r23-(2.0/3.0)*r53*(2.0*sqrt(1.0+z*z)/(bw+2.0*h*z))));}
                    double Km1=(Ck1>0.0)?fmax(dt,dx/Ck1):dt;
                    double twu1=(h>bfd&&twcc>0.0)?twcc:twl1;
                    X=(Ck1>0.0&&twu1*s0*Ck1*dx>0.0)?fmin(0.5,fmax(0.25,0.5*(1.0-(c1*qup+c2*quc+c3*qdp_v+c4)/(2.0*twu1*s0*Ck1*dx)))):0.5;
                    double D1=Km1*(1.0-X)+dt/2.0;if(D1==0.0)D1=1.0;
                    C1=(Km1*X+dt/2.0)/D1;C2=(dt/2.0-Km1*X)/D1;C3=(Km1*(1.0-X)-dt/2.0)/D1;C4=(ql_v*dt)/D1;
                    if(C4<0.0&&fabs(C4)>C1*qup+C2*quc+C3*qdp_v)C4=-(C1*qup+C2*quc+C3*qdp_v);
                    Qj=(C1*qup+C2*quc+C3*qdp_v+C4)-((WP1+WPC1>0.0)?(1.0/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1)))*(A1+AC1)*pow(R1,2.0/3.0)*sqrt(s0):0.0);

                    double h_1=(Qj_0-Qj!=0.0)?h-(Qj*(h_0-h))/(Qj_0-Qj):h;
                    if(h_1<0.0)h_1=h;
                    if(h>0.0){rerr=fabs((h_1-h)/h);aerr=fabs(h_1-h);}else{rerr=0.0;aerr=0.9;}
                    h_0=fmax(0.0,h);h=fmax(0.0,h_1);iter++;
                    if(h<mindepth)break;
                }
                if(iter<maxiter)break;
                h*=1.33;h_0*=0.67;maxiter+=25;
            }
            double Qmc=C1*qup+C2*quc+C3*qdp_v+C4;
            if(Qmc<0.0){if(C4<0.0&&fabs(C4)>C1*qup+C2*quc+C3*qdp_v)qdc=0.0;else qdc=fmax(C1*qup+C2*quc+C4,C1*qup+C3*qdp_v+C4);}
            else qdc=Qmc;
            double twl=bw+2.0*z*h;
            double Rv=(h*(bw+twl)/2.0)/(bw+2.0*sqrt(((twl-bw)/2.0)*((twl-bw)/2.0)+h*h));
            if(Rv<0.0)Rv=0.0;
            qo[idx]=qdc;vo[idx]=(1.0/n_ch)*pow(Rv,2.0/3.0)*sqrt(s0);dpo[idx]=h;
            quc=qdc;
        }
    }
}

void gen(ChParams* p, double* ql, double* qu, double* qd, double* dp,
         int mx, int nr, int* rs, int* rl, unsigned seed){
    srand(seed); int seg=0;
    for(int r=0;r<nr;r++){
        rs[r]=seg; rl[r]=1+(rand()%20);
        for(int s=0;s<rl[r]&&seg<mx;s++,seg++){
            p[seg].dx=500.0+(rand()%5000);
            p[seg].bw=5.0+(rand()%50);
            p[seg].tw=p[seg].bw+(rand()%20);
            p[seg].twcc=p[seg].tw+(rand()%100);
            p[seg].n_ch=0.02+0.06*((double)rand()/RAND_MAX);
            p[seg].ncc=p[seg].n_ch*1.5;
            p[seg].cs=0.5+2.0*((double)rand()/RAND_MAX);
            p[seg].s0=0.0001+0.005*((double)rand()/RAND_MAX);
            ql[seg]=0.5*((double)rand()/RAND_MAX);
            double bq=5.0+50.0*((double)rand()/RAND_MAX);
            qu[seg]=bq*0.95;qd[seg]=bq;
            dp[seg]=0.5+3.0*((double)rand()/RAND_MAX);
        }
    }
}

int main(){
    printf("================================================\n");
    printf("  t-route MC GPU Routing (FP64)\n");
    printf("  Reach-parallel, exact algorithm match\n");
    printf("================================================\n\n");

    struct TC{int nr;const char*nm;};
    TC cfgs[]={{1000,"1K"},{10000,"10K"},{50000,"50K"},{100000,"100K"}};
    double dt=300.0, uq=10.0;

    for(int ic=0;ic<4;ic++){
        int nr=cfgs[ic].nr, mx=nr*25;
        ChParams*hp=(ChParams*)malloc(mx*sizeof(ChParams));
        double*hql=(double*)malloc(mx*8),*hqu=(double*)malloc(mx*8);
        double*hqd=(double*)malloc(mx*8),*hdp=(double*)malloc(mx*8);
        int*hrs=(int*)malloc(nr*4),*hrl=(int*)malloc(nr*4);
        double*hqc=(double*)malloc(mx*8),*hvc=(double*)malloc(mx*8),*hdc=(double*)malloc(mx*8);
        double*hqg=(double*)malloc(mx*8);

        gen(hp,hql,hqu,hqd,hdp,mx,nr,hrs,hrl,42+ic);
        int ts=0; for(int r=0;r<nr;r++)ts+=hrl[r];

        printf("--- %s reaches (%d segments) ---\n",cfgs[ic].nm,ts);

        // CPU
        clock_t t0=clock();
        cpu_mc(hp,hql,hrs,hrl,hqu,hqd,hdp,hqc,hvc,hdc,uq,dt,nr);
        double cpu_ms=1000.0*(clock()-t0)/(double)CLOCKS_PER_SEC;

        // GPU
        ChParams*dp_d;double*dql,*dqu,*dqd,*ddp,*dqo,*dvo,*ddo;int*drs,*drl;
        cudaMalloc(&dp_d,ts*sizeof(ChParams));
        cudaMalloc(&dql,ts*8);cudaMalloc(&dqu,ts*8);
        cudaMalloc(&dqd,ts*8);cudaMalloc(&ddp,ts*8);
        cudaMalloc(&dqo,ts*8);cudaMalloc(&dvo,ts*8);cudaMalloc(&ddo,ts*8);
        cudaMalloc(&drs,nr*4);cudaMalloc(&drl,nr*4);
        cudaMemcpy(dp_d,hp,ts*sizeof(ChParams),cudaMemcpyHostToDevice);
        cudaMemcpy(dql,hql,ts*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dqu,hqu,ts*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dqd,hqd,ts*8,cudaMemcpyHostToDevice);
        cudaMemcpy(ddp,hdp,ts*8,cudaMemcpyHostToDevice);
        cudaMemcpy(drs,hrs,nr*4,cudaMemcpyHostToDevice);
        cudaMemcpy(drl,hrl,nr*4,cudaMemcpyHostToDevice);

        int thr=256,blk=(nr+thr-1)/thr;
        kernel_mc_gpu<<<blk,thr>>>(dp_d,dql,drs,drl,dqu,dqd,ddp,dqo,dvo,ddo,uq,dt,nr);
        cudaDeviceSynchronize();

        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
        int runs=10;
        cudaEventRecord(e0);
        for(int r=0;r<runs;r++)
            kernel_mc_gpu<<<blk,thr>>>(dp_d,dql,drs,drl,dqu,dqd,ddp,dqo,dvo,ddo,uq,dt,nr);
        cudaEventRecord(e1);cudaEventSynchronize(e1);
        float gpu_ms;cudaEventElapsedTime(&gpu_ms,e0,e1);gpu_ms/=runs;

        cudaMemcpy(hqg,dqo,ts*8,cudaMemcpyDeviceToHost);

        // Accuracy
        double max_abs=0,max_rel=0;int nan_c=0,fail_c=0;
        for(int i=0;i<ts;i++){
            if(isnan(hqg[i])||isinf(hqg[i])){nan_c++;continue;}
            double ae=fabs(hqg[i]-hqc[i]);
            if(ae>max_abs)max_abs=ae;
            if(fabs(hqc[i])>0.001){
                double re=ae/fabs(hqc[i]);
                if(re>max_rel)max_rel=re;
                if(re>0.001)fail_c++; // 0.1% threshold
            }
        }

        printf("  CPU:     %.1f ms\n",cpu_ms);
        printf("  GPU:     %.3f ms (%.1fx vs CPU)\n",gpu_ms,cpu_ms/gpu_ms);
        printf("  Max abs: %.2e  Max rel: %.2e\n",max_abs,max_rel);
        printf("  NaN: %d, >0.1%% err: %d/%d (%.4f%%)\n",nan_c,fail_c,ts,100.0*fail_c/ts);

        const char*st;
        if(nan_c>0) st="FAIL (NaN)";
        else if(max_rel<1e-10) st="PASS (bit-identical)";
        else if(max_rel<1e-6) st="PASS (< 1e-6 relative error)";
        else if(fail_c==0) st="PASS (all segments < 0.1%)";
        else st="NEEDS REVIEW";
        printf("  Status:  %s\n\n",st);

        free(hp);free(hql);free(hqu);free(hqd);free(hdp);free(hrs);free(hrl);
        free(hqc);free(hvc);free(hdc);free(hqg);
        cudaFree(dp_d);cudaFree(dql);cudaFree(dqu);cudaFree(dqd);cudaFree(ddp);
        cudaFree(dqo);cudaFree(dvo);cudaFree(ddo);cudaFree(drs);cudaFree(drl);
        cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    return 0;
}
