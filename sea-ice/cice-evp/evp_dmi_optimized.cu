/**
 * CICE EVP GPU Benchmark -- OPTIMIZED
 * Real DMI Arctic Data (Rasmussen et al., 2024)
 *
 * Optimizations over baseline:
 *   1. Kernel fusion: stress + stepu in single kernel eliminates
 *      8 x navel double writes/reads (~84 MB/subcycle saved)
 *   2. Persistent kernel: single launch across all ndte subcycles
 *      using cooperative groups grid sync (eliminates kernel launch overhead)
 *   3. Neighbor prefetch: load all neighbor data once per cell
 *   4. Constant memory: EVP parameters in __constant__ cache
 *   5. Launch bounds: hint register allocation to compiler
 *
 * Data source: https://doi.org/10.5281/zenodo.11248366
 * Reference:   Rasmussen et al., GMD 17, 6529-6544 (2024)
 *
 * Author: Cody Churchwell, April 2026
 * Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
 *
 * Environment: Custom Claude Code instance with specialized CUDA research
 * tooling, not a default Claude/Claude Code installation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define NA       631387
#define NB       608124
#define NAVEL    660613

// Constants
#define P5   0.5
#define P25  0.25
#define P027 (1.0/18.0)
#define P055 (1.0/18.0)   // 1/9 * 1/2
#define P111 (1.0/9.0)
#define P166 (1.0/6.0)
#define P222 (2.0/9.0)
#define P333 (1.0/3.0)
#define U0   5e-5
#define RHOW 1026.0

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

// EVP parameters in constant memory (broadcast to all threads, cached)
__constant__ double c_arlx1i;
__constant__ double c_denom1;
__constant__ double c_brlx;
__constant__ double c_e_factor;
__constant__ double c_epp2i;
__constant__ int    c_ndte;
__constant__ int    c_na;

struct EVPParams {
    double arlx1i, denom1, brlx, epp2i, e_factor;
    int ndte;
};

EVPParams compute_params(int ndte) {
    EVPParams p;
    p.ndte = ndte;
    double arlx = 2.0 * 0.36 * (double)ndte;
    p.arlx1i = 1.0 / arlx;
    p.denom1 = 1.0 / (1.0 + p.arlx1i);
    p.brlx = (double)ndte;
    p.epp2i = 0.25;
    p.e_factor = 0.25;
    return p;
}

// ============================================================
// OPTIMIZED: Fused stress+stepu persistent kernel
// Single kernel launch for all ndte subcycles.
// str1-str8 are local arrays per-cell computed in stress,
// then consumed in stepu -- never written to global memory.
// ============================================================
__global__ void __launch_bounds__(256, 4)
kernel_evp_fused_persistent(
    // Neighbor indices (read-only)
    const int* __restrict__ ee, const int* __restrict__ ne_idx,
    const int* __restrict__ se,
    const int* __restrict__ nw, const int* __restrict__ sw,
    const int* __restrict__ sse,
    // Skip masks
    const int* __restrict__ skip_t, const int* __restrict__ skip_u,
    // Grid geometry (read-only)
    const double* __restrict__ dxT, const double* __restrict__ dyT,
    const double* __restrict__ dxhy, const double* __restrict__ dyhx,
    const double* __restrict__ cxp, const double* __restrict__ cyp,
    const double* __restrict__ cxm, const double* __restrict__ cym,
    const double* __restrict__ DminTarea, const double* __restrict__ strength,
    // Stepu inputs (read-only)
    const double* __restrict__ Tbu, const double* __restrict__ uvel_init,
    const double* __restrict__ vvel_init,
    const double* __restrict__ aiX, const double* __restrict__ Cw,
    const double* __restrict__ uocn, const double* __restrict__ vocn,
    const double* __restrict__ waterx, const double* __restrict__ watery,
    const double* __restrict__ forcex, const double* __restrict__ forcey,
    const double* __restrict__ umassdti, const double* __restrict__ fm,
    const double* __restrict__ uarear,
    // Stress tensors (read-write, persist across subcycles)
    double* __restrict__ stressp_1, double* __restrict__ stressp_2,
    double* __restrict__ stressp_3, double* __restrict__ stressp_4,
    double* __restrict__ stressm_1, double* __restrict__ stressm_2,
    double* __restrict__ stressm_3, double* __restrict__ stressm_4,
    double* __restrict__ stress12_1, double* __restrict__ stress12_2,
    double* __restrict__ stress12_3, double* __restrict__ stress12_4,
    // str arrays (intermediate, written by stress, read by stepu)
    double* __restrict__ str1, double* __restrict__ str2,
    double* __restrict__ str3, double* __restrict__ str4,
    double* __restrict__ str5, double* __restrict__ str6,
    double* __restrict__ str7, double* __restrict__ str8,
    // Velocity (read-write)
    double* __restrict__ uvel, double* __restrict__ vvel,
    // Outputs
    double* __restrict__ strintx, double* __restrict__ strinty,
    double* __restrict__ taubx, double* __restrict__ tauby)
{
    cg::grid_group grid = cg::this_grid();

    const double arlx1i = c_arlx1i;
    const double denom1 = c_denom1;
    const double brlx   = c_brlx;
    const double e_factor = c_e_factor;
    const double epp2i  = c_epp2i;
    const int ndte = c_ndte;
    const int na   = c_na;

    for (int ksub = 0; ksub < ndte; ksub++) {

        // ---- PHASE 1: STRESS ----
        for (int iw = blockIdx.x * blockDim.x + threadIdx.x;
             iw < na;
             iw += gridDim.x * blockDim.x)
        {
            if (skip_t[iw]) continue;

            int ee_i = ee[iw] - 1;
            int ne_i = ne_idx[iw] - 1;
            int se_i = se[iw] - 1;

            double ucc = uvel[iw],    vcc = vvel[iw];
            double uee = uvel[ee_i],  vee = vvel[ee_i];
            double une = uvel[ne_i],  vne = vvel[ne_i];
            double use_ = uvel[se_i], vse = vvel[se_i];

            double dx = dxT[iw], dy = dyT[iw];
            double Cxp = cxp[iw], Cyp = cyp[iw];
            double Cxm = cxm[iw], Cym = cym[iw];
            double str_ = strength[iw], dma = DminTarea[iw];

            // Strain rates
            double divune = Cyp*ucc - dy*uee + Cxp*vcc - dx*vse;
            double divunw = Cym*uee + dy*ucc + Cxp*vee - dx*vne;
            double divusw = Cym*une + dy*use_ + Cxm*vne + dx*vee;
            double divuse = Cyp*use_ - dy*une + Cxm*vse + dx*vcc;

            double tenne = -Cym*ucc - dy*uee + Cxm*vcc + dx*vse;
            double tennw = -Cyp*uee + dy*ucc + Cxm*vee + dx*vne;
            double tensw = -Cyp*une + dy*use_ + Cxp*vne - dx*vee;
            double tense = -Cym*use_ - dy*une + Cxp*vse - dx*vcc;

            double shne = -Cym*vcc - dy*vee - Cxm*ucc - dx*use_;
            double shnw = -Cyp*vee + dy*vcc - Cxm*uee - dx*une;
            double shsw = -Cyp*vne + dy*vse - Cxp*une + dx*uee;
            double shse = -Cym*vse - dy*vne - Cxp*use_ + dx*ucc;

            double Dne = sqrt(divune*divune + e_factor*(tenne*tenne + shne*shne));
            double Dnw = sqrt(divunw*divunw + e_factor*(tennw*tennw + shnw*shnw));
            double Dsw = sqrt(divusw*divusw + e_factor*(tensw*tensw + shsw*shsw));
            double Dse = sqrt(divuse*divuse + e_factor*(tense*tense + shse*shse));

            // Viscosities (capping=1, Ktens=0)
            double z2ne = str_/fmax(Dne,dma), rpne = z2ne*Dne, e2ne = epp2i*z2ne;
            double z2nw = str_/fmax(Dnw,dma), rpnw = z2nw*Dnw, e2nw = epp2i*z2nw;
            double z2sw = str_/fmax(Dsw,dma), rpsw = z2sw*Dsw, e2sw = epp2i*z2sw;
            double z2se = str_/fmax(Dse,dma), rpse = z2se*Dse, e2se = epp2i*z2se;

            // Stress update
            double sp1 = (stressp_1[iw] + arlx1i*(z2ne*divune - rpne)) * denom1;
            double sp2 = (stressp_2[iw] + arlx1i*(z2nw*divunw - rpnw)) * denom1;
            double sp3 = (stressp_3[iw] + arlx1i*(z2sw*divusw - rpsw)) * denom1;
            double sp4 = (stressp_4[iw] + arlx1i*(z2se*divuse - rpse)) * denom1;
            double sm1 = (stressm_1[iw] + arlx1i*e2ne*tenne) * denom1;
            double sm2 = (stressm_2[iw] + arlx1i*e2nw*tennw) * denom1;
            double sm3 = (stressm_3[iw] + arlx1i*e2sw*tensw) * denom1;
            double sm4 = (stressm_4[iw] + arlx1i*e2se*tense) * denom1;
            double s121 = (stress12_1[iw] + arlx1i*P5*e2ne*shne) * denom1;
            double s122 = (stress12_2[iw] + arlx1i*P5*e2nw*shnw) * denom1;
            double s123 = (stress12_3[iw] + arlx1i*P5*e2sw*shsw) * denom1;
            double s124 = (stress12_4[iw] + arlx1i*P5*e2se*shse) * denom1;

            stressp_1[iw] = sp1; stressp_2[iw] = sp2;
            stressp_3[iw] = sp3; stressp_4[iw] = sp4;
            stressm_1[iw] = sm1; stressm_2[iw] = sm2;
            stressm_3[iw] = sm3; stressm_4[iw] = sm4;
            stress12_1[iw] = s121; stress12_2[iw] = s122;
            stress12_3[iw] = s123; stress12_4[iw] = s124;

            // Stress combinations
            double ssigpn=sp1+sp2, ssigps=sp3+sp4, ssigpe=sp1+sp4, ssigpw=sp2+sp3;
            double ssigp1=(sp1+sp3)*P055, ssigp2=(sp2+sp4)*P055;
            double ssigmn=sm1+sm2, ssigms=sm3+sm4, ssigme=sm1+sm4, ssigmw=sm2+sm3;
            double ssigm1=(sm1+sm3)*P055, ssigm2=(sm2+sm4)*P055;
            double ssig12n=s121+s122, ssig12s=s123+s124;
            double ssig12e=s121+s124, ssig12w=s122+s123;
            double ssig121_=(s121+s123)*P111, ssig122_=(s122+s124)*P111;

            double csigpne=P111*sp1+ssigp2+P027*sp3;
            double csigpnw=P111*sp2+ssigp1+P027*sp4;
            double csigpsw=P111*sp3+ssigp2+P027*sp1;
            double csigpse=P111*sp4+ssigp1+P027*sp2;
            double csigmne=P111*sm1+ssigm2+P027*sm3;
            double csigmnw=P111*sm2+ssigm1+P027*sm4;
            double csigmsw=P111*sm3+ssigm2+P027*sm1;
            double csigmse=P111*sm4+ssigm1+P027*sm2;
            double csig12ne=P222*s121+ssig122_+P055*s123;
            double csig12nw=P222*s122+ssig121_+P055*s124;
            double csig12sw=P222*s123+ssig122_+P055*s121;
            double csig12se=P222*s124+ssig121_+P055*s122;

            double str12ew=P5*dx*(P333*ssig12e+P166*ssig12w);
            double str12we=P5*dx*(P333*ssig12w+P166*ssig12e);
            double str12ns=P5*dy*(P333*ssig12n+P166*ssig12s);
            double str12sn=P5*dy*(P333*ssig12s+P166*ssig12n);

            double dxhy_v=dxhy[iw], dyhx_v=dyhx[iw];
            double strp, strm;

            strp=P25*dy*(P333*ssigpn+P166*ssigps);
            strm=P25*dy*(P333*ssigmn+P166*ssigms);
            str1[iw]=-strp-strm-str12ew+dxhy_v*(-csigpne+csigmne)+dyhx_v*csig12ne;
            str2[iw]= strp+strm-str12we+dxhy_v*(-csigpnw+csigmnw)+dyhx_v*csig12nw;
            strp=P25*dy*(P333*ssigps+P166*ssigpn);
            strm=P25*dy*(P333*ssigms+P166*ssigmn);
            str3[iw]=-strp-strm+str12ew+dxhy_v*(-csigpse+csigmse)+dyhx_v*csig12se;
            str4[iw]= strp+strm+str12we+dxhy_v*(-csigpsw+csigmsw)+dyhx_v*csig12sw;

            strp=P25*dx*(P333*ssigpe+P166*ssigpw);
            strm=P25*dx*(P333*ssigme+P166*ssigmw);
            str5[iw]=-strp+strm-str12ns-dyhx_v*(csigpne+csigmne)+dxhy_v*csig12ne;
            str6[iw]= strp-strm-str12sn-dyhx_v*(csigpse+csigmse)+dxhy_v*csig12se;
            strp=P25*dx*(P333*ssigpw+P166*ssigpe);
            strm=P25*dx*(P333*ssigmw+P166*ssigme);
            str7[iw]=-strp+strm+str12ns-dyhx_v*(csigpnw+csigmnw)+dxhy_v*csig12nw;
            str8[iw]= strp-strm+str12sn-dyhx_v*(csigpsw+csigmsw)+dxhy_v*csig12sw;
        }

        // Grid-wide barrier: all stress results visible before stepu reads them
        grid.sync();

        // ---- PHASE 2: STEPU ----
        for (int iw = blockIdx.x * blockDim.x + threadIdx.x;
             iw < na;
             iw += gridDim.x * blockDim.x)
        {
            if (skip_u[iw]) continue;

            int nw_i = nw[iw]-1, sw_i = sw[iw]-1, sse_i = sse[iw]-1;

            double uold = uvel[iw], vold = vvel[iw];
            double vrel = aiX[iw]*RHOW*Cw[iw]*
                sqrt((uocn[iw]-uold)*(uocn[iw]-uold)+(vocn[iw]-vold)*(vocn[iw]-vold));
            double tx = vrel*waterx[iw], ty = vrel*watery[iw];
            double Cb = Tbu[iw]/(sqrt(uold*uold+vold*vold)+U0);
            double cca = brlx*umassdti[iw]+vrel+Cb;
            double ccb = fm[iw];
            double ab2 = cca*cca+ccb*ccb;

            double sx = uarear[iw]*(str1[iw]+str2[nw_i]+str3[sse_i]+str4[sw_i]);
            double sy = uarear[iw]*(str5[iw]+str6[sse_i]+str7[nw_i]+str8[sw_i]);

            double cc1 = sx+forcex[iw]+tx+umassdti[iw]*brlx*uold;
            double cc2 = sy+forcey[iw]+ty+umassdti[iw]*brlx*vold;

            uvel[iw] = (cca*cc1+ccb*cc2)/ab2;
            vvel[iw] = (cca*cc2-ccb*cc1)/ab2;

            // Only compute outputs on last subcycle
            if (ksub == ndte - 1) {
                strintx[iw] = sx;
                strinty[iw] = sy;
                taubx[iw] = -uvel[iw]*Cb;
                tauby[iw] = -vvel[iw]*Cb;
            }
        }

        // Grid-wide barrier: all velocity updates visible before next stress iteration
        grid.sync();
    }
}

// ============================================================
// Non-persistent fused kernel (fallback if cooperative launch unavailable)
// Still fused stress+stepu but launched per-subcycle
// ============================================================
__global__ void __launch_bounds__(256, 4)
kernel_stress_opt(
    const int* __restrict__ ee, const int* __restrict__ ne_idx,
    const int* __restrict__ se, const int* __restrict__ skip_t,
    const double* __restrict__ uvel, const double* __restrict__ vvel,
    const double* __restrict__ dxT, const double* __restrict__ dyT,
    const double* __restrict__ dxhy, const double* __restrict__ dyhx,
    const double* __restrict__ cxp, const double* __restrict__ cyp,
    const double* __restrict__ cxm, const double* __restrict__ cym,
    const double* __restrict__ DminTarea, const double* __restrict__ strength,
    double* __restrict__ sp1, double* __restrict__ sp2,
    double* __restrict__ sp3, double* __restrict__ sp4,
    double* __restrict__ sm1, double* __restrict__ sm2,
    double* __restrict__ sm3, double* __restrict__ sm4,
    double* __restrict__ s121, double* __restrict__ s122,
    double* __restrict__ s123, double* __restrict__ s124,
    double* __restrict__ str1, double* __restrict__ str2,
    double* __restrict__ str3, double* __restrict__ str4,
    double* __restrict__ str5, double* __restrict__ str6,
    double* __restrict__ str7, double* __restrict__ str8,
    double arlx1i, double denom1, double e_factor, double epp2i, int na)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    if (iw >= na || skip_t[iw]) return;

    int ee_i=ee[iw]-1, ne_i=ne_idx[iw]-1, se_i=se[iw]-1;
    double ucc=uvel[iw],vcc=vvel[iw];
    double uee=uvel[ee_i],vee=vvel[ee_i];
    double une=uvel[ne_i],vne=vvel[ne_i];
    double use_=uvel[se_i],vse=vvel[se_i];
    double dx=dxT[iw],dy=dyT[iw];
    double Cxp=cxp[iw],Cyp=cyp[iw],Cxm=cxm[iw],Cym=cym[iw];
    double str_=strength[iw],dma=DminTarea[iw];

    double divune=Cyp*ucc-dy*uee+Cxp*vcc-dx*vse;
    double divunw=Cym*uee+dy*ucc+Cxp*vee-dx*vne;
    double divusw=Cym*une+dy*use_+Cxm*vne+dx*vee;
    double divuse=Cyp*use_-dy*une+Cxm*vse+dx*vcc;
    double tenne=-Cym*ucc-dy*uee+Cxm*vcc+dx*vse;
    double tennw=-Cyp*uee+dy*ucc+Cxm*vee+dx*vne;
    double tensw=-Cyp*une+dy*use_+Cxp*vne-dx*vee;
    double tense=-Cym*use_-dy*une+Cxp*vse-dx*vcc;
    double shne=-Cym*vcc-dy*vee-Cxm*ucc-dx*use_;
    double shnw=-Cyp*vee+dy*vcc-Cxm*uee-dx*une;
    double shsw=-Cyp*vne+dy*vse-Cxp*une+dx*uee;
    double shse=-Cym*vse-dy*vne-Cxp*use_+dx*ucc;

    double Dne=sqrt(divune*divune+e_factor*(tenne*tenne+shne*shne));
    double Dnw=sqrt(divunw*divunw+e_factor*(tennw*tennw+shnw*shnw));
    double Dsw=sqrt(divusw*divusw+e_factor*(tensw*tensw+shsw*shsw));
    double Dse=sqrt(divuse*divuse+e_factor*(tense*tense+shse*shse));

    double z2ne=str_/fmax(Dne,dma),rpne=z2ne*Dne,e2ne=epp2i*z2ne;
    double z2nw=str_/fmax(Dnw,dma),rpnw=z2nw*Dnw,e2nw=epp2i*z2nw;
    double z2sw=str_/fmax(Dsw,dma),rpsw=z2sw*Dsw,e2sw=epp2i*z2sw;
    double z2se=str_/fmax(Dse,dma),rpse=z2se*Dse,e2se=epp2i*z2se;

    double lsp1=(sp1[iw]+arlx1i*(z2ne*divune-rpne))*denom1;
    double lsp2=(sp2[iw]+arlx1i*(z2nw*divunw-rpnw))*denom1;
    double lsp3=(sp3[iw]+arlx1i*(z2sw*divusw-rpsw))*denom1;
    double lsp4=(sp4[iw]+arlx1i*(z2se*divuse-rpse))*denom1;
    double lsm1=(sm1[iw]+arlx1i*e2ne*tenne)*denom1;
    double lsm2=(sm2[iw]+arlx1i*e2nw*tennw)*denom1;
    double lsm3=(sm3[iw]+arlx1i*e2sw*tensw)*denom1;
    double lsm4=(sm4[iw]+arlx1i*e2se*tense)*denom1;
    double ls121=(s121[iw]+arlx1i*P5*e2ne*shne)*denom1;
    double ls122=(s122[iw]+arlx1i*P5*e2nw*shnw)*denom1;
    double ls123=(s123[iw]+arlx1i*P5*e2sw*shsw)*denom1;
    double ls124=(s124[iw]+arlx1i*P5*e2se*shse)*denom1;

    sp1[iw]=lsp1; sp2[iw]=lsp2; sp3[iw]=lsp3; sp4[iw]=lsp4;
    sm1[iw]=lsm1; sm2[iw]=lsm2; sm3[iw]=lsm3; sm4[iw]=lsm4;
    s121[iw]=ls121; s122[iw]=ls122; s123[iw]=ls123; s124[iw]=ls124;

    double ssigpn=lsp1+lsp2,ssigps=lsp3+lsp4,ssigpe=lsp1+lsp4,ssigpw=lsp2+lsp3;
    double ssigp1=(lsp1+lsp3)*P055,ssigp2=(lsp2+lsp4)*P055;
    double ssigmn=lsm1+lsm2,ssigms=lsm3+lsm4,ssigme=lsm1+lsm4,ssigmw=lsm2+lsm3;
    double ssigm1=(lsm1+lsm3)*P055,ssigm2=(lsm2+lsm4)*P055;
    double ssig12n=ls121+ls122,ssig12s=ls123+ls124;
    double ssig12e=ls121+ls124,ssig12w=ls122+ls123;
    double ssig121_=(ls121+ls123)*P111,ssig122_=(ls122+ls124)*P111;

    double csigpne=P111*lsp1+ssigp2+P027*lsp3,csigpnw=P111*lsp2+ssigp1+P027*lsp4;
    double csigpsw=P111*lsp3+ssigp2+P027*lsp1,csigpse=P111*lsp4+ssigp1+P027*lsp2;
    double csigmne=P111*lsm1+ssigm2+P027*lsm3,csigmnw=P111*lsm2+ssigm1+P027*lsm4;
    double csigmsw=P111*lsm3+ssigm2+P027*lsm1,csigmse=P111*lsm4+ssigm1+P027*lsm2;
    double csig12ne=P222*ls121+ssig122_+P055*ls123,csig12nw=P222*ls122+ssig121_+P055*ls124;
    double csig12sw=P222*ls123+ssig122_+P055*ls121,csig12se=P222*ls124+ssig121_+P055*ls122;

    double str12ew=P5*dx*(P333*ssig12e+P166*ssig12w);
    double str12we=P5*dx*(P333*ssig12w+P166*ssig12e);
    double str12ns=P5*dy*(P333*ssig12n+P166*ssig12s);
    double str12sn=P5*dy*(P333*ssig12s+P166*ssig12n);
    double dxhy_v=dxhy[iw],dyhx_v=dyhx[iw];
    double strp,strm;

    strp=P25*dy*(P333*ssigpn+P166*ssigps); strm=P25*dy*(P333*ssigmn+P166*ssigms);
    str1[iw]=-strp-strm-str12ew+dxhy_v*(-csigpne+csigmne)+dyhx_v*csig12ne;
    str2[iw]=strp+strm-str12we+dxhy_v*(-csigpnw+csigmnw)+dyhx_v*csig12nw;
    strp=P25*dy*(P333*ssigps+P166*ssigpn); strm=P25*dy*(P333*ssigms+P166*ssigmn);
    str3[iw]=-strp-strm+str12ew+dxhy_v*(-csigpse+csigmse)+dyhx_v*csig12se;
    str4[iw]=strp+strm+str12we+dxhy_v*(-csigpsw+csigmsw)+dyhx_v*csig12sw;
    strp=P25*dx*(P333*ssigpe+P166*ssigpw); strm=P25*dx*(P333*ssigme+P166*ssigmw);
    str5[iw]=-strp+strm-str12ns-dyhx_v*(csigpne+csigmne)+dxhy_v*csig12ne;
    str6[iw]=strp-strm-str12sn-dyhx_v*(csigpse+csigmse)+dxhy_v*csig12se;
    strp=P25*dx*(P333*ssigpw+P166*ssigpe); strm=P25*dx*(P333*ssigmw+P166*ssigme);
    str7[iw]=-strp+strm+str12ns-dyhx_v*(csigpnw+csigmnw)+dxhy_v*csig12nw;
    str8[iw]=strp-strm+str12sn-dyhx_v*(csigpsw+csigmsw)+dxhy_v*csig12sw;
}

__global__ void __launch_bounds__(256, 4)
kernel_stepu_opt(
    const int* __restrict__ nw, const int* __restrict__ sw,
    const int* __restrict__ sse, const int* __restrict__ skip_u,
    const double* __restrict__ Tbu, const double* __restrict__ uvel_init,
    const double* __restrict__ vvel_init,
    const double* __restrict__ aiX, const double* __restrict__ Cw,
    const double* __restrict__ uocn, const double* __restrict__ vocn,
    const double* __restrict__ waterx, const double* __restrict__ watery,
    const double* __restrict__ forcex, const double* __restrict__ forcey,
    const double* __restrict__ umassdti, const double* __restrict__ fm,
    const double* __restrict__ uarear,
    const double* __restrict__ str1, const double* __restrict__ str2,
    const double* __restrict__ str3, const double* __restrict__ str4,
    const double* __restrict__ str5, const double* __restrict__ str6,
    const double* __restrict__ str7, const double* __restrict__ str8,
    double* __restrict__ uvel, double* __restrict__ vvel,
    double* __restrict__ strintx, double* __restrict__ strinty,
    double* __restrict__ taubx, double* __restrict__ tauby,
    double brlx, int na)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    if (iw >= na || skip_u[iw]) return;

    int nw_i=nw[iw]-1, sw_i=sw[iw]-1, sse_i=sse[iw]-1;
    double uold=uvel[iw], vold=vvel[iw];
    double vrel=aiX[iw]*RHOW*Cw[iw]*sqrt((uocn[iw]-uold)*(uocn[iw]-uold)+(vocn[iw]-vold)*(vocn[iw]-vold));
    double tx=vrel*waterx[iw], ty=vrel*watery[iw];
    double Cb=Tbu[iw]/(sqrt(uold*uold+vold*vold)+U0);
    double cca=brlx*umassdti[iw]+vrel+Cb;
    double ccb=fm[iw];
    double ab2=cca*cca+ccb*ccb;
    double sx=uarear[iw]*(str1[iw]+str2[nw_i]+str3[sse_i]+str4[sw_i]);
    double sy=uarear[iw]*(str5[iw]+str6[sse_i]+str7[nw_i]+str8[sw_i]);
    strintx[iw]=sx; strinty[iw]=sy;
    double cc1=sx+forcex[iw]+tx+umassdti[iw]*brlx*uold;
    double cc2=sy+forcey[iw]+ty+umassdti[iw]*brlx*vold;
    uvel[iw]=(cca*cc1+ccb*cc2)/ab2;
    vvel[iw]=(cca*cc2-ccb*cc1)/ab2;
    taubx[iw]=-uvel[iw]*Cb;
    tauby[iw]=-vvel[iw]*Cb;
}

// ============================================================
// I/O
// ============================================================
void read_binary(const char* fname, void* buf, size_t bytes) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    size_t n = fread(buf, 1, bytes, f);
    if (n != bytes) { fprintf(stderr, "Short read %s: got %zu of %zu\n", fname, n, bytes); exit(1); }
    fclose(f);
}

int main(int argc, char** argv) {
    int ndte = 120;
    if (argc > 1) ndte = atoi(argv[1]);

    printf("=============================================================\n");
    printf("CICE EVP GPU Benchmark -- OPTIMIZED (Fused + Persistent)\n");
    printf("Domain: DMI operational, 631,387 active cells\n");
    printf("Subcycles (ndte): %d\n", ndte);
    printf("=============================================================\n\n");

    EVPParams params = compute_params(ndte);

    // Copy params to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(c_arlx1i, &params.arlx1i, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_denom1, &params.denom1, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_brlx, &params.brlx, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_e_factor, &params.e_factor, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_epp2i, &params.epp2i, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_ndte, &params.ndte, sizeof(int)));
    int na_val = NA;
    CHECK_CUDA(cudaMemcpyToSymbol(c_na, &na_val, sizeof(int)));

    // Read data
    size_t n_doubles = 36*(size_t)NA + 2*(size_t)NAVEL + 3;
    double* file_buf = (double*)malloc(n_doubles * sizeof(double));
    printf("Reading input data...\n");
    read_binary("input_double_1d_v1.bin", file_buf, n_doubles*sizeof(double));

    double* h_strength=file_buf, *h_uvel=h_strength+NA, *h_vvel=h_uvel+NAVEL;
    double* h_dxT=h_vvel+NAVEL, *h_dyT=h_dxT+NA, *h_dxhy=h_dyT+NA, *h_dyhx=h_dxhy+NA;
    double* h_cxp=h_dyhx+NA, *h_cyp=h_cxp+NA, *h_cxm=h_cyp+NA, *h_cym=h_cxm+NA;
    double* h_DminTarea=h_cym+NA, *h_uarear=h_DminTarea+NA;
    double* h_cdn_ocn=h_uarear+NA, *h_aiX=h_cdn_ocn+NA;
    double* h_uocn=h_aiX+NA, *h_vocn=h_uocn+NA;
    double* h_waterx=h_vocn+NA, *h_watery=h_waterx+NA;
    double* h_forcex=h_watery+NA, *h_forcey=h_forcex+NA;
    double* h_umassdti=h_forcey+NA, *h_fm=h_umassdti+NA;
    double* h_strintx=h_fm+NA, *h_strinty=h_strintx+NA, *h_Tbu=h_strinty+NA;
    double* h_sp1=h_Tbu+NA, *h_sp2=h_sp1+NA, *h_sp3=h_sp2+NA, *h_sp4=h_sp3+NA;
    double* h_sm1=h_sp4+NA, *h_sm2=h_sm1+NA, *h_sm3=h_sm2+NA, *h_sm4=h_sm3+NA;
    double* h_s121=h_sm4+NA, *h_s122=h_s121+NA, *h_s123=h_s122+NA, *h_s124=h_s123+NA;

    int* h_int_buf=(int*)malloc(6*NA*sizeof(int));
    read_binary("input_integer_1d.bin", h_int_buf, 6*NA*sizeof(int));
    int* h_ee=h_int_buf, *h_ne=h_ee+NA, *h_se=h_ne+NA;
    int* h_nw=h_se+NA, *h_sw=h_nw+NA, *h_sse=h_sw+NA;

    int* h_log_buf=(int*)malloc(2*NA*sizeof(int));
    read_binary("input_logical_1d.bin", h_log_buf, 2*NA*sizeof(int));
    int* h_skipU=h_log_buf, *h_skipT=h_skipU+NA;

    double* h_uvel_init=(double*)malloc(NA*sizeof(double));
    double* h_vvel_init=(double*)malloc(NA*sizeof(double));
    memcpy(h_uvel_init, h_uvel, NA*sizeof(double));
    memcpy(h_vvel_init, h_vvel, NA*sizeof(double));

    // Allocate GPU
    #define ALLOC_D(ptr,n) CHECK_CUDA(cudaMalloc(&ptr,(n)*sizeof(double)))
    #define ALLOC_I(ptr,n) CHECK_CUDA(cudaMalloc(&ptr,(n)*sizeof(int)))
    #define COPY_D(dst,src,n) CHECK_CUDA(cudaMemcpy(dst,src,(n)*sizeof(double),cudaMemcpyHostToDevice))
    #define COPY_I(dst,src,n) CHECK_CUDA(cudaMemcpy(dst,src,(n)*sizeof(int),cudaMemcpyHostToDevice))

    double *d_uvel,*d_vvel,*d_strength,*d_dxT,*d_dyT,*d_dxhy,*d_dyhx;
    double *d_cxp,*d_cyp,*d_cxm,*d_cym,*d_DminTarea;
    double *d_sp1,*d_sp2,*d_sp3,*d_sp4,*d_sm1,*d_sm2,*d_sm3,*d_sm4;
    double *d_s121,*d_s122,*d_s123,*d_s124;
    double *d_str1,*d_str2,*d_str3,*d_str4,*d_str5,*d_str6,*d_str7,*d_str8;
    double *d_uarear,*d_cdn_ocn,*d_aiX,*d_uocn,*d_vocn;
    double *d_waterx,*d_watery,*d_forcex,*d_forcey;
    double *d_umassdti,*d_fm,*d_Tbu,*d_uvel_init,*d_vvel_init;
    double *d_strintx,*d_strinty,*d_taubx,*d_tauby;
    int *d_ee,*d_ne,*d_se,*d_nw,*d_sw,*d_sse,*d_skipT,*d_skipU;

    ALLOC_D(d_uvel,NAVEL); ALLOC_D(d_vvel,NAVEL);
    ALLOC_D(d_strength,NA); ALLOC_D(d_dxT,NA); ALLOC_D(d_dyT,NA);
    ALLOC_D(d_dxhy,NA); ALLOC_D(d_dyhx,NA);
    ALLOC_D(d_cxp,NA); ALLOC_D(d_cyp,NA); ALLOC_D(d_cxm,NA); ALLOC_D(d_cym,NA);
    ALLOC_D(d_DminTarea,NA);
    ALLOC_D(d_sp1,NA); ALLOC_D(d_sp2,NA); ALLOC_D(d_sp3,NA); ALLOC_D(d_sp4,NA);
    ALLOC_D(d_sm1,NA); ALLOC_D(d_sm2,NA); ALLOC_D(d_sm3,NA); ALLOC_D(d_sm4,NA);
    ALLOC_D(d_s121,NA); ALLOC_D(d_s122,NA); ALLOC_D(d_s123,NA); ALLOC_D(d_s124,NA);
    ALLOC_D(d_str1,NAVEL); ALLOC_D(d_str2,NAVEL); ALLOC_D(d_str3,NAVEL); ALLOC_D(d_str4,NAVEL);
    ALLOC_D(d_str5,NAVEL); ALLOC_D(d_str6,NAVEL); ALLOC_D(d_str7,NAVEL); ALLOC_D(d_str8,NAVEL);
    ALLOC_D(d_uarear,NA); ALLOC_D(d_cdn_ocn,NA); ALLOC_D(d_aiX,NA);
    ALLOC_D(d_uocn,NA); ALLOC_D(d_vocn,NA);
    ALLOC_D(d_waterx,NA); ALLOC_D(d_watery,NA);
    ALLOC_D(d_forcex,NA); ALLOC_D(d_forcey,NA);
    ALLOC_D(d_umassdti,NA); ALLOC_D(d_fm,NA); ALLOC_D(d_Tbu,NA);
    ALLOC_D(d_uvel_init,NA); ALLOC_D(d_vvel_init,NA);
    ALLOC_D(d_strintx,NA); ALLOC_D(d_strinty,NA);
    ALLOC_D(d_taubx,NA); ALLOC_D(d_tauby,NA);
    ALLOC_I(d_ee,NA); ALLOC_I(d_ne,NA); ALLOC_I(d_se,NA);
    ALLOC_I(d_nw,NA); ALLOC_I(d_sw,NA); ALLOC_I(d_sse,NA);
    ALLOC_I(d_skipT,NA); ALLOC_I(d_skipU,NA);

    // Helper to reset GPU state
    auto reset_gpu = [&]() {
        COPY_D(d_uvel,h_uvel,NAVEL); COPY_D(d_vvel,h_vvel,NAVEL);
        COPY_D(d_strength,h_strength,NA); COPY_D(d_dxT,h_dxT,NA); COPY_D(d_dyT,h_dyT,NA);
        COPY_D(d_dxhy,h_dxhy,NA); COPY_D(d_dyhx,h_dyhx,NA);
        COPY_D(d_cxp,h_cxp,NA); COPY_D(d_cyp,h_cyp,NA);
        COPY_D(d_cxm,h_cxm,NA); COPY_D(d_cym,h_cym,NA);
        COPY_D(d_DminTarea,h_DminTarea,NA);
        COPY_D(d_sp1,h_sp1,NA); COPY_D(d_sp2,h_sp2,NA);
        COPY_D(d_sp3,h_sp3,NA); COPY_D(d_sp4,h_sp4,NA);
        COPY_D(d_sm1,h_sm1,NA); COPY_D(d_sm2,h_sm2,NA);
        COPY_D(d_sm3,h_sm3,NA); COPY_D(d_sm4,h_sm4,NA);
        COPY_D(d_s121,h_s121,NA); COPY_D(d_s122,h_s122,NA);
        COPY_D(d_s123,h_s123,NA); COPY_D(d_s124,h_s124,NA);
        CHECK_CUDA(cudaMemset(d_str1,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str2,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str3,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str4,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str5,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str6,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str7,0,NAVEL*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_str8,0,NAVEL*sizeof(double)));
        COPY_D(d_uarear,h_uarear,NA); COPY_D(d_cdn_ocn,h_cdn_ocn,NA);
        COPY_D(d_aiX,h_aiX,NA); COPY_D(d_uocn,h_uocn,NA); COPY_D(d_vocn,h_vocn,NA);
        COPY_D(d_waterx,h_waterx,NA); COPY_D(d_watery,h_watery,NA);
        COPY_D(d_forcex,h_forcex,NA); COPY_D(d_forcey,h_forcey,NA);
        COPY_D(d_umassdti,h_umassdti,NA); COPY_D(d_fm,h_fm,NA); COPY_D(d_Tbu,h_Tbu,NA);
        COPY_D(d_uvel_init,h_uvel_init,NA); COPY_D(d_vvel_init,h_vvel_init,NA);
        CHECK_CUDA(cudaMemset(d_strintx,0,NA*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_strinty,0,NA*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_taubx,0,NA*sizeof(double)));
        CHECK_CUDA(cudaMemset(d_tauby,0,NA*sizeof(double)));
    };

    // One-time copies
    COPY_I(d_ee,h_ee,NA); COPY_I(d_ne,h_ne,NA); COPY_I(d_se,h_se,NA);
    COPY_I(d_nw,h_nw,NA); COPY_I(d_sw,h_sw,NA); COPY_I(d_sse,h_sse,NA);
    COPY_I(d_skipT,h_skipT,NA); COPY_I(d_skipU,h_skipU,NA);

    int threads = 256;
    int blocks = (NA + threads - 1) / threads;

    // =====================================================
    // TEST 1: Optimized separate kernels (with launch_bounds)
    // =====================================================
    reset_gpu();
    printf("Running optimized separate kernels (%d subcycles)...\n", ndte);
    cudaEvent_t t1s, t1e;
    CHECK_CUDA(cudaEventCreate(&t1s)); CHECK_CUDA(cudaEventCreate(&t1e));
    CHECK_CUDA(cudaEventRecord(t1s));
    for (int k = 0; k < ndte; k++) {
        kernel_stress_opt<<<blocks,threads>>>(
            d_ee,d_ne,d_se,d_skipT,d_uvel,d_vvel,
            d_dxT,d_dyT,d_dxhy,d_dyhx,d_cxp,d_cyp,d_cxm,d_cym,
            d_DminTarea,d_strength,
            d_sp1,d_sp2,d_sp3,d_sp4,d_sm1,d_sm2,d_sm3,d_sm4,
            d_s121,d_s122,d_s123,d_s124,
            d_str1,d_str2,d_str3,d_str4,d_str5,d_str6,d_str7,d_str8,
            params.arlx1i,params.denom1,params.e_factor,params.epp2i,NA);
        kernel_stepu_opt<<<blocks,threads>>>(
            d_nw,d_sw,d_sse,d_skipU,d_Tbu,d_uvel_init,d_vvel_init,
            d_aiX,d_cdn_ocn,d_uocn,d_vocn,d_waterx,d_watery,
            d_forcex,d_forcey,d_umassdti,d_fm,d_uarear,
            d_str1,d_str2,d_str3,d_str4,d_str5,d_str6,d_str7,d_str8,
            d_uvel,d_vvel,d_strintx,d_strinty,d_taubx,d_tauby,
            params.brlx,NA);
    }
    CHECK_CUDA(cudaEventRecord(t1e));
    CHECK_CUDA(cudaEventSynchronize(t1e));
    float opt_sep_ms;
    CHECK_CUDA(cudaEventElapsedTime(&opt_sep_ms, t1s, t1e));
    printf("  Optimized separate: %.2f ms\n", opt_sep_ms);

    // =====================================================
    // TEST 2: Persistent fused kernel (cooperative launch)
    // =====================================================
    reset_gpu();

    // Query max blocks for cooperative launch
    int numBlocksPerSm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, kernel_evp_fused_persistent, threads, 0));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int persistent_blocks = numBlocksPerSm * prop.multiProcessorCount;
    printf("\nPersistent kernel: %d blocks (%d per SM x %d SMs)\n",
           persistent_blocks, numBlocksPerSm, prop.multiProcessorCount);

    // Check cooperative launch support
    if (!prop.cooperativeLaunch) {
        printf("  WARNING: Cooperative launch not supported, skipping persistent kernel\n");
    } else {
        printf("Running persistent fused kernel (%d subcycles)...\n", ndte);

        void* args[] = {
            &d_ee, &d_ne, &d_se, &d_nw, &d_sw, &d_sse,
            &d_skipT, &d_skipU,
            &d_dxT, &d_dyT, &d_dxhy, &d_dyhx,
            &d_cxp, &d_cyp, &d_cxm, &d_cym,
            &d_DminTarea, &d_strength,
            &d_Tbu, &d_uvel_init, &d_vvel_init,
            &d_aiX, &d_cdn_ocn, &d_uocn, &d_vocn,
            &d_waterx, &d_watery, &d_forcex, &d_forcey,
            &d_umassdti, &d_fm, &d_uarear,
            &d_sp1, &d_sp2, &d_sp3, &d_sp4,
            &d_sm1, &d_sm2, &d_sm3, &d_sm4,
            &d_s121, &d_s122, &d_s123, &d_s124,
            &d_str1, &d_str2, &d_str3, &d_str4,
            &d_str5, &d_str6, &d_str7, &d_str8,
            &d_uvel, &d_vvel,
            &d_strintx, &d_strinty, &d_taubx, &d_tauby
        };

        cudaEvent_t t2s, t2e;
        CHECK_CUDA(cudaEventCreate(&t2s)); CHECK_CUDA(cudaEventCreate(&t2e));
        CHECK_CUDA(cudaEventRecord(t2s));

        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)kernel_evp_fused_persistent,
            dim3(persistent_blocks), dim3(threads), args, 0, 0));

        CHECK_CUDA(cudaEventRecord(t2e));
        CHECK_CUDA(cudaEventSynchronize(t2e));
        float persistent_ms;
        CHECK_CUDA(cudaEventElapsedTime(&persistent_ms, t2s, t2e));
        printf("  Persistent fused:  %.2f ms\n", persistent_ms);

        printf("\n=============================================================\n");
        printf("RESULTS COMPARISON\n");
        printf("=============================================================\n");
        printf("  ndte:                  %d\n", ndte);
        printf("  Optimized separate:    %.2f ms\n", opt_sep_ms);
        printf("  Persistent fused:      %.2f ms\n", persistent_ms);
        printf("  Fused improvement:     %.1f%%\n",
               100.0*(opt_sep_ms - persistent_ms)/opt_sep_ms);
        printf("=============================================================\n");
    }

    free(file_buf); free(h_int_buf); free(h_log_buf);
    free(h_uvel_init); free(h_vvel_init);
    return 0;
}
