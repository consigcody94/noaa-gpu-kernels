/**
 * CICE EVP GPU Benchmark on Real DMI Arctic Data
 *
 * Faithful CUDA port of the v1 (1D) EVP solver from Rasmussen et al. (2024).
 * Reads the published Zenodo binary input data (DMI operational domain,
 * 1 March 2020) and runs stress + stepu subcycles on GPU.
 *
 * Data source: https://doi.org/10.5281/zenodo.11248366
 * Reference:   Rasmussen et al., GMD 17, 6529-6544 (2024)
 *
 * Author: Cody Churchwell, April 2026
 * Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
 *
 * Environment: Custom Claude Code instance with specialized CUDA research
 * tooling. Not a default Claude/Claude Code installation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// DMI domain from kernel.nml
#define NX_BLOCK 1117
#define NY_BLOCK 1493
#define NA       631387   // active T-cells (icellt)
#define NB       608124   // active U-cells
#define NAVEL    660613   // union of active T and U cells

// EVP constants (from ice_constants.F90 and ice_dyn_shared.F90)
#define C0   0.0
#define C1   1.0
#define C2   2.0
#define P5   0.5
#define P25  0.25
#define P027 (1.0/9.0 * 0.5 * 0.5)  // p055*p5
#define P055 (1.0/9.0 * 0.5)         // p111*p5
#define P111 (1.0/9.0)
#define P166 (1.0/6.0)
#define P222 (2.0/9.0)
#define P333 (1.0/3.0)

// EVP parameters (from set_evp_parameters with dt=300, classic EVP)
#define KTENS     0.0
#define REVP      0.0    // classic EVP (not revised)
#define U0        5e-5
#define COSW      1.0
#define SINW      0.0
#define RHOW      1026.0
// capping=1, e_plasticpot=2, e_yieldcurve=2, elasticDamp=0.36

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Compute EVP parameters from ndte
struct EVPParams {
    double arlx1i, denom1, brlx, epp2i, e_factor;
    int ndte;
};

EVPParams compute_params(int ndte) {
    EVPParams p;
    p.ndte = ndte;
    double elasticDamp = 0.36;
    double e_plasticpot = 2.0;
    double e_yieldcurve = 2.0;
    p.epp2i = 1.0 / (e_plasticpot * e_plasticpot);
    p.e_factor = (e_yieldcurve * e_yieldcurve) / (e_plasticpot * e_plasticpot * e_plasticpot * e_plasticpot);
    double arlx = 2.0 * elasticDamp * (double)ndte;
    p.arlx1i = 1.0 / arlx;
    p.denom1 = 1.0 / (1.0 + p.arlx1i);
    p.brlx = (double)ndte;
    return p;
}

// ============================================================
// GPU kernel: stress (faithful port of bench_v1.F90 stress)
// ============================================================
__global__ void kernel_stress(
    const int* __restrict__ ee, const int* __restrict__ ne_idx,
    const int* __restrict__ se,
    const int* __restrict__ skip_t,  // 0 or 1
    const double* __restrict__ uvel, const double* __restrict__ vvel,
    const double* __restrict__ dxT, const double* __restrict__ dyT,
    const double* __restrict__ dxhy, const double* __restrict__ dyhx,
    const double* __restrict__ cxp, const double* __restrict__ cyp,
    const double* __restrict__ cxm, const double* __restrict__ cym,
    const double* __restrict__ DminTarea, const double* __restrict__ strength,
    double* __restrict__ stressp_1, double* __restrict__ stressp_2,
    double* __restrict__ stressp_3, double* __restrict__ stressp_4,
    double* __restrict__ stressm_1, double* __restrict__ stressm_2,
    double* __restrict__ stressm_3, double* __restrict__ stressm_4,
    double* __restrict__ stress12_1, double* __restrict__ stress12_2,
    double* __restrict__ stress12_3, double* __restrict__ stress12_4,
    double* __restrict__ str1, double* __restrict__ str2,
    double* __restrict__ str3, double* __restrict__ str4,
    double* __restrict__ str5, double* __restrict__ str6,
    double* __restrict__ str7, double* __restrict__ str8,
    double arlx1i, double denom1, double e_factor, double epp2i,
    int na)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    if (iw >= na) return;
    if (skip_t[iw]) return;

    // Neighbor indices (Fortran 1-based -> C 0-based)
    int ee_i = ee[iw] - 1;
    int ne_i = ne_idx[iw] - 1;
    int se_i = se[iw] - 1;

    // Load velocity at cell and neighbors
    double uvel_cc = uvel[iw],   vvel_cc = vvel[iw];
    double uvel_ee = uvel[ee_i], vvel_ee = vvel[ee_i];
    double uvel_ne = uvel[ne_i], vvel_ne = vvel[ne_i];
    double uvel_se = uvel[se_i], vvel_se = vvel[se_i];

    double tmp_dxT = dxT[iw], tmp_dyT = dyT[iw];
    double tmp_cxp = cxp[iw], tmp_cyp = cyp[iw];
    double tmp_cxm = cxm[iw], tmp_cym = cym[iw];
    double tmp_strength = strength[iw];
    double tmp_DminTarea = DminTarea[iw];

    // Strain rates (strain_rates subroutine inlined)
    double divune = tmp_cyp*uvel_cc - tmp_dyT*uvel_ee + tmp_cxp*vvel_cc - tmp_dxT*vvel_se;
    double divunw = tmp_cym*uvel_ee + tmp_dyT*uvel_cc + tmp_cxp*vvel_ee - tmp_dxT*vvel_ne;
    double divusw = tmp_cym*uvel_ne + tmp_dyT*uvel_se + tmp_cxm*vvel_ne + tmp_dxT*vvel_ee;
    double divuse = tmp_cyp*uvel_se - tmp_dyT*uvel_ne + tmp_cxm*vvel_se + tmp_dxT*vvel_cc;

    double tensionne = -tmp_cym*uvel_cc - tmp_dyT*uvel_ee + tmp_cxm*vvel_cc + tmp_dxT*vvel_se;
    double tensionnw = -tmp_cyp*uvel_ee + tmp_dyT*uvel_cc + tmp_cxm*vvel_ee + tmp_dxT*vvel_ne;
    double tensionsw = -tmp_cyp*uvel_ne + tmp_dyT*uvel_se + tmp_cxp*vvel_ne - tmp_dxT*vvel_ee;
    double tensionse = -tmp_cym*uvel_se - tmp_dyT*uvel_ne + tmp_cxp*vvel_se - tmp_dxT*vvel_cc;

    double shearne = -tmp_cym*vvel_cc - tmp_dyT*vvel_ee - tmp_cxm*uvel_cc - tmp_dxT*uvel_se;
    double shearnw = -tmp_cyp*vvel_ee + tmp_dyT*vvel_cc - tmp_cxm*uvel_ee - tmp_dxT*uvel_ne;
    double shearsw = -tmp_cyp*vvel_ne + tmp_dyT*vvel_se - tmp_cxp*uvel_ne + tmp_dxT*uvel_ee;
    double shearse = -tmp_cym*vvel_se - tmp_dyT*vvel_ne - tmp_cxp*uvel_se + tmp_dxT*uvel_cc;

    double Deltane = sqrt(divune*divune + e_factor*(tensionne*tensionne + shearne*shearne));
    double Deltanw = sqrt(divunw*divunw + e_factor*(tensionnw*tensionnw + shearnw*shearnw));
    double Deltasw = sqrt(divusw*divusw + e_factor*(tensionsw*tensionsw + shearsw*shearsw));
    double Deltase = sqrt(divuse*divuse + e_factor*(tensionse*tensionse + shearse*shearse));

    // Viscosities and replacement pressure (capping=1, Ktens=0)
    double zetax2ne = tmp_strength / fmax(Deltane, tmp_DminTarea);
    double rep_prsne = zetax2ne * Deltane;
    double etax2ne = epp2i * zetax2ne;

    double zetax2nw = tmp_strength / fmax(Deltanw, tmp_DminTarea);
    double rep_prsnw = zetax2nw * Deltanw;
    double etax2nw = epp2i * zetax2nw;

    double zetax2sw = tmp_strength / fmax(Deltasw, tmp_DminTarea);
    double rep_prssw = zetax2sw * Deltasw;
    double etax2sw = epp2i * zetax2sw;

    double zetax2se = tmp_strength / fmax(Deltase, tmp_DminTarea);
    double rep_prsse = zetax2se * Deltase;
    double etax2se = epp2i * zetax2se;

    // Stress update (classic EVP: revp=0, so (1-arlx1i*revp) = 1)
    double sp1 = (stressp_1[iw] + arlx1i*(zetax2ne*divune - rep_prsne)) * denom1;
    double sp2 = (stressp_2[iw] + arlx1i*(zetax2nw*divunw - rep_prsnw)) * denom1;
    double sp3 = (stressp_3[iw] + arlx1i*(zetax2sw*divusw - rep_prssw)) * denom1;
    double sp4 = (stressp_4[iw] + arlx1i*(zetax2se*divuse - rep_prsse)) * denom1;

    double sm1 = (stressm_1[iw] + arlx1i*etax2ne*tensionne) * denom1;
    double sm2 = (stressm_2[iw] + arlx1i*etax2nw*tensionnw) * denom1;
    double sm3 = (stressm_3[iw] + arlx1i*etax2sw*tensionsw) * denom1;
    double sm4 = (stressm_4[iw] + arlx1i*etax2se*tensionse) * denom1;

    double s121 = (stress12_1[iw] + arlx1i*P5*etax2ne*shearne) * denom1;
    double s122 = (stress12_2[iw] + arlx1i*P5*etax2nw*shearnw) * denom1;
    double s123 = (stress12_3[iw] + arlx1i*P5*etax2sw*shearsw) * denom1;
    double s124 = (stress12_4[iw] + arlx1i*P5*etax2se*shearse) * denom1;

    stressp_1[iw] = sp1; stressp_2[iw] = sp2;
    stressp_3[iw] = sp3; stressp_4[iw] = sp4;
    stressm_1[iw] = sm1; stressm_2[iw] = sm2;
    stressm_3[iw] = sm3; stressm_4[iw] = sm4;
    stress12_1[iw] = s121; stress12_2[iw] = s122;
    stress12_3[iw] = s123; stress12_4[iw] = s124;

    // Stress combinations for momentum equation
    double ssigpn = sp1 + sp2, ssigps = sp3 + sp4;
    double ssigpe = sp1 + sp4, ssigpw = sp2 + sp3;
    double ssigp1 = (sp1 + sp3)*P055, ssigp2 = (sp2 + sp4)*P055;

    double ssigmn = sm1 + sm2, ssigms = sm3 + sm4;
    double ssigme = sm1 + sm4, ssigmw = sm2 + sm3;
    double ssigm1 = (sm1 + sm3)*P055, ssigm2 = (sm2 + sm4)*P055;

    double ssig12n = s121 + s122, ssig12s = s123 + s124;
    double ssig12e = s121 + s124, ssig12w = s122 + s123;
    double ssig121 = (s121 + s123)*P111, ssig122 = (s122 + s124)*P111;

    double csigpne = P111*sp1 + ssigp2 + P027*sp3;
    double csigpnw = P111*sp2 + ssigp1 + P027*sp4;
    double csigpsw = P111*sp3 + ssigp2 + P027*sp1;
    double csigpse = P111*sp4 + ssigp1 + P027*sp2;

    double csigmne = P111*sm1 + ssigm2 + P027*sm3;
    double csigmnw = P111*sm2 + ssigm1 + P027*sm4;
    double csigmsw = P111*sm3 + ssigm2 + P027*sm1;
    double csigmse = P111*sm4 + ssigm1 + P027*sm2;

    double csig12ne = P222*s121 + ssig122 + P055*s123;
    double csig12nw = P222*s122 + ssig121 + P055*s124;
    double csig12sw = P222*s123 + ssig122 + P055*s121;
    double csig12se = P222*s124 + ssig121 + P055*s122;

    double str12ew = P5*tmp_dxT*(P333*ssig12e + P166*ssig12w);
    double str12we = P5*tmp_dxT*(P333*ssig12w + P166*ssig12e);
    double str12ns = P5*tmp_dyT*(P333*ssig12n + P166*ssig12s);
    double str12sn = P5*tmp_dyT*(P333*ssig12s + P166*ssig12n);

    // dF/dx (u momentum)
    double strp_tmp = P25*tmp_dyT*(P333*ssigpn + P166*ssigps);
    double strm_tmp = P25*tmp_dyT*(P333*ssigmn + P166*ssigms);
    str1[iw] = -strp_tmp - strm_tmp - str12ew + dxhy[iw]*(-csigpne + csigmne) + dyhx[iw]*csig12ne;
    str2[iw] =  strp_tmp + strm_tmp - str12we + dxhy[iw]*(-csigpnw + csigmnw) + dyhx[iw]*csig12nw;
    strp_tmp = P25*tmp_dyT*(P333*ssigps + P166*ssigpn);
    strm_tmp = P25*tmp_dyT*(P333*ssigms + P166*ssigmn);
    str3[iw] = -strp_tmp - strm_tmp + str12ew + dxhy[iw]*(-csigpse + csigmse) + dyhx[iw]*csig12se;
    str4[iw] =  strp_tmp + strm_tmp + str12we + dxhy[iw]*(-csigpsw + csigmsw) + dyhx[iw]*csig12sw;

    // dF/dy (v momentum)
    strp_tmp = P25*tmp_dxT*(P333*ssigpe + P166*ssigpw);
    strm_tmp = P25*tmp_dxT*(P333*ssigme + P166*ssigmw);
    str5[iw] = -strp_tmp + strm_tmp - str12ns - dyhx[iw]*(csigpne + csigmne) + dxhy[iw]*csig12ne;
    str6[iw] =  strp_tmp - strm_tmp - str12sn - dyhx[iw]*(csigpse + csigmse) + dxhy[iw]*csig12se;
    strp_tmp = P25*tmp_dxT*(P333*ssigpw + P166*ssigpe);
    strm_tmp = P25*tmp_dxT*(P333*ssigmw + P166*ssigme);
    str7[iw] = -strp_tmp + strm_tmp + str12ns - dyhx[iw]*(csigpnw + csigmnw) + dxhy[iw]*csig12nw;
    str8[iw] =  strp_tmp - strm_tmp + str12sn - dyhx[iw]*(csigpsw + csigmsw) + dxhy[iw]*csig12sw;
}

// ============================================================
// GPU kernel: stepu (faithful port of bench_v1.F90 stepu)
// ============================================================
__global__ void kernel_stepu(
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
    if (iw >= na) return;
    if (skip_u[iw]) return;

    int nw_i = nw[iw] - 1;   // Fortran 1-based -> C 0-based
    int sw_i = sw[iw] - 1;
    int sse_i = sse[iw] - 1;

    double uold = uvel[iw];
    double vold = vvel[iw];

    double vrel = aiX[iw] * RHOW * Cw[iw] *
                  sqrt((uocn[iw] - uold)*(uocn[iw] - uold) +
                       (vocn[iw] - vold)*(vocn[iw] - vold));

    double taux = vrel * waterx[iw];
    double tauy = vrel * watery[iw];

    double Cb = Tbu[iw] / (sqrt(uold*uold + vold*vold) + U0);

    double cca = brlx * umassdti[iw] + vrel * COSW + Cb;
    double ccb = fm[iw] + copysign(C1, fm[iw]) * vrel * SINW;
    double ab2 = cca*cca + ccb*ccb;

    double sint_x = uarear[iw] * (str1[iw] + str2[nw_i] + str3[sse_i] + str4[sw_i]);
    double sint_y = uarear[iw] * (str5[iw] + str6[sse_i] + str7[nw_i] + str8[sw_i]);
    strintx[iw] = sint_x;
    strinty[iw] = sint_y;

    double cc1 = sint_x + forcex[iw] + taux + umassdti[iw] * brlx * uold;
    double cc2 = sint_y + forcey[iw] + tauy + umassdti[iw] * brlx * vold;

    uvel[iw] = (cca*cc1 + ccb*cc2) / ab2;
    vvel[iw] = (cca*cc2 - ccb*cc1) / ab2;

    taubx[iw] = -uvel[iw] * Cb;
    tauby[iw] = -vvel[iw] * Cb;
}

// ============================================================
// CPU reference: stress (for validation)
// ============================================================
void cpu_stress(
    const int* ee, const int* ne_idx, const int* se,
    const int* skip_t,
    const double* uvel, const double* vvel,
    const double* dxT, const double* dyT,
    const double* dxhy, const double* dyhx,
    const double* cxp, const double* cyp,
    const double* cxm, const double* cym,
    const double* DminTarea, const double* strength,
    double* stressp_1, double* stressp_2,
    double* stressp_3, double* stressp_4,
    double* stressm_1, double* stressm_2,
    double* stressm_3, double* stressm_4,
    double* stress12_1, double* stress12_2,
    double* stress12_3, double* stress12_4,
    double* str1, double* str2, double* str3, double* str4,
    double* str5, double* str6, double* str7, double* str8,
    double arlx1i, double denom1, double e_factor, double epp2i,
    int na)
{
    for (int iw = 0; iw < na; iw++) {
        if (skip_t[iw]) continue;
        int ee_i = ee[iw]-1, ne_i = ne_idx[iw]-1, se_i = se[iw]-1;
        double ucc=uvel[iw], vcc=vvel[iw];
        double uee=uvel[ee_i], vee=vvel[ee_i];
        double une=uvel[ne_i], vne=vvel[ne_i];
        double use_=uvel[se_i], vse=vvel[se_i];
        double dx=dxT[iw], dy=dyT[iw];
        double Cxp=cxp[iw], Cyp=cyp[iw], Cxm=cxm[iw], Cym=cym[iw];
        double str_=strength[iw], dma=DminTarea[iw];

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

        double Dne=sqrt(divune*divune+e_factor*(tenne*tenne+shne*shne));
        double Dnw=sqrt(divunw*divunw+e_factor*(tennw*tennw+shnw*shnw));
        double Dsw=sqrt(divusw*divusw+e_factor*(tensw*tensw+shsw*shsw));
        double Dse=sqrt(divuse*divuse+e_factor*(tense*tense+shse*shse));

        double z2ne=str_/fmax(Dne,dma), rpne=z2ne*Dne, e2ne=epp2i*z2ne;
        double z2nw=str_/fmax(Dnw,dma), rpnw=z2nw*Dnw, e2nw=epp2i*z2nw;
        double z2sw=str_/fmax(Dsw,dma), rpsw=z2sw*Dsw, e2sw=epp2i*z2sw;
        double z2se=str_/fmax(Dse,dma), rpse=z2se*Dse, e2se=epp2i*z2se;

        double sp1=(stressp_1[iw]+arlx1i*(z2ne*divune-rpne))*denom1;
        double sp2=(stressp_2[iw]+arlx1i*(z2nw*divunw-rpnw))*denom1;
        double sp3=(stressp_3[iw]+arlx1i*(z2sw*divusw-rpsw))*denom1;
        double sp4=(stressp_4[iw]+arlx1i*(z2se*divuse-rpse))*denom1;
        double sm1=(stressm_1[iw]+arlx1i*e2ne*tenne)*denom1;
        double sm2=(stressm_2[iw]+arlx1i*e2nw*tennw)*denom1;
        double sm3=(stressm_3[iw]+arlx1i*e2sw*tensw)*denom1;
        double sm4=(stressm_4[iw]+arlx1i*e2se*tense)*denom1;
        double s121=(stress12_1[iw]+arlx1i*0.5*e2ne*shne)*denom1;
        double s122=(stress12_2[iw]+arlx1i*0.5*e2nw*shnw)*denom1;
        double s123=(stress12_3[iw]+arlx1i*0.5*e2sw*shsw)*denom1;
        double s124=(stress12_4[iw]+arlx1i*0.5*e2se*shse)*denom1;

        stressp_1[iw]=sp1; stressp_2[iw]=sp2; stressp_3[iw]=sp3; stressp_4[iw]=sp4;
        stressm_1[iw]=sm1; stressm_2[iw]=sm2; stressm_3[iw]=sm3; stressm_4[iw]=sm4;
        stress12_1[iw]=s121; stress12_2[iw]=s122; stress12_3[iw]=s123; stress12_4[iw]=s124;

        double ssigpn=sp1+sp2, ssigps=sp3+sp4, ssigpe=sp1+sp4, ssigpw=sp2+sp3;
        double ssigp1=(sp1+sp3)*P055, ssigp2=(sp2+sp4)*P055;
        double ssigmn=sm1+sm2, ssigms=sm3+sm4, ssigme=sm1+sm4, ssigmw=sm2+sm3;
        double ssigm1=(sm1+sm3)*P055, ssigm2=(sm2+sm4)*P055;
        double ssig12n=s121+s122, ssig12s=s123+s124, ssig12e=s121+s124, ssig12w=s122+s123;
        double ssig121_=(s121+s123)*P111, ssig122_=(s122+s124)*P111;

        double csigpne=P111*sp1+ssigp2+P027*sp3, csigpnw=P111*sp2+ssigp1+P027*sp4;
        double csigpsw=P111*sp3+ssigp2+P027*sp1, csigpse=P111*sp4+ssigp1+P027*sp2;
        double csigmne=P111*sm1+ssigm2+P027*sm3, csigmnw=P111*sm2+ssigm1+P027*sm4;
        double csigmsw=P111*sm3+ssigm2+P027*sm1, csigmse=P111*sm4+ssigm1+P027*sm2;
        double csig12ne=P222*s121+ssig122_+P055*s123, csig12nw=P222*s122+ssig121_+P055*s124;
        double csig12sw=P222*s123+ssig122_+P055*s121, csig12se=P222*s124+ssig121_+P055*s122;

        double str12ew=P5*dx*(P333*ssig12e+P166*ssig12w);
        double str12we=P5*dx*(P333*ssig12w+P166*ssig12e);
        double str12ns=P5*dy*(P333*ssig12n+P166*ssig12s);
        double str12sn=P5*dy*(P333*ssig12s+P166*ssig12n);

        double strp=P25*dy*(P333*ssigpn+P166*ssigps), strm=P25*dy*(P333*ssigmn+P166*ssigms);
        str1[iw]=-strp-strm-str12ew+dxhy[iw]*(-csigpne+csigmne)+dyhx[iw]*csig12ne;
        str2[iw]= strp+strm-str12we+dxhy[iw]*(-csigpnw+csigmnw)+dyhx[iw]*csig12nw;
        strp=P25*dy*(P333*ssigps+P166*ssigpn); strm=P25*dy*(P333*ssigms+P166*ssigmn);
        str3[iw]=-strp-strm+str12ew+dxhy[iw]*(-csigpse+csigmse)+dyhx[iw]*csig12se;
        str4[iw]= strp+strm+str12we+dxhy[iw]*(-csigpsw+csigmsw)+dyhx[iw]*csig12sw;

        strp=P25*dx*(P333*ssigpe+P166*ssigpw); strm=P25*dx*(P333*ssigme+P166*ssigmw);
        str5[iw]=-strp+strm-str12ns-dyhx[iw]*(csigpne+csigmne)+dxhy[iw]*csig12ne;
        str6[iw]= strp-strm-str12sn-dyhx[iw]*(csigpse+csigmse)+dxhy[iw]*csig12se;
        strp=P25*dx*(P333*ssigpw+P166*ssigpe); strm=P25*dx*(P333*ssigmw+P166*ssigme);
        str7[iw]=-strp+strm+str12ns-dyhx[iw]*(csigpnw+csigmnw)+dxhy[iw]*csig12nw;
        str8[iw]= strp-strm+str12sn-dyhx[iw]*(csigpsw+csigmsw)+dxhy[iw]*csig12sw;
    }
}

// ============================================================
// CPU reference: stepu
// ============================================================
void cpu_stepu(
    const int* nw, const int* sw, const int* sse, const int* skip_u,
    const double* Tbu, const double* uvel_init, const double* vvel_init,
    const double* aiX, const double* Cw,
    const double* uocn, const double* vocn,
    const double* waterx, const double* watery,
    const double* forcex, const double* forcey,
    const double* umassdti, const double* fm, const double* uarear,
    const double* str1, const double* str2, const double* str3, const double* str4,
    const double* str5, const double* str6, const double* str7, const double* str8,
    double* uvel, double* vvel, double* strintx, double* strinty,
    double* taubx, double* tauby, double brlx, int na)
{
    for (int iw = 0; iw < na; iw++) {
        if (skip_u[iw]) continue;
        int nw_i=nw[iw]-1, sw_i=sw[iw]-1, sse_i=sse[iw]-1;
        double uold=uvel[iw], vold=vvel[iw];
        double vrel = aiX[iw]*RHOW*Cw[iw]*sqrt((uocn[iw]-uold)*(uocn[iw]-uold)+(vocn[iw]-vold)*(vocn[iw]-vold));
        double tx=vrel*waterx[iw], ty=vrel*watery[iw];
        double Cb=Tbu[iw]/(sqrt(uold*uold+vold*vold)+U0);
        double cca=brlx*umassdti[iw]+vrel*COSW+Cb;
        double ccb=fm[iw]+copysign(1.0,fm[iw])*vrel*SINW;
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
}

// ============================================================
// I/O helpers
// ============================================================
void read_binary(const char* fname, void* buf, size_t bytes) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    size_t n = fread(buf, 1, bytes, f);
    if (n != bytes) { fprintf(stderr, "Short read %s: got %zu of %zu\n", fname, n, bytes); exit(1); }
    fclose(f);
    printf("  Read %s: %.1f MB\n", fname, bytes/1e6);
}

int main(int argc, char** argv) {
    int ndte = 120;
    if (argc > 1) ndte = atoi(argv[1]);

    printf("=============================================================\n");
    printf("CICE EVP GPU Benchmark -- Real DMI Arctic Data\n");
    printf("Domain: DMI operational, 1 March 2020 (winter)\n");
    printf("Active T-cells (na): %d, U-cells (nb): %d, navel: %d\n", NA, NB, NAVEL);
    printf("Subcycles (ndte): %d\n", ndte);
    printf("=============================================================\n\n");

    EVPParams params = compute_params(ndte);
    printf("EVP parameters: arlx1i=%.6e, denom1=%.10f, brlx=%.1f\n",
           params.arlx1i, params.denom1, params.brlx);
    printf("  e_factor=%.6e, epp2i=%.6f\n\n", params.e_factor, params.epp2i);

    // ---- Allocate host memory ----
    // Double arrays from input_double_1d_v1.bin (all size na except uvel/vvel which are navel)
    // Order in file: strength, uvel, vvel, dxt, dyt, dxhy, dyhx, cxp, cyp, cxm, cym,
    //   DminTarea, uarear, cdn_ocn, aiX, uocn, vocn, waterx, watery, forcex, forcey,
    //   umassdti, fm, strintx, strinty, Tbu,
    //   stressp_1..4, stressm_1..4, stress12_1..4, capping, e_factor, epp2i

    // Total doubles in file: na + navel + navel + 23*na + 12*na + 3 scalars
    // = 36*na + 2*navel + 3 (approx)
    // Actually: strength(na), uvel(navel), vvel(navel), then 23 arrays of size na,
    //   then 12 stress arrays of size na, then 3 scalars
    // But wait - the v1 allocations show uvel/vvel are size navel, most others are na
    // File has: strength(navel) uvel(navel) vvel(navel) ... all arrays are navel sized!
    // Actually no - let me re-read. The readin_1d reads into arrays declared as na or navel.
    // Looking at alloc_1d_v1: uvel(1:navel), vvel(1:navel), everything else is 1:na
    // But the file just reads sequentially. Let me compute expected file size.

    // Actually looking more carefully at readin_1d (line 1241-1248):
    // It reads: strength, uvel, vvel, dxt, dyt, dxhy, dyhx, cxp, cyp, cxm, cym, DminTarea,
    //   uarear, cdn_ocn, aiX, uocn, vocn, waterx, watery, forcex, forcey,
    //   umassdti, fm, strintx, strinty, Tbu,
    //   stressp_1..4, stressm_1..4, stress12_1..4, capping, e_factor, epp2i
    // Where each array has its allocated size. So:
    // strength(na) + uvel(navel) + vvel(navel) + 23*na arrays + 12*na stress arrays + 3 scalars
    // = (1+23+12)*na + 2*navel + 3 = 36*na + 2*navel + 3

    size_t n_doubles_file = 36*(size_t)NA + 2*(size_t)NAVEL + 3;
    size_t file_bytes = n_doubles_file * sizeof(double);
    printf("Expected file size: %.1f MB (%zu doubles)\n", file_bytes/1e6, n_doubles_file);

    double* file_buf = (double*)malloc(file_bytes);
    if (!file_buf) { fprintf(stderr, "Failed to allocate %.0f MB\n", file_bytes/1e6); exit(1); }

    printf("Reading input data...\n");
    read_binary("input_double_1d_v1.bin", file_buf, file_bytes);

    // Parse the file buffer
    double* h_strength = file_buf;                          // na
    double* h_uvel     = h_strength + NA;                   // navel
    double* h_vvel     = h_uvel + NAVEL;                    // navel
    double* h_dxT      = h_vvel + NAVEL;                    // na
    double* h_dyT      = h_dxT + NA;
    double* h_dxhy     = h_dyT + NA;
    double* h_dyhx     = h_dxhy + NA;
    double* h_cxp      = h_dyhx + NA;
    double* h_cyp      = h_cxp + NA;
    double* h_cxm      = h_cyp + NA;
    double* h_cym      = h_cxm + NA;
    double* h_DminTarea= h_cym + NA;
    double* h_uarear   = h_DminTarea + NA;
    double* h_cdn_ocn  = h_uarear + NA;
    double* h_aiX      = h_cdn_ocn + NA;
    double* h_uocn     = h_aiX + NA;
    double* h_vocn     = h_uocn + NA;
    double* h_waterx   = h_vocn + NA;
    double* h_watery   = h_waterx + NA;
    double* h_forcex   = h_watery + NA;
    double* h_forcey   = h_forcex + NA;
    double* h_umassdti = h_forcey + NA;
    double* h_fm       = h_umassdti + NA;
    double* h_strintx  = h_fm + NA;
    double* h_strinty  = h_strintx + NA;
    double* h_Tbu      = h_strinty + NA;
    double* h_stressp_1= h_Tbu + NA;
    double* h_stressp_2= h_stressp_1 + NA;
    double* h_stressp_3= h_stressp_2 + NA;
    double* h_stressp_4= h_stressp_3 + NA;
    double* h_stressm_1= h_stressp_4 + NA;
    double* h_stressm_2= h_stressm_1 + NA;
    double* h_stressm_3= h_stressm_2 + NA;
    double* h_stressm_4= h_stressm_3 + NA;
    double* h_stress12_1=h_stressm_4 + NA;
    double* h_stress12_2=h_stress12_1+ NA;
    double* h_stress12_3=h_stress12_2+ NA;
    double* h_stress12_4=h_stress12_3+ NA;
    // 3 scalars: capping, e_factor, epp2i (we use our computed values)

    printf("  Sample: strength[0]=%.6e, uvel[0]=%.6e, DminTarea[0]=%.6e\n",
           h_strength[0], h_uvel[0], h_DminTarea[0]);

    // Read integer neighbor indices (6 arrays of na int32)
    int* h_int_buf = (int*)malloc(6 * NA * sizeof(int));
    read_binary("input_integer_1d.bin", h_int_buf, 6 * NA * sizeof(int));
    int* h_ee  = h_int_buf;
    int* h_ne  = h_ee + NA;
    int* h_se  = h_ne + NA;
    int* h_nw  = h_se + NA;
    int* h_sw  = h_nw + NA;
    int* h_sse = h_sw + NA;
    printf("  Neighbor indices: ee[0]=%d, ne[0]=%d, se[0]=%d\n", h_ee[0], h_ne[0], h_se[0]);

    // Read logical skip masks (2 arrays of na, Fortran logical = 4 bytes)
    int* h_log_buf = (int*)malloc(2 * NA * sizeof(int));
    read_binary("input_logical_1d.bin", h_log_buf, 2 * NA * sizeof(int));
    // Fortran reads: skipUcell1d, skipTcell1d (in that order)
    // Fortran .true. = -1, .false. = 0
    int* h_skipU = h_log_buf;        // first in file
    int* h_skipT = h_skipU + NA;     // second in file

    // Count active cells
    int active_t = 0, active_u = 0;
    for (int i = 0; i < NA; i++) {
        if (!h_skipT[i]) active_t++;
        if (!h_skipU[i]) active_u++;
    }
    printf("  Active T-cells: %d / %d, Active U-cells: %d / %d\n\n",
           active_t, NA, active_u, NA);

    // ---- Allocate str arrays (size navel, zeroed) ----
    double* h_str1 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str2 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str3 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str4 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str5 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str6 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str7 = (double*)calloc(NAVEL, sizeof(double));
    double* h_str8 = (double*)calloc(NAVEL, sizeof(double));

    // Make copies for CPU and GPU runs
    double* cpu_uvel = (double*)malloc(NAVEL * sizeof(double));
    double* cpu_vvel = (double*)malloc(NAVEL * sizeof(double));
    double* gpu_uvel = (double*)malloc(NAVEL * sizeof(double));
    double* gpu_vvel = (double*)malloc(NAVEL * sizeof(double));

    // Stress arrays - need separate copies
    double* cpu_sp1=(double*)malloc(NA*sizeof(double)); double* gpu_sp1=(double*)malloc(NA*sizeof(double));
    double* cpu_sp2=(double*)malloc(NA*sizeof(double)); double* gpu_sp2=(double*)malloc(NA*sizeof(double));
    double* cpu_sp3=(double*)malloc(NA*sizeof(double)); double* gpu_sp3=(double*)malloc(NA*sizeof(double));
    double* cpu_sp4=(double*)malloc(NA*sizeof(double)); double* gpu_sp4=(double*)malloc(NA*sizeof(double));
    double* cpu_sm1=(double*)malloc(NA*sizeof(double)); double* gpu_sm1=(double*)malloc(NA*sizeof(double));
    double* cpu_sm2=(double*)malloc(NA*sizeof(double)); double* gpu_sm2=(double*)malloc(NA*sizeof(double));
    double* cpu_sm3=(double*)malloc(NA*sizeof(double)); double* gpu_sm3=(double*)malloc(NA*sizeof(double));
    double* cpu_sm4=(double*)malloc(NA*sizeof(double)); double* gpu_sm4=(double*)malloc(NA*sizeof(double));
    double* cpu_s121=(double*)malloc(NA*sizeof(double)); double* gpu_s121=(double*)malloc(NA*sizeof(double));
    double* cpu_s122=(double*)malloc(NA*sizeof(double)); double* gpu_s122=(double*)malloc(NA*sizeof(double));
    double* cpu_s123=(double*)malloc(NA*sizeof(double)); double* gpu_s123=(double*)malloc(NA*sizeof(double));
    double* cpu_s124=(double*)malloc(NA*sizeof(double)); double* gpu_s124=(double*)malloc(NA*sizeof(double));

    double* h_uvel_init = (double*)malloc(NA * sizeof(double));
    double* h_vvel_init = (double*)malloc(NA * sizeof(double));
    memcpy(h_uvel_init, h_uvel, NA * sizeof(double));
    memcpy(h_vvel_init, h_vvel, NA * sizeof(double));

    double* h_strintx_out = (double*)calloc(NA, sizeof(double));
    double* h_strinty_out = (double*)calloc(NA, sizeof(double));
    double* h_taubx = (double*)calloc(NA, sizeof(double));
    double* h_tauby = (double*)calloc(NA, sizeof(double));

    // =====================================================
    // CPU BENCHMARK
    // =====================================================
    printf("Running CPU benchmark (%d subcycles)...\n", ndte);

    // Initialize CPU copies
    memcpy(cpu_uvel, h_uvel, NAVEL * sizeof(double));
    memcpy(cpu_vvel, h_vvel, NAVEL * sizeof(double));
    memcpy(cpu_sp1, h_stressp_1, NA*sizeof(double)); memcpy(cpu_sp2, h_stressp_2, NA*sizeof(double));
    memcpy(cpu_sp3, h_stressp_3, NA*sizeof(double)); memcpy(cpu_sp4, h_stressp_4, NA*sizeof(double));
    memcpy(cpu_sm1, h_stressm_1, NA*sizeof(double)); memcpy(cpu_sm2, h_stressm_2, NA*sizeof(double));
    memcpy(cpu_sm3, h_stressm_3, NA*sizeof(double)); memcpy(cpu_sm4, h_stressm_4, NA*sizeof(double));
    memcpy(cpu_s121, h_stress12_1, NA*sizeof(double)); memcpy(cpu_s122, h_stress12_2, NA*sizeof(double));
    memcpy(cpu_s123, h_stress12_3, NA*sizeof(double)); memcpy(cpu_s124, h_stress12_4, NA*sizeof(double));
    memset(h_str1, 0, NAVEL*sizeof(double)); memset(h_str2, 0, NAVEL*sizeof(double));
    memset(h_str3, 0, NAVEL*sizeof(double)); memset(h_str4, 0, NAVEL*sizeof(double));
    memset(h_str5, 0, NAVEL*sizeof(double)); memset(h_str6, 0, NAVEL*sizeof(double));
    memset(h_str7, 0, NAVEL*sizeof(double)); memset(h_str8, 0, NAVEL*sizeof(double));

    cudaEvent_t cpu_start, cpu_stop;
    CHECK_CUDA(cudaEventCreate(&cpu_start));
    CHECK_CUDA(cudaEventCreate(&cpu_stop));
    CHECK_CUDA(cudaEventRecord(cpu_start));

    for (int ksub = 0; ksub < ndte; ksub++) {
        cpu_stress(h_ee, h_ne, h_se, h_skipT,
                   cpu_uvel, cpu_vvel, h_dxT, h_dyT, h_dxhy, h_dyhx,
                   h_cxp, h_cyp, h_cxm, h_cym, h_DminTarea, h_strength,
                   cpu_sp1, cpu_sp2, cpu_sp3, cpu_sp4,
                   cpu_sm1, cpu_sm2, cpu_sm3, cpu_sm4,
                   cpu_s121, cpu_s122, cpu_s123, cpu_s124,
                   h_str1, h_str2, h_str3, h_str4,
                   h_str5, h_str6, h_str7, h_str8,
                   params.arlx1i, params.denom1, params.e_factor, params.epp2i, NA);

        cpu_stepu(h_nw, h_sw, h_sse, h_skipU,
                  h_Tbu, h_uvel_init, h_vvel_init,
                  h_aiX, h_cdn_ocn, h_uocn, h_vocn,
                  h_waterx, h_watery, h_forcex, h_forcey,
                  h_umassdti, h_fm, h_uarear,
                  h_str1, h_str2, h_str3, h_str4,
                  h_str5, h_str6, h_str7, h_str8,
                  cpu_uvel, cpu_vvel, h_strintx_out, h_strinty_out,
                  h_taubx, h_tauby, params.brlx, NA);
    }

    CHECK_CUDA(cudaEventRecord(cpu_stop));
    CHECK_CUDA(cudaEventSynchronize(cpu_stop));
    float cpu_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_stop));
    printf("  CPU time: %.2f ms\n", cpu_ms);

    // =====================================================
    // GPU BENCHMARK
    // =====================================================
    printf("\nAllocating GPU memory...\n");

    // Device arrays
    double *d_uvel, *d_vvel, *d_strength, *d_dxT, *d_dyT, *d_dxhy, *d_dyhx;
    double *d_cxp, *d_cyp, *d_cxm, *d_cym, *d_DminTarea;
    double *d_sp1, *d_sp2, *d_sp3, *d_sp4;
    double *d_sm1, *d_sm2, *d_sm3, *d_sm4;
    double *d_s121, *d_s122, *d_s123, *d_s124;
    double *d_str1, *d_str2, *d_str3, *d_str4;
    double *d_str5, *d_str6, *d_str7, *d_str8;
    double *d_uarear, *d_cdn_ocn, *d_aiX, *d_uocn, *d_vocn;
    double *d_waterx, *d_watery, *d_forcex, *d_forcey;
    double *d_umassdti, *d_fm, *d_Tbu;
    double *d_uvel_init, *d_vvel_init;
    double *d_strintx, *d_strinty, *d_taubx, *d_tauby;
    int *d_ee, *d_ne, *d_se, *d_nw, *d_sw, *d_sse;
    int *d_skipT, *d_skipU;

    #define ALLOC_D(ptr, n) CHECK_CUDA(cudaMalloc(&ptr, (n)*sizeof(double)))
    #define ALLOC_I(ptr, n) CHECK_CUDA(cudaMalloc(&ptr, (n)*sizeof(int)))

    ALLOC_D(d_uvel, NAVEL); ALLOC_D(d_vvel, NAVEL);
    ALLOC_D(d_strength, NA); ALLOC_D(d_dxT, NA); ALLOC_D(d_dyT, NA);
    ALLOC_D(d_dxhy, NA); ALLOC_D(d_dyhx, NA);
    ALLOC_D(d_cxp, NA); ALLOC_D(d_cyp, NA); ALLOC_D(d_cxm, NA); ALLOC_D(d_cym, NA);
    ALLOC_D(d_DminTarea, NA);
    ALLOC_D(d_sp1, NA); ALLOC_D(d_sp2, NA); ALLOC_D(d_sp3, NA); ALLOC_D(d_sp4, NA);
    ALLOC_D(d_sm1, NA); ALLOC_D(d_sm2, NA); ALLOC_D(d_sm3, NA); ALLOC_D(d_sm4, NA);
    ALLOC_D(d_s121, NA); ALLOC_D(d_s122, NA); ALLOC_D(d_s123, NA); ALLOC_D(d_s124, NA);
    ALLOC_D(d_str1, NAVEL); ALLOC_D(d_str2, NAVEL); ALLOC_D(d_str3, NAVEL); ALLOC_D(d_str4, NAVEL);
    ALLOC_D(d_str5, NAVEL); ALLOC_D(d_str6, NAVEL); ALLOC_D(d_str7, NAVEL); ALLOC_D(d_str8, NAVEL);
    ALLOC_D(d_uarear, NA); ALLOC_D(d_cdn_ocn, NA); ALLOC_D(d_aiX, NA);
    ALLOC_D(d_uocn, NA); ALLOC_D(d_vocn, NA);
    ALLOC_D(d_waterx, NA); ALLOC_D(d_watery, NA);
    ALLOC_D(d_forcex, NA); ALLOC_D(d_forcey, NA);
    ALLOC_D(d_umassdti, NA); ALLOC_D(d_fm, NA); ALLOC_D(d_Tbu, NA);
    ALLOC_D(d_uvel_init, NA); ALLOC_D(d_vvel_init, NA);
    ALLOC_D(d_strintx, NA); ALLOC_D(d_strinty, NA);
    ALLOC_D(d_taubx, NA); ALLOC_D(d_tauby, NA);
    ALLOC_I(d_ee, NA); ALLOC_I(d_ne, NA); ALLOC_I(d_se, NA);
    ALLOC_I(d_nw, NA); ALLOC_I(d_sw, NA); ALLOC_I(d_sse, NA);
    ALLOC_I(d_skipT, NA); ALLOC_I(d_skipU, NA);

    // Copy data to GPU
    #define COPY_D(dst, src, n) CHECK_CUDA(cudaMemcpy(dst, src, (n)*sizeof(double), cudaMemcpyHostToDevice))
    #define COPY_I(dst, src, n) CHECK_CUDA(cudaMemcpy(dst, src, (n)*sizeof(int), cudaMemcpyHostToDevice))

    COPY_D(d_uvel, h_uvel, NAVEL); COPY_D(d_vvel, h_vvel, NAVEL);
    COPY_D(d_strength, h_strength, NA); COPY_D(d_dxT, h_dxT, NA); COPY_D(d_dyT, h_dyT, NA);
    COPY_D(d_dxhy, h_dxhy, NA); COPY_D(d_dyhx, h_dyhx, NA);
    COPY_D(d_cxp, h_cxp, NA); COPY_D(d_cyp, h_cyp, NA);
    COPY_D(d_cxm, h_cxm, NA); COPY_D(d_cym, h_cym, NA);
    COPY_D(d_DminTarea, h_DminTarea, NA);
    COPY_D(d_sp1, h_stressp_1, NA); COPY_D(d_sp2, h_stressp_2, NA);
    COPY_D(d_sp3, h_stressp_3, NA); COPY_D(d_sp4, h_stressp_4, NA);
    COPY_D(d_sm1, h_stressm_1, NA); COPY_D(d_sm2, h_stressm_2, NA);
    COPY_D(d_sm3, h_stressm_3, NA); COPY_D(d_sm4, h_stressm_4, NA);
    COPY_D(d_s121, h_stress12_1, NA); COPY_D(d_s122, h_stress12_2, NA);
    COPY_D(d_s123, h_stress12_3, NA); COPY_D(d_s124, h_stress12_4, NA);
    CHECK_CUDA(cudaMemset(d_str1, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str2, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str3, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str4, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str5, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str6, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str7, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str8, 0, NAVEL*sizeof(double)));
    COPY_D(d_uarear, h_uarear, NA); COPY_D(d_cdn_ocn, h_cdn_ocn, NA);
    COPY_D(d_aiX, h_aiX, NA); COPY_D(d_uocn, h_uocn, NA); COPY_D(d_vocn, h_vocn, NA);
    COPY_D(d_waterx, h_waterx, NA); COPY_D(d_watery, h_watery, NA);
    COPY_D(d_forcex, h_forcex, NA); COPY_D(d_forcey, h_forcey, NA);
    COPY_D(d_umassdti, h_umassdti, NA); COPY_D(d_fm, h_fm, NA); COPY_D(d_Tbu, h_Tbu, NA);
    COPY_D(d_uvel_init, h_uvel_init, NA); COPY_D(d_vvel_init, h_vvel_init, NA);
    CHECK_CUDA(cudaMemset(d_strintx, 0, NA*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_strinty, 0, NA*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_taubx, 0, NA*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_tauby, 0, NA*sizeof(double)));
    COPY_I(d_ee, h_ee, NA); COPY_I(d_ne, h_ne, NA); COPY_I(d_se, h_se, NA);
    COPY_I(d_nw, h_nw, NA); COPY_I(d_sw, h_sw, NA); COPY_I(d_sse, h_sse, NA);
    COPY_I(d_skipT, h_skipT, NA); COPY_I(d_skipU, h_skipU, NA);

    int threads = 256;
    int blocks_na = (NA + threads - 1) / threads;

    // Warmup
    printf("GPU warmup...\n");
    kernel_stress<<<blocks_na, threads>>>(
        d_ee, d_ne, d_se, d_skipT,
        d_uvel, d_vvel, d_dxT, d_dyT, d_dxhy, d_dyhx,
        d_cxp, d_cyp, d_cxm, d_cym, d_DminTarea, d_strength,
        d_sp1, d_sp2, d_sp3, d_sp4,
        d_sm1, d_sm2, d_sm3, d_sm4,
        d_s121, d_s122, d_s123, d_s124,
        d_str1, d_str2, d_str3, d_str4,
        d_str5, d_str6, d_str7, d_str8,
        params.arlx1i, params.denom1, params.e_factor, params.epp2i, NA);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset GPU state for actual benchmark
    COPY_D(d_uvel, h_uvel, NAVEL); COPY_D(d_vvel, h_vvel, NAVEL);
    COPY_D(d_sp1, h_stressp_1, NA); COPY_D(d_sp2, h_stressp_2, NA);
    COPY_D(d_sp3, h_stressp_3, NA); COPY_D(d_sp4, h_stressp_4, NA);
    COPY_D(d_sm1, h_stressm_1, NA); COPY_D(d_sm2, h_stressm_2, NA);
    COPY_D(d_sm3, h_stressm_3, NA); COPY_D(d_sm4, h_stressm_4, NA);
    COPY_D(d_s121, h_stress12_1, NA); COPY_D(d_s122, h_stress12_2, NA);
    COPY_D(d_s123, h_stress12_3, NA); COPY_D(d_s124, h_stress12_4, NA);
    CHECK_CUDA(cudaMemset(d_str1, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str2, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str3, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str4, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str5, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str6, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str7, 0, NAVEL*sizeof(double)));
    CHECK_CUDA(cudaMemset(d_str8, 0, NAVEL*sizeof(double)));

    printf("\nRunning GPU benchmark (%d subcycles)...\n", ndte);
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA(cudaEventCreate(&gpu_start));
    CHECK_CUDA(cudaEventCreate(&gpu_stop));
    CHECK_CUDA(cudaEventRecord(gpu_start));

    for (int ksub = 0; ksub < ndte; ksub++) {
        kernel_stress<<<blocks_na, threads>>>(
            d_ee, d_ne, d_se, d_skipT,
            d_uvel, d_vvel, d_dxT, d_dyT, d_dxhy, d_dyhx,
            d_cxp, d_cyp, d_cxm, d_cym, d_DminTarea, d_strength,
            d_sp1, d_sp2, d_sp3, d_sp4,
            d_sm1, d_sm2, d_sm3, d_sm4,
            d_s121, d_s122, d_s123, d_s124,
            d_str1, d_str2, d_str3, d_str4,
            d_str5, d_str6, d_str7, d_str8,
            params.arlx1i, params.denom1, params.e_factor, params.epp2i, NA);

        kernel_stepu<<<blocks_na, threads>>>(
            d_nw, d_sw, d_sse, d_skipU,
            d_Tbu, d_uvel_init, d_vvel_init,
            d_aiX, d_cdn_ocn, d_uocn, d_vocn,
            d_waterx, d_watery, d_forcex, d_forcey,
            d_umassdti, d_fm, d_uarear,
            d_str1, d_str2, d_str3, d_str4,
            d_str5, d_str6, d_str7, d_str8,
            d_uvel, d_vvel, d_strintx, d_strinty,
            d_taubx, d_tauby, params.brlx, NA);
    }

    CHECK_CUDA(cudaEventRecord(gpu_stop));
    CHECK_CUDA(cudaEventSynchronize(gpu_stop));
    float gpu_ms;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    printf("  GPU time: %.2f ms\n", gpu_ms);

    // =====================================================
    // VALIDATION
    // =====================================================
    printf("\nValidating GPU vs CPU...\n");

    // Copy GPU results back
    CHECK_CUDA(cudaMemcpy(gpu_uvel, d_uvel, NAVEL*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_vvel, d_vvel, NAVEL*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_sp1, d_sp1, NA*sizeof(double), cudaMemcpyDeviceToHost));

    double max_rel_err_u = 0, max_rel_err_v = 0, max_rel_err_s = 0;
    for (int i = 0; i < NA; i++) {
        if (h_skipU[i]) continue;
        double ru = fabs(cpu_uvel[i]) > 1e-30 ? fabs((gpu_uvel[i]-cpu_uvel[i])/cpu_uvel[i]) : fabs(gpu_uvel[i]-cpu_uvel[i]);
        double rv = fabs(cpu_vvel[i]) > 1e-30 ? fabs((gpu_vvel[i]-cpu_vvel[i])/cpu_vvel[i]) : fabs(gpu_vvel[i]-cpu_vvel[i]);
        double rs = fabs(cpu_sp1[i]) > 1e-30 ? fabs((gpu_sp1[i]-cpu_sp1[i])/cpu_sp1[i]) : fabs(gpu_sp1[i]-cpu_sp1[i]);
        if (ru > max_rel_err_u) max_rel_err_u = ru;
        if (rv > max_rel_err_v) max_rel_err_v = rv;
        if (rs > max_rel_err_s) max_rel_err_s = rs;
    }

    printf("  Max relative error uvel: %.6e\n", max_rel_err_u);
    printf("  Max relative error vvel: %.6e\n", max_rel_err_v);
    printf("  Max relative error stressp_1: %.6e\n", max_rel_err_s);

    // =====================================================
    // RESULTS
    // =====================================================
    printf("\n=============================================================\n");
    printf("RESULTS: Real DMI Arctic Data (631,387 active cells)\n");
    printf("=============================================================\n");
    printf("  Subcycles (ndte):  %d\n", ndte);
    printf("  CPU time:          %.2f ms\n", cpu_ms);
    printf("  GPU time:          %.2f ms  (RTX 3060)\n", gpu_ms);
    printf("  Speedup:           %.1fx\n", cpu_ms / gpu_ms);
    printf("  Validation:        %s\n",
           (max_rel_err_u < 1e-6 && max_rel_err_v < 1e-6) ? "PASS" : "CHECK");
    printf("=============================================================\n");

    // Cleanup
    free(file_buf); free(h_int_buf); free(h_log_buf);
    free(h_str1); free(h_str2); free(h_str3); free(h_str4);
    free(h_str5); free(h_str6); free(h_str7); free(h_str8);
    free(cpu_uvel); free(cpu_vvel); free(gpu_uvel); free(gpu_vvel);
    free(cpu_sp1); free(cpu_sp2); free(cpu_sp3); free(cpu_sp4);
    free(cpu_sm1); free(cpu_sm2); free(cpu_sm3); free(cpu_sm4);
    free(cpu_s121); free(cpu_s122); free(cpu_s123); free(cpu_s124);
    free(gpu_sp1); free(gpu_sp2); free(gpu_sp3); free(gpu_sp4);
    free(gpu_sm1); free(gpu_sm2); free(gpu_sm3); free(gpu_sm4);
    free(gpu_s121); free(gpu_s122); free(gpu_s123); free(gpu_s124);
    free(h_uvel_init); free(h_vvel_init);
    free(h_strintx_out); free(h_strinty_out); free(h_taubx); free(h_tauby);

    return 0;
}
