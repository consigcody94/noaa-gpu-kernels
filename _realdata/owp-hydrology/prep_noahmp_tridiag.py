"""
prep_noahmp_tridiag.py — Build real Richards-equation tridiagonal systems for the
NOAH-MP ROSR12 solver kernel in owp_batched_kernels_realdata.cu, from the real
Bondville, IL flux-tower forcing and parameter tables shipped in
NOAA-OWP/noah-owp-modular.

Data provenance (accessed 2026-06-10, shallow clone of github.com/NOAA-OWP/noah-owp-modular master):
  - data/bondville.dat        : 1 year (1998) of 30-minute observed meteorological forcing
                                at Bondville, Illinois (40.01N, -88.37E); precipitation
                                column in kg m-2 s-1 (= mm/s).
  - run/namelist.input        : official single-point test configuration: nsoil=4,
                                dz = [0.1, 0.3, 0.6, 1.0] m, initial sh2o = 0.3 m3/m3,
                                isltyp=1, dt = 1800 s, OPT_INF=1? (frozen_soil_option=1),
                                runoff/drainage option 8.
  - parameters/SOILPARM.TBL   : STAS class 1 (SAND): BB(BEXP)=2.79, MAXSMC=0.339,
                                SATPSI=0.069 m, SATDK=4.66e-5 m/s, SATDW=2.65e-5 m2/s,
                                DRYSMC=0.010.
  - parameters/GENPARM.TBL    : SLOPE_DATA category 1 = 0.1 (parameters%SLOPE).

Method — exactly the model's own matrix assembly (src/SoilWaterMovement.f90, SRT + SSTEP
scaling, with OPT_INF=2 / WDFCND2 from src/SoilWaterRetentionCoeff.f90, no soil ice,
OPT_DRN=8 so QDRAIN = SLOPE * WCND(nsoil)):

  FACTR2 = max(0.01, SMC/SMCMAX)
  WDF    = DWSAT * FACTR2**(BEXP+2)
  WCND   = DKSAT * FACTR2**(2*BEXP+3)
  k=1:        DENOM=-z1, DDZ=2/(-z2), DSMDZ=2*(S1-S2)/(-z2)
              WFLUX = WDF1*DSMDZ1 + WCND1 - PDDUM + ETRANI + QSEVA
  1<k<n:      DENOM=z(k-1)-z(k), DDZ=2/(z(k-1)-z(k+1)), DSMDZ=2*(Sk-Sk+1)/(z(k-1)-z(k+1))
              WFLUX = WDFk*DSMDZk + WCNDk - WDFk-1*DSMDZk-1 - WCNDk-1 + ETRANI
  k=n:        DENOM=z(n-1)-z(n), QDRAIN = SLOPE*WCNDn
              WFLUX = -(WDFn-1*DSMDZn-1) - WCNDn-1 + ETRANI + QDRAIN
  AI(1)=0;          BI(1)=WDF1*DDZ1/DENOM1;        CI(1)=-BI(1)
  AI(k)=-WDFk-1*DDZk-1/DENOMk; CI(k)=-WDFk*DDZk/DENOMk; BI(k)=-(AI+CI); CI(n)=0
  RHSTT(k) = WFLUX(k)/(-DENOM(k))
  SSTEP scaling: A=AI*dt, B=1+BI*dt, C=CI*dt, D=RHSTT*dt   <-- the system the kernel solves.

Forcing coupling: PDDUM (infiltration into surface, m/s) = observed precipitation rate
(mm/s * 1e-3); ETRANI and QSEVA set to 0 (no observed ET partitioning in bondville.dat;
documented simplification — affects RHS magnitude only, not provenance of forcing).
The soil moisture state is advanced through the full year with the model's own implicit
update (Thomas solve + SH2O += solution), clipped to [DRYSMC, SMCMAX], so each 30-minute
timestep yields a distinct, real-forcing-driven tridiagonal system.

Output: data/noahmp_tridiag_realdata.bin
  int32 magic=0x4E4F4148, int32 ncol, int32 nsoil
  float32 A[ncol*10], B[ncol*10], C[ncol*10], D[ncol*10]  (padded to MAX_SOIL=10)
"""
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
NOAH = HERE / "upstream" / "noah-owp-modular"
OUT = HERE / "data" / "noahmp_tridiag_realdata.bin"
MAX_SOIL = 10

# --- real soil parameters: SOILPARM.TBL STAS class 1 (SAND), namelist isltyp=1 ---
BEXP = 2.79
MAXSMC = 0.339
SATDK = 4.66e-5     # m/s
SATDW = 2.65e-5     # m2/s
DRYSMC = 0.010
SLOPE = 0.1         # GENPARM.TBL SLOPE_DATA category 1
DT = 1800.0         # s, namelist
DZ = np.array([0.1, 0.3, 0.6, 1.0])
ZSOIL = -np.cumsum(DZ)          # [-0.1, -0.4, -1.0, -2.0]
NSOIL = 4
SH2O0 = np.array([0.3, 0.3, 0.3, 0.3])  # namelist initial profile

# --- parse bondville.dat precipitation (last column, kg m-2 s-1) ---
precip = []
in_data = False
for line in (NOAH / "data" / "bondville.dat").read_text().splitlines():
    if not in_data:
        if line.strip().lower().startswith("<forcing>"):
            in_data = True
        continue
    parts = line.split()
    if len(parts) >= 13:
        precip.append(float(parts[12]))   # kg m-2 s-1 = mm/s
precip = np.array(precip)
print(f"bondville records: {len(precip)}, wet steps: {(precip>0).sum()}, "
      f"max rate {precip.max()*3600:.2f} mm/h")


def wdfcnd2(smc):
    factr2 = np.maximum(0.01, smc / MAXSMC)
    wdf = SATDW * factr2 ** (BEXP + 2.0)
    wcnd = SATDK * factr2 ** (2.0 * BEXP + 3.0)
    return wdf, wcnd


def srt_assemble(sh2o, pddum):
    """SRT (OPT_INF=2, no ice, OPT_DRN=8) + SSTEP dt-scaling. Returns A,B,C,D."""
    wdf, wcnd = wdfcnd2(sh2o)
    n = NSOIL
    denom = np.zeros(n); ddz = np.zeros(n); dsmdz = np.zeros(n); wflux = np.zeros(n)
    for k in range(n):
        if k == 0:
            denom[k] = -ZSOIL[k]
            t1 = -ZSOIL[k + 1]
            ddz[k] = 2.0 / t1
            dsmdz[k] = 2.0 * (sh2o[k] - sh2o[k + 1]) / t1
            wflux[k] = wdf[k] * dsmdz[k] + wcnd[k] - pddum  # ETRANI=QSEVA=0
        elif k < n - 1:
            denom[k] = ZSOIL[k - 1] - ZSOIL[k]
            t1 = ZSOIL[k - 1] - ZSOIL[k + 1]
            ddz[k] = 2.0 / t1
            dsmdz[k] = 2.0 * (sh2o[k] - sh2o[k + 1]) / t1
            wflux[k] = (wdf[k] * dsmdz[k] + wcnd[k]
                        - wdf[k - 1] * dsmdz[k - 1] - wcnd[k - 1])
        else:
            denom[k] = ZSOIL[k - 1] - ZSOIL[k]
            qdrain = SLOPE * wcnd[k]
            wflux[k] = -(wdf[k - 1] * dsmdz[k - 1]) - wcnd[k - 1] + qdrain
    ai = np.zeros(n); bi = np.zeros(n); ci = np.zeros(n)
    for k in range(n):
        if k == 0:
            ai[k] = 0.0
            bi[k] = wdf[k] * ddz[k] / denom[k]
            ci[k] = -bi[k]
        elif k < n - 1:
            ai[k] = -wdf[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = -wdf[k] * ddz[k] / denom[k]
            bi[k] = -(ai[k] + ci[k])
        else:
            ai[k] = -wdf[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = 0.0
            bi[k] = -(ai[k] + ci[k])
    rhstt = wflux / (-denom)
    # SSTEP scaling -> system actually handed to ROSR12
    A = ai * DT
    B = 1.0 + bi * DT
    C = ci * DT
    D = rhstt * DT
    return A, B, C, D


def thomas(A, B, C, D):
    n = len(B)
    P = np.zeros(n); Q = np.zeros(n); X = np.zeros(n)
    P[0] = -C[0] / B[0]
    Q[0] = D[0] / B[0]
    for k in range(1, n):
        den = B[k] + A[k] * P[k - 1]
        P[k] = -C[k] / den
        Q[k] = (D[k] - A[k] * Q[k - 1]) / den
    X[n - 1] = Q[n - 1]
    for k in range(n - 2, -1, -1):
        X[k] = P[k] * X[k + 1] + Q[k]
    return X


ncol = len(precip)
A_all = np.zeros((ncol, MAX_SOIL), dtype="<f4")
B_all = np.zeros((ncol, MAX_SOIL), dtype="<f4")
C_all = np.zeros((ncol, MAX_SOIL), dtype="<f4")
D_all = np.zeros((ncol, MAX_SOIL), dtype="<f4")

sh2o = SH2O0.copy()
for t in range(ncol):
    pddum = precip[t] * 1e-3   # mm/s -> m/s into soil surface
    A, B, C, D = srt_assemble(sh2o, pddum)
    A_all[t, :NSOIL] = A; B_all[t, :NSOIL] = B
    C_all[t, :NSOIL] = C; D_all[t, :NSOIL] = D
    # advance state with the model's own implicit update (SSTEP)
    sh2o = np.clip(sh2o + thomas(A, B, C, D), DRYSMC, MAXSMC)

print(f"final sh2o profile after 1 yr: {np.round(sh2o,4)}")
print(f"B diag range [{B_all[:, :NSOIL].min():.4f}, {B_all[:, :NSOIL].max():.4f}], "
      f"D range [{D_all[:, :NSOIL].min():.3e}, {D_all[:, :NSOIL].max():.3e}]")

with open(OUT, "wb") as f:
    f.write(struct.pack("<iii", 0x4E4F4148, ncol, NSOIL))
    f.write(A_all.tobytes()); f.write(B_all.tobytes())
    f.write(C_all.tobytes()); f.write(D_all.tobytes())
print(f"wrote {OUT} ({OUT.stat().st_size} bytes), ncol={ncol}")
