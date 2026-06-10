#!/usr/bin/env python3
"""
prep_igra2_tridi.py — build REAL-DATA tridiagonal PBL diffusion systems for the
CCPP tridi1 kernel (noaa_multi_kernel.cu) from IGRA2 radiosonde soundings.

DATA SOURCE (real, public, authoritative):
  NOAA NCEI Integrated Global Radiosonde Archive v2 (IGRA2)
  https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/
  Files <STATION>-data-beg2025.txt.zip  (soundings 2025-01-01 .. present)
  Accessed 2026-06-10.
  Stations:
    USM00072403  STERLING VA (Washington Dulles), USA       38.98N  77.49W   88 m
    USM00072250  BROWNSVILLE/INT TX, USA                    25.92N  97.42W    7 m
    USM00070026  BARROW/W. POST W. ROGERS AK, USA (Arctic)  71.29N 156.78W   12 m
    GMM00010868  OBERSCHLEISSHEIM (Munich), Germany         48.24N  11.55E  484 m
    RQM00078526  SAN JUAN/INT, Puerto Rico (tropics)        18.43N  65.99W    4 m

PIPELINE (standard PBL preprocessing — NOT kernel math):
 1. Parse IGRA2 fixed-width soundings (PRESS Pa, GPH m, TEMP 0.1C, RH 0.1%,
    DPDP 0.1C, WDIR deg, WSPD 0.1 m/s; -9999/-8888 = missing).
 2. QC: keep soundings with surface pressure >= 850 hPa and >= 25 valid
    T+GPH levels and >= 15 valid wind levels spanning the sigma grid
    (surface .. 0.20*psfc). No gap-filling beyond interpolation between
    valid levels of the same sounding; no extrapolation.
 3. Interpolate T, q, u, v, z linearly in ln(p) to a 64-level sigma grid
    (layer midpoints; interfaces stretched, denser near the surface):
       s_if[j] = 1 - 0.8*(j/64)**1.3 ,  j=0..64   (s=1.0 .. 0.20)
       sigma_mid[k] = 0.5*(s_if[k]+s_if[k+1]),  p_k = sigma_mid[k]*psfc
    q from RH (preferred) or dewpoint depression via Magnus saturation
    vapor pressure over water; q = 0.622 e / (p - 0.378 e).
 4. Eddy diffusivity at interior interfaces via a bulk Richardson
    formulation (first-order local K closure, Louis 1979-type stability
    functions with a Blackadar mixing length — standard textbook/GFS-style
    local scheme):
       Ri  = (g/thv_m) * d(thv) * dz / (du^2 + dv^2 + 1e-9)
       l   = k*z / (1 + k*z/lambda),  k=0.4, lambda=150 m
       S   = sqrt(du^2+dv^2)/dz
       fm  = sqrt(1-16Ri)          (Ri<0, unstable)
           = 1/(1+5Ri)^2           (Ri>=0, stable)
       Km  = l^2 * S * fm,  bounded to [0.01, 1000] m^2/s
       Kh  = Km/Pr ; Pr = 0.85 (Ri<0) ; min(1+2.1*Ri, 3) (Ri>=0)
             (Kim & Mahrt 1992 stable-Prandtl form)
 5. Assemble implicit-Euler vertical diffusion tridiagonal systems with
    dt = 600 s, exactly the cl/cm/cu/r1 structure that cpu_tridi1/
    kernel_tridi1 in noaa_multi_kernel.cu solve (cl[0]=0, cu[n-1]=0,
    diagonally dominant cm = 1 - cl - cu; same sign convention as the
    synthetic generator gen_tridi: negative off-diagonals):
       cl[k] = -dt*K(k-1/2) / (dzl[k] * dzi[k-1/2])
       cu[k] = -dt*K(k+1/2) / (dzl[k] * dzi[k+1/2])
       cm[k] = 1 - cl[k] - cu[k]
       r1[k] = field value (T[K], q[kg/kg], u[m/s] or v[m/s])
    Each accepted sounding yields 4 REAL columns:
       (Kh-matrix, T), (Kh-matrix, q), (Km-matrix, u), (Km-matrix, v).
 6. Write little-endian binary 'tridi_real.bin':
       int32 ncol, int32 nlev, then float32 cl[ncol*nlev], cm[...],
       cu[...], r1[...]  (row-major: column-by-column, k fastest).

No random numbers anywhere. Every value in cl/cm/cu/r1 derives from
measured radiosonde T/RH/wind/geopotential plus the documented constants.
"""
import sys, glob, os, json
import numpy as np

NLEV   = 64
SIG_TOP_FRAC = 0.20      # sigma grid spans 1.0 .. 0.20
DT     = 600.0           # s, physics timestep for the implicit solve
GRAV   = 9.80665
RD     = 287.05
KAPPA  = 0.2854          # Rd/cp
VONK   = 0.4
LAMBDA = 150.0           # m, Blackadar asymptotic mixing length
KMIN, KMAX = 0.01, 1000.0
MISS   = (-9999, -8888)

def es_hPa(T_C):
    """Magnus saturation vapor pressure over water, hPa (T in deg C)."""
    return 6.112 * np.exp(17.62 * T_C / (243.12 + T_C))

def parse_station(path):
    """Yield soundings: dict with arrays p(Pa), z(m), T(K), q(kg/kg), u, v."""
    out = []
    with open(path, "r") as f:
        lines = f.readlines()
    i, nlines = 0, len(lines)
    while i < nlines:
        line = lines[i]
        if not line.startswith("#"):
            i += 1; continue
        # header: #ID(12) YEAR MONTH DAY HOUR RELTIME NUMLEV ...
        sid   = line[1:12]
        year  = int(line[13:17]); month = int(line[18:20]); day = int(line[21:23])
        hour  = int(line[24:26])
        numlev = int(line[32:36])
        recs = lines[i+1 : i+1+numlev]
        i += 1 + numlev
        p_, z_, t_, rh_, dpdp_, wd_, ws_ = [], [], [], [], [], [], []
        for r in recs:
            try:
                press = int(r[9:15]); gph = int(r[16:21]); temp = int(r[22:27])
                rh    = int(r[28:33]); dpdp = int(r[34:39])
                wdir  = int(r[40:45]); wspd = int(r[46:51])
            except (ValueError, IndexError):
                continue
            if press in MISS or press <= 0:   # need pressure
                continue
            p_.append(press); z_.append(gph); t_.append(temp)
            rh_.append(rh); dpdp_.append(dpdp); wd_.append(wdir); ws_.append(wspd)
        if not p_:
            continue
        out.append(dict(id=sid, ymdh=(year, month, day, hour),
                        p=np.array(p_, float), z=np.array(z_, float),
                        t=np.array(t_, float), rh=np.array(rh_, float),
                        dpdp=np.array(dpdp_, float),
                        wd=np.array(wd_, float), ws=np.array(ws_, float)))
    return out

def build_column(s):
    """Interpolate one sounding to the sigma grid; return dict or None."""
    p = s["p"]
    # sort by decreasing pressure (surface first); drop duplicate pressures
    order = np.argsort(-p)
    p = p[order]
    z, t, rh, dpdp, wd, ws = (s[k][order] for k in ("z", "t", "rh", "dpdp", "wd", "ws"))
    keep = np.concatenate(([True], np.diff(p) < 0))
    p, z, t, rh, dpdp, wd, ws = p[keep], z[keep], t[keep], rh[keep], dpdp[keep], wd[keep], ws[keep]

    psfc = p[0]
    if psfc < 85000.0:          # station too high / bad surface
        return None
    ptop_need = SIG_TOP_FRAC * psfc

    valid_tz = ~np.isin(t, MISS) & ~np.isin(z, MISS)
    valid_w  = ~np.isin(wd, MISS) & ~np.isin(ws, MISS) & (wd >= 0) & (ws >= 0)
    if valid_tz.sum() < 25 or valid_w.sum() < 15:
        return None
    # require coverage of the whole sigma grid by valid T/Z and wind levels
    if p[valid_tz].min() > ptop_need or p[valid_w].min() > ptop_need:
        return None
    # humidity: prefer RH, fall back to dewpoint depression
    valid_rh = ~np.isin(rh, MISS) & (rh >= 0)
    valid_dp = ~np.isin(dpdp, MISS) & (dpdp >= 0) & valid_tz
    if (valid_rh | valid_dp).sum() < 10:
        return None

    lnp = np.log(p)
    # target grid
    j = np.arange(NLEV + 1)
    s_if = 1.0 - (1.0 - SIG_TOP_FRAC) * (j / NLEV) ** 1.3
    sig_mid = 0.5 * (s_if[:-1] + s_if[1:])
    p_mid = sig_mid * psfc
    p_if = s_if * psfc
    lnp_mid = np.log(p_mid)
    lnp_if = np.log(p_if)

    def interp(lnp_src, vals, lnp_dst):
        # np.interp needs increasing x; lnp decreases with index -> flip
        return np.interp(lnp_dst, lnp_src[::-1], vals[::-1])

    # T (K), z (m) on layer mids and interfaces
    T_mid = interp(lnp[valid_tz], t[valid_tz] / 10.0 + 273.15, lnp_mid)
    z_mid = interp(lnp[valid_tz], z[valid_tz], lnp_mid)
    z_if  = interp(lnp[valid_tz], z[valid_tz], lnp_if)
    if not (np.all(np.diff(z_mid) > 0) and np.all(np.diff(z_if) > 0)):
        return None  # non-monotone geopotential after interpolation

    # vapor pressure e (hPa) at observed levels, then interpolate ln(e+eps)
    T_C = t / 10.0
    e_obs = np.full_like(p, np.nan, dtype=float)
    use_rh = valid_rh & valid_tz
    e_obs[use_rh] = (rh[use_rh] / 1000.0) * es_hPa(T_C[use_rh])
    use_dp = valid_dp & ~use_rh
    e_obs[use_dp] = es_hPa(T_C[use_dp] - dpdp[use_dp] / 10.0)
    has_e = ~np.isnan(e_obs)
    if has_e.sum() < 10:
        return None
    e_mid = np.exp(interp(lnp[has_e], np.log(np.maximum(e_obs[has_e], 1e-6)), lnp_mid))
    # above the highest humidity level, hold the (tiny) topmost value: clamp
    p_e_top = p[has_e].min()
    e_mid = np.where(p_mid < p_e_top, e_mid.min(), e_mid)
    q_mid = 0.622 * e_mid / (p_mid / 100.0 - 0.378 * e_mid)   # p in hPa here
    q_mid = np.clip(q_mid, 1e-7, 0.035)

    # wind components
    wd_r = np.deg2rad(wd[valid_w])
    u_obs = -ws[valid_w] / 10.0 * np.sin(wd_r)
    v_obs = -ws[valid_w] / 10.0 * np.cos(wd_r)
    u_mid = interp(lnp[valid_w], u_obs, lnp_mid)
    v_mid = interp(lnp[valid_w], v_obs, lnp_mid)

    # virtual potential temperature
    thv = T_mid * (1.0 + 0.61 * q_mid) * (100000.0 / p_mid) ** KAPPA

    # --- K profiles at interior interfaces (j = 1..NLEV-1 between k-1,k) ---
    dz_i = z_mid[1:] - z_mid[:-1]                      # between layer mids
    du = u_mid[1:] - u_mid[:-1]
    dv = v_mid[1:] - v_mid[:-1]
    dthv = thv[1:] - thv[:-1]
    thv_m = 0.5 * (thv[1:] + thv[:-1])
    shr2 = du**2 + dv**2 + 1e-9
    Ri = (GRAV / thv_m) * dthv * dz_i / shr2
    z_int = z_if[1:-1] - z_if[0]                        # height AGL of interfaces
    z_int = np.maximum(z_int, 1.0)
    l_mix = VONK * z_int / (1.0 + VONK * z_int / LAMBDA)
    S = np.sqrt(shr2) / dz_i
    fm = np.where(Ri < 0, np.sqrt(np.maximum(1.0 - 16.0 * Ri, 1.0)),
                  1.0 / (1.0 + 5.0 * np.maximum(Ri, 0.0)) ** 2)
    Km = np.clip(l_mix**2 * S * fm, KMIN, KMAX)
    Pr = np.where(Ri < 0, 0.85, np.minimum(1.0 + 2.1 * np.maximum(Ri, 0.0), 3.0))
    Kh = np.clip(Km / Pr, KMIN, KMAX)

    dz_l = z_if[1:] - z_if[:-1]                         # layer thickness
    return dict(T=T_mid, q=q_mid, u=u_mid, v=v_mid,
                Km=Km, Kh=Kh, dz_l=dz_l, dz_i=dz_i)

def assemble(col):
    """Return 4 (cl, cm, cu, r1) systems for one column."""
    n = NLEV
    out = []
    for K, field in ((col["Kh"], col["T"]), (col["Kh"], col["q"]),
                     (col["Km"], col["u"]), (col["Km"], col["v"])):
        cl = np.zeros(n, np.float32); cu = np.zeros(n, np.float32)
        cl[1:]  = -DT * K / (col["dz_l"][1:]  * col["dz_i"])
        cu[:-1] = -DT * K / (col["dz_l"][:-1] * col["dz_i"])
        cm = (1.0 - cl - cu).astype(np.float32)
        out.append((cl, cm, cu, field.astype(np.float32)))
    return out

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "data", "*-data.txt")))
    if not files:
        sys.exit("no IGRA2 station files found under data/")
    systems = []
    stats = {}
    for fp in files:
        st = os.path.basename(fp).split("-")[0]
        snd = parse_station(fp)
        ok = 0
        for s in snd:
            col = build_column(s)
            if col is None:
                continue
            systems.extend(assemble(col))
            ok += 1
        stats[st] = dict(soundings=len(snd), accepted=ok)
        print(f"{st}: {len(snd)} soundings, {ok} accepted -> {4*ok} columns")
    ncol = len(systems)
    print(f"TOTAL real columns: {ncol} ({ncol//4} soundings x 4 fields)")

    cl = np.stack([s[0] for s in systems]); cm = np.stack([s[1] for s in systems])
    cu = np.stack([s[2] for s in systems]); r1 = np.stack([s[3] for s in systems])
    # sanity: diagonal dominance + finite
    assert np.all(np.isfinite(cl)) and np.all(np.isfinite(cm)) and \
           np.all(np.isfinite(cu)) and np.all(np.isfinite(r1))
    dom = cm - (np.abs(cl) + np.abs(cu))
    print(f"diag dominance margin: min {dom.min():.6f} (must be ~1.0)")
    print(f"|cl| range: {np.abs(cl[cl!=0]).min():.3e} .. {np.abs(cl).max():.3e}")
    print(f"cm  range: {cm.min():.3e} .. {cm.max():.3e}")
    print(f"r1  range: {r1.min():.3e} .. {r1.max():.3e}")

    outbin = os.path.join(here, "tridi_real.bin")
    with open(outbin, "wb") as f:
        np.array([ncol, NLEV], np.int32).tofile(f)
        cl.astype(np.float32).tofile(f); cm.astype(np.float32).tofile(f)
        cu.astype(np.float32).tofile(f); r1.astype(np.float32).tofile(f)
    print(f"wrote {outbin} ({os.path.getsize(outbin)/1e6:.1f} MB)")

    meta = dict(source="NOAA NCEI IGRA2 data-y2d (2025-01-01..2026-06, accessed 2026-06-10)",
                url="https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/",
                stations=stats, ncol=ncol, nlev=NLEV, dt_s=DT,
                sigma_top=SIG_TOP_FRAC, fields=["T", "q", "u", "v"],
                K_scheme="bulk-Ri local closure: Louis-79 fm, Blackadar l (lambda=150m), "
                         "Kim-Mahrt stable Pr; K in [0.01,1000] m2/s")
    with open(os.path.join(here, "tridi_real_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
