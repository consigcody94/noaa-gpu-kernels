"""
prep_igra2.py — Parse real IGRA2 radiosonde soundings (NOAA NCEI) and write a flat
binary input file for upp_cape_realdata.cu.

Data source (public, authoritative):
  https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/
  Files: <STATION>-data-beg2025.txt.zip  (period 2025-01-01 .. present; accessed 2026-06-10)

Stations (from igra2-station-list.txt):
  USM00072357  35.1808  -97.4378   344.9 m  NORMAN/MAX WESTHEIMER A, OK   (OUN — convective Great Plains)
  USM00072261  29.3744 -100.9183   314.0 m  DEL RIO/INT., TX              (DRT — subtropical, high CAPE)
  USM00072649  44.8497  -93.5647   290.2 m  CHANHASSEN, MN                (MPX — midlatitude)
  RQM00078526  18.4317  -65.9919     4.0 m  SAN JUAN/INT., PUERTO RICO    (TJSJ — tropical)

IGRA2 fixed-width data format (igra2-data-format.txt):
  Header: '#' ID(2-12) YEAR(14-17) MONTH(19-20) DAY(22-23) HOUR(25-26) RELTIME(28-31) NUMLEV(32-36) ...
  Data:   LVLTYP1(1) LVLTYP2(2) ETIME(4-8) PRESS(10-15,Pa) PFLAG(16) GPH(17-21,m) ZFLAG(22)
          TEMP(23-27,degC*10) TFLAG(28) RH(29-33,%*10) DPDP(35-39,degC*10) WDIR(41-45) WSPD(47-51,m/s*10)
  Missing: -9999 ; removed-by-QC: -8888.

Mapping to the kernel's Column struct (upp_cape.cu — read first, unmodified):
  - Kernel indexing: l = nlev-1 is the SURFACE (max pressure); l = 0 is the column top.
    (The most-unstable-parcel search starts at l = nlev-1 and breaks once
     pmid[nlev-1] - pmid[l] > 30000 Pa, i.e. lowest 300 hPa, so pmid must
     increase with l.)
  - pmid[l]: uniform 1000-Pa grid from the observed surface pressure up to <=10000 Pa
    (>=100 hPa). nlev = floor((psfc - 10000)/1000) + 1, always < MAX_LEV=128.
  - T: linear interpolation in ln(p) from observed IGRA2 temperatures.
  - Q: specific humidity computed at observed levels from dewpoint depression
    (Td = T - DPDP) via the SAME Tetens saturation-vapor-pressure formula the
    kernel uses (es = 611.2*exp(17.67*tc/(tc+243.5)); q = 0.622 e/(p-0.378 e)).
    If DPDP missing but RH present: e = (RH/100)*es(T). Levels outside the span
    of valid moisture observations get q = 0 (dry), interpolated in ln(p) inside.
  - zint[0..nlev]: layer interfaces from observed geopotential heights (GPH),
    interpolated in ln(p) to the grid midpoints z[l], then
       zint[l]   = 0.5*(z[l-1] + z[l])   for l = 1..nlev-1
       zint[0]   = z[0] + 0.5*(z[0] - z[1])          (top extrapolation)
       zint[nlev]= z[nlev-1]                         (surface geopotential height)
    so dz = zint[l] - zint[l+1] > 0 is the thickness attributed to level l.

Sounding acceptance criteria (no value fabrication — reject instead of fill):
  - nominal HOUR 00 or 12 UTC
  - surface level (LVLTYP2==1) present with valid PRESS, TEMP, GPH
  - at least 30 levels with valid PRESS+TEMP, monotonically decreasing pressure
  - profile reaches 10000 Pa (100 hPa) or higher with valid temperature
  - valid GPH at enough levels to span the grid; z strictly increasing with height

Output:
  cape_input.bin : int32 magic 0x43415045, int32 nsound, then per sounding:
                   int32 nlev, float32 pmid[nlev], T[nlev], Q[nlev], zint[nlev+1]
                   (index 0 = top, nlev-1 = surface, matching kernel expectation)
  manifest.csv   : sounding index -> station, date, hour, nlev, psfc, raw level count
"""
import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path(r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\upp_cape\data")
OUT_BIN = Path(r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\upp_cape\cape_input.bin")
OUT_MANIFEST = Path(r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\upp_cape\manifest.csv")

STATIONS = ["USM00072357", "USM00072261", "USM00072649", "RQM00078526"]

MAX_LEV = 128
DP = 1000.0          # Pa, uniform grid spacing
P_TOP_MIN = 10000.0  # grid must stop at >= 100 hPa
EPS = 0.622


def esat_tetens(t_k):
    """Same Tetens formula as the kernel's esat() (Pa)."""
    tc = t_k - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def parse_int(line, lo, hi):
    """1-based inclusive column slice -> int or None for missing/QC-removed."""
    s = line[lo - 1:hi].strip()
    if not s:
        return None
    v = int(s)
    if v in (-9999, -8888):
        return None
    return v


def iter_soundings(text):
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("#"):
            i += 1
            continue
        sid = line[1:12]
        year = int(line[13:17])
        month = int(line[18:20])
        day = int(line[21:23])
        hour = int(line[24:26])
        numlev = int(line[31:36])
        levels = lines[i + 1:i + 1 + numlev]
        i += 1 + numlev
        yield sid, year, month, day, hour, levels


def build_profile(levels):
    """Return raw arrays (p desc, T K, q, z) or None if unusable."""
    p_l, t_l, q_l, z_l = [], [], [], []
    sfc_seen = False
    for ln in levels:
        lvltyp2 = ln[1:2]
        press = parse_int(ln, 10, 15)
        gph = parse_int(ln, 17, 21)
        temp = parse_int(ln, 23, 27)
        rh = parse_int(ln, 29, 33)
        dpdp = parse_int(ln, 35, 39)
        if press is None or press <= 0 or temp is None:
            continue
        p = float(press)
        t = temp / 10.0 + 273.15
        if not (150.0 < t < 340.0):
            continue
        if lvltyp2 == "1":
            if gph is None:
                return None  # need surface height
            sfc_seen = True
        # moisture
        q = np.nan
        if dpdp is not None and dpdp >= 0:
            td = t - dpdp / 10.0
            e = esat_tetens(np.float64(td))
            q = EPS * e / (p - (1.0 - EPS) * e)
        elif rh is not None and 0 <= rh <= 1000:
            e = (rh / 1000.0) * esat_tetens(np.float64(t))
            q = EPS * e / (p - (1.0 - EPS) * e)
        if np.isfinite(q):
            q = min(max(q, 0.0), 0.04)
        p_l.append(p)
        t_l.append(t)
        q_l.append(q)
        z_l.append(float(gph) if gph is not None else np.nan)
    if not sfc_seen or len(p_l) < 30:
        return None
    p = np.array(p_l)
    t = np.array(t_l)
    q = np.array(q_l)
    z = np.array(z_l)
    # sort by descending pressure, drop duplicate pressures
    order = np.argsort(-p, kind="stable")
    p, t, q, z = p[order], t[order], q[order], z[order]
    keep = np.concatenate(([True], np.diff(p) < -0.5))
    p, t, q, z = p[keep], t[keep], q[keep], z[keep]
    if p[0] < 60000.0 or p[-1] > P_TOP_MIN:
        return None  # no real surface or doesn't reach 100 hPa
    return p, t, q, z


def to_grid(p, t, q, z):
    """Interpolate to the uniform-in-p grid; return (pmid, T, Q, zint) kernel-ordered."""
    psfc = p[0]
    nlev = int((psfc - P_TOP_MIN) // DP) + 1
    if nlev < 30 or nlev > MAX_LEV - 1:
        return None
    pg_desc = psfc - DP * np.arange(nlev)        # descending: surface .. top
    lnp = np.log(p)                               # decreasing
    lnpg = np.log(pg_desc)
    # np.interp needs increasing x: flip
    x = lnp[::-1]
    tg = np.interp(lnpg, x, t[::-1])
    # heights: only levels with valid z
    zok = np.isfinite(z)
    if zok.sum() < 10:
        return None
    xz = lnp[zok][::-1]
    zz = z[zok][::-1]
    if pg_desc[-1] < p[zok][-1] - 0.5 * DP or not np.all(np.diff(zz) < 0):
        return None  # grid extends above topmost valid height, or z not monotone
        # (zz runs top -> surface as ln(p) increases, so heights must decrease)
    zg = np.interp(lnpg, xz, zz)
    if not np.all(np.diff(zg) > 0):
        return None
    # moisture: interpolate inside the valid-moisture span, q=0 outside
    qok = np.isfinite(q)
    qg = np.zeros(nlev)
    if qok.sum() >= 2:
        xq = lnp[qok][::-1]
        qq = q[qok][::-1]
        inside = (lnpg >= xq[0]) & (lnpg <= xq[-1])
        qg[inside] = np.interp(lnpg[inside], xq, qq)
    # kernel ordering: index 0 = top, nlev-1 = surface
    pmid = pg_desc[::-1].astype(np.float32)       # increasing with l
    T = tg[::-1].astype(np.float32)
    Q = qg[::-1].astype(np.float32)
    zmid = zg[::-1]                               # decreasing with l (z[0] = top)
    zint = np.empty(nlev + 1)
    zint[1:nlev] = 0.5 * (zmid[:-1] + zmid[1:])
    zint[0] = zmid[0] + 0.5 * (zmid[0] - zmid[1])
    zint[nlev] = zmid[nlev - 1]                   # surface geopotential height
    if not np.all(np.diff(zint) < 0):
        return None
    return pmid, T, Q, zint.astype(np.float32)


def main():
    records = []   # (station, ymdh, pmid, T, Q, zint)
    stats = {}
    for st in STATIONS:
        zpath = DATA_DIR / f"{st}-data-beg2025.txt.zip"
        with zipfile.ZipFile(zpath) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("ascii", errors="replace")
        tot = acc = 0
        for sid, yr, mo, dy, hr, levels in iter_soundings(text):
            if hr not in (0, 12):
                continue
            tot += 1
            prof = build_profile(levels)
            if prof is None:
                continue
            grid = to_grid(*prof)
            if grid is None:
                continue
            acc += 1
            records.append((st, f"{yr:04d}-{mo:02d}-{dy:02d} {hr:02d}Z",
                            len(levels), *grid))
        stats[st] = (tot, acc)
        print(f"{st}: {acc}/{tot} soundings accepted")

    print(f"TOTAL real soundings: {len(records)}")
    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack("<ii", 0x43415045, len(records)))
        for st, ymdh, nraw, pmid, T, Q, zint in records:
            nlev = len(pmid)
            f.write(struct.pack("<i", nlev))
            f.write(pmid.tobytes())
            f.write(T.tobytes())
            f.write(Q.tobytes())
            f.write(zint.tobytes())
    with open(OUT_MANIFEST, "w") as f:
        f.write("index,station,datetime,nlev,psfc_pa,raw_levels\n")
        for i, (st, ymdh, nraw, pmid, T, Q, zint) in enumerate(records):
            f.write(f"{i},{st},{ymdh},{len(pmid)},{pmid[-1]:.0f},{nraw}\n")
    print(f"wrote {OUT_BIN} ({OUT_BIN.stat().st_size} bytes) and {OUT_MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())
