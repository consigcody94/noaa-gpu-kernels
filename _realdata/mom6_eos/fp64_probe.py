"""
Independent FP32-vs-FP64 conditioning probe for the triDiagTS Thomas recurrence
on the real Argo columns (does NOT touch the CUDA kernels — pure diagnostics).

Reimplements the exact recurrence from cpu_tridiag_ts in numpy at float32 and
float64, with ea/eb set per the harness scenarios, and reports how far FP32
drifts from FP64. If FP32-vs-FP64 drift is the same order as the observed
CPU-vs-GPU disagreement, the NEEDS REVIEW status is an intrinsic FP32
conditioning limit of the algorithm, not a GPU bug.
"""
import numpy as np

DATA = r"U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\mom6_eos\data\columns.bin"

cols = []
with open(DATA, "rb") as f:
    ncol = np.fromfile(f, np.int32, 1)[0]
    for _ in range(ncol):
        nz = np.fromfile(f, np.int32, 1)[0]
        h = np.fromfile(f, np.float32, nz)
        T = np.fromfile(f, np.float32, nz)
        S = np.fromfile(f, np.float32, nz)
        cols.append((h, T, S))

def thomas(h, ea, eb, x, dtype):
    h, ea, eb, x = (a.astype(dtype) for a in (h, ea, eb, x))
    nz = h.size
    fwd = np.zeros(nz, dtype); cu = np.zeros(nz, dtype)
    one = dtype(1)
    hp = h[0] + ea[0] + eb[0]
    bet = hp
    fwd[0] = h[0]*x[0]/bet
    cu[0] = eb[0]/bet
    for k in range(1, nz):
        hk = h[k] + ea[k] + eb[k]
        bet = hk - ea[k]*cu[k-1]
        cu[k] = eb[k]/bet
        fwd[k] = (h[k]*x[k] + ea[k]*fwd[k-1])/bet
    out = np.zeros(nz, dtype)
    out[nz-1] = fwd[nz-1]
    for k in range(nz-2, -1, -1):
        out[k] = fwd[k] + cu[k]*out[k+1]
    return out

def make_eaeb(h, K=None, dt=3600.0, const=None):
    nz = h.size
    ea = np.zeros(nz); eb = np.zeros(nz)
    if const is not None:
        ea[:] = const; eb[:] = const
        return ea, eb
    e = np.zeros(nz+1)
    for k in range(1, nz):
        e[k] = K*dt/(0.5*(h[k-1]+h[k]))
    for k in range(nz):
        ea[k] = e[k]; eb[k] = e[k+1]
    return ea, eb

def thomas_cpu_clobber(h, ea, eb, x, dtype):
    """Emulates the ORIGINAL cpu_tridiag_ts exactly: the first (incomplete)
    forward sweep overwrites x[nz-1] with the solve result, THEN the proper
    Thomas pass runs on the clobbered input."""
    h, ea, eb, x = (a.astype(dtype) for a in (h, ea, eb, x))
    x = x.copy()
    nz = h.size
    # first pass (reciprocal-multiply form), result clobbers bottom level
    hp = h[0] + ea[0] + eb[0]
    b1 = dtype(1)/hp
    d1 = b1*(h[0]*x[0])
    c1 = eb[0]*b1
    for k in range(1, nz):
        hk = h[k] + ea[k] + eb[k]
        bet = dtype(1)/(hk - ea[k]*c1)
        c1 = eb[k]*bet
        d1 = bet*(h[k]*x[k] + ea[k]*d1)
    x[nz-1] = d1                      # <-- the clobber
    return thomas(h, ea, eb, x, dtype)

scenarios = [("K=1e-5", dict(K=1e-5)), ("K=1e-4", dict(K=1e-4)),
             ("K=1e-3", dict(K=1e-3)), ("const=1.0", dict(const=1.0))]

print("A) intrinsic FP32 conditioning (clean Thomas, FP32 vs FP64):")
for name, kw in scenarios:
    worst = 0.0; wcol = -1
    hmin_w = None
    for ci, (h, T, S) in enumerate(cols):
        ea, eb = make_eaeb(h.astype(np.float64), **kw)
        t32 = thomas(h, ea, eb, T, np.float32)
        t64 = thomas(h, ea, eb, T, np.float64)
        m = np.abs(t64) > 0.01
        if not m.any():
            continue
        rel = np.max(np.abs(t32[m].astype(np.float64) - t64[m]) / np.abs(t64[m]))
        if rel > worst:
            worst = rel; wcol = ci; hmin_w = float(h.min())
    print(f"  {name:10s} max FP32-vs-FP64 rel err (T) over {len(cols)} real cols: "
          f"{worst:.2e}  (col {wcol}, min h = {hmin_w:.2f} m)")

print("\nB) clobber hypothesis (CPU-with-clobber vs clean Thomas, both FP64):")
print("   if this matches the observed CPU-vs-GPU error, the discrepancy is the")
print("   CPU reference's bottom-level overwrite, not GPU/FP32 error.")
for name, kw in scenarios:
    worst = 0.0; wcol = -1; wk = -1
    for ci, (h, T, S) in enumerate(cols):
        ea, eb = make_eaeb(h.astype(np.float64), **kw)
        t_cl = thomas_cpu_clobber(h, ea, eb, T, np.float64)
        t_ok = thomas(h, ea, eb, T, np.float64)
        m = np.abs(t_cl) > 0.01
        if not m.any():
            continue
        rel = np.abs(t_ok - t_cl) / np.where(m, np.abs(t_cl), np.inf)
        r = float(rel.max())
        if r > worst:
            worst = r; wcol = ci; wk = int(rel.argmax())
    print(f"  {name:10s} max clean-vs-clobbered rel diff (T): {worst:.2e}  "
          f"(col {wcol}, level {wk})")
