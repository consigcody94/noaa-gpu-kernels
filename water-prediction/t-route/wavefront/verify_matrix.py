#!/usr/bin/env python
"""
Verify matrix_Q_final.bin (from linear_mc_matrix.exe) against a FP64 CPU
reference linear-MC solve on the same topology + forcings + coefficients.
"""
from __future__ import annotations
import struct
import numpy as np
import sys
import numba

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


@numba.njit(cache=True, fastmath=False)
def run_reference(n_r, n_ts, n_l, qdp0, C1, C2, C3, C4oq,
                  up_ptr, up_idx, level_ptr, level_reach, qlat):
    Qp = qdp0.copy()
    Qn = np.zeros(n_r, dtype=np.float64)
    for ts in range(n_ts):
        qlat_t = qlat[ts]
        for L in range(n_l):
            lb = level_ptr[L]
            le = level_ptr[L + 1]
            for ii in range(lb, le):
                r = level_reach[ii]
                sum_new = 0.0
                sum_old = 0.0
                for k in range(up_ptr[r], up_ptr[r + 1]):
                    u = up_idx[k]
                    sum_new += Qn[u]
                    sum_old += Qp[u]
                Qn[r] = C1[r] * sum_new + C2[r] * sum_old + C3[r] * Qp[r] + C4oq[r] * qlat_t[r]
        tmp = Qp
        Qp = Qn
        Qn = tmp
        for i in range(n_r):
            Qn[i] = 0.0
    return Qp


def compute_mc_coeffs(dx, bw, tw, n_ch, ncc, cs, s0, dt=300.0):
    z = 1.0 if cs == 0.0 else 1.0 / cs
    if bw > tw:
        bfd = bw / 0.00001
    elif bw == tw:
        bfd = bw / (2.0 * z)
    else:
        bfd = (tw - bw) / (2.0 * z)
    h = max(0.5 * bfd, 0.1)
    A = (bw + h * z) * h
    WP = bw + 2.0 * h * np.sqrt(1.0 + z * z)
    R = A / WP if WP > 0 else 0.0
    r23 = R ** (2.0 / 3.0)
    r53 = R ** (5.0 / 3.0)
    Ck = max(1e-6, (np.sqrt(s0) / n_ch) *
         ((5.0 / 3.0) * r23 - (2.0 / 3.0) * r53 *
          (2.0 * np.sqrt(1.0 + z * z) / (bw + 2.0 * h * z))))
    Km = max(dt, dx / Ck)
    X = 0.25
    D = Km * (1.0 - X) + dt / 2.0
    if D == 0.0:
        D = 1.0
    C1 = (Km * X + dt / 2.0) / D
    C2 = (dt / 2.0 - Km * X) / D
    C3 = (Km * (1.0 - X) - dt / 2.0) / D
    C4oq = dt / D
    return C1, C2, C3, C4oq


def main():
    dir = "net_real_309k"
    with open(f"{dir}/topo.bin", "rb") as f:
        n_r, n_s, n_l, m_up = struct.unpack("<iiii", f.read(16))
        seg_start = np.frombuffer(f.read(4 * n_r), dtype=np.int32).copy()
        seg_len = np.frombuffer(f.read(4 * n_r), dtype=np.int32).copy()
        level = np.frombuffer(f.read(4 * n_r), dtype=np.int32).copy()
        n_up = np.frombuffer(f.read(4 * n_r), dtype=np.int32).copy()
        up_ptr = np.frombuffer(f.read(4 * (n_r + 1)), dtype=np.int32).copy()
        nnz = int(up_ptr[-1])
        up_idx = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        level_ptr = np.frombuffer(f.read(4 * (n_l + 1)), dtype=np.int32).copy()
        level_reach = np.frombuffer(f.read(4 * n_r), dtype=np.int32).copy()

    params = np.fromfile(f"{dir}/params.bin", dtype=np.float32).reshape(-1, 8)
    with open(f"{dir}/forcings.bin", "rb") as f:
        (n_ts,) = struct.unpack("<i", f.read(4))
        qlat = np.frombuffer(f.read(4 * n_ts * n_s), dtype=np.float32).copy()
        qlat = qlat.reshape(n_ts, n_s)
        qup0 = np.frombuffer(f.read(4 * n_r), dtype=np.float32).copy()
        qdp0 = np.frombuffer(f.read(4 * n_s), dtype=np.float32).copy()
        dp0 = np.frombuffer(f.read(4 * n_r), dtype=np.float32).copy()

    # Compute C1..C4 in FP64
    C1 = np.zeros(n_r, dtype=np.float64)
    C2 = np.zeros(n_r, dtype=np.float64)
    C3 = np.zeros(n_r, dtype=np.float64)
    C4oq = np.zeros(n_r, dtype=np.float64)
    for r in range(n_r):
        dx, bw, tw, twcc, n_ch, ncc, cs, s0 = params[r].astype(np.float64)
        c1, c2, c3, c4 = compute_mc_coeffs(dx, bw, tw, n_ch, ncc, cs, s0)
        C1[r], C2[r], C3[r], C4oq[r] = c1, c2, c3, c4

    # FP64 reference: level-scheduled linear MC (same algorithm as GPU, but FP64)
    qlat_f64 = qlat.astype(np.float64)
    print(f"[verify] running FP64 reference for {n_ts} timesteps, {n_r} reaches (numba JIT)")
    truth = run_reference(
        n_r, n_ts, n_l, qdp0.astype(np.float64),
        C1, C2, C3, C4oq,
        up_ptr, up_idx, level_ptr, level_reach, qlat_f64
    )
    print(f"[verify] FP64 truth: sum={truth.sum():.3e} max={truth.max():.3e}")

    # Load matrix output
    gpu = np.fromfile("matrix_Q_final.bin", dtype=np.float32).astype(np.float64)
    print(f"[verify] GPU matrix:  sum={gpu.sum():.3e} max={gpu.max():.3e}")

    # Comparison
    diff = np.abs(gpu - truth)
    max_abs = diff.max()
    mask = np.abs(truth) > 1e-3
    rel = diff[mask] / np.abs(truth[mask])
    pcts = np.percentile(rel, [50, 90, 99, 99.9, 100])
    within_1pct = (rel < 0.01).mean() * 100
    within_10pct = (rel < 0.1).mean() * 100
    print(f"[verify] abs max err: {max_abs:.3e}")
    print(f"[verify] rel err percentiles (p50, p90, p99, p99.9, max): {pcts}")
    print(f"[verify] within 1%:  {within_1pct:.2f}%")
    print(f"[verify] within 10%: {within_10pct:.2f}%")


if __name__ == "__main__":
    main()
