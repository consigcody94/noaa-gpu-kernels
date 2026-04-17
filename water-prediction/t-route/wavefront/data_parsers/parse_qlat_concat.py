#!/usr/bin/env python
"""
Strategy: concat all 142K nex-*_output.csv into ONE pre-sorted binary
(wbid, ts, qlat) tuples, then stream-read. Eliminates 142K file opens.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

NEX_RE = re.compile(r"nex-(\d+)_output\.csv$")


def read_and_pack(path: str) -> bytes:
    """Read one CSV, emit a compact binary: int32 wbid, int32 n_vals, float32[n_vals]."""
    m = NEX_RE.search(path)
    if not m:
        return b""
    wbid = int(m.group(1))
    with open(path, "rb") as f:
        data = f.read()
    vals = []
    for line in data.decode("ascii", errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            int(parts[0])
            vals.append(float(parts[-1]))
        except ValueError:
            continue
    arr = np.asarray(vals, dtype=np.float32).tobytes()
    return struct.pack("<ii", wbid, len(vals)) + arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="nex_csvs/big_one_fixed_1/outputs/ngen")
    ap.add_argument("--net-dir", default="net_real_309k")
    ap.add_argument("--out", default="net_real_309k/forcings.bin")
    ap.add_argument("--packed-out", default="net_real_309k/packed.bin")
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--skip-pack", action="store_true",
                    help="reuse existing packed file")
    args = ap.parse_args()

    topo_to_wbid = np.load(os.path.join(args.net_dir, "topo_to_wbid.npy"))
    n_reaches = len(topo_to_wbid)
    wbid_to_topo = {int(w): i for i, w in enumerate(topo_to_wbid)}

    t0 = time.perf_counter()
    paths = [
        os.path.join(args.csv_dir, name)
        for name in os.listdir(args.csv_dir)
        if name.startswith("nex-") and name.endswith("_output.csv")
    ]
    t1 = time.perf_counter()
    print(f"[cat] listed {len(paths)} files in {(t1-t0)*1000:.1f} ms")

    if not args.skip_pack:
        # Pack stage (parallel read + pack)
        t2 = time.perf_counter()
        total_bytes = 0
        with open(args.packed_out, "wb") as fout:
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                for chunk in ex.map(read_and_pack, paths, chunksize=256):
                    if chunk:
                        fout.write(chunk)
                        total_bytes += len(chunk)
        t3 = time.perf_counter()
        print(f"[cat] packed into {args.packed_out} ({total_bytes/1e6:.1f} MB) in {(t3-t2)*1000:.1f} ms "
              f"using {args.threads} threads ({len(paths)/(t3-t2):.0f} files/s)")
    else:
        t2 = t3 = time.perf_counter()

    # Read packed file
    t4 = time.perf_counter()
    qlat = np.zeros((args.timesteps, n_reaches), dtype=np.float32)
    with open(args.packed_out, "rb") as f:
        data = f.read()
    matched = 0
    missing = 0
    pos = 0
    while pos < len(data):
        wbid, n_vals = struct.unpack_from("<ii", data, pos)
        pos += 8
        vals_bytes = 4 * n_vals
        if wbid not in wbid_to_topo:
            missing += 1
        else:
            tidx = wbid_to_topo[wbid]
            vals = np.frombuffer(data, dtype=np.float32, count=n_vals, offset=pos)
            m = min(args.timesteps, n_vals)
            qlat[:m, tidx] = vals[:m]
            matched += 1
        pos += vals_bytes
    t5 = time.perf_counter()
    print(f"[cat] unpacked {matched}/{len(paths)} (missing {missing}) in {(t5-t4)*1000:.1f} ms")

    # Write forcings.bin
    t6 = time.perf_counter()
    qdp0 = np.zeros(n_reaches, dtype=np.float32)
    dp0 = np.ones(n_reaches, dtype=np.float32) * 0.5
    with open(args.out, "wb") as f:
        f.write(struct.pack("<i", args.timesteps))
        f.write(qlat.astype(np.float32).tobytes())
        f.write(np.zeros(n_reaches, dtype=np.float32).tobytes())
        f.write(qdp0.tobytes())
        f.write(dp0.tobytes())
    t7 = time.perf_counter()
    print(f"[cat] wrote {args.out} in {(t7-t6)*1000:.1f} ms")
    print(f"[cat] TOTAL (pack + unpack + write): {(t7-t0)*1000:.1f} ms")
    print(f"[cat] If skipping pack (warm-run): {(t7-t4+t1-t0)*1000:.1f} ms")


if __name__ == "__main__":
    main()
