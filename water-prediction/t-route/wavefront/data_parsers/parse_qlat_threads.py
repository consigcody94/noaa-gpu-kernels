#!/usr/bin/env python
"""
Multi-threaded Python version of parse_qlat.py.

Uses ThreadPoolExecutor: Python releases the GIL during file I/O (open, read),
so threads genuinely run file reads in parallel. The Python parsing portion
still holds the GIL but it's a small fraction of per-file time.

Two variants:
  --mode=threads: pre-extracted directory, threadpool reads files
  --mode=tarstream: stream from tar.gz, single-threaded (fallback baseline)

Expected: ~5-10x faster than single-threaded Python on 8-16 core hosts,
because the bottleneck is OS filesystem overhead on 142K small files,
not CPU parsing.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

NEX_RE = re.compile(r"nex-(\d+)_output\.csv$")


def parse_one_file(path: str) -> tuple[int, np.ndarray]:
    """Parse a single nex-*_output.csv: returns (wbid, qlat_values)."""
    m = NEX_RE.search(path)
    if not m:
        return -1, np.empty(0, dtype=np.float32)
    wbid = int(m.group(1))
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return -1, np.empty(0, dtype=np.float32)
    vals: list[float] = []
    for line in data.decode("ascii", errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            int(parts[0])
            vals.append(float(parts[-1]))
        except ValueError:
            continue
    return wbid, np.asarray(vals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="nex_csvs/big_one_fixed_1/outputs/ngen")
    ap.add_argument("--net-dir", default="net_real_309k")
    ap.add_argument("--out", default="net_real_309k/forcings.bin")
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = auto (os.cpu_count())")
    args = ap.parse_args()

    topo_to_wbid = np.load(os.path.join(args.net_dir, "topo_to_wbid.npy"))
    n_reaches = len(topo_to_wbid)
    wbid_to_topo = {int(w): i for i, w in enumerate(topo_to_wbid)}

    # List all paths
    t0 = time.perf_counter()
    paths = [
        os.path.join(args.csv_dir, name)
        for name in os.listdir(args.csv_dir)
        if name.startswith("nex-") and name.endswith("_output.csv")
    ]
    t1 = time.perf_counter()
    print(f"[thr] listed {len(paths)} files in {(t1-t0)*1000:.1f} ms")

    n_workers = args.threads if args.threads > 0 else os.cpu_count() or 4
    qlat = np.zeros((args.timesteps, n_reaches), dtype=np.float32)

    t2 = time.perf_counter()
    matched = 0
    missing = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Use executor.map for lazy streaming (no 142K futures in memory at once)
        for (wbid, vals) in ex.map(parse_one_file, paths, chunksize=128):
            if wbid < 0 or wbid not in wbid_to_topo:
                missing += 1
                continue
            tidx = wbid_to_topo[wbid]
            m = min(args.timesteps, len(vals))
            qlat[:m, tidx] = vals[:m]
            matched += 1
    t3 = time.perf_counter()
    parse_sec = t3 - t2
    print(f"[thr] parsed {matched} csv (missing {missing}) in {parse_sec*1000:.1f} ms "
          f"using {n_workers} threads ({len(paths)/parse_sec:.0f} files/s)")

    # Write forcings.bin
    t4 = time.perf_counter()
    qdp0 = np.zeros(n_reaches, dtype=np.float32)
    dp0 = np.ones(n_reaches, dtype=np.float32) * 0.5
    with open(args.out, "wb") as f:
        f.write(struct.pack("<i", args.timesteps))
        f.write(qlat.astype(np.float32).tobytes())
        f.write(np.zeros(n_reaches, dtype=np.float32).tobytes())
        f.write(qdp0.tobytes())
        f.write(dp0.tobytes())
    t5 = time.perf_counter()
    print(f"[thr] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) in {(t5-t4)*1000:.1f} ms")
    print(f"[thr] TOTAL end-to-end: {(t5-t0)*1000:.1f} ms")


if __name__ == "__main__":
    main()
