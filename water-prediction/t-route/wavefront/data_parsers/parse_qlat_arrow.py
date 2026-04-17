#!/usr/bin/env python
"""
pyarrow-backed parallel CSV preprocessor. pyarrow.csv.read_csv is
C-implemented and releases the GIL during parsing, so ThreadPoolExecutor
genuinely parallelizes.
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
import pyarrow as pa
import pyarrow.csv as pa_csv

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

NEX_RE = re.compile(r"nex-(\d+)_output\.csv$")


def parse_one(path: str) -> tuple[int, np.ndarray]:
    m = NEX_RE.search(path)
    if not m:
        return -1, np.empty(0, dtype=np.float32)
    wbid = int(m.group(1))
    try:
        # Our CSVs have no header, columns: "ts, time, qlat"
        read_opts = pa_csv.ReadOptions(
            column_names=["ts", "time", "qlat"], use_threads=False
        )
        parse_opts = pa_csv.ParseOptions(delimiter=",")
        conv_opts = pa_csv.ConvertOptions(
            column_types={"ts": pa.int32(), "qlat": pa.float32()},
            include_columns=["qlat"],
        )
        tbl = pa_csv.read_csv(path, read_options=read_opts,
                              parse_options=parse_opts,
                              convert_options=conv_opts)
        vals = tbl.column("qlat").to_numpy().astype(np.float32, copy=False)
        return wbid, vals
    except Exception:
        return -1, np.empty(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="nex_csvs/big_one_fixed_1/outputs/ngen")
    ap.add_argument("--net-dir", default="net_real_309k")
    ap.add_argument("--out", default="net_real_309k/forcings.bin")
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
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
    print(f"[arw] listed {len(paths)} files in {(t1-t0)*1000:.1f} ms")

    n_workers = args.threads if args.threads > 0 else os.cpu_count() or 4
    qlat = np.zeros((args.timesteps, n_reaches), dtype=np.float32)

    t2 = time.perf_counter()
    matched = 0
    missing = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for (wbid, vals) in ex.map(parse_one, paths, chunksize=128):
            if wbid < 0 or wbid not in wbid_to_topo:
                missing += 1
                continue
            tidx = wbid_to_topo[wbid]
            m = min(args.timesteps, len(vals))
            qlat[:m, tidx] = vals[:m]
            matched += 1
    t3 = time.perf_counter()
    print(f"[arw] parsed {matched} csv (missing {missing}) in {(t3-t2)*1000:.1f} ms "
          f"using {n_workers} threads ({len(paths)/(t3-t2):.0f} files/s)")

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
    print(f"[arw] wrote {args.out} in {(t5-t4)*1000:.1f} ms")
    print(f"[arw] TOTAL: {(t5-t0)*1000:.1f} ms")


if __name__ == "__main__":
    main()
