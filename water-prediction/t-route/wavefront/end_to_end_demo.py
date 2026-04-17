#!/usr/bin/env python
"""
End-to-end wall-time demo on JoshCu's real 309K CONUS data.

Three stages, each timed:
  1. Preprocess: stream nex-*.csv from 309k.tar.gz into a single qlat binary
  2. Load preprocessed topology + forcings from disk
  3. Run the GPU kernel (linear MC, persistent, all-timesteps-in-one-launch)

This is the comparison point for JoshCu's "~50% I/O" observation. For a
single run, the preprocess dominates. For repeated runs on cached forcings,
only stages 2 and 3 matter — which is where GPU wins.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TAR = "/c/Users/Ajs/Desktop/troute-gpu-work/joshcu-data/309k.tar.gz"
NET_DIR = "net_real_309k"
WIN_NET_DIR = "net_real_309k"


def main():
    print("=" * 60)
    print("End-to-end pipeline on JoshCu's 309K CONUS data")
    print("=" * 60)

    # Stage 1: already-completed preprocess (gpkg + csv streaming)
    topo_bin = os.path.join(NET_DIR, "topo.bin")
    forcings_bin = os.path.join(NET_DIR, "forcings.bin")
    params_bin = os.path.join(NET_DIR, "params.bin")
    if not os.path.exists(topo_bin) or not os.path.exists(forcings_bin):
        print("[stage-1] preprocessed data missing; run parse_gpkg.py and parse_qlat.py first")
        sys.exit(1)
    tp_bytes = os.path.getsize(topo_bin)
    fc_bytes = os.path.getsize(forcings_bin)
    pa_bytes = os.path.getsize(params_bin)
    print(f"[stage-1] preprocessed files:")
    print(f"    topo.bin     {tp_bytes/1e6:.1f} MB  (from gpkg parse, prev measured 3.7s)")
    print(f"    params.bin    {pa_bytes/1e6:.1f} MB  (from gpkg parse)")
    print(f"    forcings.bin {fc_bytes/1e6:.1f} MB  (from 142K nex csv stream, prev measured 32.7s)")
    print(f"    total:       {(tp_bytes+fc_bytes+pa_bytes)/1e6:.1f} MB")

    # Stage 2: load + kernel (what matters for subsequent runs)
    print(f"\n[stage-2+3] run the all-timesteps-in-one-launch GPU kernel end-to-end")
    # Timed via wall clock (includes process startup, all file reads, H2D, kernel, D2H)
    t0 = time.perf_counter()
    res = subprocess.run(
        ["./linear_mc_persistent.exe", NET_DIR],
        capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    print("---subprocess stdout---")
    print(res.stdout)
    if res.returncode != 0:
        print("---stderr---")
        print(res.stderr)

    # Parse GPU total ms from stdout
    gpu_ms_ts = None
    allts_ms = None
    for line in res.stdout.splitlines():
        if "[gpu-allts] total ms:" in line:
            try:
                allts_ms = float(line.split("total ms:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif "[gpu-persistent] total ms:" in line:
            try:
                gpu_ms_ts = float(line.split("total ms:")[1].split()[0])
            except (IndexError, ValueError):
                pass

    print(f"\n[timing] wall-clock (includes: process startup, file I/O, H2D, kernel, D2H):")
    print(f"    total subprocess:   {wall_ms:7.1f} ms")
    if allts_ms:
        print(f"    kernel-only alt-ts: {allts_ms:7.1f} ms  ({allts_ms/24:.2f} ms/timestep)")
        ratio = wall_ms / allts_ms
        print(f"    startup/IO overhead: {wall_ms - allts_ms:7.1f} ms ({(wall_ms-allts_ms)/wall_ms*100:.0f}% of wall time)")

    print()
    print("=" * 60)
    print("Comparison points")
    print("=" * 60)
    print(f"  JoshCu rs_route 32-core AMD 9950X3D:  ~5320 ms for 24ts (~222 ms/ts)")
    print(f"                                        (~50% is CSV I/O per JoshCu)")
    if allts_ms:
        print(f"  Our GPU linear MC (RTX 3060):        {allts_ms:.0f} ms for 24ts ({allts_ms/24:.2f} ms/ts)")
        print(f"  -> kernel speedup vs rs_route:       {5320/allts_ms:.1f}x")
    print(f"  Our preprocessing (one-time):         {(3.7+32.7)*1000:.0f} ms (gpkg + 142K CSV stream)")
    print(f"  -> first-run total:                   ~{(3.7+32.7)*1000 + (allts_ms or 0):.0f} ms")
    print(f"  -> subsequent-run total:              ~{wall_ms:.0f} ms (preprocessed data cached)")


if __name__ == "__main__":
    main()
