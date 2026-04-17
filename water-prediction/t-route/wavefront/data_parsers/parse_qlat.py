#!/usr/bin/env python
"""
Stream nex-*_output.csv from 309k.tar.gz into a (timesteps, n_reaches) qlat tensor.

Avoids extracting 450K files to disk — reads directly from the tar stream.

Per JoshCu: nex-M_output.csv provides qlat for wb-M (1:1 mapping), with columns
    timestep, time, q_lateral_m3_per_s.

Output: forcings.bin in our binary format for the wavefront/linear kernels.
"""
from __future__ import annotations

import os
import re
import struct
import sys
import tarfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

NEX_RE = re.compile(r"/nex-(\d+)_output\.csv$")


def main():
    tar_path = "309k.tar.gz"
    out_dir = "net_real_309k"
    topo_to_wbid = np.load(os.path.join(out_dir, "topo_to_wbid.npy"))
    n_reaches = len(topo_to_wbid)

    # Build wbid -> topo index map (vectorized via dict — ~300K entries)
    wbid_to_topo = {int(wbid): i for i, wbid in enumerate(topo_to_wbid)}

    n_timesteps = 24  # per the dataset (we'll resize if needed)
    # Pre-allocate
    qlat = np.zeros((n_timesteps, n_reaches), dtype=np.float32)

    print(f"[qlat] streaming {tar_path} for nex-*_output.csv ({n_reaches} reaches)")
    t0 = time.time()
    matched = 0
    missing = 0
    processed = 0
    sample_timestamps = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for ti, member in enumerate(tar):
            if not member.isfile():
                continue
            m = NEX_RE.search(member.name)
            if not m:
                continue
            processed += 1
            wbid = int(m.group(1))
            tidx = wbid_to_topo.get(wbid)
            if tidx is None:
                missing += 1
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            # Parse: "ts, time, qlat" one per line
            lines = data.decode("ascii", errors="replace").splitlines()
            # Some files begin with a header row; detect by non-numeric first field
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                try:
                    ts = int(parts[0])
                    q = float(parts[-1])
                except ValueError:
                    continue
                if 0 <= ts < n_timesteps:
                    qlat[ts, tidx] = q
                if len(sample_timestamps) < 5 and ts == 0:
                    sample_timestamps.append(parts[1])
            matched += 1

            if processed % 25000 == 0:
                elapsed = time.time() - t0
                rate = processed / max(0.001, elapsed)
                print(f"  processed {processed:7d} nex csv, matched {matched}, missing {missing} ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print(f"[qlat] streamed {processed} nex csv in {elapsed:.1f}s ({processed/elapsed:.0f}/s)")
    print(f"[qlat]  matched={matched} missing={missing}")
    print(f"[qlat]  sample first-timestep times: {sample_timestamps[:3]}")
    print(f"[qlat]  qlat shape: {qlat.shape}, sum={qlat.sum():.3f}, nonzero frac={(qlat!=0).mean():.3f}")

    # Seed qdp0 and dp0 with zeros (cold-start baseline; real NWM cold-start uses upstream accumulation but 0 is fine for wall-time timing)
    qdp0 = np.zeros(n_reaches, dtype=np.float32)
    dp0 = np.ones(n_reaches, dtype=np.float32) * 0.5

    # reach_last0 seeded with 0 (first timestep's downstream state assumed 0)
    # (matches synthetic generator's "previous-timestep last-seg qdc")
    with open(os.path.join(out_dir, "forcings.bin"), "wb") as f:
        f.write(struct.pack("<i", n_timesteps))
        f.write(qlat.astype(np.float32).tobytes())
        f.write(np.zeros(n_reaches, dtype=np.float32).tobytes())   # qup0_reach (unused)
        f.write(qdp0.astype(np.float32).tobytes())
        f.write(dp0.astype(np.float32).tobytes())

    print(f"[qlat] wrote {out_dir}/forcings.bin ({os.path.getsize(os.path.join(out_dir, 'forcings.bin'))/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
