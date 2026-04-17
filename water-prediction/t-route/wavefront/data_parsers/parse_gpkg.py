#!/usr/bin/env python
"""
Parse JoshCu's big_one_fixed_subset.gpkg into our topology format.

Approach:
  - Read the flowpaths table: each row is a "waterbody" (wb-N) that routes
    downstream through a nexus (nex-M). Per JoshCu, nexus-to-wb mapping is
    1:1 so nex-M feeds wb-M. So the edge is wb-X.toid=nex-M implies flow
    goes wb-X -> wb-M.
  - Some wb's have toid pointing to a terminal nexus (tnx-* or tnex-*) with
    no downstream wb — those are outlet reaches.
  - Join flowpath-attributes for channel parameters (Length_m, n, BtmWdth,
    TopWdth, TopWdthCC, nCC, ChSlp, So).
  - Sort reaches topologically (by `hydroseq` which is NWM's topological
    order; smaller = further downstream, so we iterate in reverse).
  - Build CSR upstream adjacency, assign levels via Kahn's, emit our
    binary format compatible with wavefront_mc / linear_mc.
"""
from __future__ import annotations

import sqlite3
import struct
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def strip_prefix(s: str) -> int | None:
    if s is None:
        return None
    # split on hyphen — "wb-123", "nex-123", "tnx-123", "cat-123"
    parts = s.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def main():
    gpkg = "extract/big_one_fixed_1/config/big_one_fixed_subset.gpkg"
    out_dir = "net_real_309k"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[parse] opening {gpkg}")
    con = sqlite3.connect(gpkg)
    cur = con.cursor()

    # Read flowpaths: id -> toid, plus hydroseq, lengthkm, areasqkm
    print("[parse] reading flowpaths")
    cur.execute("SELECT id, toid, hydroseq, lengthkm, areasqkm FROM flowpaths")
    rows = cur.fetchall()
    print(f"[parse]  {len(rows)} flowpaths")

    # Build id -> info map
    wb_ids = []  # integer ids
    wb_toids = []
    wb_hydroseq = []
    wb_lengthkm = []
    wb_areasqkm = []
    for id_s, toid_s, hydroseq, lengthkm, areasqkm in rows:
        wid = strip_prefix(id_s)
        tid = strip_prefix(toid_s)  # this is nex-N or tnex-N
        if wid is None:
            continue
        wb_ids.append(wid)
        wb_toids.append(tid)
        wb_hydroseq.append(hydroseq)
        wb_lengthkm.append(lengthkm or 0.0)
        wb_areasqkm.append(areasqkm or 0.0)

    wb_ids_arr = np.array(wb_ids, dtype=np.int64)
    wb_toids_arr = np.array([-1 if x is None else x for x in wb_toids], dtype=np.int64)
    wb_hydroseq_arr = np.array([0 if x is None else x for x in wb_hydroseq], dtype=np.int64)

    # Check for duplicate ids
    unique_ids, counts = np.unique(wb_ids_arr, return_counts=True)
    if (counts > 1).any():
        dup = unique_ids[counts > 1]
        print(f"[parse]  WARNING: {len(dup)} duplicate wb ids; keeping first of each")
        seen = set()
        keep = np.zeros(len(wb_ids_arr), dtype=bool)
        for i, w in enumerate(wb_ids_arr):
            if int(w) not in seen:
                seen.add(int(w))
                keep[i] = True
        wb_ids_arr = wb_ids_arr[keep]
        wb_toids_arr = wb_toids_arr[keep]
        wb_hydroseq_arr = wb_hydroseq_arr[keep]
        wb_lengthkm = [x for i, x in enumerate(wb_lengthkm) if keep[i]]
        wb_areasqkm = [x for i, x in enumerate(wb_areasqkm) if keep[i]]

    n = len(wb_ids_arr)
    print(f"[parse]  unique wb count: {n}")

    # Build id -> local index
    idx_of = {int(w): i for i, w in enumerate(wb_ids_arr)}

    # Edges: wb-X -> nex-M with M being a wb id (per JoshCu's mapping)
    # So parent(X) has downstream = M (if M is a known wb).
    # Build upstream list per wb.
    upstreams: list[list[int]] = [[] for _ in range(n)]
    terminal_count = 0
    for i, tid in enumerate(wb_toids_arr):
        if tid < 0:
            continue
        # Downstream wb id is `tid` (because nex-M maps to wb-M)
        d = idx_of.get(int(tid))
        if d is None:
            terminal_count += 1
            continue
        upstreams[d].append(i)

    print(f"[parse]  edges wired: {sum(len(u) for u in upstreams)} | terminals: {terminal_count}")
    print(f"[parse]  fan-in distribution:")
    fan = np.bincount([len(u) for u in upstreams])
    for k, c in enumerate(fan):
        if c > 0:
            print(f"    {k} upstreams: {c} reaches")

    # Topological order. Use hydroseq sort (descending — larger hydroseq = upstream).
    # But hydroseq may not be strictly consistent for distributed subsets; fall back
    # to Kahn's over our derived graph.
    print("[parse] topological sort via Kahn's")
    in_deg = np.array([len(u) for u in upstreams], dtype=np.int32)
    # Build downstream list for Kahn
    downs: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        for u in upstreams[v]:
            downs[u].append(v)

    topo_order = []
    ready = [i for i in range(n) if in_deg[i] == 0]
    while ready:
        u = ready.pop()
        topo_order.append(u)
        for d in downs[u]:
            in_deg[d] -= 1
            if in_deg[d] == 0:
                ready.append(d)

    if len(topo_order) != n:
        raise RuntimeError(
            f"Topo sort incomplete: {len(topo_order)} / {n} — network has cycles?"
        )

    # Relabel to topological order
    old_to_new = np.empty(n, dtype=np.int64)
    for new_idx, old_idx in enumerate(topo_order):
        old_to_new[old_idx] = new_idx

    new_upstreams: list[list[int]] = [[] for _ in range(n)]
    for old_idx in range(n):
        new_idx = int(old_to_new[old_idx])
        new_upstreams[new_idx] = sorted(int(old_to_new[u]) for u in upstreams[old_idx])

    # Assign levels
    level = np.zeros(n, dtype=np.int32)
    for r in range(n):
        ups = new_upstreams[r]
        if ups:
            level[r] = 1 + max(level[u] for u in ups)
    n_levels = int(level.max()) + 1
    level_counts = np.bincount(level, minlength=n_levels).astype(np.int32)
    level_ptr = np.concatenate([[0], np.cumsum(level_counts)]).astype(np.int32)
    level_reach = np.argsort(level, kind="stable").astype(np.int32)

    print(f"[parse]  n_levels={n_levels} widest level has {int(level_counts.max())} reaches")
    # Level-size distribution
    print(f"[parse]  level-size head: {level_counts[:15].tolist()}")
    deep_tail = level_counts[20:] if n_levels > 20 else np.array([])
    print(f"[parse]  deep-tail (level 20+) levels: {len(deep_tail)}, mean size: {deep_tail.mean() if len(deep_tail) else 0:.1f}")

    # Upstream CSR
    reach_n_up = np.array([len(new_upstreams[r]) for r in range(n)], dtype=np.int32)
    reach_up_ptr = np.concatenate([[0], np.cumsum(reach_n_up)]).astype(np.int32)
    nnz = int(reach_n_up.sum())
    reach_up_idx = np.empty(nnz, dtype=np.int32)
    pos = 0
    for r in range(n):
        for u in new_upstreams[r]:
            reach_up_idx[pos] = u
            pos += 1

    # Read flowpath-attributes: join on id==wb-N
    print("[parse] reading flowpath-attributes")
    cur.execute(
        'SELECT link, "Length_m", "n", "nCC", "BtmWdth", "TopWdth", "TopWdthCC", "ChSlp", "So" '
        'FROM "flowpath-attributes"'
    )
    attrs = cur.fetchall()
    print(f"[parse]  {len(attrs)} attribute rows")
    con.close()

    # Default channel params (used if attribute missing)
    params = np.zeros((n, 8), dtype=np.float32)
    params[:, 0] = 1000.0  # dx
    params[:, 1] = 20.0    # bw
    params[:, 2] = 25.0    # tw
    params[:, 3] = 60.0    # twcc
    params[:, 4] = 0.06    # n
    params[:, 5] = 0.12    # ncc
    params[:, 6] = 2.0     # cs (ChSlp)
    params[:, 7] = 0.001   # s0

    filled = 0
    for link_s, length_m, n_ch, nCC, bw, tw, twcc, cs, s0 in attrs:
        wid = strip_prefix(link_s)
        if wid is None or wid not in idx_of:
            continue
        old_i = idx_of[wid]
        new_i = int(old_to_new[old_i])
        p = params[new_i]
        if length_m and length_m > 0:
            p[0] = max(100.0, float(length_m))
        if bw and bw > 0:
            p[1] = float(bw)
        if tw and tw > 0 and tw > p[1]:
            p[2] = float(tw)
        else:
            p[2] = p[1] * 1.2
        if twcc and twcc > 0 and twcc > p[2]:
            p[3] = float(twcc)
        else:
            p[3] = p[2] * 3.0
        if n_ch and n_ch > 0:
            p[4] = max(0.01, float(n_ch))
        if nCC and nCC > 0:
            p[5] = max(0.02, float(nCC))
        if cs and cs > 0:
            p[6] = max(0.1, float(cs))
        if s0 and s0 > 0:
            p[7] = max(1e-5, float(s0))
        filled += 1

    print(f"[parse]  attributes matched to {filled}/{n} reaches")

    # Each reach is a single segment (1 segment per wb). reach_seg_start = i, reach_seg_len = 1
    reach_seg_start = np.arange(n, dtype=np.int32)
    reach_seg_len = np.ones(n, dtype=np.int32)

    # Write topology binary (same format as gen_network.py)
    print(f"[parse] writing to {out_dir}/")
    with open(os.path.join(out_dir, "topo.bin"), "wb") as f:
        f.write(struct.pack("<iiii", n, n, n_levels, int(reach_n_up.max())))
        f.write(reach_seg_start.astype(np.int32).tobytes())
        f.write(reach_seg_len.astype(np.int32).tobytes())
        f.write(level.astype(np.int32).tobytes())
        f.write(reach_n_up.astype(np.int32).tobytes())
        f.write(reach_up_ptr.astype(np.int32).tobytes())
        f.write(reach_up_idx.astype(np.int32).tobytes())
        f.write(level_ptr.astype(np.int32).tobytes())
        f.write(level_reach.astype(np.int32).tobytes())

    with open(os.path.join(out_dir, "params.bin"), "wb") as f:
        f.write(params.astype(np.float32).tobytes())

    # Save the id mapping so we can join qlat later
    # old_to_new: for each input wb index, its new topo index
    # We also need: for each topo index, the original wb integer id
    topo_to_wbid = np.empty(n, dtype=np.int64)
    for old_idx in range(n):
        new_idx = int(old_to_new[old_idx])
        topo_to_wbid[new_idx] = int(wb_ids_arr[old_idx])
    np.save(os.path.join(out_dir, "topo_to_wbid.npy"), topo_to_wbid)
    print(f"[parse]  wrote {out_dir}/topo.bin, params.bin, topo_to_wbid.npy")
    print(f"[parse] done")


if __name__ == "__main__":
    main()
