#!/usr/bin/env python
"""
Generate a synthetic NWM-like river network topology and channel parameters.

Produces reach-level adjacency in topological order, with realistic
tree-depth characteristics. Each reach is a sequence of channel segments.
Reaches are ordered so that every reach's index is greater than all its
upstream reaches' indices (topological order).

Writes three binary files consumed by wavefront_mc.cu:

  {out}/topo.bin
    Header (int32 little-endian):
      n_reaches, n_segments, n_levels, max_upstream
    Per-reach arrays (length n_reaches):
      reach_seg_start  int32   start index into segment arrays
      reach_seg_len    int32   number of segments in this reach
      reach_level      int32   topological level (0 = headwater)
      reach_n_up       int32   number of upstream reaches
    Upstream CSR (length nnz = sum(reach_n_up)):
      reach_up_ptr     int32[n_reaches+1]  CSR row pointer
      reach_up_idx     int32[nnz]          upstream reach indices
    Level CSR (length n_reaches):
      level_ptr        int32[n_levels+1]   start index in level_reach
      level_reach      int32[n_reaches]    reach indices per level

  {out}/params.bin
    n_segments float64 records, 8 doubles each:
      dx, bw, tw, twcc, n_ch, ncc, cs, s0

  {out}/forcings.bin
    For every timestep (n_timesteps):
      qlat[n_segments]  float32  lateral inflow per segment
    And per-reach at t=0:
      qup0[n_reaches]   float32  previous-timestep upstream flow
      qdp0[n_segments]  float32  previous-timestep reach-segment outflow
      dp0[n_segments]   float32  previous-timestep depth
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

# Force unbuffered stdout for Windows consoles
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


@dataclass
class NetworkSpec:
    n_reaches: int
    seed: int = 42
    # Chain length: number of sequential reaches before a confluence
    mean_chain: int = 3
    # Probability distribution over fan-in at confluences (0=no confluence, 1=single pred, 2-N=multi)
    # Real NWM: ~65% single-upstream, ~34% two-upstream, ~1% 3+
    # Headwaters are separate (level 0, no upstream)
    fanin_p: tuple[float, float, float] = (0.0, 0.65, 0.34)  # probs for fan-in 1,2,3
    mean_segments_per_reach: int = 12


def build_network(spec: NetworkSpec):
    rng = np.random.default_rng(spec.seed)

    # We build the network bottom-up (headwaters -> outlet) using a Shreve-like
    # construction. Start with a pool of "active" nodes (each is a single
    # headwater reach). Repeatedly: either extend a chain (one node -> one
    # longer node) OR merge two nodes into a confluence (two nodes -> one).
    # Continue until only N_outlets nodes remain (or just 1 outlet).

    # To control exact n_reaches, we instead precompute structure differently:
    # generate a tree via a Galton-Watson-like process and adjust.

    # Simpler, controllable method: generate by the "elongated binary tree" pattern.
    # - build a binary tree with B leaves where B = roughly n_reaches / (mean_chain)
    #   and number of internal confluences = B - 1
    # - then inject chains of mean_chain reaches along each edge
    #
    # Target n_reaches ≈ mean_chain * (2B - 1). Solve for B.

    B = max(2, int(round(spec.n_reaches / (spec.mean_chain * 2.0))))

    # --- build binary tree with B leaves ---
    # Use a simple iterative merge: start with B leaves, merge pairs at each level
    # until one root remains. With random pairings we'll get realistic depth.
    current = list(range(B))
    reach_counter = B
    downstream_of = [-1] * B  # node -> its immediate downstream node
    upstreams_of = [[] for _ in range(B)]  # node -> list of upstream nodes

    def new_node():
        nonlocal reach_counter
        idx = reach_counter
        reach_counter += 1
        downstream_of.append(-1)
        upstreams_of.append([])
        return idx

    while len(current) > 1:
        rng.shuffle(current)
        next_level = []
        i = 0
        while i < len(current):
            # Decide fan-in for the new confluence: 2 most of the time, 3 rarely, 1 rarely
            # (1 = pure chain extension at this level — usually not needed since we'll add chains later)
            r = rng.random()
            if r < spec.fanin_p[2] and i + 3 <= len(current):
                k = 3
            else:
                k = 2 if i + 2 <= len(current) else 1
            children = current[i : i + k]
            i += k
            if k == 1:
                # single carryover — just keep it in next_level
                next_level.append(children[0])
                continue
            parent = new_node()
            for c in children:
                downstream_of[c] = parent
                upstreams_of[parent].append(c)
            next_level.append(parent)
        current = next_level

    # Now the tree has reach_counter nodes.
    # Inject chains: for every edge (u -> d), insert random number of intermediate reaches.
    # Chain lengths follow Poisson(mean_chain - 1) so the edge itself counts as 1 reach.

    # Collect edges (upstream, downstream) by walking the tree
    edges: list[tuple[int, int]] = []
    for u in range(reach_counter):
        d = downstream_of[u]
        if d >= 0:
            edges.append((u, d))

    # After chain injection, replace each edge (u,d) with a chain:
    # u -> c1 -> c2 -> ... -> ck -> d
    # Where c1..ck are newly allocated reach indices.
    # Then link u->c1 (as upstream), and d gets ck (and removes u from its upstreams).

    # We'll operate directly on upstreams_of, so that new_node() (which appends
    # a new empty list to upstreams_of) keeps indices consistent.
    for u, d in edges:
        # chain length = edge (1) + Poisson(mean_chain - 1) extra segments
        extra = int(rng.poisson(max(0.0, spec.mean_chain - 1.0)))
        if extra == 0:
            continue  # no chain to inject; keep u->d directly

        # create extra new reaches as chain c1 .. c_extra
        chain = []
        for _ in range(extra):
            idx = new_node()
            chain.append(idx)

        # Before: u feeds d (d.upstreams contains u)
        # After: u -> c1 -> c2 -> ... -> c_extra -> d
        # Remove u from d.upstreams, add c_extra instead
        try:
            upstreams_of[d].remove(u)
        except ValueError:
            pass
        upstreams_of[d].append(chain[-1])

        # Wire the chain
        upstreams_of[chain[0]].append(u)
        for a, b in zip(chain[:-1], chain[1:]):
            upstreams_of[b].append(a)

        # downstream_of bookkeeping (not strictly needed after this)
        downstream_of[u] = chain[0]
        for a, b in zip(chain[:-1], chain[1:]):
            downstream_of[a] = b
        downstream_of[chain[-1]] = d

    total = reach_counter
    new_upstreams = upstreams_of

    # Truncate or pad? We want exactly spec.n_reaches. Most convenient: use what we have.
    # We'll return actual count.

    # Topological sort (Kahn's algorithm) using upstream counts
    in_deg = np.array([len(new_upstreams[i]) for i in range(total)], dtype=np.int32)
    topo_order: list[int] = []
    ready: list[int] = [i for i in range(total) if in_deg[i] == 0]

    # Build downstream adjacency for fast walking
    downs: list[list[int]] = [[] for _ in range(total)]
    for r in range(total):
        for u in new_upstreams[r]:
            downs[u].append(r)

    while ready:
        # pop in random order (preserves determinism under seed)
        j = int(rng.integers(0, len(ready)))
        ready[-1], ready[j] = ready[j], ready[-1]
        u = ready.pop()
        topo_order.append(u)
        for d in downs[u]:
            in_deg[d] -= 1
            if in_deg[d] == 0:
                ready.append(d)

    if len(topo_order) != total:
        raise RuntimeError(f"Topological sort produced {len(topo_order)} nodes, expected {total}")

    # Relabel reaches to topo order
    old_to_new = np.empty(total, dtype=np.int64)
    for new_idx, old_idx in enumerate(topo_order):
        old_to_new[old_idx] = new_idx

    relabeled_up: list[list[int]] = [[] for _ in range(total)]
    for old_idx in range(total):
        new_idx = int(old_to_new[old_idx])
        relabeled_up[new_idx] = sorted(int(old_to_new[u]) for u in new_upstreams[old_idx])

    # Assign levels (0 = headwater)
    level = np.zeros(total, dtype=np.int32)
    for r in range(total):
        ups = relabeled_up[r]
        if ups:
            level[r] = 1 + max(level[u] for u in ups)

    n_levels = int(level.max()) + 1 if total > 0 else 0

    # Group reaches by level for level-scheduled launches
    level_reach = np.argsort(level, kind="stable").astype(np.int32)
    level_counts = np.bincount(level, minlength=n_levels).astype(np.int32)
    level_ptr = np.concatenate([[0], np.cumsum(level_counts)]).astype(np.int32)

    # Number of segments per reach: Poisson around mean, floor 1
    reach_seg_len = np.maximum(1, rng.poisson(spec.mean_segments_per_reach, size=total)).astype(
        np.int32
    )
    reach_seg_start = np.concatenate([[0], np.cumsum(reach_seg_len[:-1])]).astype(np.int32)
    n_segments = int(reach_seg_len.sum())

    # Upstream CSR
    reach_n_up = np.array([len(relabeled_up[r]) for r in range(total)], dtype=np.int32)
    reach_up_ptr = np.concatenate([[0], np.cumsum(reach_n_up)]).astype(np.int32)
    nnz = int(reach_n_up.sum())
    reach_up_idx = np.empty(nnz, dtype=np.int32)
    pos = 0
    for r in range(total):
        for u in relabeled_up[r]:
            reach_up_idx[pos] = u
            pos += 1

    return {
        "n_reaches": total,
        "n_segments": n_segments,
        "n_levels": n_levels,
        "max_upstream": int(reach_n_up.max()) if total > 0 else 0,
        "reach_seg_start": reach_seg_start,
        "reach_seg_len": reach_seg_len,
        "reach_level": level,
        "reach_n_up": reach_n_up,
        "reach_up_ptr": reach_up_ptr,
        "reach_up_idx": reach_up_idx,
        "level_ptr": level_ptr,
        "level_reach": level_reach,
    }


def generate_params(n_segments: int, seed: int):
    rng = np.random.default_rng(seed ^ 0x5A5A)
    # Channel parameters drawn from NWM-like ranges
    dx = rng.uniform(300.0, 5000.0, size=n_segments)          # reach length in meters
    bw = rng.uniform(5.0, 50.0, size=n_segments)              # bottom width
    tw = bw + rng.uniform(2.0, 20.0, size=n_segments)         # top width > bw
    twcc = tw + rng.uniform(5.0, 100.0, size=n_segments)      # compound
    n_ch = rng.uniform(0.025, 0.08, size=n_segments)          # Manning's n
    ncc = n_ch * rng.uniform(1.2, 2.0, size=n_segments)       # compound n
    cs = rng.uniform(0.5, 3.0, size=n_segments)               # side slope
    s0 = rng.uniform(1e-4, 5e-3, size=n_segments)             # bed slope

    params = np.stack([dx, bw, tw, twcc, n_ch, ncc, cs, s0], axis=1).astype(np.float32)
    return params  # (n_segments, 8) float32 — matches t-route Fortran REAL


def generate_forcings(n_reaches: int, n_segments: int, n_timesteps: int, seed: int):
    rng = np.random.default_rng(seed ^ 0xA1A1)
    qlat = rng.uniform(0.0, 2.0, size=(n_timesteps, n_segments)).astype(np.float32)
    qup0 = rng.uniform(0.0, 10.0, size=n_reaches).astype(np.float32)
    qdp0 = rng.uniform(0.0, 15.0, size=n_segments).astype(np.float32)
    dp0 = rng.uniform(0.3, 3.0, size=n_segments).astype(np.float32)
    return qlat, qup0, qdp0, dp0


def write_network(out_dir: str, net: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "topo.bin"), "wb") as f:
        f.write(
            struct.pack(
                "<iiii",
                net["n_reaches"],
                net["n_segments"],
                net["n_levels"],
                net["max_upstream"],
            )
        )
        for key in (
            "reach_seg_start",
            "reach_seg_len",
            "reach_level",
            "reach_n_up",
        ):
            f.write(net[key].astype(np.int32).tobytes())
        # Upstream CSR
        f.write(net["reach_up_ptr"].astype(np.int32).tobytes())
        f.write(net["reach_up_idx"].astype(np.int32).tobytes())
        # Level CSR
        f.write(net["level_ptr"].astype(np.int32).tobytes())
        f.write(net["level_reach"].astype(np.int32).tobytes())


def write_params(out_dir: str, params: np.ndarray):
    with open(os.path.join(out_dir, "params.bin"), "wb") as f:
        f.write(params.astype(np.float32).tobytes())


def write_forcings(
    out_dir: str,
    n_timesteps: int,
    qlat: np.ndarray,
    qup0: np.ndarray,
    qdp0: np.ndarray,
    dp0: np.ndarray,
):
    with open(os.path.join(out_dir, "forcings.bin"), "wb") as f:
        f.write(struct.pack("<i", n_timesteps))
        f.write(qlat.astype(np.float32).tobytes())
        f.write(qup0.astype(np.float32).tobytes())
        f.write(qdp0.astype(np.float32).tobytes())
        f.write(dp0.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaches", type=int, default=309_000, help="target reach count")
    ap.add_argument("--mean-chain", type=int, default=3)
    ap.add_argument("--mean-segs", type=int, default=12)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="net")
    args = ap.parse_args()

    spec = NetworkSpec(
        n_reaches=args.reaches,
        seed=args.seed,
        mean_chain=args.mean_chain,
        mean_segments_per_reach=args.mean_segs,
    )

    print(f"[gen] building network (target {args.reaches} reaches)...", flush=True)
    net = build_network(spec)
    print(
        f"[gen]  got n_reaches={net['n_reaches']} n_segments={net['n_segments']} "
        f"n_levels={net['n_levels']} max_upstream={net['max_upstream']}",
        flush=True,
    )

    print("[gen] generating channel params...", flush=True)
    params = generate_params(net["n_segments"], args.seed)

    print("[gen] generating forcings...", flush=True)
    qlat, qup0, qdp0, dp0 = generate_forcings(
        net["n_reaches"], net["n_segments"], args.timesteps, args.seed
    )

    print(f"[gen] writing to {args.out}/...", flush=True)
    write_network(args.out, net)
    write_params(args.out, params)
    write_forcings(args.out, args.timesteps, qlat, qup0, qdp0, dp0)

    # Level-size distribution (first 20 levels + tail summary)
    counts = np.bincount(net["reach_level"], minlength=net["n_levels"])
    print("[gen] level sizes (first 20):", counts[:20].tolist(), flush=True)
    print(
        f"[gen] parallel reaches at widest level: {int(counts.max())} "
        f"(at level {int(counts.argmax())})",
        flush=True,
    )
    print(f"[gen] tree depth (n_levels): {net['n_levels']}", flush=True)
    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
