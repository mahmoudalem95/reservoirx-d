"""
Directed graph construction with exact degree sequences.

Everything here builds *simple* digraphs: no self-loops, no parallel edges,
and the requested in/out degree sequence is honoured exactly.

Why this module exists at all
----------------------------
NetworkX has no directed analogue of `random_regular_graph`. The tempting
workaround -- `nx.random_regular_graph(d, n).to_directed()` -- produces a
perfectly symmetric adjacency matrix, which destroys the directedness that
the whole ReservoirX-D thesis depends on. So we build our own.

Method: configuration-model stub pairing, followed by directed double-edge
swaps that repair self-loops and parallel edges. Swaps preserve the degree
sequence exactly, so the repaired graph still has the degrees you asked for.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

__all__ = [
    "GraphConstructionError",
    "simple_digraph_from_degrees",
    "regular_degree_sequence",
    "near_regular_degree_sequence",
    "skewed_degree_sequence",
    "erdos_renyi_digraph",
    "edges_to_matrix",
    "degree_cv",
]


class GraphConstructionError(RuntimeError):
    """Raised when a simple digraph with the requested degrees cannot be built."""


# --------------------------------------------------------------------------
# degree sequences
# --------------------------------------------------------------------------


def regular_degree_sequence(n: int, d: int) -> np.ndarray:
    """Constant degree `d` on every node. CV is exactly 0."""
    if not (0 < d < n):
        raise GraphConstructionError(
            f"degree d={d} must satisfy 0 < d < n (n={n}); a simple digraph "
            f"cannot give a node more than n-1 distinct targets"
        )
    return np.full(n, d, dtype=np.int64)


def near_regular_degree_sequence(n: int, mean_degree: float, rng: np.random.Generator) -> np.ndarray:
    """
    Fractional mean degree with the lowest CV achievable on integers.

    A strictly degree-regular digraph requires an integer degree. When a caller
    asks for mean_degree=2.7 we split the nodes between floor=2 and ceil=3 so
    the mean is exactly 2.7 and only two distinct degrees ever appear. CV stays
    near zero (~0.17 at 2.7) rather than the ~1.2 of a genuinely skewed graph.
    """
    lo = int(math.floor(mean_degree))
    hi = lo + 1
    if not (0 < mean_degree < n):
        raise GraphConstructionError(f"mean_degree={mean_degree} must satisfy 0 < mean_degree < n (n={n})")
    total = int(round(mean_degree * n))
    n_hi = total - lo * n
    if n_hi < 0 or n_hi > n:
        raise GraphConstructionError(f"cannot realise mean_degree={mean_degree} on n={n} nodes")
    seq = np.full(n, lo, dtype=np.int64)
    if n_hi:
        idx = rng.choice(n, size=n_hi, replace=False)
        seq[idx] = hi
    if seq.min() < 1 or seq.max() > n - 1:
        raise GraphConstructionError(f"degree sequence out of range [1, {n - 1}] for n={n}")
    return seq


def skewed_degree_sequence(
    n: int,
    mean_degree: float,
    rng: np.random.Generator,
    target_cv: float = 1.2,
    min_degree: int = 1,
) -> np.ndarray:
    """
    Log-normal degree sequence with a controlled coefficient of variation.

    This is the *contrast* condition: same edge budget, heavy-tailed degrees.
    `target_cv` defaults to 1.2 to match the skew present in the CIFAR sparse-CNN
    baseline (CV=1.137-1.212 in the benchmark logs).

    The realised CV is usually a little below target because of integer
    rounding and the clip to [min_degree, n-1]. The achieved value is what
    diagnostics report -- never assume the target was hit.
    """
    if target_cv <= 0:
        raise GraphConstructionError("target_cv must be > 0 for a skewed sequence")
    total = int(round(mean_degree * n))
    if total < n * min_degree:
        raise GraphConstructionError(
            f"edge budget {total} cannot give every node at least min_degree={min_degree}"
        )

    sigma = math.sqrt(math.log(1.0 + target_cv**2))
    mu = math.log(max(mean_degree, 1e-9)) - 0.5 * sigma**2

    seq = None
    for _ in range(64):
        raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
        cand = np.clip(np.rint(raw), min_degree, n - 1).astype(np.int64)
        cand = _force_sum(cand, total, lo=min_degree, hi=n - 1, rng=rng)
        if cand is not None:
            seq = cand
            break
    if seq is None:
        raise GraphConstructionError(
            f"could not realise a skewed sequence with sum={total} on n={n} nodes"
        )
    return seq


def _force_sum(
    seq: np.ndarray, total: int, lo: int, hi: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Nudge integers up or down until they sum to `total`, respecting [lo, hi]."""
    seq = seq.copy()
    guard = 200 * len(seq)
    while seq.sum() != total and guard > 0:
        guard -= 1
        delta = total - int(seq.sum())
        if delta > 0:
            movable = np.flatnonzero(seq < hi)
            if movable.size == 0:
                return None
            seq[rng.choice(movable)] += 1
        else:
            movable = np.flatnonzero(seq > lo)
            if movable.size == 0:
                return None
            seq[rng.choice(movable)] -= 1
    return seq if seq.sum() == total else None


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def simple_digraph_from_degrees(
    out_deg: Sequence[int],
    in_deg: Sequence[int],
    rng: np.random.Generator,
    max_restarts: int = 12,
    max_swaps_per_bad_edge: int = 400,
) -> np.ndarray:
    """
    Build a simple digraph realising `out_deg` and `in_deg` exactly.

    Returns an (E, 2) int array of edges. Raises GraphConstructionError if the
    degree sequences are infeasible or repair fails on every restart.
    """
    out_deg = np.asarray(out_deg, dtype=np.int64)
    in_deg = np.asarray(in_deg, dtype=np.int64)
    n = out_deg.size

    if in_deg.size != n:
        raise GraphConstructionError("out_deg and in_deg must have the same length")
    if out_deg.sum() != in_deg.sum():
        raise GraphConstructionError(
            f"degree sequences must have equal sums (out={out_deg.sum()}, in={in_deg.sum()})"
        )
    if out_deg.max(initial=0) > n - 1 or in_deg.max(initial=0) > n - 1:
        raise GraphConstructionError(f"no node can exceed degree n-1={n - 1} in a simple digraph")
    if (out_deg < 0).any() or (in_deg < 0).any():
        raise GraphConstructionError("degrees must be non-negative")

    for _ in range(max_restarts):
        edges = _pair_stubs(out_deg, in_deg, rng)
        if _repair(edges, rng, max_swaps_per_bad_edge):
            return edges
    raise GraphConstructionError(
        "failed to repair the graph into a simple digraph; the degree sequence is "
        "likely too dense or too heavy-tailed for n nodes"
    )


def _pair_stubs(out_deg: np.ndarray, in_deg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    src = np.repeat(np.arange(out_deg.size), out_deg)
    dst = np.repeat(np.arange(in_deg.size), in_deg)
    rng.shuffle(src)
    rng.shuffle(dst)
    return np.column_stack([src, dst])


def _repair(edges: np.ndarray, rng: np.random.Generator, max_swaps_per_bad_edge: int) -> bool:
    """
    Remove self-loops and parallel edges with directed double-edge swaps.

    A swap takes (u,v) and (x,y) to (u,y) and (x,v). Every node keeps its
    out-degree and in-degree, so the sequence survives untouched.
    """
    counts = Counter(map(tuple, edges))
    n_edges = len(edges)

    def bad_indices() -> list[int]:
        return [
            i
            for i, (u, v) in enumerate(map(tuple, edges))
            if u == v or counts[(u, v)] > 1
        ]

    bad = bad_indices()
    budget = max_swaps_per_bad_edge * max(len(bad), 1)

    while bad and budget > 0:
        budget -= 1
        b = bad[int(rng.integers(len(bad)))]
        u, v = int(edges[b, 0]), int(edges[b, 1])
        k = int(rng.integers(n_edges))
        if k == b:
            continue
        x, y = int(edges[k, 0]), int(edges[k, 1])

        # proposed replacements must be loop-free and not already present
        if u == y or x == v:
            continue
        if counts[(u, y)] > 0 or counts[(x, v)] > 0:
            continue

        counts[(u, v)] -= 1
        counts[(x, y)] -= 1
        edges[b] = (u, y)
        edges[k] = (x, v)
        counts[(u, y)] += 1
        counts[(x, v)] += 1

        bad = bad_indices()

    return not bad


def erdos_renyi_digraph(n: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random simple digraph with exactly `n_edges` edges (null model)."""
    max_edges = n * (n - 1)
    if n_edges > max_edges:
        raise GraphConstructionError(f"cannot place {n_edges} edges in a simple digraph on {n} nodes")
    chosen = rng.choice(max_edges, size=n_edges, replace=False)
    src = chosen // (n - 1)
    off = chosen % (n - 1)
    dst = off + (off >= src)  # skip the diagonal
    return np.column_stack([src, dst]).astype(np.int64)


def edges_to_matrix(edges: np.ndarray, n: int) -> np.ndarray:
    """Binary adjacency mask, A[i, j] = 1 for edge i -> j."""
    A = np.zeros((n, n), dtype=np.float64)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = 1.0
    return A


def degree_cv(degrees: np.ndarray) -> float:
    """Coefficient of variation; 0 means perfectly regular."""
    degrees = np.asarray(degrees, dtype=np.float64)
    mean = degrees.mean()
    if mean <= 0:
        return 0.0
    return float(degrees.std() / mean)
