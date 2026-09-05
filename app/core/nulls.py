"""
Null models for /diagnose.

Two nulls, because they answer different questions:

* `erdos_renyi` holds only the edge count fixed. A metric that is extreme here
  might be extreme purely because of the degree sequence.
* `degree_preserving` holds the exact in/out degree sequence fixed and rewires
  everything else. A metric still extreme against this null is telling you
  something about wiring beyond degrees -- which is the interesting case.
"""

from __future__ import annotations

import numpy as np

from .graphs import edges_to_matrix, erdos_renyi_digraph
from .metrics import compute_metrics

__all__ = ["NULL_MODELS", "run_null_analysis", "randomize_preserving_degrees"]

NULL_MODELS: tuple[str, ...] = ("erdos_renyi", "degree_preserving")

# Metrics whose null distribution is degenerate or meaningless to compare.
_SKIP = {"n_nodes", "n_edges", "density", "mean_out_degree", "numerical_rank"}


def randomize_preserving_degrees(
    edges: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10
) -> np.ndarray:
    """
    Directed double-edge swap randomisation.

    Repeatedly replaces (u,v),(x,y) with (u,y),(x,v). Both degree sequences are
    invariant under this move, so the result is a uniform-ish draw from the set
    of simple digraphs with the observed degrees.
    """
    edges = edges.copy()
    n_edges = len(edges)
    if n_edges < 2:
        return edges
    present = {(int(a), int(b)) for a, b in edges}
    target_swaps = swaps_per_edge * n_edges
    attempts = 0
    done = 0
    while done < target_swaps and attempts < 20 * target_swaps:
        attempts += 1
        i, j = rng.integers(0, n_edges, size=2)
        if i == j:
            continue
        u, v = int(edges[i, 0]), int(edges[i, 1])
        x, y = int(edges[j, 0]), int(edges[j, 1])
        if u == y or x == v:
            continue
        if (u, y) in present or (x, v) in present:
            continue
        present.discard((u, v))
        present.discard((x, y))
        present.add((u, y))
        present.add((x, v))
        edges[i] = (u, y)
        edges[j] = (x, v)
        done += 1
    return edges


def run_null_analysis(
    W: np.ndarray,
    observed: dict[str, float],
    models: list[str],
    iterations: int,
    seed: int | None = None,
) -> dict[str, dict]:
    """
    Compare observed metrics against each null model.

    Returns, per model and per metric, the null mean and std, a Z-score, and a
    two-sided empirical p-value using the (r+1)/(m+1) correction.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    mask = W != 0.0
    edges = np.column_stack(np.nonzero(mask)).astype(np.int64)
    weights = W[mask]

    results: dict[str, dict] = {}
    for model in models:
        draws: list[dict[str, float]] = []
        for _ in range(iterations):
            if model == "erdos_renyi":
                new_edges = erdos_renyi_digraph(n, len(edges), rng)
            elif model == "degree_preserving":
                new_edges = randomize_preserving_degrees(edges, rng, swaps_per_edge=5)
            else:
                raise ValueError(f"unknown null model {model!r}")

            A = edges_to_matrix(new_edges, n)
            # reuse the observed weight multiset so only topology varies
            shuffled = rng.permutation(weights)
            A[new_edges[:, 0], new_edges[:, 1]] = shuffled[: len(new_edges)]
            draws.append(compute_metrics(A))

        results[model] = _summarise(observed, draws, iterations)
    return results


def _summarise(observed: dict[str, float], draws: list[dict[str, float]], iterations: int) -> dict:
    out: dict[str, dict] = {}
    for key, obs in observed.items():
        if key in _SKIP or not isinstance(obs, (int, float)):
            continue
        values = np.array([d.get(key, np.nan) for d in draws], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2 or not np.isfinite(obs):
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        z = float((obs - mean) / std) if std > 1e-12 else 0.0
        n_extreme = int(np.sum(np.abs(values - mean) >= abs(obs - mean) - 1e-15))
        out[key] = {
            "observed": float(obs),
            "null_mean": mean,
            "null_std": std,
            "z_score": z,
            "p_value": float((n_extreme + 1) / (values.size + 1)),
        }
    return {"iterations": iterations, "metrics": out}
