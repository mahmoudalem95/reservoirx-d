"""
Structural metrics for a reservoir matrix.

Definitions are spelled out because several of them appear in the literature
with more than one convention, and a diagnostic you cannot compare against a
published number is not much of a diagnostic.
"""

from __future__ import annotations

import numpy as np

from .reservoir import spectral_radius

__all__ = ["compute_metrics", "METRIC_DEFINITIONS"]

METRIC_DEFINITIONS: dict[str, str] = {
    "density": "edges / (n * (n-1)); self-loops are excluded from the denominator",
    "n_edges": "count of non-zero entries",
    "mean_out_degree": "edges / n",
    "degree_cv_out": "std(out-degree) / mean(out-degree); 0 = every node emits the same number of edges",
    "degree_cv_in": "std(in-degree) / mean(in-degree); 0 = every node receives the same number",
    "degree_cv": "mean of degree_cv_out and degree_cv_in",
    "reciprocity": "reciprocated directed edges / total directed edges (Newman convention). "
    "A mutual pair counts as 2 reciprocated edges. 1.0 = fully symmetric topology.",
    "symmetry_index": "<W, W^T>_F / ||W||_F^2, in [-1, 1]. 1 = symmetric, 0 = no weight-level "
    "symmetry, -1 = antisymmetric. Unlike reciprocity this is weight-aware.",
    "spectral_radius": "max |eigenvalue|",
    "spectral_gap": "|lambda_1| - |lambda_2|",
    "henrici_departure": "sqrt(max(0, ||W||_F^2 - sum |lambda_i|^2)) -- the standard Henrici "
    "departure-from-normality. 0 for a normal matrix. Scale-dependent.",
    "henrici_normalised": "henrici_departure / ||W||_F, in [0, 1). Scale-free, so this is the one "
    "to compare across matrix sizes.",
    "normality_ratio": "||W^T W - W W^T||_F / ||W||_F^2. A second, cheaper departure measure. "
    "NOT the Henrici index -- reported separately so the two are never conflated.",
    "effective_rank": "exp(entropy of the normalised singular value spectrum). Preferred over "
    "numerical rank, which saturates at n for almost every random matrix and "
    "therefore carries no information.",
    "numerical_rank": "np.linalg.matrix_rank; retained for completeness, expect it to equal n",
}


def compute_metrics(W: np.ndarray, include_spectral: bool = True) -> dict[str, float]:
    """Compute the full structural metric set for a weighted adjacency matrix."""
    n = W.shape[0]
    mask = W != 0.0
    n_edges = int(mask.sum())

    out_deg = mask.sum(axis=1).astype(np.float64)
    in_deg = mask.sum(axis=0).astype(np.float64)

    metrics: dict[str, float] = {
        "n_nodes": float(n),
        "n_edges": float(n_edges),
        "density": float(n_edges / (n * (n - 1))) if n > 1 else 0.0,
        "mean_out_degree": float(n_edges / n) if n else 0.0,
        "degree_cv_out": _cv(out_deg),
        "degree_cv_in": _cv(in_deg),
    }
    metrics["degree_cv"] = 0.5 * (metrics["degree_cv_out"] + metrics["degree_cv_in"])

    # Newman reciprocity: reciprocated directed edges over all directed edges.
    reciprocated = int(np.sum(mask & mask.T))
    metrics["reciprocity"] = float(reciprocated / n_edges) if n_edges else 0.0

    fro_sq = float(np.sum(W * W))
    metrics["frobenius_norm"] = float(np.sqrt(fro_sq))
    metrics["symmetry_index"] = float(np.sum(W * W.T) / fro_sq) if fro_sq > 1e-30 else 0.0

    commutator = W.T @ W - W @ W.T
    metrics["normality_ratio"] = (
        float(np.linalg.norm(commutator, ord="fro") / fro_sq) if fro_sq > 1e-30 else 0.0
    )

    if include_spectral:
        eigenvalues = np.linalg.eigvals(W) if n <= 900 else None
        if eigenvalues is not None:
            moduli = np.sort(np.abs(eigenvalues))[::-1]
            metrics["spectral_radius"] = float(moduli[0])
            metrics["spectral_gap"] = float(moduli[0] - moduli[1]) if n > 1 else 0.0
            henrici_sq = max(0.0, fro_sq - float(np.sum(moduli**2)))
            metrics["henrici_departure"] = float(np.sqrt(henrici_sq))
        else:
            metrics["spectral_radius"] = spectral_radius(W)
            metrics["spectral_gap"] = float("nan")
            metrics["henrici_departure"] = float("nan")
        norm = metrics["frobenius_norm"]
        metrics["henrici_normalised"] = (
            float(metrics["henrici_departure"] / norm) if norm > 1e-30 else 0.0
        )

        svals = np.linalg.svd(W, compute_uv=False)
        metrics["effective_rank"] = _effective_rank(svals)
        metrics["numerical_rank"] = float(np.sum(svals > svals.max() * max(W.shape) * 1e-15))

    return metrics


def _cv(degrees: np.ndarray) -> float:
    mean = degrees.mean() if degrees.size else 0.0
    if mean <= 1e-12:
        return 0.0
    return float(degrees.std() / mean)


def _effective_rank(svals: np.ndarray) -> float:
    """exp(Shannon entropy of the normalised singular value distribution)."""
    total = svals.sum()
    if total <= 1e-30:
        return 0.0
    p = svals / total
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))
