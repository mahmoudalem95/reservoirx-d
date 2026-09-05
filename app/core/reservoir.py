"""Reservoir weight matrix generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .graphs import (
    GraphConstructionError,
    edges_to_matrix,
    erdos_renyi_digraph,
    near_regular_degree_sequence,
    regular_degree_sequence,
    simple_digraph_from_degrees,
    skewed_degree_sequence,
)

WiringScheme = Literal["degree_regular", "near_regular", "skewed", "erdos_renyi"]
WeightDistribution = Literal["uniform", "normal", "bimodal"]

DENSE_EIGEN_LIMIT = 900  # above this we switch to sparse ARPACK


@dataclass
class ReservoirMatrix:
    """A generated reservoir plus the parameters that produced it."""

    W: np.ndarray
    n: int
    wiring_scheme: str
    mean_degree: float
    spectral_radius_target: float
    spectral_radius_achieved: float
    weight_distribution: str
    seed: int | None
    n_edges: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return (self.W != 0.0).astype(np.float64)


def build_adjacency(
    n: int,
    mean_degree: float,
    wiring_scheme: WiringScheme,
    rng: np.random.Generator,
    skew_cv: float = 1.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return (binary adjacency, construction info) for the requested wiring.

    All schemes are matched on edge count as closely as integer degrees allow,
    which is what makes regular-vs-skewed a fair comparison rather than a
    budget comparison.
    """
    if n < 3:
        raise GraphConstructionError("node_count must be at least 3")

    info: dict[str, Any] = {"wiring_scheme": wiring_scheme}

    if wiring_scheme == "degree_regular":
        d = int(round(mean_degree))
        if abs(mean_degree - d) > 1e-9:
            raise GraphConstructionError(
                f"wiring_scheme='degree_regular' needs an integer mean_degree; got {mean_degree}. "
                f"A digraph in which every node has identical in- and out-degree cannot have a "
                f"fractional degree. Use wiring_scheme='near_regular' for {mean_degree}, which "
                f"splits nodes between degree {int(mean_degree)} and {int(mean_degree) + 1}."
            )
        seq = regular_degree_sequence(n, d)
        edges = simple_digraph_from_degrees(seq, seq, rng)
        info["degree_sequence_kind"] = "constant"

    elif wiring_scheme == "near_regular":
        seq = near_regular_degree_sequence(n, mean_degree, rng)
        # independent shuffle so a node's in-degree is not tied to its out-degree
        in_seq = rng.permutation(seq)
        edges = simple_digraph_from_degrees(seq, in_seq, rng)
        info["degree_sequence_kind"] = "two-valued"

    elif wiring_scheme == "skewed":
        out_seq = skewed_degree_sequence(n, mean_degree, rng, target_cv=skew_cv)
        in_seq = rng.permutation(out_seq)
        edges = simple_digraph_from_degrees(out_seq, in_seq, rng)
        info["degree_sequence_kind"] = "lognormal"
        info["skew_cv_target"] = skew_cv

    elif wiring_scheme == "erdos_renyi":
        edges = erdos_renyi_digraph(n, int(round(mean_degree * n)), rng)
        info["degree_sequence_kind"] = "binomial"

    else:  # pragma: no cover - guarded by pydantic
        raise GraphConstructionError(f"unknown wiring_scheme {wiring_scheme!r}")

    A = edges_to_matrix(edges, n)
    info["n_edges"] = int(A.sum())
    return A, info


def draw_weights(
    mask: np.ndarray, distribution: WeightDistribution, rng: np.random.Generator
) -> np.ndarray:
    """Attach random weights to the existing edges only."""
    n = mask.shape[0]
    if distribution == "uniform":
        W = rng.uniform(-1.0, 1.0, size=(n, n))
    elif distribution == "normal":
        W = rng.normal(0.0, 1.0, size=(n, n))
    elif distribution == "bimodal":
        W = rng.choice([-1.0, 1.0], size=(n, n))
    else:  # pragma: no cover - guarded by pydantic
        raise ValueError(f"unknown weight distribution {distribution!r}")
    return mask * W


def spectral_radius(A: np.ndarray) -> float:
    """
    Largest eigenvalue modulus.

    Dense eigenvalues below DENSE_EIGEN_LIMIT, ARPACK above it, power iteration
    if ARPACK will not converge. Computed once per matrix -- rescaling is a
    linear operation, so the achieved radius after scaling is known exactly and
    does not need a second O(n^3) solve.
    """
    n = A.shape[0]
    if not np.any(A):
        return 0.0
    if n <= DENSE_EIGEN_LIMIT:
        return float(np.max(np.abs(np.linalg.eigvals(A))))
    try:
        vals = eigs(csr_matrix(A), k=1, which="LM", return_eigenvectors=False, maxiter=5000)
        return float(np.max(np.abs(vals)))
    except (ArpackNoConvergence, ValueError):
        return _power_iteration_radius(A)


def _power_iteration_radius(A: np.ndarray, iters: int = 500, tol: float = 1e-9) -> float:
    rng = np.random.default_rng(0)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v) + 1e-30
    prev = 0.0
    for _ in range(iters):
        w = A @ v
        norm = np.linalg.norm(w)
        if norm < 1e-30:
            return 0.0
        v = w / norm
        if abs(norm - prev) < tol * max(norm, 1.0):
            break
        prev = norm
    return float(norm)


def rescale_spectral_radius(A: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Scale A so its spectral radius equals `target`. Returns (A_scaled, achieved)."""
    current = spectral_radius(A)
    if current <= 1e-12:
        return A, 0.0
    return A * (target / current), target


def generate_reservoir(
    node_count: int,
    mean_degree: float,
    wiring_scheme: WiringScheme = "degree_regular",
    spectral_radius_target: float = 0.9,
    weight_distribution: WeightDistribution = "uniform",
    seed: int | None = None,
    skew_cv: float = 1.2,
) -> ReservoirMatrix:
    """Generate one reservoir matrix end to end."""
    rng = np.random.default_rng(seed)
    A, info = build_adjacency(node_count, mean_degree, wiring_scheme, rng, skew_cv=skew_cv)
    W = draw_weights(A, weight_distribution, rng)
    W, achieved = rescale_spectral_radius(W, spectral_radius_target)

    return ReservoirMatrix(
        W=W,
        n=node_count,
        wiring_scheme=wiring_scheme,
        mean_degree=mean_degree,
        spectral_radius_target=spectral_radius_target,
        spectral_radius_achieved=achieved,
        weight_distribution=weight_distribution,
        seed=seed,
        n_edges=info["n_edges"],
        extra=info,
    )


def symmetrize(W: np.ndarray, preserve_spectral_radius: bool = True) -> np.ndarray:
    """
    Symmetrise a reservoir: (W + W^T) / 2, optionally rescaled to the original radius.

    This is the control condition. The benchmark's reservoir arm reported the
    regular-vs-skewed effect collapsing under symmetrisation (d 3.05 -> 0.74),
    which is why directedness is treated as load-bearing. Rescaling to the
    original radius removes the trivial explanation that symmetrisation simply
    changed the operator's gain.
    """
    target = spectral_radius(W) if preserve_spectral_radius else None
    S = 0.5 * (W + W.T)
    if target is not None and target > 1e-12:
        S, _ = rescale_spectral_radius(S, target)
    return S
