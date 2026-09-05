"""Graph construction invariants. These are the properties the whole service rests on."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.graphs import (
    GraphConstructionError,
    degree_cv,
    edges_to_matrix,
    erdos_renyi_digraph,
    near_regular_degree_sequence,
    regular_degree_sequence,
    simple_digraph_from_degrees,
    skewed_degree_sequence,
)
from app.core.reservoir import build_adjacency, generate_reservoir, symmetrize


@pytest.mark.parametrize("n,d", [(50, 3), (100, 4), (200, 3), (30, 6)])
def test_regular_digraph_has_exact_degrees(n, d):
    rng = np.random.default_rng(0)
    seq = regular_degree_sequence(n, d)
    edges = simple_digraph_from_degrees(seq, seq, rng)
    A = edges_to_matrix(edges, n)

    assert np.all(A.sum(axis=1) == d), "every node must have out-degree exactly d"
    assert np.all(A.sum(axis=0) == d), "every node must have in-degree exactly d"
    assert A.sum() == n * d


@pytest.mark.parametrize("n,d", [(50, 3), (120, 5)])
def test_no_self_loops_or_parallel_edges(n, d):
    rng = np.random.default_rng(7)
    seq = regular_degree_sequence(n, d)
    edges = simple_digraph_from_degrees(seq, seq, rng)

    assert not any(u == v for u, v in edges), "self-loops must be repaired away"
    assert len({tuple(e) for e in edges}) == len(edges), "no parallel edges"
    assert set(np.unique(edges_to_matrix(edges, n))) <= {0.0, 1.0}


def test_regular_wiring_is_not_symmetric():
    """
    The failure mode this service exists to avoid.

    random_regular_graph(...).to_directed() would give symmetry_index 1.0. A real
    directed construction should reciprocate only about d/(n-1) of its edges.
    """
    rng = np.random.default_rng(3)
    n, d = 200, 3
    A, _ = build_adjacency(n, d, "degree_regular", rng)
    mask = A > 0
    reciprocity = np.sum(mask & mask.T) / mask.sum()
    assert reciprocity < 0.1, f"graph is far too reciprocal ({reciprocity:.3f}); is it symmetric?"


def test_degree_cv_separates_regular_from_skewed():
    rng = np.random.default_rng(11)
    n, d = 300, 4
    regular, _ = build_adjacency(n, d, "degree_regular", rng)
    skewed, _ = build_adjacency(n, d, "skewed", rng)

    cv_regular = degree_cv((regular > 0).sum(axis=1))
    cv_skewed = degree_cv((skewed > 0).sum(axis=1))

    assert cv_regular == pytest.approx(0.0, abs=1e-12)
    assert cv_skewed > 0.5, f"skewed CV={cv_skewed:.3f} is not meaningfully skewed"


def test_edge_budgets_match_across_schemes():
    rng = np.random.default_rng(5)
    n, d = 200, 3
    regular, info_r = build_adjacency(n, d, "degree_regular", rng)
    skewed, info_s = build_adjacency(n, d, "skewed", rng)
    er, info_e = build_adjacency(n, d, "erdos_renyi", rng)

    assert info_r["n_edges"] == info_s["n_edges"] == info_e["n_edges"] == n * d


def test_near_regular_hits_fractional_mean():
    rng = np.random.default_rng(2)
    seq = near_regular_degree_sequence(200, 2.7, rng)
    assert seq.mean() == pytest.approx(2.7, abs=1e-9)
    assert set(np.unique(seq)) <= {2, 3}
    assert degree_cv(seq) < 0.2, "near_regular should stay close to regular, not become skewed"


def test_skewed_sequence_matches_budget_and_is_heavy_tailed():
    rng = np.random.default_rng(4)
    seq = skewed_degree_sequence(400, 4.0, rng, target_cv=1.2)
    assert seq.sum() == 1600
    assert seq.min() >= 1
    assert degree_cv(seq) > 0.6


def test_degree_sequence_feasibility_errors():
    rng = np.random.default_rng(0)
    with pytest.raises(GraphConstructionError):
        regular_degree_sequence(10, 10)  # d must be < n
    with pytest.raises(GraphConstructionError):
        simple_digraph_from_degrees([3, 3, 3], [3, 3, 2], rng)  # unequal sums


def test_erdos_renyi_is_simple_and_exact():
    rng = np.random.default_rng(9)
    edges = erdos_renyi_digraph(80, 240, rng)
    assert len(edges) == 240
    assert not any(u == v for u, v in edges)
    assert len({tuple(e) for e in edges}) == 240


def test_spectral_radius_is_rescaled_accurately():
    for seed in range(3):
        reservoir = generate_reservoir(120, 3, "degree_regular", 0.9, seed=seed)
        achieved = np.max(np.abs(np.linalg.eigvals(reservoir.W)))
        assert achieved == pytest.approx(0.9, rel=1e-8)


def test_generation_is_deterministic_given_seed():
    a = generate_reservoir(100, 3, "degree_regular", 0.9, seed=123)
    b = generate_reservoir(100, 3, "degree_regular", 0.9, seed=123)
    assert np.array_equal(a.W, b.W)

    c = generate_reservoir(100, 3, "degree_regular", 0.9, seed=124)
    assert not np.array_equal(a.W, c.W)


def test_no_global_random_state_leak():
    """Generation must not depend on or disturb numpy's global RNG."""
    np.random.seed(0)
    first = generate_reservoir(60, 3, "degree_regular", 0.9, seed=5).W
    np.random.seed(999)
    second = generate_reservoir(60, 3, "degree_regular", 0.9, seed=5).W
    assert np.array_equal(first, second)


def test_symmetrize_produces_symmetric_matrix_at_same_radius():
    reservoir = generate_reservoir(100, 3, "degree_regular", 0.9, seed=1)
    S = symmetrize(reservoir.W)
    assert np.allclose(S, S.T)
    assert np.max(np.abs(np.linalg.eigvals(S))) == pytest.approx(0.9, rel=1e-6)


def test_fractional_degree_rejected_for_regular_scheme():
    rng = np.random.default_rng(0)
    with pytest.raises(GraphConstructionError, match="near_regular"):
        build_adjacency(100, 2.7, "degree_regular", rng)
