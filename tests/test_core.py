"""Metrics, evaluation, and statistics."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.evaluation import drive_reservoir, evaluate_task
from app.core.metrics import compute_metrics
from app.core.reservoir import generate_reservoir
from app.core.stats import (
    bootstrap_ci_paired,
    cohens_d_paired,
    holm_bonferroni,
    permutation_test_paired,
)
from app.core.tasks import build_task


def test_reciprocity_uses_newman_convention():
    """A fully mutual pair means both directed edges are reciprocated: reciprocity 1.0."""
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 1.0
    W[2, 3] = W[3, 2] = 1.0
    assert compute_metrics(W)["reciprocity"] == pytest.approx(1.0)

    W2 = np.zeros((4, 4))
    W2[0, 1] = W2[1, 2] = W2[2, 3] = 1.0
    assert compute_metrics(W2)["reciprocity"] == pytest.approx(0.0)


def test_symmetry_index_bounds():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 30))
    sym = 0.5 * (A + A.T)
    anti = 0.5 * (A - A.T)
    assert compute_metrics(sym)["symmetry_index"] == pytest.approx(1.0, abs=1e-9)
    assert compute_metrics(anti)["symmetry_index"] == pytest.approx(-1.0, abs=1e-9)


def test_henrici_is_zero_for_normal_matrices():
    """A symmetric matrix is normal, so departure from normality must vanish."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(40, 40))
    sym = 0.5 * (A + A.T)
    metrics = compute_metrics(sym)
    assert metrics["henrici_departure"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["normality_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_effective_rank_is_informative_where_numerical_rank_is_not():
    """Numerical rank saturates at n; effective rank should distinguish these two."""
    rng = np.random.default_rng(2)
    full = rng.normal(size=(50, 50))
    concentrated = np.outer(rng.normal(size=50), rng.normal(size=50)) + 1e-3 * full

    m_full = compute_metrics(full)
    m_conc = compute_metrics(concentrated)
    assert m_full["numerical_rank"] == m_conc["numerical_rank"] == 50
    assert m_conc["effective_rank"] < m_full["effective_rank"] / 5


def test_states_are_bounded_so_divergence_cannot_occur():
    """
    tanh bounds the state in [-1, 1]. This is why the original rejection-sampling
    guard could never have fired for the reason given, and why it was removed.
    """
    rng = np.random.default_rng(3)
    W = rng.normal(size=(50, 50)) * 5.0  # deliberately far outside the echo-state regime
    u = rng.uniform(-10, 10, size=500)
    W_in = rng.uniform(-10, 10, size=50)
    states = drive_reservoir(W, u, W_in)
    assert np.all(np.isfinite(states))
    assert np.max(np.abs(states)) <= 1.0 + 1e-12


def test_evaluation_reports_saturation_rather_than_hiding_it():
    rng = np.random.default_rng(4)
    reservoir = generate_reservoir(60, 3, "degree_regular", 0.9, seed=0)
    task = build_task("memory_capacity", 800, rng, max_lag=5)
    result = evaluate_task(reservoir.W, task, input_scaling=40.0, seed=0)
    assert result.stability["saturated_fraction"] > 0.5
    assert any("saturated" in w for w in result.warnings)


def test_memory_capacity_is_positive_and_decays_with_lag():
    rng = np.random.default_rng(5)
    reservoir = generate_reservoir(120, 3, "degree_regular", 0.9, seed=0)
    task = build_task("memory_capacity", 3000, rng, max_lag=15)
    result = evaluate_task(reservoir.W, task, seed=0)

    assert result.score > 0.5
    assert result.per_target["lag_1"] > result.per_target["lag_15"]
    assert all(0.0 <= v <= 1.0 for v in result.per_target.values())
    assert result.score <= reservoir.n


def test_evaluation_input_is_never_resampled():
    """Same matrix, same seed, same task -> identical score. No hidden retries."""
    rng_a = np.random.default_rng(6)
    rng_b = np.random.default_rng(6)
    reservoir = generate_reservoir(80, 3, "degree_regular", 0.9, seed=0)
    task_a = build_task("memory_capacity", 1200, rng_a, max_lag=10)
    task_b = build_task("memory_capacity", 1200, rng_b, max_lag=10)
    assert np.array_equal(task_a.inputs, task_b.inputs)
    a = evaluate_task(reservoir.W, task_a, seed=1).score
    b = evaluate_task(reservoir.W, task_b, seed=1).score
    assert a == pytest.approx(b)


@pytest.mark.parametrize("task_name", ["memory_capacity", "delay_parity", "delay_xor", "narma10"])
def test_all_tasks_run_and_score_finitely(task_name):
    rng = np.random.default_rng(7)
    reservoir = generate_reservoir(80, 3, "degree_regular", 0.9, seed=0)
    task = build_task(task_name, 1500, rng, max_lag=8)
    result = evaluate_task(reservoir.W, task, seed=0)
    assert np.isfinite(result.score)
    assert result.n_test > 0


def test_paired_statistics_detect_a_planted_effect():
    rng = np.random.default_rng(8)
    base = rng.normal(size=12)
    treat = base + 1.0
    assert cohens_d_paired(treat, base) > 5
    assert permutation_test_paired(treat, base, n_permutations=5000) < 0.01
    lo, hi = bootstrap_ci_paired(treat, base, n_boot=5000)
    assert lo > 0.5 and hi < 1.5


def test_paired_statistics_find_nothing_in_noise():
    rng = np.random.default_rng(9)
    a = rng.normal(size=12)
    b = rng.normal(size=12)
    p = permutation_test_paired(a, b, n_permutations=5000)
    assert p > 0.05
    lo, hi = bootstrap_ci_paired(a, b, n_boot=5000)
    assert lo < 0 < hi


def test_permutation_p_is_never_exactly_zero():
    rng = np.random.default_rng(10)
    base = rng.normal(size=20)
    treat = base + 100.0
    assert permutation_test_paired(treat, base, n_permutations=1000) > 0


def test_holm_correction_penalises_a_single_hit_in_a_sweep():
    """
    The MoE density sweep found one hit at p=0.0183 across four tests. Corrected,
    that is not significant -- which is the whole point of applying the correction.
    """
    corrected = holm_bonferroni({"d015": 0.0183, "d030": 0.0852, "d050": 0.3239, "d075": 0.7473})
    assert corrected["d015"]["p_holm"] == pytest.approx(0.0732, abs=1e-4)
    assert not corrected["d015"]["significant_at_05"]


def test_holm_is_monotone():
    corrected = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03})
    assert corrected["a"]["p_holm"] <= corrected["b"]["p_holm"] <= corrected["c"]["p_holm"]
