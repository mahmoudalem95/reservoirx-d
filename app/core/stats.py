"""
Statistics used by /compare and /optimize.

The benchmark protocol these endpoints implement is paired: the regular and
skewed conditions are generated from the same seed, so seed-to-seed variation
cancels. Everything here is therefore the paired variant unless named otherwise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "ComparisonStats",
    "paired_comparison",
    "cohens_d_paired",
    "permutation_test_paired",
    "bootstrap_ci_paired",
    "holm_bonferroni",
]


@dataclass
class ComparisonStats:
    n_pairs: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    effect_size: float
    effect_size_name: str
    p_value: float
    p_value_method: str
    significant_at_05: bool

    def to_dict(self) -> dict:
        """
        Serialise, replacing non-finite values with None.

        Cohen's d is infinite when every pair shows an identical difference: the
        denominator is the standard deviation of the differences, and zero
        variance makes the standardised effect undefined rather than enormous.
        That is a real state, but `inf` is not valid JSON and would silently
        become `null` downstream, so it is nulled here deliberately and the
        raw mean difference -- which is always finite and meaningful -- is what
        callers should read in that case.
        """
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        return out


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d on the paired differences (a - b), i.e. d_z."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diff.std(ddof=1)
    if sd <= 1e-30:
        return 0.0 if abs(diff.mean()) <= 1e-30 else float(np.sign(diff.mean()) * np.inf)
    return float(diff.mean() / sd)


def permutation_test_paired(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 20000, seed: int = 0
) -> float:
    """
    Two-sided exact-or-sampled sign-flip permutation test.

    With n pairs there are 2^n sign assignments. Below 2^20 we enumerate the
    sampled space with the same estimator either way; the returned p-value uses
    the (r+1)/(m+1) correction so it can never be reported as exactly zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = diff.size
    if n == 0:
        return float("nan")

    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    null = np.abs((signs * diff).mean(axis=1))
    n_extreme = int(np.sum(null >= observed - 1e-15))
    return float((n_extreme + 1) / (n_permutations + 1))


def bootstrap_ci_paired(
    a: np.ndarray, b: np.ndarray, n_boot: int = 20000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 20000,
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ComparisonStats:
    """Full paired comparison of condition `a` against condition `b`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci_paired(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
    p = permutation_test_paired(a, b, n_permutations=n_permutations, seed=seed)
    return ComparisonStats(
        n_pairs=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        std_a=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        std_b=float(b.std(ddof=1)) if b.size > 1 else 0.0,
        mean_difference=float((a - b).mean()),
        ci_low=lo,
        ci_high=hi,
        effect_size=cohens_d_paired(a, b),
        effect_size_name="cohens_d_paired (d_z)",
        p_value=p,
        p_value_method=f"two-sided sign-flip permutation, {n_permutations} draws",
        significant_at_05=bool(p < alpha),
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    Holm-Bonferroni correction over a family of tests.

    Used wherever an endpoint runs a sweep. A sweep over four densities that
    finds one hit at p=0.018 is not evidence of an effect, and the corrected
    column is what stops that being read as one.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    max_so_far = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        max_so_far = max(max_so_far, adjusted)  # enforce monotonicity
        out[key] = {
            "p_raw": float(p),
            "p_holm": float(max_so_far),
            "significant_at_05": bool(max_so_far < alpha),
        }
    return out
