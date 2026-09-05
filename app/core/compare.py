"""
The /compare engine: does regular-degree wiring actually beat skewed wiring here?

Design commitments
------------------
1. Paired by seed. Condition A and condition B share a seed, so the pair differs
   only in wiring, not in the input signal or the readout initialisation.
2. Multi-task by default. The source benchmark's reservoir arm found the effect
   on some tasks and not others; running a single favourable task would turn a
   test into a demonstration.
3. Holm-corrected across tasks. One hit out of four uncorrected tests is what
   chance produces.
4. The symmetrisation control is *measured*, not estimated from a structural
   proxy. If directedness is load-bearing, symmetrising the same matrices should
   shrink the effect, and that is an experiment, not an extrapolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation import evaluate_task
from .reservoir import generate_reservoir, symmetrize
from .stats import holm_bonferroni, paired_comparison
from .tasks import build_task

__all__ = ["run_comparison", "EvaluationParams"]


class EvaluationParams(dict):
    """Plain dict of evaluation knobs, kept as a type alias for readability."""


DEFAULT_EVAL_PARAMS: dict[str, Any] = {
    "ridge_alpha": 1e-6,
    "washout_steps": 100,
    "train_split": 0.7,
    "max_lag": 20,
    "input_scaling": 0.1,
    "leak_rate": 1.0,
    "sequence_length": 2000,
}


def _score_one(
    node_count: int,
    mean_degree: float,
    wiring_scheme: str,
    spectral_radius: float,
    weight_distribution: str,
    seed: int,
    task_name: str,
    eval_params: dict[str, Any],
    skew_cv: float,
    symmetrise: bool = False,
) -> tuple[float, dict[str, Any]]:
    reservoir = generate_reservoir(
        node_count=node_count,
        mean_degree=mean_degree,
        wiring_scheme=wiring_scheme,
        spectral_radius_target=spectral_radius,
        weight_distribution=weight_distribution,
        seed=seed,
        skew_cv=skew_cv,
    )
    W = symmetrize(reservoir.W) if symmetrise else reservoir.W

    task_rng = np.random.default_rng(seed + 999_331)  # shared across conditions at equal seed
    task = build_task(
        task_name,
        sequence_length=eval_params["sequence_length"],
        rng=task_rng,
        max_lag=eval_params["max_lag"],
    )
    result = evaluate_task(
        W,
        task,
        ridge_alpha=eval_params["ridge_alpha"],
        washout_steps=eval_params["washout_steps"],
        train_split=eval_params["train_split"],
        input_scaling=eval_params["input_scaling"],
        leak_rate=eval_params["leak_rate"],
        seed=seed + 555_101,
    )
    meta = {
        "n_edges": reservoir.n_edges,
        "warnings": result.warnings,
        "saturated_fraction": result.stability["saturated_fraction"],
    }
    return result.score, meta


def run_comparison(
    node_count: int,
    mean_degree: float,
    spectral_radius: float = 0.9,
    weight_distribution: str = "uniform",
    iterations: int = 8,
    tasks: list[str] | None = None,
    baseline_scheme: str = "skewed",
    treatment_scheme: str = "degree_regular",
    skew_cv: float = 1.2,
    include_symmetrization_control: bool = False,
    eval_params: dict[str, Any] | None = None,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Run the paired comparison and return a fully-specified result document."""
    tasks = tasks or ["memory_capacity"]
    params = {**DEFAULT_EVAL_PARAMS, **(eval_params or {})}
    seeds = [base_seed + i for i in range(iterations)]

    per_task: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    all_warnings: set[str] = set()

    for task_name in tasks:
        treat_scores, base_scores = [], []
        treat_edges, base_edges = [], []

        for seed in seeds:
            t_score, t_meta = _score_one(
                node_count, mean_degree, treatment_scheme, spectral_radius,
                weight_distribution, seed, task_name, params, skew_cv,
            )
            b_score, b_meta = _score_one(
                node_count, mean_degree, baseline_scheme, spectral_radius,
                weight_distribution, seed, task_name, params, skew_cv,
            )
            treat_scores.append(t_score)
            base_scores.append(b_score)
            treat_edges.append(t_meta["n_edges"])
            base_edges.append(b_meta["n_edges"])
            all_warnings.update(t_meta["warnings"])
            all_warnings.update(b_meta["warnings"])

        treat = np.array(treat_scores)
        base = np.array(base_scores)
        stats = paired_comparison(treat, base, seed=base_seed)
        raw_p[task_name] = stats.p_value

        entry: dict[str, Any] = {
            "treatment": {
                "wiring_scheme": treatment_scheme,
                "mean_score": float(treat.mean()),
                "std": float(treat.std(ddof=1)) if treat.size > 1 else 0.0,
                "scores": [float(s) for s in treat],
                "mean_edges": float(np.mean(treat_edges)),
            },
            "baseline": {
                "wiring_scheme": baseline_scheme,
                "mean_score": float(base.mean()),
                "std": float(base.std(ddof=1)) if base.size > 1 else 0.0,
                "scores": [float(s) for s in base],
                "mean_edges": float(np.mean(base_edges)),
            },
            "statistics": stats.to_dict(),
            "edge_budget_matched": bool(abs(np.mean(treat_edges) - np.mean(base_edges)) < 1e-9),
        }

        if include_symmetrization_control:
            sym_treat, sym_base = [], []
            for seed in seeds:
                s_t, _ = _score_one(
                    node_count, mean_degree, treatment_scheme, spectral_radius,
                    weight_distribution, seed, task_name, params, skew_cv, symmetrise=True,
                )
                s_b, _ = _score_one(
                    node_count, mean_degree, baseline_scheme, spectral_radius,
                    weight_distribution, seed, task_name, params, skew_cv, symmetrise=True,
                )
                sym_treat.append(s_t)
                sym_base.append(s_b)
            sym_stats = paired_comparison(np.array(sym_treat), np.array(sym_base), seed=base_seed)
            entry["symmetrization_control"] = {
                "statistics": sym_stats.to_dict(),
                "effect_size_directed": stats.effect_size,
                "effect_size_symmetrised": sym_stats.effect_size,
                "effect_retained_fraction": (
                    float(sym_stats.effect_size / stats.effect_size)
                    if abs(stats.effect_size) > 1e-9
                    else None
                ),
                "interpretation": _symmetrisation_verdict(stats.effect_size, sym_stats.effect_size),
            }

        per_task[task_name] = entry

    corrected = holm_bonferroni(raw_p)
    for task_name, adj in corrected.items():
        per_task[task_name]["statistics"]["p_holm"] = adj["p_holm"]
        per_task[task_name]["statistics"]["significant_after_correction"] = adj["significant_at_05"]

    n_sig = sum(1 for t in per_task.values() if t["statistics"]["significant_after_correction"])
    n_favouring = sum(1 for t in per_task.values() if t["statistics"]["mean_difference"] > 0)

    return {
        "configuration": {
            "node_count": node_count,
            "mean_degree": mean_degree,
            "spectral_radius": spectral_radius,
            "weight_distribution": weight_distribution,
            "iterations": iterations,
            "seeds": seeds,
            "tasks": tasks,
            "treatment_scheme": treatment_scheme,
            "baseline_scheme": baseline_scheme,
            "skew_cv_target": skew_cv,
            "evaluation_params": params,
        },
        "per_task": per_task,
        "multiple_comparison_correction": {
            "method": "holm-bonferroni",
            "n_tests": len(raw_p),
            "detail": corrected,
        },
        "summary": {
            "tasks_tested": len(tasks),
            "tasks_significant_after_correction": n_sig,
            "tasks_favouring_treatment": n_favouring,
            "verdict": _overall_verdict(n_sig, n_favouring, len(tasks)),
        },
        "warnings": sorted(all_warnings),
    }


def _symmetrisation_verdict(directed_d: float, symmetric_d: float) -> str:
    if abs(directed_d) < 0.2:
        return (
            "No directed effect to speak of, so the symmetrisation control cannot say anything. "
            "Interpret neither number."
        )
    ratio = abs(symmetric_d) / abs(directed_d)
    if ratio < 0.4:
        return (
            f"The effect shrinks substantially when the same matrices are symmetrised "
            f"(|d| {abs(directed_d):.2f} -> {abs(symmetric_d):.2f}). Consistent with directedness "
            f"being necessary for the effect in this configuration."
        )
    if ratio < 0.8:
        return (
            f"The effect is partially reduced under symmetrisation (|d| {abs(directed_d):.2f} -> "
            f"{abs(symmetric_d):.2f}). Directedness contributes but does not appear to be the "
            f"whole story here."
        )
    return (
        f"The effect largely survives symmetrisation (|d| {abs(directed_d):.2f} -> "
        f"{abs(symmetric_d):.2f}). In this configuration directedness is not what is driving it, "
        f"which contradicts the directedness hypothesis rather than supporting it."
    )


def _overall_verdict(n_sig: int, n_favouring: int, n_tasks: int) -> str:
    if n_sig == 0:
        return (
            f"No task shows a significant difference after correction for {n_tasks} test(s). "
            f"On this configuration the wiring schemes are indistinguishable."
        )
    if n_sig == n_tasks:
        return (
            f"All {n_tasks} task(s) show a significant difference after correction, "
            f"{n_favouring} favouring the treatment wiring."
        )
    return (
        f"{n_sig} of {n_tasks} tasks show a significant difference after correction. "
        f"The effect is task-specific on this configuration and should not be described as general."
    )
