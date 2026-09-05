"""
Width optimisation: how far can the reservoir shrink before performance drops?

Protocol
--------
This mirrors the selection protocol from the source benchmark rather than a
plain binary search, because a search that both picks the winner and reports
its score has selected on the same data it reports.

1. Selection seeds establish the baseline at ratio 1.0 and score every candidate
   ratio.
2. A ratio passes if it is non-inferior to baseline: the lower bound of the
   bootstrap CI on the paired difference sits above -margin.
3. The smallest passing ratio is selected.
4. Confirmation seeds -- never used in step 2 -- re-run baseline and the selected
   ratio. If the difference does not hold there, the response says so.

Savings are computed from actual edge counts of the generated graphs. No fixed
constant is used: the 16.2% figure in the original blueprint came from a
CIFAR-10 sparse CNN sweep and has no bearing on a reservoir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .compare import DEFAULT_EVAL_PARAMS
from .evaluation import evaluate_task
from .reservoir import generate_reservoir
from .stats import bootstrap_ci_paired, paired_comparison
from .tasks import build_task

__all__ = ["run_optimization"]


def _score_at_ratio(
    base_nodes: int,
    ratio: float,
    mean_degree: float,
    wiring_scheme: str,
    spectral_radius: float,
    weight_distribution: str,
    seed: int,
    task_name: str,
    params: dict[str, Any],
) -> tuple[float, int, int, list[str]]:
    n = max(3, int(round(base_nodes * ratio)))
    reservoir = generate_reservoir(
        node_count=n,
        mean_degree=mean_degree,
        wiring_scheme=wiring_scheme,
        spectral_radius_target=spectral_radius,
        weight_distribution=weight_distribution,
        seed=seed,
    )
    task = build_task(
        task_name,
        sequence_length=params["sequence_length"],
        rng=np.random.default_rng(seed + 999_331),
        max_lag=params["max_lag"],
    )
    result = evaluate_task(
        reservoir.W,
        task,
        ridge_alpha=params["ridge_alpha"],
        washout_steps=params["washout_steps"],
        train_split=params["train_split"],
        input_scaling=params["input_scaling"],
        leak_rate=params["leak_rate"],
        seed=seed + 555_101,
    )
    return result.score, n, reservoir.n_edges, result.warnings


def run_optimization(
    base_nodes: int,
    mean_degree: float,
    spectral_radius: float = 0.9,
    wiring_scheme: str = "degree_regular",
    weight_distribution: str = "uniform",
    target_metric: str = "memory_capacity",
    ratios: list[float] | None = None,
    min_acceptable_ratio: float = 0.5,
    non_inferiority_margin_fraction: float = 0.02,
    selection_seeds: int = 8,
    confirmation_seeds: int = 4,
    eval_params: dict[str, Any] | None = None,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Search for the smallest width ratio that is non-inferior to full width."""
    params = {**DEFAULT_EVAL_PARAMS, **(eval_params or {})}
    if ratios is None:
        ratios = [1.0, 0.8, 0.6]  # demo-sized; pass your own for a full sweep
    ratios = sorted({round(float(r), 4) for r in ratios if r >= min_acceptable_ratio}, reverse=True)
    if not ratios or abs(ratios[0] - 1.0) > 1e-9:
        ratios = [1.0] + [r for r in ratios if r < 1.0]

    sel_seeds = [base_seed + i for i in range(selection_seeds)]
    conf_seeds = [base_seed + 10_000 + i for i in range(confirmation_seeds)]

    warnings: set[str] = set()
    by_ratio: dict[float, dict[str, Any]] = {}

    for ratio in ratios:
        scores, nodes, edges = [], [], []
        for seed in sel_seeds:
            score, n, n_edges, warn = _score_at_ratio(
                base_nodes, ratio, mean_degree, wiring_scheme, spectral_radius,
                weight_distribution, seed, target_metric, params,
            )
            scores.append(score)
            nodes.append(n)
            edges.append(n_edges)
            warnings.update(warn)
        by_ratio[ratio] = {
            "scores": np.array(scores),
            "node_count": int(nodes[0]),
            "mean_edges": float(np.mean(edges)),
        }

    baseline = by_ratio[1.0]
    margin = non_inferiority_margin_fraction * abs(float(baseline["scores"].mean()))

    sweep: list[dict[str, Any]] = []
    passing: list[float] = []
    for ratio in ratios:
        entry = by_ratio[ratio]
        lo, hi = bootstrap_ci_paired(entry["scores"], baseline["scores"], seed=base_seed)
        diff = float((entry["scores"] - baseline["scores"]).mean())
        non_inferior = bool(lo > -margin)
        if non_inferior:
            passing.append(ratio)
        sweep.append(
            {
                "ratio": ratio,
                "node_count": entry["node_count"],
                "mean_edges": entry["mean_edges"],
                "mean_score": float(entry["scores"].mean()),
                "std": float(entry["scores"].std(ddof=1)) if len(sel_seeds) > 1 else 0.0,
                "difference_vs_full_width": diff,
                "ci_low": lo,
                "ci_high": hi,
                "non_inferior": non_inferior,
                "verdict": "keeps" if non_inferior else "drops",
            }
        )

    selected = min(passing) if passing else 1.0
    selected_entry = by_ratio[selected]

    # ---- confirmation on seeds never used for selection -------------------
    conf_selected, conf_baseline = [], []
    for seed in conf_seeds:
        s, _, _, warn = _score_at_ratio(
            base_nodes, selected, mean_degree, wiring_scheme, spectral_radius,
            weight_distribution, seed, target_metric, params,
        )
        b, _, _, warn_b = _score_at_ratio(
            base_nodes, 1.0, mean_degree, wiring_scheme, spectral_radius,
            weight_distribution, seed, target_metric, params,
        )
        conf_selected.append(s)
        conf_baseline.append(b)
        warnings.update(warn)
        warnings.update(warn_b)

    conf_stats = paired_comparison(
        np.array(conf_selected), np.array(conf_baseline), seed=base_seed + 7
    )
    conf_margin = non_inferiority_margin_fraction * abs(float(np.mean(conf_baseline)))
    holds = bool(conf_stats.ci_low > -conf_margin)

    edge_saving = (
        1.0 - selected_entry["mean_edges"] / baseline["mean_edges"]
        if baseline["mean_edges"] > 0
        else 0.0
    )

    # A saturated metric makes every ratio look non-inferior, which would hand
    # back an aggressive shrink recommendation built on nothing.
    metric_saturated = any("ceiling" in w for w in warnings)
    caveats = [
        "Connection savings are counts of graph edges. Whether they translate into memory, "
        "latency, or FLOP savings depends entirely on whether your runtime uses a sparse "
        "kernel; a dense matrix with zeros in it saves nothing.",
        "Savings are measured for this configuration only. No fixed shrink ratio is assumed.",
    ]
    if metric_saturated:
        caveats.insert(
            0,
            "The target metric is at its ceiling, so every ratio passed the non-inferiority "
            "test trivially. This selection carries no information -- raise max_lag (or pick a "
            "harder task) and re-run before acting on the recommended ratio.",
        )

    return {
        "configuration": {
            "base_nodes": base_nodes,
            "mean_degree": mean_degree,
            "wiring_scheme": wiring_scheme,
            "spectral_radius": spectral_radius,
            "target_metric": target_metric,
            "ratios_tested": ratios,
            "non_inferiority_margin_fraction": non_inferiority_margin_fraction,
            "selection_seeds": sel_seeds,
            "confirmation_seeds": conf_seeds,
            "evaluation_params": params,
        },
        "sweep": sweep,
        "selection": {
            "optimal_ratio": selected,
            "node_count": selected_entry["node_count"],
            "mean_edges": selected_entry["mean_edges"],
            "baseline_mean_edges": baseline["mean_edges"],
            "measured_connection_saving_fraction": float(edge_saving),
            "measured_connection_saving_percent": float(100.0 * edge_saving),
            "selection_basis": "non-inferiority on selection seeds only",
            "metric_saturated": metric_saturated,
        },
        "confirmation": {
            "seeds": conf_seeds,
            "selected_mean_score": float(np.mean(conf_selected)),
            "baseline_mean_score": float(np.mean(conf_baseline)),
            "statistics": conf_stats.to_dict(),
            "non_inferiority_margin": conf_margin,
            "holds": holds,
            "interpretation": (
                "The selected ratio remained non-inferior on seeds not used to select it."
                if holds
                else "The selected ratio did NOT stay non-inferior on fresh seeds. Treat the "
                "selection as unconfirmed and do not report the saving as achieved."
            ),
        },
        "caveats": caveats,
        "warnings": sorted(warnings),
    }
