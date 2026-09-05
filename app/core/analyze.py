"""
Analysis of externally-produced results.

Why this exists
---------------
/compare runs the whole experiment in-process because a reservoir simulation
takes milliseconds. A CIFAR-10 sparse CNN takes ~100 minutes per run and a MoE
transformer ~150; sixteen paired runs is a night of compute, not an HTTP
request. Those arms have to train wherever the hardware is.

So the statistical protocol is separated from the thing being measured. A local
runner trains the models and emits per-seed scores; this module applies exactly
the same analysis /compare uses -- paired Cohen's d, sign-flip permutation,
bootstrap CIs, Holm correction across tasks, ceiling detection, and held-out
confirmation. One protocol across all three architectures, so the CNN, MoE and
reservoir results are directly comparable rather than three different analyses
wearing the same vocabulary.

The trade-off is honest and worth stating: /compare controls the whole
experiment, this endpoint trusts numbers it did not produce. It validates
structure and flags what it can see, but it cannot know whether the conditions
were genuinely paired or the seeds genuinely held out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stats import holm_bonferroni, paired_comparison

__all__ = ["analyze_results", "AnalysisError"]


class AnalysisError(ValueError):
    """Raised when submitted results cannot be analysed as a paired design."""


def analyze_results(
    architecture: str,
    metric_name: str,
    results: list[dict[str, Any]],
    higher_is_better: bool = True,
    ceiling: float | None = None,
    confirmation: list[dict[str, Any]] | None = None,
    treatment_label: str = "regular",
    baseline_label: str = "skewed",
    non_inferiority_margin_fraction: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Apply the paired protocol to results produced elsewhere.

    `results` is a flat list of {task, seed, treatment, baseline} records. Pairing
    is by (task, seed): both conditions must be present for the same seed, which
    is what makes the paired test valid.
    """
    by_task = _group_and_validate(results, "results")

    per_task: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    notes: list[str] = []

    for task, rows in by_task.items():
        treat = np.array([r["treatment"] for r in rows], dtype=float)
        base = np.array([r["baseline"] for r in rows], dtype=float)
        seeds = [r["seed"] for r in rows]

        stats = paired_comparison(treat, base, alpha=alpha, seed=seed)
        raw_p[task] = stats.p_value

        entry: dict[str, Any] = {
            "n_pairs": len(rows),
            "seeds": seeds,
            "treatment": {
                "label": treatment_label,
                "mean": float(treat.mean()),
                "std": float(treat.std(ddof=1)) if treat.size > 1 else 0.0,
                "scores": [float(x) for x in treat],
            },
            "baseline": {
                "label": baseline_label,
                "mean": float(base.mean()),
                "std": float(base.std(ddof=1)) if base.size > 1 else 0.0,
                "scores": [float(x) for x in base],
            },
            "statistics": stats.to_dict(),
            "favours": _favours(
                stats.mean_difference, stats.significant_at_05, higher_is_better,
                treatment_label, baseline_label,
            ),
        }

        entry["diagnostics"] = _diagnostics(treat, base, ceiling, len(rows))
        effect = entry["statistics"]["effect_size"]
        if effect is None or abs(effect) > 1e6:
            # Cohen's d divides by the standard deviation of the paired differences.
            # When every seed shows the same difference that denominator collapses,
            # and d comes out either infinite or -- because floating point rarely
            # gives bit-identical deltas -- as an enormous finite number that reads
            # like a colossal effect. Neither is meaningful.
            entry["statistics"]["effect_size_degenerate"] = True
            entry["diagnostics"]["warnings"].append(
                "the paired differences have essentially zero variance, so the standardised "
                "effect size is not meaningful (it reflects a collapsed denominator, not a large "
                "effect). Read mean_difference instead. In real training runs this usually means "
                "the scores were duplicated or the conditions were not independently re-seeded."
            )
        per_task[task] = entry

    corrected = holm_bonferroni(raw_p, alpha=alpha)
    for task, adj in corrected.items():
        per_task[task]["statistics"]["p_holm"] = adj["p_holm"]
        per_task[task]["statistics"]["significant_after_correction"] = adj["significant_at_05"]
        per_task[task]["favours"] = _favours(
            per_task[task]["statistics"]["mean_difference"],
            adj["significant_at_05"],
            higher_is_better,
            treatment_label,
            baseline_label,
        )

    # ---- held-out confirmation ------------------------------------------
    confirmation_block = None
    if confirmation:
        confirmation_block = _confirm(
            confirmation, by_task, non_inferiority_margin_fraction, higher_is_better, alpha, seed
        )
        notes.extend(confirmation_block.pop("_notes", []))
    else:
        notes.append(
            "No confirmation seeds supplied. Effects measured on the same seeds used to choose "
            "what to report are selection-prone; submit a held-out set under 'confirmation'."
        )

    n_sig = sum(1 for t in per_task.values() if t["statistics"]["significant_after_correction"])
    directions = {t["favours"] for t in per_task.values() if t["favours"] != "neither"}
    if len(directions) > 1:
        notes.append(
            "The effect changes direction across tasks. Do not summarise this as one condition "
            "being better; report the per-task signs."
        )

    return {
        "architecture": architecture,
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
        "conditions": {"treatment": treatment_label, "baseline": baseline_label},
        "per_task": per_task,
        "multiple_comparison_correction": {
            "method": "holm-bonferroni",
            "n_tests": len(raw_p),
            "detail": corrected,
        },
        "confirmation": confirmation_block,
        "summary": {
            "tasks_tested": len(per_task),
            "tasks_significant_after_correction": n_sig,
            "directions_observed": sorted(directions),
            "verdict": _verdict(n_sig, len(per_task), directions, confirmation_block),
        },
        "provenance": {
            "analysis": "performed by ReservoirX-D using the same protocol as /compare",
            "measurement": "performed externally; this service did not produce these scores and "
                           "cannot verify that the conditions were paired or the confirmation "
                           "seeds genuinely held out",
        },
        "notes": notes,
    }


def _group_and_validate(
    records: list[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    if not records:
        raise AnalysisError(f"'{field}' must contain at least one record")

    by_task: dict[str, list[dict[str, Any]]] = {}
    for i, row in enumerate(records):
        for key in ("task", "seed", "treatment", "baseline"):
            if key not in row:
                raise AnalysisError(f"{field}[{i}] is missing required field '{key}'")
        for key in ("treatment", "baseline"):
            if not np.isfinite(float(row[key])):
                raise AnalysisError(f"{field}[{i}]['{key}'] is not a finite number")
        by_task.setdefault(str(row["task"]), []).append(row)

    for task, rows in by_task.items():
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            dupes = sorted({s for s in seeds if seeds.count(s) > 1})
            raise AnalysisError(
                f"task '{task}' has duplicate seeds {dupes}. Each seed must appear once: the "
                f"pairing between conditions is what makes the paired test valid."
            )
        if len(rows) < 3:
            raise AnalysisError(
                f"task '{task}' has only {len(rows)} paired seed(s). At least 3 are needed, and "
                f"note that with n pairs the smallest attainable p-value is about 2/2^n -- "
                f"8 seeds bottom out near 0.008, so use more if you need resolution below that."
            )
        rows.sort(key=lambda r: r["seed"])
    return by_task


def _diagnostics(
    treat: np.ndarray, base: np.ndarray, ceiling: float | None, n_pairs: int
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []

    floor_p = 2.0 / (2**n_pairs)
    diagnostics["smallest_attainable_p"] = float(floor_p)
    if n_pairs <= 8:
        warnings.append(
            f"with {n_pairs} pairs the permutation test cannot return a p-value below "
            f"~{floor_p:.4f}; a result at that value is at the floor, not a measure of strength"
        )

    if ceiling is not None:
        headroom = min(
            (ceiling - treat.mean()) / abs(ceiling) if ceiling else 1.0,
            (ceiling - base.mean()) / abs(ceiling) if ceiling else 1.0,
        )
        diagnostics["ceiling"] = float(ceiling)
        diagnostics["fraction_of_ceiling_used"] = float(max(treat.mean(), base.mean()) / ceiling)
        if headroom < 0.05:
            warnings.append(
                f"both conditions are within 5% of the stated ceiling of {ceiling:g}. The metric "
                f"is saturated and effect sizes measured here are compressed -- the reservoir arm "
                f"measured d=+1.42 against a ceiling where the true value was +3.05"
            )

    diagnostics["warnings"] = warnings
    return diagnostics


def _confirm(
    confirmation: list[dict[str, Any]],
    selection_by_task: dict[str, list[dict[str, Any]]],
    margin_fraction: float | None,
    higher_is_better: bool,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    by_task = _group_and_validate(confirmation, "confirmation")
    notes: list[str] = []
    out: dict[str, Any] = {"_notes": notes, "per_task": {}}

    for task, rows in by_task.items():
        selection_seeds = {r["seed"] for r in selection_by_task.get(task, [])}
        overlap = sorted(selection_seeds & {r["seed"] for r in rows})
        if overlap:
            notes.append(
                f"task '{task}': confirmation seeds {overlap} were also used in the main results. "
                f"Confirmation on reused seeds is not held out and does not support the claim."
            )

        treat = np.array([r["treatment"] for r in rows], dtype=float)
        base = np.array([r["baseline"] for r in rows], dtype=float)
        stats = paired_comparison(treat, base, alpha=alpha, seed=seed + 7)

        direction_holds = bool(
            (stats.mean_difference > 0) == (higher_is_better)
            if abs(stats.mean_difference) > 0
            else False
        )
        entry: dict[str, Any] = {
            "n_pairs": len(rows),
            "seeds": [r["seed"] for r in rows],
            "held_out": not overlap,
            "statistics": stats.to_dict(),
            "sign_matches_main_result": direction_holds,
        }
        if margin_fraction is not None:
            margin = margin_fraction * abs(float(base.mean()))
            entry["non_inferiority_margin"] = float(margin)
            entry["non_inferior"] = bool(stats.ci_low > -margin)
        out["per_task"][task] = entry

    return out


def _favours(
    mean_difference: float,
    significant: bool,
    higher_is_better: bool,
    treatment_label: str,
    baseline_label: str,
) -> str:
    if not significant or mean_difference == 0:
        return "neither"
    better_is_treatment = (mean_difference > 0) == higher_is_better
    return treatment_label if better_is_treatment else baseline_label


def _verdict(
    n_sig: int, n_tasks: int, directions: set[str], confirmation: dict[str, Any] | None
) -> str:
    if n_sig == 0:
        return (
            f"No task shows a significant difference after correction for {n_tasks} test(s). "
            f"On this evidence the conditions are indistinguishable."
        )
    base = f"{n_sig} of {n_tasks} task(s) significant after correction"
    if len(directions) > 1:
        base += (
            f", and the direction is not consistent: {' and '.join(sorted(directions))} each win "
            f"on at least one task. This cannot be summarised as one condition being better."
        )
    else:
        base += f", all favouring {next(iter(directions))}."
    if confirmation:
        unheld = [t for t, v in confirmation["per_task"].items() if not v["held_out"]]
        if unheld:
            base += f" Confirmation for {', '.join(unheld)} reused selection seeds and is not held out."
    return base
