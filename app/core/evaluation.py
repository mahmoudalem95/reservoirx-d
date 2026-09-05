"""
Driving the reservoir and fitting the readout.

A note on the rejection sampling this replaces
----------------------------------------------
The original blueprint caught divergent states and responded by drawing a new
input signal. That is unsound twice over. First, with a tanh activation the
state is analytically bounded in [-1, 1], so it cannot diverge -- the guard
could never fire for the stated reason. Second, resampling the input silently
changes the task: the returned states then correspond to a different signal
than the caller supplied, and any readout fitted against the original targets
is fitting mismatched data.

The real failure modes are saturation (states pinned near +/-1, destroying the
linear readout) and collapse (states pinned near 0). Both are measured and
reported here as `stability`, and neither is silently "fixed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tasks import Task

__all__ = ["EvaluationResult", "drive_reservoir", "evaluate_task"]


@dataclass
class EvaluationResult:
    task: str
    aggregate_name: str
    score: float
    per_target: dict[str, float]
    stability: dict[str, float]
    n_train: int
    n_test: int
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def drive_reservoir(
    W: np.ndarray,
    u: np.ndarray,
    W_in: np.ndarray,
    leak_rate: float = 1.0,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run x[t] = (1-a) x[t-1] + a * tanh(W x[t-1] + W_in u[t] + b).

    Returns the state matrix with shape (T, n). States are bounded by
    construction; no rejection sampling, no input resampling.
    """
    n = W.shape[0]
    T = u.shape[0]
    W_in = W_in.reshape(n)
    b = np.zeros(n) if bias is None else bias.reshape(n)

    states = np.zeros((T, n))
    x = np.zeros(n)
    for t in range(T):
        pre = W @ x + W_in * u[t] + b
        x = (1.0 - leak_rate) * x + leak_rate * np.tanh(pre)
        states[t] = x
    return states


def _stability_report(states: np.ndarray) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    absolute = np.abs(states)
    saturated = float(np.mean(absolute > 0.99))
    collapsed = float(np.mean(absolute < 1e-4))
    report = {
        "mean_abs_state": float(absolute.mean()),
        "max_abs_state": float(absolute.max()),
        "saturated_fraction": saturated,
        "collapsed_fraction": collapsed,
        "state_std": float(states.std()),
        "non_finite_fraction": float(np.mean(~np.isfinite(states))),
    }
    if saturated > 0.5:
        warnings.append(
            f"{saturated:.0%} of state values exceed 0.99 in magnitude; the reservoir is "
            f"saturated. Lower input_scaling or spectral_radius -- scores from a saturated "
            f"reservoir mostly measure the clipping, not the dynamics."
        )
    if collapsed > 0.5:
        warnings.append(
            f"{collapsed:.0%} of state values are effectively zero; the reservoir is not being "
            f"driven. Raise input_scaling."
        )
    if report["non_finite_fraction"] > 0:
        warnings.append("non-finite states encountered; results for this run are not trustworthy")
    return report, warnings


def evaluate_task(
    W: np.ndarray,
    task: Task,
    ridge_alpha: float = 1e-6,
    washout_steps: int = 100,
    train_split: float = 0.7,
    input_scaling: float = 0.1,
    leak_rate: float = 1.0,
    seed: int | None = None,
) -> EvaluationResult:
    """Drive the reservoir on `task`, fit a ridge readout, score on held-out data."""
    n = W.shape[0]
    T = task.inputs.shape[0]
    if washout_steps >= T:
        raise ValueError(f"washout_steps={washout_steps} must be smaller than sequence length {T}")

    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-1.0, 1.0, size=n) * input_scaling

    states = drive_reservoir(W, task.inputs, W_in, leak_rate=leak_rate)
    stability, warnings = _stability_report(states)

    # discard washout, then split chronologically (never shuffled: this is a time series)
    X = states[washout_steps:]
    Y = task.targets[washout_steps:]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_usable = X.shape[0]
    n_train = int(round(train_split * n_usable))
    if n_train < n + 2 or n_usable - n_train < 10:
        warnings.append(
            f"only {n_train} training rows for {n} state dimensions; the readout is "
            f"under-determined and the score will be optimistic. Increase sequence_length."
        )
    n_train = max(2, min(n_train, n_usable - 2))

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    coef, intercept = _ridge_fit(X_train, Y_train, ridge_alpha)
    predictions = X_test @ coef + intercept

    per_target: dict[str, float] = {}
    if task.scoring == "sum_r2":
        for i, label in enumerate(task.target_labels):
            per_target[label] = _squared_correlation(predictions[:, i], Y_test[:, i])
        score = float(sum(per_target.values()))
        # A sum of per-lag r^2 is bounded by the number of lags scored. If the
        # reservoir is already near that bound, the task has hit its ceiling and
        # can no longer tell configurations apart -- every candidate will look
        # equally good, including ones that are genuinely worse.
        ceiling = float(len(task.target_labels))
        if ceiling > 0 and score > 0.95 * ceiling:
            warnings.append(
                f"score {score:.2f} is within 5% of this task's ceiling of {ceiling:.0f} "
                f"(one point per lag). The task is saturated and cannot discriminate between "
                f"configurations. Raise max_lag until the score falls below the ceiling before "
                f"drawing any comparison from it."
            )
    elif task.scoring == "one_minus_nrmse":
        for i, label in enumerate(task.target_labels):
            per_target[label] = _one_minus_nrmse(predictions[:, i], Y_test[:, i])
        score = float(np.mean(list(per_target.values())))
    else:  # pragma: no cover
        raise ValueError(f"unknown scoring {task.scoring!r}")

    return EvaluationResult(
        task=task.name,
        aggregate_name=task.aggregate_name,
        score=score,
        per_target=per_target,
        stability=stability,
        n_train=int(n_train),
        n_test=int(X_test.shape[0]),
        warnings=warnings,
        extra={"input_scaling": input_scaling, "ridge_alpha": ridge_alpha, "leak_rate": leak_rate},
    )


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ridge with an unpenalised intercept.

    Centring the data and solving on the centred system is equivalent to fitting
    an intercept that ridge never shrinks, which is the convention memory-capacity
    numbers in the literature assume.
    """
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    n_features = Xc.shape[1]
    gram = Xc.T @ Xc + alpha * np.eye(n_features)
    try:
        coef = np.linalg.solve(gram, Xc.T @ Yc)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, Xc.T @ Yc, rcond=None)[0]
    intercept = y_mean - x_mean @ coef
    return coef, intercept


def _squared_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Squared Pearson correlation, the standard memory-capacity per-lag score."""
    if pred.size < 2:
        return 0.0
    ps, ts = pred.std(), target.std()
    if ps < 1e-12 or ts < 1e-12:
        return 0.0
    r = float(np.corrcoef(pred, target)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return float(min(1.0, max(0.0, r * r)))


def _one_minus_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    denom = target.std()
    if denom < 1e-12:
        return 0.0
    nrmse = float(np.sqrt(np.mean((pred - target) ** 2)) / denom)
    return float(max(0.0, 1.0 - nrmse))
