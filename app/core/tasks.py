"""
Reservoir benchmark tasks.

Four tasks are offered rather than one. The reservoir arm of the source
benchmark found the regular-vs-skewed effect on some tasks and not on others
(parity and XOR showed nothing), so an API that only exposed memory capacity
would only ever be able to confirm the favourable case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TaskName = Literal["memory_capacity", "delay_parity", "delay_xor", "narma10"]

__all__ = ["Task", "build_task", "TASK_DESCRIPTIONS", "TASK_NAMES"]

TASK_NAMES: tuple[str, ...] = ("memory_capacity", "delay_parity", "delay_xor", "narma10")

TASK_DESCRIPTIONS: dict[str, str] = {
    "memory_capacity": "Linear short-term memory (Jaeger). Input is i.i.d. uniform; target at lag k "
    "is u[t-k]. Score per lag is the squared Pearson correlation on held-out "
    "data; total capacity is the sum over lags, bounded above by n.",
    "delay_parity": "Non-linear memory. Input is i.i.d. binary in {-1,+1}; target at order k is the "
    "product of the last k inputs. Requires non-linear mixing, not just memory.",
    "delay_xor": "Two-tap non-linear memory; target at lag k is u[t-k] * u[t-k-1].",
    "narma10": "Non-linear autoregressive moving average of order 10. A single regression target "
    "with both memory and non-linearity; scored as 1 - NRMSE (clipped at 0).",
}


@dataclass
class Task:
    name: str
    inputs: np.ndarray  # (T,)
    targets: np.ndarray  # (T, n_targets)
    target_labels: list[str]
    scoring: str  # "sum_r2" or "one_minus_nrmse"
    aggregate_name: str


def build_task(
    name: TaskName,
    sequence_length: int,
    rng: np.random.Generator,
    max_lag: int = 20,
) -> Task:
    if name == "memory_capacity":
        return _memory_capacity(sequence_length, rng, max_lag)
    if name == "delay_parity":
        return _delay_parity(sequence_length, rng, max_lag)
    if name == "delay_xor":
        return _delay_xor(sequence_length, rng, max_lag)
    if name == "narma10":
        return _narma10(sequence_length, rng)
    raise ValueError(f"unknown task {name!r}")


def _memory_capacity(T: int, rng: np.random.Generator, max_lag: int) -> Task:
    u = rng.uniform(-1.0, 1.0, size=T)
    targets = np.zeros((T, max_lag))
    labels = []
    for i, k in enumerate(range(1, max_lag + 1)):
        targets[k:, i] = u[:-k]
        labels.append(f"lag_{k}")
    return Task("memory_capacity", u, targets, labels, "sum_r2", "memory_capacity")


def _delay_parity(T: int, rng: np.random.Generator, max_lag: int) -> Task:
    u = rng.choice([-1.0, 1.0], size=T)
    targets = np.zeros((T, max_lag))
    labels = []
    for i, k in enumerate(range(1, max_lag + 1)):
        prod = np.ones(T)
        for shift in range(k):
            shifted = np.zeros(T)
            if shift == 0:
                shifted = u.copy()
            else:
                shifted[shift:] = u[:-shift]
            prod = prod * shifted
        targets[:, i] = prod
        labels.append(f"order_{k}")
    return Task("delay_parity", u, targets, labels, "sum_r2", "parity_capacity")


def _delay_xor(T: int, rng: np.random.Generator, max_lag: int) -> Task:
    u = rng.choice([-1.0, 1.0], size=T)
    targets = np.zeros((T, max_lag))
    labels = []
    for i, k in enumerate(range(1, max_lag + 1)):
        a = np.zeros(T)
        b = np.zeros(T)
        a[k:] = u[:-k]
        b[k + 1 :] = u[: -(k + 1)]
        targets[:, i] = a * b
        labels.append(f"xor_lag_{k}")
    return Task("delay_xor", u, targets, labels, "sum_r2", "xor_capacity")


def _narma10(T: int, rng: np.random.Generator) -> Task:
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T)
    order = 10
    for t in range(order, T - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - order + 1 : t + 1])
            + 1.5 * u[t - order + 1] * u[t]
            + 0.1
        )
        # NARMA-10 is famously prone to blow-up; clip rather than emit inf
        if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 1e3:
            y[t + 1] = np.clip(np.nan_to_num(y[t + 1]), -1e3, 1e3)
    return Task("narma10", u, y[:, None], ["narma10"], "one_minus_nrmse", "narma10_score")
