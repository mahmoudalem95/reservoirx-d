#!/usr/bin/env python3
"""
Convert a benchmark results.json into an /analyze request.

The CNN and MoE harnesses each write per-seed score arrays. This pulls those
out, pairs them by seed position, and emits the payload /analyze expects, so all
three architectures go through one statistical protocol instead of three.

    python3 to_analysis_payload.py results/results.json -o cnn_payload.json
    curl -X POST https://reservoirx-d.fly.dev/v1/reservoirx-d/analyze \\
      -H 'Content-Type: application/json' -d @cnn_payload.json

A note on the pairing, because it is the assumption everything rests on: the
harnesses redraw the graph every seed and run both conditions at the same seed,
so index i of the regular array and index i of the baseline array are a genuine
pair. If you ever edit the harness so the two arrays come from different seed
sets, this conversion silently becomes wrong and the paired test with it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _pair(task: str, treatment: list, baseline: list, seeds: list | None = None) -> list[dict]:
    if len(treatment) != len(baseline):
        raise SystemExit(
            f"task '{task}': {len(treatment)} treatment scores vs {len(baseline)} baseline. "
            f"Paired analysis needs one of each per seed."
        )
    seeds = seeds if seeds is not None else list(range(len(treatment)))
    return [
        {"task": task, "seed": int(s), "treatment": float(t), "baseline": float(b)}
        for s, t, b in zip(seeds, treatment, baseline)
    ]


def from_cnn(payload: dict, split: str = "test") -> dict:
    """
    Sparse CNN harness.

    Every width ratio becomes its own task, so Holm correction applies across the
    sweep -- which matters, because a six-ratio sweep is six tests and reading the
    best one uncorrected is how a null becomes a finding.
    """
    baseline = payload["baseline"][split]
    results: list[dict] = []
    for entry in payload["sweep"]:
        results += _pair(f"ratio_{entry['ratio']:g}", entry[split], baseline)

    confirmation = None
    fresh = payload.get("fresh")
    if fresh:
        confirmation = _pair(
            f"ratio_{payload['selected']['ratio']:g}",
            fresh["regular"], fresh["baseline"], fresh.get("seeds"),
        )

    return {
        "architecture": "sparse_cnn_cifar10",
        "metric_name": f"{split}_accuracy_pct",
        "higher_is_better": True,
        "ceiling": 100.0,
        "treatment_label": "regular",
        "baseline_label": "skewed",
        "results": results,
        "confirmation": confirmation,
        "non_inferiority_margin_fraction": 0.02,
    }


def from_moe(payload: dict) -> dict:
    """
    MoE harness. Loss, so lower is better -- higher_is_better=False.

    Each density in the sweep becomes a task. The original run found one hit at
    density 0.15 across four densities; under Holm that is not significant, which
    is the whole reason the correction is applied here.
    """
    results: list[dict] = []
    for entry in payload.get("densities", payload.get("sweep", [])):
        label = entry.get("density", entry.get("ratio"))
        treatment = entry.get("regular") or entry.get("treatment")
        baseline = entry.get("skewed") or entry.get("baseline")
        if treatment is None or baseline is None:
            continue
        results += _pair(f"density_{label:g}", treatment, baseline)

    confirmation = None
    fresh = payload.get("fresh")
    if fresh and fresh.get("regular"):
        confirmation = _pair(
            "fresh_seeds", fresh["regular"], fresh["baseline"], fresh.get("seeds")
        )

    return {
        "architecture": "moe_hash_routing",
        "metric_name": "val_loss",
        "higher_is_better": False,
        "treatment_label": "regular",
        "baseline_label": "skewed",
        "results": results,
        "confirmation": confirmation,
    }


def detect(payload: dict) -> str:
    if "sweep" in payload and "baseline" in payload and "efficiency" in payload:
        return "cnn"
    if "densities" in payload or "val_loss" in json.dumps(payload)[:2000]:
        return "moe"
    raise SystemExit(
        "could not tell which harness produced this file. Pass --arch cnn or --arch moe."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("payload.json"))
    parser.add_argument("--arch", choices=["cnn", "moe"], help="override auto-detection")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="CNN only; which accuracy to analyse (default: test)")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    arch = args.arch or detect(payload)
    request = from_cnn(payload, args.split) if arch == "cnn" else from_moe(payload)

    if not request["results"]:
        raise SystemExit("no paired results found -- is this the right file?")

    args.output.write_text(json.dumps(request, indent=2))

    tasks = sorted({r["task"] for r in request["results"]})
    print(f"architecture : {request['architecture']}")
    print(f"metric       : {request['metric_name']} "
          f"({'higher' if request['higher_is_better'] else 'lower'} is better)")
    print(f"tasks        : {len(tasks)} ({', '.join(tasks[:6])}{'...' if len(tasks) > 6 else ''})")
    print(f"pairs        : {len(request['results'])}")
    print(f"confirmation : {len(request['confirmation']) if request['confirmation'] else 0} pairs")
    print(f"wrote        : {args.output}")

    n_seeds = len(request["results"]) // max(len(tasks), 1)
    if n_seeds <= 8:
        print(f"\nnote: {n_seeds} seeds per task. The permutation floor is ~{2 / 2**n_seeds:.4f}, "
              f"so p-values cannot go below that no matter how large the effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
