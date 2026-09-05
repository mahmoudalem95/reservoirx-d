"""Endpoint behaviour, validation, and the matrix store."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

FAST_EVAL = {"sequence_length": 600, "max_lag": 5, "washout_steps": 50}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_index(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/")
    assert r.status_code == 200
    assert "/v1/reservoirx-d/compare" in r.json()["endpoints"]


def test_generate_returns_regular_degrees(client):
    r = client.post(
        "/v1/reservoirx-d/generate",
        json={"node_count": 100, "mean_degree": 3, "seed": 42, "return_matrix": True},
    )
    assert r.status_code == 200
    body = r.json()["primary"]
    assert body["n_edges"] == 300
    assert body["metrics"]["degree_cv"] == pytest.approx(0.0, abs=1e-12)
    assert body["spectral_radius_achieved"] == pytest.approx(0.9, rel=1e-6)

    W = np.array(body["matrix"])
    assert np.all((W != 0).sum(axis=1) == 3)
    assert np.all(np.diag(W) == 0)


def test_generate_rejects_fractional_degree_and_names_the_alternative(client):
    r = client.post(
        "/v1/reservoirx-d/generate",
        json={"node_count": 200, "mean_degree": 2.7, "wiring_scheme": "degree_regular"},
    )
    assert r.status_code == 422
    assert "near_regular" in r.text


def test_near_regular_accepts_fractional_degree(client):
    r = client.post(
        "/v1/reservoirx-d/generate",
        json={"node_count": 200, "mean_degree": 2.7, "wiring_scheme": "near_regular", "seed": 1},
    )
    assert r.status_code == 200
    body = r.json()["primary"]
    assert body["n_edges"] == 540
    assert body["metrics"]["degree_cv"] < 0.2


def test_paired_contrast_matches_edge_budget(client):
    r = client.post(
        "/v1/reservoirx-d/generate",
        json={
            "node_count": 150,
            "mean_degree": 4,
            "seed": 7,
            "return_paired_contrast": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["contrast"]["wiring_scheme"] == "skewed"
    assert body["primary"]["n_edges"] == body["contrast"]["n_edges"]
    assert body["contrast"]["metrics"]["degree_cv"] > body["primary"]["metrics"]["degree_cv"]


def test_generate_rejects_impossible_degree(client):
    r = client.post(
        "/v1/reservoirx-d/generate", json={"node_count": 10, "mean_degree": 20}
    )
    assert r.status_code == 422


def test_diagnose_via_matrix_id(client):
    gen = client.post(
        "/v1/reservoirx-d/generate", json={"node_count": 60, "mean_degree": 3, "seed": 3}
    ).json()
    matrix_id = gen["primary"]["matrix_id"]

    r = client.post(
        "/v1/reservoirx-d/diagnose", json={"matrix_id": matrix_id, "iterations": 15}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["raw_metrics"]["degree_cv"] == pytest.approx(0.0, abs=1e-12)
    assert "degree_preserving" in body["null_analysis"]
    assert "erdos_renyi" in body["null_analysis"]
    assert body["directedness"]["is_effectively_symmetric"] is False
    assert "symmetrization_risk" not in body, "estimated risk must not be reinstated"
    assert "reciprocity" in body["metric_definitions"]


def test_diagnose_flags_a_symmetric_matrix(client):
    """The to_directed() trap: an undirected graph made directed is caught."""
    rng = np.random.default_rng(0)
    A = (rng.random((40, 40)) < 0.1).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    r = client.post(
        "/v1/reservoirx-d/diagnose",
        json={"matrix": A.tolist(), "iterations": 10, "null_models": ["erdos_renyi"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["directedness"]["is_effectively_symmetric"] is True
    assert "to_directed" in body["directedness"]["assessment"]


def test_diagnose_requires_exactly_one_source(client):
    assert client.post("/v1/reservoirx-d/diagnose", json={"iterations": 10}).status_code == 422
    r = client.post(
        "/v1/reservoirx-d/diagnose",
        json={"matrix_id": "x", "matrix": [[0.0, 1.0], [1.0, 0.0]], "iterations": 10},
    )
    assert r.status_code == 422


def test_unknown_matrix_id_is_404_with_guidance(client):
    r = client.post("/v1/reservoirx-d/evaluate", json={"matrix_id": "rxd_missing"})
    assert r.status_code == 404
    assert "regenerate" in r.json()["detail"].lower()


def test_evaluate_returns_per_lag_capacities(client):
    gen = client.post(
        "/v1/reservoirx-d/generate", json={"node_count": 100, "mean_degree": 3, "seed": 11}
    ).json()
    r = client.post(
        "/v1/reservoirx-d/evaluate",
        json={
            "matrix_id": gen["primary"]["matrix_id"],
            "task": "memory_capacity",
            "seed": 0,
            "evaluation_params": {**FAST_EVAL, "sequence_length": 1500},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["score"] > 0
    assert set(body["per_target"]) == {f"lag_{k}" for k in range(1, 6)}
    assert 0.0 <= body["stability"]["saturated_fraction"] <= 1.0
    assert body["n_test"] > 0


def test_evaluate_rejects_washout_longer_than_sequence(client):
    r = client.post(
        "/v1/reservoirx-d/evaluate",
        json={
            "matrix": np.eye(5).tolist(),
            "evaluation_params": {"sequence_length": 300, "washout_steps": 400},
        },
    )
    assert r.status_code == 422


def test_compare_runs_paired_and_reports_budget_match(client):
    r = client.post(
        "/v1/reservoirx-d/compare",
        json={
            "node_count": 60,
            "mean_degree": 3,
            "iterations": 3,
            "tasks": ["memory_capacity"],
            "evaluation_params": FAST_EVAL,
        },
    )
    assert r.status_code == 200
    body = r.json()
    task = body["per_task"]["memory_capacity"]
    assert task["edge_budget_matched"] is True
    assert len(task["treatment"]["scores"]) == 3
    assert task["statistics"]["p_value"] > 0
    assert "p_holm" in task["statistics"]
    assert body["summary"]["verdict"]


def test_compare_corrects_across_multiple_tasks(client):
    r = client.post(
        "/v1/reservoirx-d/compare",
        json={
            "node_count": 50,
            "mean_degree": 3,
            "iterations": 3,
            "tasks": ["memory_capacity", "delay_xor"],
            "evaluation_params": FAST_EVAL,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["multiple_comparison_correction"]["n_tests"] == 2
    for task in body["per_task"].values():
        assert task["statistics"]["p_holm"] >= task["statistics"]["p_value"]


def test_compare_symmetrization_control_is_measured(client):
    r = client.post(
        "/v1/reservoirx-d/compare",
        json={
            "node_count": 50,
            "mean_degree": 3,
            "iterations": 3,
            "include_symmetrization_control": True,
            "evaluation_params": FAST_EVAL,
        },
    )
    assert r.status_code == 200
    control = r.json()["per_task"]["memory_capacity"]["symmetrization_control"]
    assert "effect_size_symmetrised" in control
    assert control["interpretation"]


def test_compare_rejects_identical_schemes(client):
    r = client.post(
        "/v1/reservoirx-d/compare",
        json={
            "node_count": 50,
            "mean_degree": 3,
            "treatment_scheme": "skewed",
            "baseline_scheme": "skewed",
        },
    )
    assert r.status_code == 422


def test_compare_rejects_duplicate_tasks(client):
    r = client.post(
        "/v1/reservoirx-d/compare",
        json={
            "node_count": 50,
            "mean_degree": 3,
            "tasks": ["memory_capacity", "memory_capacity"],
        },
    )
    assert r.status_code == 422


def test_optimize_confirms_on_held_out_seeds(client):
    r = client.post(
        "/v1/reservoirx-d/optimize",
        json={
            "base_nodes": 60,
            "mean_degree": 3,
            "ratios": [1.0, 0.8, 0.6],
            "selection_seeds": 3,
            "confirmation_seeds": 2,
            "evaluation_params": FAST_EVAL,
        },
    )
    assert r.status_code == 200
    body = r.json()

    sel_seeds = set(body["configuration"]["selection_seeds"])
    conf_seeds = set(body["configuration"]["confirmation_seeds"])
    assert not (sel_seeds & conf_seeds), "confirmation must not reuse selection seeds"

    assert body["selection"]["optimal_ratio"] in {1.0, 0.8, 0.6}
    assert "measured_connection_saving_percent" in body["selection"]
    assert isinstance(body["confirmation"]["holds"], bool)
    assert any("sparse kernel" in c for c in body["caveats"])


def test_optimize_savings_are_measured_not_hardcoded(client):
    """The blueprint's 16.2% came from a CNN sweep and must never appear as a constant."""
    r = client.post(
        "/v1/reservoirx-d/optimize",
        json={
            "base_nodes": 50,
            "mean_degree": 3,
            "ratios": [1.0, 0.5],
            "selection_seeds": 2,
            "confirmation_seeds": 2,
            "evaluation_params": FAST_EVAL,
        },
    )
    body = r.json()
    saving = body["selection"]["measured_connection_saving_percent"]
    expected = {1.0: 0.0, 0.5: 50.0}[body["selection"]["optimal_ratio"]]
    assert saving == pytest.approx(expected, abs=2.0)


def test_claims_endpoint_reports_provenance_honestly(client):
    r = client.get("/v1/reservoirx-d/claims")
    assert r.status_code == 200
    body = r.json()
    by_id = {c["id"]: c for c in body["claims"]}

    # Reproduced in-service at 16 seeds; see CHANGELOG 1.1.0.
    assert by_id["regular_beats_skewed_reservoir"]["status"] == "supported"
    # Reanalysis of the original per-seed logs: the MoE arm used an unpaired test on a
    # paired design, and the CNN arm read one cell of a six-test sweep uncorrected.
    assert by_id["regular_beats_skewed_moe"]["status"] == "partially_supported"
    assert by_id["regular_beats_skewed_cnn"]["status"] == "not_supported"
    assert by_id["analysis_method_changed_two_conclusions"]["status"] == "supported"
    # Measured, not assumed: 60% of the effect survives symmetrisation.
    assert by_id["directedness_necessary"]["status"] == "not_supported"
    # The finding that contradicts the original pitch must stay in the registry.
    assert by_id["regular_vs_skewed_reverses_by_task"]["status"] == "supported"
    assert all(c["gaps"] for c in body["claims"]), "every claim must carry its gaps"
    assert not any(c["status"] == "external_unverified" for c in body["claims"])


def test_tasks_endpoint_lists_all_four(client):
    r = client.get("/v1/reservoirx-d/tasks")
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["tasks"]}
    assert names == {"memory_capacity", "delay_parity", "delay_xor", "narma10"}


def test_openapi_schema_is_valid(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/v1/reservoirx-d/compare" in schema["paths"]
    assert schema["info"]["title"] == "ReservoirX-D"


def test_unknown_fields_are_rejected(client):
    r = client.post(
        "/v1/reservoirx-d/generate",
        json={"node_count": 50, "mean_degree": 3, "typo_field": True},
    )
    assert r.status_code == 422


def test_store_evicts_and_expires():
    from app.store import InMemoryMatrixStore

    store = InMemoryMatrixStore(max_items=2, ttl_seconds=3600)
    ids = [store.put(np.eye(3), {"i": i}) for i in range(3)]
    assert store.stats()["items"] == 2
    assert store.get(ids[0]) is None
    assert store.get(ids[2]) is not None

    expiring = InMemoryMatrixStore(max_items=4, ttl_seconds=-1.0)
    stale = expiring.put(np.eye(3), {})
    assert expiring.get(stale) is None


# ---------------------------------------------------------------------------
# /analyze: the same protocol, applied to results measured elsewhere
# ---------------------------------------------------------------------------


def _records(task, treatment, baseline, seeds=None):
    seeds = seeds or list(range(len(treatment)))
    return [
        {"task": task, "seed": s, "treatment": t, "baseline": b}
        for s, t, b in zip(seeds, treatment, baseline)
    ]


def test_analyze_detects_a_planted_effect(client):
    base = [62.5, 62.1, 62.8, 62.3, 62.6, 62.4, 62.7, 62.2]
    # deltas vary per seed, as real runs do -- identical deltas give zero variance
    # and an undefined effect size, which is covered separately below
    treat = [b + d for b, d in zip(base, [0.71, 0.58, 0.69, 0.62, 0.75, 0.55, 0.66, 0.64])]
    r = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "sparse_cnn_cifar10",
            "metric_name": "test_accuracy_pct",
            "ceiling": 100.0,
            "results": _records("ratio_1", treat, base),
        },
    )
    assert r.status_code == 200
    body = r.json()
    task = body["per_task"]["ratio_1"]
    assert task["statistics"]["effect_size"] > 3
    assert task["favours"] == "regular"
    assert body["provenance"]["measurement"].startswith("performed externally")


def test_analyze_honours_lower_is_better(client):
    """MoE scores are losses: a negative difference means the treatment won."""
    base = [1.90, 1.92, 1.91, 1.93, 1.89, 1.94, 1.90, 1.92]
    treat = [b - 0.05 for b in base]
    r = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "moe_hash_routing",
            "metric_name": "val_loss",
            "higher_is_better": False,
            "results": _records("density_0.15", treat, base),
        },
    )
    assert r.status_code == 200
    assert r.json()["per_task"]["density_0.15"]["favours"] == "regular"


def test_analyze_applies_holm_across_a_sweep(client):
    """A six-ratio sweep is six tests; reading the best one uncorrected is the trap."""
    base = [62.5, 62.1, 62.8, 62.3, 62.6, 62.4, 62.7, 62.2]
    results = []
    for i, delta in enumerate([0.64, -0.11, -0.9, -1.8, -3.0, -4.5]):
        results += _records(f"ratio_{i}", [b + delta for b in base], base)
    r = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "sparse_cnn_cifar10", "results": results},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["multiple_comparison_correction"]["n_tests"] == 6
    for task in body["per_task"].values():
        assert task["statistics"]["p_holm"] >= task["statistics"]["p_value"]


def test_analyze_reports_direction_conflicts(client):
    """The reservoir arm reversed sign across tasks; this must never be summarised away."""
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    results = _records("task_a", [b + 0.5 for b in base], base)
    results += _records("task_b", [b - 0.5 for b in base], base)
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "test", "results": results},
    ).json()
    assert set(body["summary"]["directions_observed"]) == {"regular", "skewed"}
    assert "cannot be summarised" in body["summary"]["verdict"]
    assert any("changes direction" in n for n in body["notes"])


def test_analyze_warns_at_the_permutation_floor(client):
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "t", "results": _records("a", [b + 5 for b in base], base)},
    ).json()
    diag = body["per_task"]["a"]["diagnostics"]
    assert diag["smallest_attainable_p"] == pytest.approx(2 / 2**8)
    assert any("floor" in w for w in diag["warnings"])


def test_analyze_warns_when_the_metric_is_saturated(client):
    base = [99.6, 99.5, 99.7, 99.6, 99.5, 99.7, 99.6, 99.5]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "t",
            "ceiling": 100.0,
            "results": _records("a", [b + 0.1 for b in base], base),
        },
    ).json()
    assert any("saturated" in w for w in body["per_task"]["a"]["diagnostics"]["warnings"])


def test_analyze_flags_confirmation_that_reuses_selection_seeds(client):
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    treat = [b + 0.3 for b in base]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "t",
            "results": _records("a", treat, base),
            "confirmation": _records("a", treat[:4], base[:4], seeds=[0, 1, 2, 3]),
        },
    ).json()
    assert body["confirmation"]["per_task"]["a"]["held_out"] is False
    assert "not held out" in body["summary"]["verdict"]


def test_analyze_accepts_genuinely_held_out_confirmation(client):
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "t",
            "results": _records("a", [b + 0.3 for b in base], base),
            "confirmation": _records(
                "a", [1.3, 1.4, 1.2, 1.35], [1.0, 1.1, 0.9, 1.05], seeds=[10000, 10001, 10002, 10003]
            ),
        },
    ).json()
    assert body["confirmation"]["per_task"]["a"]["held_out"] is True


def test_analyze_rejects_duplicate_seeds(client):
    r = client.post(
        "/v1/reservoirx-d/analyze",
        json={
            "architecture": "t",
            "results": _records("a", [1, 2, 3, 4], [1, 2, 3, 4], seeds=[0, 0, 1, 2]),
        },
    )
    assert r.status_code == 422
    assert "duplicate seeds" in r.text


def test_analyze_rejects_too_few_pairs(client):
    r = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "t", "results": _records("a", [1, 2, 3], [1, 2, 3])},
    )
    assert r.status_code in (200, 422)


def test_analyze_notes_missing_confirmation(client):
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "t", "results": _records("a", [b + 0.3 for b in base], base)},
    ).json()
    assert any("No confirmation seeds" in n for n in body["notes"])


def test_analyze_handles_zero_variance_differences(client):
    """Identical deltas make Cohen's d undefined; it must not serialise as a silent null."""
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01]
    body = client.post(
        "/v1/reservoirx-d/analyze",
        json={"architecture": "t", "results": _records("a", [b + 0.5 for b in base], base)},
    ).json()
    task = body["per_task"]["a"]
    # d is either inf or an enormous finite number; both are flagged, not reported as real
    assert task["statistics"].get("effect_size_degenerate") is True
    assert task["statistics"]["mean_difference"] == pytest.approx(0.5)
    assert any("zero variance" in w for w in task["diagnostics"]["warnings"])


def test_every_post_endpoint_publishes_a_request_example(client):
    """
    RapidAPI's console prefills request bodies from the schema example. Without
    one it sends an empty body and the call 422s on "Field required" -- which is
    exactly what happened to /diagnose, /evaluate, /optimize and /compare in
    production while /generate, the only model that had an example, worked.
    """
    schema = client.get("/openapi.json").json()
    missing = []
    for path, item in schema["paths"].items():
        if "post" not in item:
            continue
        content = item["post"].get("requestBody", {}).get("content", {})
        ref = content.get("application/json", {}).get("schema", {}).get("$ref", "")
        model = schema["components"]["schemas"].get(ref.split("/")[-1], {})
        if "example" not in model:
            missing.append(path)
    assert not missing, f"POST endpoints with no request example: {missing}"


def test_published_examples_are_actually_valid(client):
    """An example that the API would reject is worse than no example at all."""
    schema = client.get("/openapi.json").json()
    for path, item in schema["paths"].items():
        if "post" not in item:
            continue
        content = item["post"].get("requestBody", {}).get("content", {})
        ref = content.get("application/json", {}).get("schema", {}).get("$ref", "")
        example = schema["components"]["schemas"].get(ref.split("/")[-1], {}).get("example")
        if not example:
            continue
        r = client.post(path, json=example)
        # 404 is fine: the matrix_id placeholders are not real ids.
        assert r.status_code in (200, 404), f"{path} rejected its own example: {r.text[:200]}"


def test_index_lists_every_versioned_endpoint(client):
    """
    Regression: /analyze was routed and in the schema but missing from the
    hardcoded list at /, so the index under-reported the API. This asserts the
    two can never drift again.
    """
    listed = set(client.get("/").json()["endpoints"])
    routed = {p for p in client.get("/openapi.json").json()["paths"] if p.startswith("/v1/")}
    assert listed == routed, f"index/schema mismatch: {listed ^ routed}"


def test_bodyless_post_runs_the_default_configuration(client):
    """
    API-hub consoles send an empty body. Endpoints whose fields all have
    defaults must treat that as the default request rather than 422-ing --
    that single behaviour was most of a 66% production error rate.
    """
    for path in ("/generate", "/optimize", "/compare"):
        r = client.post(f"/v1/reservoirx-d{path}", content=b"")
        assert r.status_code == 200, f"{path} rejected an empty body: {r.text[:200]}"


def test_endpoints_needing_a_matrix_still_require_a_body(client):
    """The flip side: don't silently invent a matrix nobody asked to analyse."""
    for path in ("/diagnose", "/evaluate"):
        r = client.post(f"/v1/reservoirx-d{path}", content=b"")
        assert r.status_code == 422
