"""
ReservoirX-D v1 endpoints.

Handlers are defined with `def`, not `async def`, on purpose: every one of them
is CPU-bound NumPy work, so FastAPI runs them in the threadpool instead of
blocking the event loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import API_VERSION
from ..core.analyze import AnalysisError, analyze_results
from ..core.compare import run_comparison
from ..core.evaluation import evaluate_task
from ..core.evidence import claims_document
from ..core.graphs import GraphConstructionError
from ..core.metrics import METRIC_DEFINITIONS, compute_metrics
from ..core.nulls import run_null_analysis
from ..core.optimization import run_optimization
from ..core.reservoir import generate_reservoir
from ..core.tasks import TASK_DESCRIPTIONS, TASK_NAMES, build_task
from ..models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompareRequest,
    CompareResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    EvaluateRequest,
    EvaluateResponse,
    GeneratedMatrix,
    GenerateRequest,
    GenerateResponse,
    OptimizeRequest,
    OptimizeResponse,
)
from ..store import MatrixStore

router = APIRouter()

CONTRAST_OF = {
    "degree_regular": "skewed",
    "near_regular": "skewed",
    "skewed": "degree_regular",
    "erdos_renyi": "degree_regular",
}


def get_store(request: Request) -> MatrixStore:
    return request.app.state.store


def _load_matrix(
    store: MatrixStore, matrix_id: str | None, inline: list[list[float]] | None
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if matrix_id is not None:
        stored = store.get(matrix_id)
        if stored is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"matrix_id '{matrix_id}' not found. Stored matrices expire and are held "
                    f"per-process; regenerate it with /generate, or pass the matrix inline."
                ),
            )
        return stored.W, stored.metadata

    W = np.asarray(inline, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise HTTPException(status_code=422, detail="matrix must be square and two-dimensional")
    if W.shape[0] < 3:
        raise HTTPException(status_code=422, detail="matrix must have at least 3 nodes")
    if not np.all(np.isfinite(W)):
        raise HTTPException(status_code=422, detail="matrix contains non-finite values")
    return W, None


def _as_generated(
    reservoir, matrix_id: str, metrics: dict[str, float], include_matrix: bool
) -> GeneratedMatrix:
    return GeneratedMatrix(
        matrix_id=matrix_id,
        node_count=reservoir.n,
        wiring_scheme=reservoir.wiring_scheme,
        mean_degree=reservoir.mean_degree,
        n_edges=reservoir.n_edges,
        spectral_radius_target=reservoir.spectral_radius_target,
        spectral_radius_achieved=reservoir.spectral_radius_achieved,
        weight_distribution=reservoir.weight_distribution,
        seed=reservoir.seed,
        metrics=metrics,
        matrix=reservoir.W.tolist() if include_matrix else None,
    )


# --------------------------------------------------------------------------


@router.post("/generate", response_model=GenerateResponse, summary="Generate a reservoir matrix")
def generate(req: GenerateRequest, store: MatrixStore = Depends(get_store)) -> GenerateResponse:
    """
    Generate a directed reservoir with an exact degree sequence and a rescaled
    spectral radius.

    `degree_regular` gives every node identical in- and out-degree. `near_regular`
    accepts a fractional mean degree. `skewed` and `erdos_renyi` are the contrast
    conditions, matched on edge budget.
    """
    try:
        reservoir = generate_reservoir(
            node_count=req.node_count,
            mean_degree=req.mean_degree,
            wiring_scheme=req.wiring_scheme,
            spectral_radius_target=req.spectral_radius,
            weight_distribution=req.weight_distribution,
            seed=req.seed,
            skew_cv=req.skew_cv,
        )
    except GraphConstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    metrics = compute_metrics(reservoir.W)
    matrix_id = store.put(
        reservoir.W,
        {
            "wiring_scheme": reservoir.wiring_scheme,
            "node_count": reservoir.n,
            "mean_degree": reservoir.mean_degree,
            "seed": reservoir.seed,
        },
    )
    primary = _as_generated(reservoir, matrix_id, metrics, req.return_matrix)

    contrast = None
    notes: list[str] = []
    if req.return_paired_contrast:
        contrast_scheme = CONTRAST_OF[req.wiring_scheme]
        try:
            other = generate_reservoir(
                node_count=req.node_count,
                mean_degree=req.mean_degree,
                wiring_scheme=contrast_scheme,
                spectral_radius_target=req.spectral_radius,
                weight_distribution=req.weight_distribution,
                seed=req.seed,
                skew_cv=req.skew_cv,
            )
        except GraphConstructionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        other_metrics = compute_metrics(other.W)
        other_id = store.put(
            other.W,
            {
                "wiring_scheme": other.wiring_scheme,
                "node_count": other.n,
                "mean_degree": other.mean_degree,
                "seed": other.seed,
            },
        )
        contrast = _as_generated(other, other_id, other_metrics, req.return_matrix)
        if primary.n_edges != contrast.n_edges:
            notes.append(
                f"Edge budgets differ ({primary.n_edges} vs {contrast.n_edges}) because integer "
                f"degree sequences cannot always match exactly. Any comparison between them is "
                f"confounded by budget; use /compare, which reports budget matching explicitly."
            )
        else:
            notes.append("Edge budgets match exactly, so the two differ only in degree shape.")

    notes.append(
        "A structural difference is not a performance difference. Use /compare to test whether "
        "this wiring actually outperforms the contrast on your task."
    )
    return GenerateResponse(primary=primary, contrast=contrast, notes=notes)


@router.post("/diagnose", response_model=DiagnoseResponse, summary="Structural diagnostics")
def diagnose(req: DiagnoseRequest, store: MatrixStore = Depends(get_store)) -> DiagnoseResponse:
    """
    Structural metrics plus null-model Z-scores and empirical p-values.

    Note what is deliberately absent: there is no `symmetrization_risk` field.
    Estimating a performance drop from a structural proxy such as Henrici
    departure would imply a calibration that does not exist -- the evidence for
    the symmetrisation collapse is a single external experiment. Measure it
    instead with `include_symmetrization_control` on /compare.
    """
    W, _ = _load_matrix(store, req.matrix_id, req.matrix)
    observed = compute_metrics(W)
    nulls = run_null_analysis(W, observed, req.null_models, req.iterations, seed=req.seed)

    symmetry = observed.get("symmetry_index", 0.0)
    reciprocity = observed.get("reciprocity", 0.0)
    directedness = {
        "symmetry_index": symmetry,
        "reciprocity": reciprocity,
        "is_effectively_symmetric": bool(symmetry > 0.95),
        "assessment": _directedness_assessment(symmetry, reciprocity),
        "how_to_test_the_consequence": (
            "Set include_symmetrization_control=true on /compare. That symmetrises these same "
            "matrices and re-measures the effect, which is an experiment rather than an "
            "extrapolation from this number."
        ),
    }

    notes = [
        "Z-scores against 'degree_preserving' are the informative ones: that null holds the exact "
        "degree sequence fixed, so anything extreme against it is about wiring beyond degrees.",
        "p-values here are uncorrected and there are many metrics. Treat them as screening, not "
        "as confirmation.",
    ]
    return DiagnoseResponse(
        matrix_id=req.matrix_id,
        raw_metrics=observed,
        metric_definitions=METRIC_DEFINITIONS,
        null_analysis=nulls,
        directedness=directedness,
        notes=notes,
    )


def _directedness_assessment(symmetry: float, reciprocity: float) -> str:
    if symmetry > 0.95:
        return (
            "This matrix is essentially symmetric. If you built it by taking an undirected graph "
            "and calling to_directed(), that is the likely cause, and it removes the directedness "
            "the ReservoirX-D construction is designed to provide."
        )
    if symmetry > 0.5:
        return "Substantially symmetric; a large share of edges are reciprocated with matched weights."
    if reciprocity > 0.3:
        return "Topologically reciprocal in part, though the weights are not mirrored."
    return "Strongly directed: few reciprocated edges and no weight-level symmetry."


@router.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate a reservoir on a task")
def evaluate(req: EvaluateRequest, store: MatrixStore = Depends(get_store)) -> EvaluateResponse:
    """
    Drive the reservoir and fit a ridge readout on held-out data.

    States are bounded by the tanh activation, so divergence is not a failure
    mode and there is no rejection sampling. The real risks -- saturation and
    collapse -- are reported in `stability` and flagged in `warnings` rather
    than silently worked around.
    """
    W, _ = _load_matrix(store, req.matrix_id, req.matrix)
    params = req.evaluation_params

    task = build_task(
        req.task,
        sequence_length=params.sequence_length,
        rng=np.random.default_rng(req.seed if req.seed is not None else 0),
        max_lag=params.max_lag,
    )
    result = evaluate_task(
        W,
        task,
        ridge_alpha=params.ridge_alpha,
        washout_steps=params.washout_steps,
        train_split=params.train_split,
        input_scaling=params.input_scaling,
        leak_rate=params.leak_rate,
        seed=req.seed,
    )
    return EvaluateResponse(
        matrix_id=req.matrix_id,
        task=result.task,
        task_description=TASK_DESCRIPTIONS[result.task],
        aggregate_name=result.aggregate_name,
        score=result.score,
        per_target=result.per_target,
        stability=result.stability,
        n_train=result.n_train,
        n_test=result.n_test,
        evaluation_params=params.model_dump(),
        warnings=result.warnings,
    )


@router.post("/optimize", response_model=OptimizeResponse, summary="Find the smallest viable width")
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """
    Search for the smallest width that stays non-inferior to full width.

    Selection runs on one set of seeds and confirmation on another, so the
    reported result is not scored on the data that chose it. Connection savings
    are counted from the generated graphs; no fixed shrink ratio is assumed.
    """
    try:
        result = run_optimization(
            base_nodes=req.base_nodes,
            mean_degree=req.mean_degree,
            spectral_radius=req.spectral_radius,
            wiring_scheme=req.wiring_scheme,
            weight_distribution=req.weight_distribution,
            target_metric=req.target_metric,
            ratios=req.ratios,
            min_acceptable_ratio=req.min_acceptable_ratio,
            non_inferiority_margin_fraction=req.non_inferiority_margin_fraction,
            selection_seeds=req.selection_seeds,
            confirmation_seeds=req.confirmation_seeds,
            eval_params=req.evaluation_params.model_dump(),
            base_seed=req.base_seed,
        )
    except GraphConstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OptimizeResponse(**result)


@router.post("/compare", response_model=CompareResponse, summary="Test regular against skewed")
def compare(req: CompareRequest) -> CompareResponse:
    """
    Paired regular-vs-skewed comparison on your configuration.

    Conditions share seeds, edge budgets are reported, multiple tasks are
    Holm-corrected, and the symmetrisation control -- when enabled -- measures
    the directedness hypothesis rather than assuming it.

    This endpoint is the point of the service. The wiring effect is not treated
    as a given; it is the thing being tested.
    """
    try:
        result = run_comparison(
            node_count=req.node_count,
            mean_degree=req.mean_degree,
            spectral_radius=req.spectral_radius,
            weight_distribution=req.weight_distribution,
            iterations=req.iterations,
            tasks=list(req.tasks),
            treatment_scheme=req.treatment_scheme,
            baseline_scheme=req.baseline_scheme,
            skew_cv=req.skew_cv,
            include_symmetrization_control=req.include_symmetrization_control,
            eval_params=req.evaluation_params.model_dump(),
            base_seed=req.base_seed,
        )
    except GraphConstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CompareResponse(**result)


@router.post("/analyze", response_model=AnalyzeResponse,
             summary="Apply the /compare protocol to results measured elsewhere")
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the paired analysis on per-seed scores this service did not produce.

    /compare handles reservoirs, where a run is milliseconds. Architectures that
    take hours per run -- the CIFAR sparse CNN, the MoE transformer -- have to
    train wherever the hardware is. Submit their per-seed scores here and they go
    through the identical protocol: paired Cohen's d, sign-flip permutation,
    bootstrap CIs, Holm correction across tasks, permutation-floor and ceiling
    detection, and held-out confirmation.

    The response says plainly that this service did not measure these numbers.
    """
    try:
        result = analyze_results(
            architecture=req.architecture,
            metric_name=req.metric_name,
            results=[r.model_dump() for r in req.results],
            higher_is_better=req.higher_is_better,
            ceiling=req.ceiling,
            confirmation=[r.model_dump() for r in req.confirmation] if req.confirmation else None,
            treatment_label=req.treatment_label,
            baseline_label=req.baseline_label,
            non_inferiority_margin_fraction=req.non_inferiority_margin_fraction,
            alpha=req.alpha,
            seed=req.seed,
        )
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AnalyzeResponse(**result)


@router.get("/claims", summary="Evidence behind every claim this service makes")
def claims() -> dict[str, Any]:
    """
    The provenance of the results ReservoirX-D is built on, with status codes.

    Served as an endpoint rather than buried in a README because two of the
    headline claims come from an external pipeline that was never reproduced,
    and anyone building on this API should be able to see that programmatically.
    """
    return claims_document()


@router.get("/tasks", summary="Available benchmark tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {"name": name, "description": TASK_DESCRIPTIONS[name]} for name in TASK_NAMES
        ],
        "note": "Run more than one. The reservoir effect reported in the source benchmark "
                "appeared on 2 of 4 tasks; parity and XOR showed nothing.",
    }


@router.get("/version", summary="Service version")
def version() -> dict[str, str]:
    return {"name": "ReservoirX-D", "version": API_VERSION}
