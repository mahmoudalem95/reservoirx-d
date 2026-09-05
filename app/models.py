"""
Request and response schemas.

The one that matters: `mean_degree` is validated against `wiring_scheme`. A
degree-regular digraph gives every node the same integer in- and out-degree, so
`mean_degree: 2.7` with `wiring_scheme: "degree_regular"` is not a rounding
question, it is a contradiction. The blueprint's example request contained
exactly that pair. Rather than silently rounding, the API rejects it and points
at `near_regular`, which realises a fractional mean by splitting nodes between
degree 2 and 3.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings

WiringScheme = Literal["degree_regular", "near_regular", "skewed", "erdos_renyi"]
WeightDistribution = Literal["uniform", "normal", "bimodal"]
TaskName = Literal["memory_capacity", "delay_parity", "delay_xor", "narma10"]
NullModel = Literal["erdos_renyi", "degree_preserving"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# shared blocks
# --------------------------------------------------------------------------


class EvaluationParams(StrictModel):
    ridge_alpha: float = Field(1e-6, gt=0, le=1e6, description="L2 penalty on the readout")
    washout_steps: int = Field(100, ge=0, le=10000)
    train_split: float = Field(0.7, gt=0.05, lt=0.99)
    max_lag: int = Field(20, ge=1, le=200, description="Number of lags/orders scored per task")
    input_scaling: float = Field(0.1, gt=0, le=10.0)
    leak_rate: float = Field(1.0, gt=0, le=1.0, description="1.0 = no leaky integration")
    sequence_length: int = Field(1000, ge=200, le=settings.max_sequence_length)

    @model_validator(mode="after")
    def _washout_fits(self) -> "EvaluationParams":
        if self.washout_steps >= self.sequence_length:
            raise ValueError("washout_steps must be smaller than sequence_length")
        return self


class WiringSpec(StrictModel):
    node_count: int = Field(200, ge=3, le=settings.max_node_count)
    mean_degree: float = Field(3.0, gt=0)
    wiring_scheme: WiringScheme = "degree_regular"
    spectral_radius: float = Field(0.9, gt=0, le=5.0)
    weight_distribution: WeightDistribution = "uniform"
    skew_cv: float = Field(1.2, gt=0, le=5.0, description="Target degree CV for the skewed scheme")

    @model_validator(mode="after")
    def _degree_is_realisable(self) -> "WiringSpec":
        _validate_degree(self.mean_degree, self.node_count, self.wiring_scheme)
        return self


def _validate_degree(mean_degree: float, node_count: int, scheme: str) -> None:
    if mean_degree >= node_count:
        raise ValueError(
            f"mean_degree={mean_degree} must be below node_count={node_count}; a simple digraph "
            f"gives each node at most {node_count - 1} distinct targets"
        )
    if scheme == "degree_regular" and abs(mean_degree - round(mean_degree)) > 1e-9:
        raise ValueError(
            f"wiring_scheme='degree_regular' requires an integer mean_degree, got {mean_degree}. "
            f"Every node in a degree-regular digraph has the same in- and out-degree, so a "
            f"fractional degree is not realisable. Use wiring_scheme='near_regular' to get a "
            f"mean of {mean_degree} by splitting nodes between degree {int(mean_degree)} and "
            f"{int(mean_degree) + 1} (CV stays near zero), or round mean_degree to an integer."
        )


# --------------------------------------------------------------------------
# /generate
# --------------------------------------------------------------------------


class GenerateRequest(WiringSpec):
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    return_matrix: bool = Field(False, description="Inline the full n x n matrix in the response")
    return_paired_contrast: bool = Field(
        False,
        description="Also generate the contrasting wiring at the same seed and edge budget, "
        "for a like-for-like structural comparison",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "node_count": 200,
                "mean_degree": 3,
                "wiring_scheme": "degree_regular",
                "spectral_radius": 0.9,
                "weight_distribution": "uniform",
                "seed": 42,
            }
        },
    )


class GeneratedMatrix(BaseModel):
    matrix_id: str
    node_count: int
    wiring_scheme: str
    mean_degree: float
    n_edges: int
    spectral_radius_target: float
    spectral_radius_achieved: float
    weight_distribution: str
    seed: int | None
    metrics: dict[str, float]
    matrix: list[list[float]] | None = None


class GenerateResponse(BaseModel):
    primary: GeneratedMatrix
    contrast: GeneratedMatrix | None = None
    notes: list[str] = []


# --------------------------------------------------------------------------
# /diagnose
# --------------------------------------------------------------------------


class DiagnoseRequest(StrictModel):
    matrix_id: str | None = None
    matrix: list[list[float]] | None = None
    null_models: list[NullModel] = ["erdos_renyi", "degree_preserving"]
    iterations: int = Field(100, ge=10, le=settings.max_null_iterations)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _one_source(self) -> "DiagnoseRequest":
        if (self.matrix_id is None) == (self.matrix is None):
            raise ValueError("provide exactly one of matrix_id or matrix")
        return self

    # Without an example the RapidAPI console sends an empty body and every call
    # 422s on "Field required". Each POST model carries one for that reason.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "matrix_id": "rxd_replace_with_an_id_from_generate",
                "null_models": ["erdos_renyi", "degree_preserving"],
                "iterations": 100,
                "seed": 0,
            }
        },
    )


class DiagnoseResponse(BaseModel):
    matrix_id: str | None
    raw_metrics: dict[str, float]
    metric_definitions: dict[str, str]
    null_analysis: dict[str, Any]
    directedness: dict[str, Any]
    notes: list[str] = []


# --------------------------------------------------------------------------
# /evaluate
# --------------------------------------------------------------------------


class EvaluateRequest(StrictModel):
    matrix_id: str | None = None
    matrix: list[list[float]] | None = None
    task: TaskName = "memory_capacity"
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    evaluation_params: EvaluationParams = EvaluationParams()

    @model_validator(mode="after")
    def _one_source(self) -> "EvaluateRequest":
        if (self.matrix_id is None) == (self.matrix is None):
            raise ValueError("provide exactly one of matrix_id or matrix")
        return self

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "matrix_id": "rxd_replace_with_an_id_from_generate",
                "task": "memory_capacity",
                "seed": 0,
                "evaluation_params": {"sequence_length": 3000, "max_lag": 60},
            }
        },
    )


class EvaluateResponse(BaseModel):
    matrix_id: str | None
    task: str
    task_description: str
    aggregate_name: str
    score: float
    per_target: dict[str, float]
    stability: dict[str, float]
    n_train: int
    n_test: int
    evaluation_params: dict[str, Any]
    warnings: list[str] = []


# --------------------------------------------------------------------------
# /optimize
# --------------------------------------------------------------------------


class OptimizeRequest(StrictModel):
    base_nodes: int = Field(80, ge=10, le=settings.max_node_count)
    mean_degree: float = Field(3.0, gt=0)
    wiring_scheme: WiringScheme = "degree_regular"
    spectral_radius: float = Field(0.9, gt=0, le=5.0)
    weight_distribution: WeightDistribution = "uniform"
    target_metric: TaskName = "memory_capacity"
    ratios: list[float] | None = Field(
        None, description="Width ratios to test. Defaults to [1.0, 0.8, 0.6] to keep a bodyless "
        "call fast; pass [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] for a full sweep."
    )
    min_acceptable_ratio: float = Field(0.5, gt=0.05, le=1.0)
    non_inferiority_margin_fraction: float = Field(
        0.02, gt=0, le=0.5, description="A ratio passes if its CI lower bound sits above this "
        "fraction of the baseline score, measured downwards"
    )
    # Defaults are a smoke test, not an experiment. A bodyless POST has to finish
    # inside an API gateway timeout (RapidAPI cuts at 180s) on a 0.1-CPU host,
    # where work taking 0.65s locally took 115s. Nothing statistical can be
    # concluded from a run at these settings -- they exist so the endpoint
    # returns a well-formed response you can read the shape of. For real work:
    # base_nodes=200, selection_seeds=8, confirmation_seeds=4, the full ratio
    # sweep, and max_lag high enough to avoid the ceiling.
    selection_seeds: int = Field(3, ge=2, le=settings.max_iterations)
    confirmation_seeds: int = Field(2, ge=2, le=settings.max_iterations)
    base_seed: int = Field(0, ge=0, le=2**31 - 1)
    evaluation_params: EvaluationParams = EvaluationParams()

    @field_validator("ratios")
    @classmethod
    def _ratio_bounds(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("ratios must not be empty")
        if len(v) > settings.max_ratios_per_request:
            raise ValueError(f"at most {settings.max_ratios_per_request} ratios per request")
        if any(r <= 0 or r > 1.0 for r in v):
            raise ValueError("every ratio must lie in (0, 1]")
        return v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "base_nodes": 200,
                "mean_degree": 3,
                "target_metric": "memory_capacity",
                "min_acceptable_ratio": 0.5,
                "selection_seeds": 4,
                "confirmation_seeds": 2,
                "ratios": [1.0, 0.8, 0.6],
                # max_lag matters: at the default of 20 a 200-node reservoir sits at
                # its ceiling and every width passes non-inferiority trivially.
                "evaluation_params": {"sequence_length": 4000, "max_lag": 120},
            }
        },
    )

    @model_validator(mode="after")
    def _degree_ok(self) -> "OptimizeRequest":
        smallest = max(3, int(round(self.base_nodes * self.min_acceptable_ratio)))
        _validate_degree(self.mean_degree, smallest, self.wiring_scheme)
        return self


class OptimizeResponse(BaseModel):
    configuration: dict[str, Any]
    sweep: list[dict[str, Any]]
    selection: dict[str, Any]
    confirmation: dict[str, Any]
    caveats: list[str]
    warnings: list[str] = []


# --------------------------------------------------------------------------
# /compare
# --------------------------------------------------------------------------


class CompareRequest(StrictModel):
    node_count: int = Field(80, ge=10, le=settings.max_node_count)
    mean_degree: float = Field(3.0, gt=0)
    spectral_radius: float = Field(0.9, gt=0, le=5.0)
    weight_distribution: WeightDistribution = "uniform"
    # 3 is a smoke-test default, not a statistical one: with n pairs the smallest
    # attainable permutation p-value is ~2/2^n, so 3 seeds bottom out at 0.25 and
    # NOTHING can reach significance. Use 8 to detect an effect at all and 16 for
    # resolution below the floor. The response warns when you are at the floor.
    iterations: int = Field(3, ge=3, le=settings.max_iterations)
    tasks: list[TaskName] = ["memory_capacity"]
    treatment_scheme: WiringScheme = "degree_regular"
    baseline_scheme: WiringScheme = "skewed"
    skew_cv: float = Field(1.2, gt=0, le=5.0)
    include_symmetrization_control: bool = Field(
        False,
        description="Also run both conditions with the same matrices symmetrised, to measure "
        "whether directedness is carrying the effect. Doubles the runtime.",
    )
    base_seed: int = Field(0, ge=0, le=2**31 - 1)
    evaluation_params: EvaluationParams = EvaluationParams()

    @field_validator("tasks")
    @classmethod
    def _task_limit(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one task is required")
        if len(set(v)) != len(v):
            raise ValueError("duplicate tasks are not allowed")
        if len(v) > settings.max_tasks_per_request:
            raise ValueError(f"at most {settings.max_tasks_per_request} tasks per request")
        return v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "node_count": 200,
                "mean_degree": 3,
                "iterations": 8,
                "tasks": ["memory_capacity", "delay_parity", "delay_xor", "narma10"],
                "include_symmetrization_control": True,
                "evaluation_params": {"sequence_length": 4000, "max_lag": 120},
            }
        },
    )

    @model_validator(mode="after")
    def _schemes_and_degree(self) -> "CompareRequest":
        if self.treatment_scheme == self.baseline_scheme:
            raise ValueError("treatment_scheme and baseline_scheme must differ")
        _validate_degree(self.mean_degree, self.node_count, self.treatment_scheme)
        _validate_degree(self.mean_degree, self.node_count, self.baseline_scheme)
        return self


class CompareResponse(BaseModel):
    configuration: dict[str, Any]
    per_task: dict[str, Any]
    multiple_comparison_correction: dict[str, Any]
    summary: dict[str, Any]
    warnings: list[str] = []


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


class ResultRecord(StrictModel):
    task: str = Field(..., min_length=1, max_length=100)
    seed: int = Field(..., ge=0, le=2**31 - 1)
    treatment: float
    baseline: float


class AnalyzeRequest(StrictModel):
    architecture: str = Field(..., min_length=1, max_length=100,
                              description="e.g. 'sparse_cnn' or 'moe_hash_routing'")
    metric_name: str = Field("score", min_length=1, max_length=100)
    higher_is_better: bool = True
    ceiling: float | None = Field(
        None, description="Theoretical maximum of the metric, if it has one (e.g. 100 for "
        "accuracy). Enables saturation detection."
    )
    treatment_label: str = Field("regular", min_length=1, max_length=60)
    baseline_label: str = Field("skewed", min_length=1, max_length=60)
    results: list[ResultRecord] = Field(..., min_length=3, max_length=2000)
    confirmation: list[ResultRecord] | None = Field(
        None, max_length=2000,
        description="Held-out seeds, never used to choose what to report."
    )
    non_inferiority_margin_fraction: float | None = Field(None, gt=0, le=0.5)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    seed: int = Field(0, ge=0, le=2**31 - 1)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "architecture": "sparse_cnn_cifar10",
                "metric_name": "test_accuracy_pct",
                "higher_is_better": True,
                "ceiling": 100.0,
                "results": [
                    {"task": "ratio_100", "seed": 0, "treatment": 63.55, "baseline": 62.87},
                    {"task": "ratio_100", "seed": 1, "treatment": 63.77, "baseline": 63.19},
                    {"task": "ratio_100", "seed": 2, "treatment": 62.93, "baseline": 62.24},
                    {"task": "ratio_100", "seed": 3, "treatment": 62.62, "baseline": 62.00},
                    {"task": "ratio_100", "seed": 4, "treatment": 63.39, "baseline": 62.64},
                    {"task": "ratio_100", "seed": 5, "treatment": 62.31, "baseline": 61.76},
                    {"task": "ratio_100", "seed": 6, "treatment": 63.66, "baseline": 62.92},
                    {"task": "ratio_100", "seed": 7, "treatment": 63.07, "baseline": 62.39},
                ],
            }
        },
    )

    @model_validator(mode="after")
    def _labels_differ(self) -> "AnalyzeRequest":
        if self.treatment_label == self.baseline_label:
            raise ValueError("treatment_label and baseline_label must differ")
        return self


class AnalyzeResponse(BaseModel):
    architecture: str
    metric_name: str
    higher_is_better: bool
    conditions: dict[str, str]
    per_task: dict[str, Any]
    multiple_comparison_correction: dict[str, Any]
    confirmation: dict[str, Any] | None
    summary: dict[str, Any]
    provenance: dict[str, str]
    notes: list[str] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    store: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None
