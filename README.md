# ReservoirX-D

Generate directed, degree-regular reservoir weight matrices — and test whether that wiring actually helps on your configuration, instead of taking it on faith.

FastAPI service. No database required. Deterministic given a seed. 59 tests.

---

## Why this is not just a matrix generator

The regular-degree connectivity thesis had mixed support across architectures, and the reservoir
result the service is named after was never independently reproduced. So ReservoirX-D ships the
experiment rather than the conclusion — and then ran it on itself.

The primary claim reproduced: d = +3.05 on memory capacity at 16 seeds, matching the external
figure almost exactly. Two others did not survive contact with measurement. Directedness turned
out not to be necessary, and the wiring advantage reverses sign on a non-linear task. `GET /claims`
returns all of it, with provenance and gaps, and no claim in the registry is marked
`external_unverified` any more.

* `POST /compare` runs the comparison on **your** parameters, paired by seed, matched on edge
  budget, Holm-corrected across tasks, with an optional symmetrisation control that measures the
  directedness hypothesis rather than estimating it.
* `POST /optimize` selects a width on one set of seeds and confirms it on another, and reports
  connection savings it actually counted.
* `GET /claims` is the evidence registry, served as JSON.

If the effect isn't there on your configuration, the API says so.

## Quick start

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8080
# interactive docs at http://localhost:8080/docs
```

Docker:

```bash
docker build -t reservoirx-d .
docker run --rm -p 8080:8080 reservoirx-d
```

Tests:

```bash
make test          # 59 tests, ~3 seconds
```

---

## Endpoints

All under `/v1/reservoirx-d`.

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/generate` | POST | Directed matrix with an exact degree sequence and rescaled spectral radius |
| `/diagnose` | POST | Structural metrics plus null-model Z-scores and p-values |
| `/evaluate` | POST | Score a matrix on a benchmark task with a ridge readout |
| `/optimize` | POST | Smallest non-inferior width, with held-out confirmation |
| `/compare` | POST | Regular vs skewed, paired and corrected |
| `/analyze` | POST | The same protocol applied to results measured elsewhere |
| `/claims` | GET | Evidence registry with provenance |
| `/tasks` | GET | The four benchmark tasks |
| `/version` | GET | Service version |
| `/health` | GET | Liveness and store state |

### Generate

```bash
curl -X POST http://localhost:8080/v1/reservoirx-d/generate \
  -H 'Content-Type: application/json' \
  -d '{"node_count": 200, "mean_degree": 3, "wiring_scheme": "degree_regular",
       "spectral_radius": 0.9, "seed": 42}'
```

Every node gets out-degree and in-degree exactly 3. No self-loops, no parallel edges, reciprocity around `d/(n-1)`.

Add `"return_paired_contrast": true` to get the skewed counterpart at the same seed and edge budget.

### Compare — the endpoint that matters

```bash
curl -X POST https://reservoirx-d.fly.dev/v1/reservoirx-d/compare \
  -H 'Content-Type: application/json' \
  -d '{"node_count": 200, "mean_degree": 3, "iterations": 16,
       "tasks": ["memory_capacity", "delay_parity", "delay_xor", "narma10"],
       "include_symmetrization_control": true,
       "evaluation_params": {"max_lag": 120, "sequence_length": 4000}}'
```

This is what the endpoint found on its own service, at 16 paired seeds, replicated:

| Task | Cohen's d | p (Holm) | Favours |
| --- | --- | --- | --- |
| memory_capacity | **+3.05** | 0.00005 | regular |
| narma10 | +1.04 | 0.0016 | regular |
| delay_parity | **−1.54** | 0.0002 | **skewed** |
| delay_xor | +0.09 | 0.73 | neither |

Two things in that table matter more than the headline.

**The advantage reverses.** Delayed parity favours *skewed* wiring at d = −1.54, with a confidence
interval clear of zero, replicated at 8 and 16 seeds. Regular-degree connectivity is not simply
better; which wiring wins depends on the task.

**Directedness is not necessary.** The symmetrisation control ran the same matrices through
`(W + Wᵀ)/2`, rescaled to the original spectral radius, and re-measured: memory capacity went
from d=3.05 to d=1.84. Sixty percent of the effect survives. The original claim — that
symmetrisation collapses the advantage — does not hold here.

### Optimize

```bash
curl -X POST http://localhost:8080/v1/reservoirx-d/optimize \
  -H 'Content-Type: application/json' \
  -d '{"base_nodes": 200, "mean_degree": 3, "target_metric": "memory_capacity",
       "min_acceptable_ratio": 0.5, "evaluation_params": {"max_lag": 120, "sequence_length": 4000}}'
```

Note `max_lag`. Memory capacity is bounded by the number of lags scored, so at `max_lag=20` a 200-node reservoir sits at 19.9 out of 20 and *every* width looks non-inferior. The API detects this and returns `metric_saturated: true` with a caveat telling you the selection carries no information. With a discriminating `max_lag`, the same configuration permits no shrink at all — so the 90% width rule from the blueprint does not transfer to reservoirs.

### Diagnose

```bash
curl -X POST http://localhost:8080/v1/reservoirx-d/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"matrix_id": "rxd_...", "iterations": 100}'
```

Two nulls. `erdos_renyi` fixes only the edge count; `degree_preserving` fixes the exact degree sequence and rewires everything else, so a metric still extreme against it is telling you about wiring beyond degrees.

The response also flags accidentally symmetric matrices — the `to_directed()` trap that silently removes the property the whole method depends on.

---

## Tasks

| Task | What it measures |
| --- | --- |
| `memory_capacity` | Linear short-term memory; sum of per-lag squared correlation, bounded by `max_lag` |
| `delay_parity` | Non-linear memory; product of the last *k* binary inputs |
| `delay_xor` | Two-tap non-linear memory |
| `narma10` | Order-10 non-linear autoregressive moving average; `1 − NRMSE` |

Run more than one. An API that only exposed memory capacity could only ever confirm the favourable case.

---

## What changed from the original blueprint

**Bugs fixed**

- `nx.random_regular_directed_graph` does not exist in NetworkX (checked against 3.6.1). The obvious substitute, `random_regular_graph(...).to_directed()`, returns a symmetric matrix and destroys the directedness the method depends on. Replaced with configuration-model stub pairing plus directed double-edge swaps, which repair self-loops and parallel edges while preserving the degree sequence exactly.
- `mean_degree: 2.7` with `wiring_scheme: "degree_regular"` is a contradiction — a degree-regular digraph has one integer degree for every node. Now rejected with a pointer to `near_regular`, which realises a fractional mean by splitting nodes between degree 2 and 3 while keeping CV near zero.
- Rejection sampling on divergence removed. With a `tanh` activation the state is analytically bounded in [−1, 1], so it cannot diverge; the guard could never fire for the stated reason. Worse, it responded by *resampling the input*, which silently changes the task so the returned states no longer correspond to the caller's signal. The real failure modes — saturation and collapse — are now measured and reported in `stability`, not worked around.
- `drive_reservoir` referenced an undefined `n` and multiplied by a no-op `A @ zeros`.
- Reciprocity divided a halved mutual-pair count by the total edge count, giving half the conventional value. Now the standard Newman definition.
- `np.random.seed()` set global state, which is a reproducibility and thread-safety hazard under async FastAPI. Everything uses `np.random.default_rng(seed)`; a test asserts no global-state leak.
- The Henrici formula in the blueprint was a normality-departure ratio, not the Henrici index. Both are now returned under separate, documented names.
- `matrix_rank` saturates at *n* for essentially every random matrix. `effective_rank` (entropy of the singular-value spectrum) is reported alongside it.
- Spectral radius was computed twice per matrix. Rescaling is linear, so once is enough.

**Claims corrected**

- The 90% width rule and the 16.2% connection saving came from a CIFAR-10 sparse-CNN sweep. They are not reservoir properties and no longer appear as constants anywhere. `/optimize` measures savings from actual edge counts.
- The `/compare` and `/optimize` example responses no longer ship the expected answer pre-filled. An endpoint that advertises its result in advance is a confirmation ritual, not a test.
- `symmetrization_risk` is gone. Estimating a performance drop from Henrici departure implies a calibration that does not exist — the evidence is one external experiment. Use `include_symmetrization_control` to measure it instead.
- The CNN result was non-inferiority at 90% width, not a win: the confidence interval spans zero. `/claims` records it as `partially_supported` with that gap stated.
- The MoE arm found one hit across four uncorrected density tests and failed its fresh-seed check. It is recorded as `not_supported`, and Holm correction is applied wherever this API runs a sweep.

**Added**

- Held-out confirmation in `/optimize`, mirroring the benchmark's own protocol.
- Ceiling detection, so a saturated metric cannot produce a bogus shrink recommendation.
- Three tasks beyond memory capacity, including the two the source benchmark found no effect on.
- `/claims` as a machine-readable evidence registry.
- Redis store backend, request caps, request IDs, structured errors.

---

## Deployment

Environment variables in `.env.example`. The defaults are single-worker: the in-memory matrix store is per-process, so `matrix_id` values issued by one worker are invisible to the others. Before scaling out, set `RXD_STORE_BACKEND=redis`.

Every endpoint is synchronous CPU work — handlers are declared `def` so FastAPI runs them in the threadpool rather than blocking the event loop. A `/compare` over 8 seeds and 4 tasks is 64 reservoir simulations. Caps in `config.py` keep one request from occupying a worker for minutes; for larger jobs, put them behind a queue rather than a request.

```bash
python scripts/export_openapi.py   # openapi.json for API-hub upload
```

## Analysing other architectures

`/compare` runs the whole experiment in-process because a reservoir simulation takes
milliseconds. A CIFAR-10 sparse CNN takes ~100 minutes per run and a MoE transformer ~150, so
those arms train wherever the hardware is — and then submit their per-seed scores to `/analyze`,
which applies the identical protocol. One analysis across all three architectures rather than
three analyses sharing a vocabulary.

```bash
# on your training machine
python3 scripts/enable_mps.py regular_degree_cnn_benchmark.py moe_routing_validation.py
QUICK=1 python3 regular_degree_cnn_benchmark.py          # smoke test first
python3 regular_degree_cnn_benchmark.py                  # the real run

python3 scripts/to_analysis_payload.py results/results.json -o cnn_payload.json
curl -X POST https://reservoirx-d.fly.dev/v1/reservoirx-d/analyze \
  -H 'Content-Type: application/json' -d @cnn_payload.json
```

`enable_mps.py` makes Apple Silicon a first-class device in both harnesses — two lines changed
per file, originals backed up. Note that switching device changes floating-point numerics and the
RNG stream, so MPS results are a fresh replication rather than a continuation of your CUDA logs;
don't mix seeds from the two in one paired test.

`/analyze` states plainly in every response that it did not produce the numbers it analysed.

## Changelog

### 1.3.1

- Fixed: every POST endpoint except `/generate` returned 422 "Field required" when called from
  an API-hub console. The console prefills request bodies from the schema `example`, and only
  `GenerateRequest` had one — so the others were called with an empty body. All six POST models
  now publish a valid example, with tests asserting both that the example exists and that the
  API accepts it.
- `/analyze` added to the endpoint list served at `/`. The route worked and was in the schema,
  but the hardcoded index list was never updated.

### 1.3.0

Reanalysis of the original CNN and MoE benchmark logs through `/analyze`. No retraining — the
per-seed numbers were already in the logs. Two of the three original conclusions changed, in
opposite directions, purely from how the numbers were analysed.

- `regular_beats_skewed_cnn`: `partially_supported` → **`not_supported`**. Reported as SUPPORTED
  at p=0.030, but that was one cell of a six-ratio sweep with no correction. Holm gives p=0.069.
- `regular_beats_skewed_moe`: `not_supported` → **`partially_supported`**. The original used an
  unpaired label-shuffle on a paired design (reproduced here at p=0.084 vs the logged 0.085).
  The correct paired sign-flip gives p=0.008, and all 8 of 8 seeds favour regular routing.
  Two of four densities significant after Holm. Held back from `supported` because the
  fresh-seed check can't be redone — the source log is truncated.
- New `analysis_method_changed_two_conclusions` recording the pattern.

### 1.2.0

- New `POST /analyze`: paired Cohen's d, sign-flip permutation, bootstrap CIs, Holm correction,
  ceiling detection, permutation-floor warnings and held-out confirmation, applied to scores
  measured outside this service.
- `scripts/enable_mps.py` and `scripts/to_analysis_payload.py` for the CNN and MoE harnesses.
- Fixed: a degenerate effect size (paired differences with near-zero variance) was reported as a
  huge finite number that read like a colossal effect. It is now flagged as
  `effect_size_degenerate` with the advice to read `mean_difference` instead. Non-finite values
  in any statistics block now serialise as `null` rather than becoming invalid JSON.
- The CNN width sweep is six tests. Holm correction now applies across it; reading the best ratio
  uncorrected was how a null could have become a finding.

### 1.1.0

The reservoir arm was reproduced in-service and the evidence registry rewritten around the
measurements. No code behaviour changed; the claims did.

- `regular_beats_skewed_reservoir`: `external_unverified` → **`supported`**. d = +3.05,
  p = 0.00005, CI [+8.00, +10.93], 16 seeds, edge budgets matched.
- `directedness_necessary`: `external_unverified` → **`not_supported`**. Symmetrisation retains
  60% of the effect (3.05 → 1.84); the source reported 24% (3.05 → 0.74).
- New `regular_vs_skewed_reverses_by_task`: delayed parity favours skewed at d = −1.54.
- New `measurement_ceiling_artifact`: at `max_lag=20` the same comparison measures d = +1.42
  instead of +3.05, because memory capacity saturates against its ceiling.
- No claim in the registry is `external_unverified` any more.

## License

MIT.
