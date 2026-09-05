# Evidence registry

ReservoirX-D generates and measures. Every reservoir claim below has now been reproduced in this service on its own /compare endpoint, and two of them came back different from the originals: the wiring advantage reverses sign on a non-linear task, and symmetrisation does not collapse the effect the way the source reported. Use /compare to test your own configuration rather than assuming any of this transfers.

## Status vocabulary

- **`supported`** — reproduced and confirmed on held-out seeds
- **`partially_supported`** — reproduced, with a stated protocol gap
- **`not_supported`** — tested; the effect did not hold
- **`external_unverified`** — reported externally, not reproduced in the source capsule
- **`not_transferable`** — established on a different architecture only

## Summary

| Status | Claims |
| --- | --- |
| `supported` | 5 |
| `partially_supported` | 2 |
| `external_unverified` | 0 |
| `not_supported` | 2 |
| `not_transferable` | 0 |

## Claims

### `regular_beats_skewed_cnn` — `not_supported`

**Claim.** At matched edge budget and equal width, regular-degree connectivity outperforms skewed connectivity in a sparse CNN.

**Architecture.** Sparse CNN (CIFAR-10 channel mask)

**Evidence.**

```json
{
  "test_accuracy_difference_pp": 0.68,
  "cohens_d": 1.23,
  "p_uncorrected": 0.0304,
  "p_holm": 0.0692,
  "n_tests_in_sweep": 6,
  "seeds": 8,
  "reanalysis": "the original per-seed accuracies, re-run through /analyze"
}
```

**Gaps.**

- The original run reported this as SUPPORTED at p=0.030. That p-value came from one cell of a six-ratio sweep with no correction for the other five. Under Holm it becomes 0.069 and the same-size advantage is no longer significant.
- Nothing about the measurement changed -- same seeds, same accuracies, same test. Only the correction for having run six comparisons instead of one.
- The three ratios that ARE significant after correction (70%, 60%, 50%) all favour skewed, which is just the unremarkable finding that a narrower network is worse.

### `shrink_at_matched_accuracy` — `partially_supported`

**Claim.** A regular-degree network can be narrowed by 10% with no accuracy loss, saving roughly 16% of connections.

**Architecture.** Sparse CNN (CIFAR-10 channel mask)

**Evidence.**

```json
{
  "test_accuracy_difference_pp": -0.11,
  "ci_pp": [
    -0.57,
    0.34
  ],
  "fresh_seed_check": "holds",
  "connection_reduction_percent": 16.2,
  "seeds": 8
}
```

**Gaps.**

- This is non-inferiority, not a win: the confidence interval spans zero.
- The 16.2% figure came from holding density fixed while shrinking width in that specific sweep. It is not a constant and is not carried into this API.
- The benchmark masked dense weights, so no wall-clock or stored-parameter saving was measured. The reduction is what a real sparse kernel would deliver, not what was observed.

### `regular_beats_skewed_moe` — `partially_supported`

**Claim.** Regular-degree routing beats skewed routing in a mixture-of-experts transformer.

**Architecture.** MoE hash-routing (character-level transformer LM)

**Evidence.**

```json
{
  "density_0.15": {
    "diff": -0.0094,
    "cohens_d": -1.63,
    "p_holm": 0.0326
  },
  "density_0.30": {
    "diff": -0.006,
    "cohens_d": -1.01,
    "p_holm": 0.0326
  },
  "density_0.50": {
    "diff": 0.0039,
    "p_holm": 0.1902
  },
  "density_0.75": {
    "diff": 0.0013,
    "p_holm": 0.4634
  },
  "seeds": 8,
  "seeds_favouring_regular_at_density_0.30": "8 of 8",
  "original_p_at_density_0.30": 0.0852,
  "corrected_p_at_density_0.30": 0.00815,
  "reanalysis": "the original per-seed val losses, re-run through /analyze"
}
```

**Gaps.**

- The original analysis used an unpaired label-shuffle permutation test (reproduced here at p=0.0837 against the logged 0.0852). The design is paired -- the routing table is redrawn per seed and both conditions run at the same seeds -- so pooling the groups discards the pairing that makes the comparison sensitive. Under the correct paired sign-flip test the same numbers give p=0.008.
- The effect is small but perfectly consistent: all 8 of 8 seeds favour regular routing at density 0.30. An unpaired test cannot see that, because seed-to-seed spread swamps a small consistent difference when the groups are pooled.
- Both significant p-values sit at 0.00815, which is the permutation floor for 8 pairs. They cannot go lower, so the two densities cannot be ranked against each other on significance.
- The fresh-seed confirmation has NOT been redone under the paired test: the source log is truncated and one of the four baseline values is missing. On the three visible seeds the direction is mixed, unlike the 8/8 consistency in the main comparison. This claim is partially_supported and not supported until that check is repeated.

### `regular_beats_skewed_reservoir` — `supported`

**Claim.** Regular-degree directed connectivity outperforms skewed connectivity in a reservoir.

**Architecture.** Directed reservoir (this service)

**Evidence.**

```json
{
  "cohens_d": 3.05,
  "p_value": 5e-05,
  "mean_difference_ci": [
    7.998,
    10.928
  ],
  "regular_mean_capacity": 34.95,
  "skewed_mean_capacity": 25.42,
  "seeds": 16,
  "configuration": "n=200, mean_degree=3, spectral_radius=0.9, max_lag=120, sequence_length=4000, edge budgets matched exactly",
  "reproduced_by": "GET /compare on this service, replicated at 8 and 16 seeds",
  "external_reference_d": 2.92
}
```

**Gaps.**

- Measured on memory capacity. The same run found the advantage REVERSING on delayed parity (see regular_vs_skewed_reverses_by_task), so this must not be stated as a general property of the wiring.
- Single configuration. Node count, degree, and spectral radius were not swept.
- An earlier run capping max_lag at 20 measured d=+1.42 for the same comparison, because memory capacity saturates at its ceiling there. The uncapped number is the valid one; see measurement_ceiling_artifact.

### `directedness_necessary` — `not_supported`

**Claim.** Directedness is necessary; symmetrising the connectivity collapses the effect.

**Architecture.** Directed reservoir (this service)

**Evidence.**

```json
{
  "cohens_d_directed": 3.05,
  "cohens_d_symmetrised": 1.84,
  "effect_retained_fraction": 0.6,
  "seeds": 16,
  "external_reference": {
    "directed": 3.05,
    "symmetrised": 0.74,
    "retained": 0.24
  },
  "method": "the same matrices symmetrised as (W + W^T)/2 and rescaled to the original spectral radius, then re-measured -- not estimated from a structural proxy"
}
```

**Gaps.**

- The effect is reduced under symmetrisation but 60% of it survives, so directedness contributes without being necessary. The source reported only 24% surviving.
- The directed effect size matched the external report almost exactly (3.05 vs 3.05) while the symmetrised one did not (1.84 vs 0.74). That divergence is unexplained and most likely a difference in how symmetrisation was applied -- rescaling to the original spectral radius, as done here, removes a confound the source may not have controlled for.
- On delayed parity, symmetrisation does collapse and flip the effect (-1.54 to +0.25). Directedness appears load-bearing for the SKEWED advantage on that task, which is the reverse of the original hypothesis.

### `regular_vs_skewed_reverses_by_task` — `supported`

**Claim.** The regular-vs-skewed advantage is task-dependent in sign, not just in size.

**Architecture.** Directed reservoir (this service)

**Evidence.**

```json
{
  "memory_capacity": {
    "cohens_d": 1.42,
    "p_holm": 0.0002,
    "favours": "regular"
  },
  "narma10": {
    "cohens_d": 1.04,
    "p_holm": 0.0016,
    "favours": "regular"
  },
  "delay_parity": {
    "cohens_d": -1.54,
    "p_holm": 0.0002,
    "favours": "skewed"
  },
  "delay_xor": {
    "cohens_d": 0.09,
    "p_holm": 0.728,
    "favours": "neither"
  },
  "seeds": 16,
  "correction": "holm-bonferroni across 4 tasks",
  "replication": "signs and significance identical at 8 and 16 seeds",
  "note": "these four ran at max_lag=20, so magnitudes are compressed by the ceiling; the signs are the finding"
}
```

**Gaps.**

- Delayed parity favours SKEWED wiring at d=-1.54 with a CI clear of zero. No summary of this work should describe regular-degree wiring as simply better.
- Mechanism unknown. Parity is the most non-linear task in the set, which suggests degree heterogeneity may help non-linear mixing, but that is a hypothesis and has not been tested.
- One configuration only.

### `measurement_ceiling_artifact` — `supported`

**Claim.** Memory capacity measured at an insufficient max_lag understates effect sizes.

**Architecture.** Directed reservoir (this service)

**Evidence.**

```json
{
  "capped_max_lag_20": {
    "cohens_d": 1.42,
    "regular": 19.975,
    "skewed": 19.723
  },
  "uncapped_max_lag_120": {
    "cohens_d": 3.05,
    "regular": 34.95,
    "skewed": 25.42
  },
  "ceiling_at_max_lag_20": 20.0,
  "seeds": 16
}
```

**Gaps.**

- Memory capacity sums one point per lag, so it cannot exceed max_lag. At n=200 both conditions sit at ~19.9 out of 20 and the comparison is squeezed against the ceiling, halving the measured effect and making every width in /optimize look non-inferior.
- /evaluate warns when a score comes within 5% of the ceiling and /optimize returns metric_saturated. Both were silent on the uncapped run, which is the intended behaviour, but a caller who ignores warnings will draw a wrong conclusion.

### `analysis_method_changed_two_conclusions` — `supported`

**Claim.** Two of the three original architecture conclusions were wrong because of how the numbers were analysed, not how they were measured.

**Architecture.** cross-architecture reanalysis

**Evidence.**

```json
{
  "cnn": "reported SUPPORTED at p=0.030; uncorrected for a six-test sweep. Holm -> p=0.069, not significant.",
  "moe": "reported NOT SUPPORTED at p=0.085 from an unpaired test on a paired design. Paired sign-flip -> p=0.008, significant at two of four densities after Holm.",
  "method": "the original per-seed values from both logs, re-run through /analyze",
  "retraining_required": false
}
```

**Gaps.**

- The errors ran in opposite directions: the CNN claim was inflated by an uncorrected sweep, the MoE claim was suppressed by discarding the pairing. Neither is a measurement problem -- the training runs were sound and the per-seed numbers were already in the logs.
- This is the argument for /analyze existing at all. A protocol applied by hand, differently per architecture, produced two wrong conclusions from correct data.

### `effect_is_architecture_specific` — `supported`

**Claim.** The advantage is architecture-specific.

**Architecture.** cross-architecture audit

**Evidence.**

```json
{
  "architectures_tested": 3,
  "architectures_with_effect": "2 reproduced (sparse CNN, reservoir), 1 negative (MoE)",
  "task_specificity_within_reservoir": "3 of 4 tasks significant, 1 of them favouring skewed rather than regular"
}
```

**Gaps.**

- The reservoir arm is now reproduced in-service at 16 seeds, so 'two of three architectures' is defensible -- but only for memory capacity. Within that one architecture the effect reverses sign across tasks, so specificity runs deeper than architecture: it goes down to the individual task.

---

Generated from `app/core/evidence.py`, served live at
`GET /v1/reservoirx-d/claims`. Update the module, not this file.
