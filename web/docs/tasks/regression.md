# Regression

Premium pricing, claims severity, claims frequency — anything with a
continuous target.

```python
context = StructuredGateContext(
    model=pricing_model,
    X=X_val,
    y_true=realised_loss,
    y_pred=quoted_premium,
    protected_df=protected_val,
    expected_loss=technical_premium,  # enables loss-ratio parity
    task="regression",
)

config = GateConfig()
config.performance.metric = "rmse"
config.performance.max_error = 45_000.0
```

## Metrics point in two directions

This is the change that catches people out. Error metrics are
**lower-is-better**, so they are gated with `max_error`, not `min_score`.

| Metric | Direction | Gated with | Use for |
|---|---|---|---|
| `r2` | higher better | `min_score` | scale-free; the `"auto"` default |
| `rmse` | lower better | `max_error` | premium, general |
| `mae` | lower better | `max_error` | robust to outliers |
| `mape` | lower better | `max_error` | skewed money — claims severity |
| `poisson_deviance` | lower better | `max_error` | counts — claims frequency |

All five are implemented in numpy, so they work on a core install without
scikit-learn.

There is deliberately **no default `max_error`** — a sensible ceiling depends
entirely on whether the target is naira or claim counts — so configuring an
error metric without one raises `GateConfigurationError` rather than passing
silently. The comparison is the entire point of the gate.

### Choosing for the target's shape

- **Severity** is money and heavily right-skewed. A handful of large claims
  dominate squared error, so `rmse` measures the tail rather than typical
  accuracy. `mape` scores relative error, which is usually what a reserving
  team cares about — but it is undefined where the actual is zero.
- **Frequency** is a count, mostly zero. Squared error understates
  over-dispersion and treats "predicted 0.1 claims, saw 3" far too gently.
  `poisson_deviance` is built for it, and needs strictly positive predictions
  since it takes their log.

## Fairness without a selected class

Demographic parity has no regression analogue. Four checks replace it, each
answering something the others cannot.

| Check | Question | Needs |
|---|---|---|
| `loss_ratio_parity` | higher **margin over own expected loss**? | `expected_loss` |
| `group_mean_gap` | systematically higher predictions? | — |
| `error_parity` | model materially less accurate for a group? | `y_true` |
| `calibration_parity` | predictions over- or under-shoot reality? | `y_true` |

All gaps are measured **relative** to the overall figure, so one threshold
works whether the target is naira or claim counts. Groups smaller than
`FairnessConfig.min_group_size` (default 30) are reported but not scored — a
three-policy segment otherwise produces a wild ratio that reads as a finding.

### Why loss-ratio parity is the one that matters

A pricing model *should* charge more in a higher-loss segment. That is
risk-based pricing, not discrimination, and a check that flags it makes the
gate noisy enough to ignore.

`LossRatioParityCheck` divides each group's premium by its **own expected
cost**, which isolates unfairness from actuarially justified variation:

```text
region: premium-to-expected-loss ratio spans 1.123 (Lagos) to 1.412 (Kano)
        — 23.6% of the overall ratio 1.225; Kano carries the higher margin
        over its own expected cost
```

Same risk, different margin. `group_mean_gap` alone would only have told you
Kano pays more, which is ambiguous.

It requires `context.expected_loss` — a per-row expected loss, technical
premium or pure premium. Without it the check reports `NOT_APPLICABLE` rather
than falling back to a raw-price comparison, which would answer a different
question under the same name.

## Robustness on a continuous output

A "prediction flip" is meaningless when the output is continuous — every
perturbation moves it, so a flip rate would be ~1.0 and every regression model
would be permanently blocked.

Regression measures the mean **relative** prediction shift instead, against
`SecurityConfig.adversarial_max_relative_shift`. Each feature is perturbed
relative to its own magnitude, so a sum-insured column in the millions does
not swamp a single-digit risk score.

!!! warning "Rescale `shap_gap_threshold` for regression"
    `FairnessConfig.shap_gap_threshold` is **absolute and in the units of the
    model output**, unlike the four thresholds above. On a model predicting
    naira in the tens of thousands the default of `0.15` flags essentially
    every feature.

    ```python
    config.fairness.shap_gap_threshold = 0.05 * float(np.mean(y_pred))
    ```

    Making it relative is planned; see the
    [roadmap](https://github.com/vanjy-eng/model-gate/blob/main/ROADMAP.md).

Worked end to end in
[notebook 03](../examples/03_regression_sklearn.ipynb).
