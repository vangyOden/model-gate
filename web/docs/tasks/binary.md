# Binary classification

The default task. Credit scoring, fraud detection, lapse prediction — anything
with a positive class.

```python
context = StructuredGateContext(
    model=model,
    X=X_val,
    y_true=y_val,
    y_pred=model.predict_proba(X_val)[:, 1],  # positive-class probability
    protected_df=protected_val,
    task="binary",
)
```

## Metrics

| Metric | Expects | Notes |
|---|---|---|
| `roc_auc` | scores | threshold-independent; the `"auto"` default |
| `average_precision` | scores | better than AUC when the positive class is rare |
| `accuracy` | labels | works without scikit-learn |
| `balanced_accuracy` | labels | for skewed classes |
| `f1`, `precision`, `recall` | labels | |

Label-based metrics binarise a continuous `y_pred` at
`config.performance.decision_threshold` (default `0.5`). Ranking metrics
ignore it.

All are gated with `min_score`, since higher is better for every one.

```python
config = GateConfig()
config.performance.metric = "average_precision"
config.performance.min_score = 0.30
```

!!! tip "Imbalanced problems"
    At a 5% base rate, ROC AUC looks comfortable while the model is barely
    usable. `average_precision` does not flatter.

### A custom metric

Any `fn(y_true, y_pred) -> float`. It is called with `y_pred` exactly as
supplied — no thresholding, since only you know what your metric expects.

```python
from sklearn.metrics import fbeta_score


def f2_at_30pct(y_true, y_pred):
    return fbeta_score(y_true, (np.asarray(y_pred) >= 0.30).astype(int), beta=2)


config.performance.metric = f2_at_30pct  # reported under the name "f2_at_30pct"
```

Custom callables are assumed higher-is-better and gated with `min_score`.
Negate inside your function if that is wrong.

## Fairness

All four checks apply.

| Check | Question |
|---|---|
| `proxy_correlation` | does a feature encode a protected attribute the model cannot see? |
| `disparate_impact` | do selection rates differ across groups? |
| `shap_subgroup_gap` | does a feature *drive* outcomes differently per group? |
| `counterfactual_flip` | does flipping the attribute change the prediction? |

`proxy_correlation` is usually the most valuable: it catches the case where a
team believes it removed a protected attribute and did not. Drop `region` from
the features and `distance_to_branch_km` will happily reconstruct it.

!!! warning "Demographic parity needs hard labels"
    It counts predictions equal to `1`. Continuous predictions are binarised
    for you at `FairnessConfig.decision_threshold`. Before 0.2.1 they were
    not, and the check silently reported `0.000` for any probability
    `y_pred` — perfectly fair, however skewed the model.

## The `"auto"` metric fallback

With `metric="auto"` and no scikit-learn installed, the gate falls back from
`roc_auc` to `accuracy`. That fallback is **never silent**: it is logged at
`WARNING`, marked `metric_is_fallback: true` in the result metadata, and named
in the detail string.

Naming a metric explicitly opts out of fallback entirely — if it cannot run,
the gate reports a blocking `CHECK_ERROR` rather than scoring you on something
you did not ask for.

Worked end to end in
[notebook 01](../examples/01_binary_classification_sklearn.ipynb).
