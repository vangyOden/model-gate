# Multiclass and ordinal

Underwriting decisions, risk tiers, product categories.

```python
context = StructuredGateContext(
    model=model,
    X=X_val,
    y_true=decisions,
    y_pred=predicted,
    protected_df=protected_val,
    task="multiclass",
    class_order=["decline", "refer", "accept"],  # ascending favourability
    favourable_classes=["accept"],
)
```

## Ordinal problems

If the outcomes sit on a scale, supply `class_order` — **least favourable
first**. That single field marks the problem ordinal and unlocks metrics that
understand distance.

Why it matters: plain multiclass metrics count errors. They cannot see that
predicting **decline** for an application that should have been **accepted**
is worse than predicting **refer**. To accuracy, both are one mistake.

| Prediction | Errors | Accuracy | `ordinal_mae` | `quadratic_kappa` |
|---|---|---|---|---|
| off by one step | 1 | 0.833 | 0.167 | 0.842 |
| off by two steps | 1 | 0.833 | 0.333 | 0.500 |

| Metric | Direction | Gated with |
|---|---|---|
| `ordinal_mae` | lower better | `max_error` |
| `quadratic_kappa` | higher better | `min_score` |

`quadratic_kappa` penalises a disagreement by the **square** of its rank
distance, so a two-step error costs four times a one-step one. Both are
numpy-native and both require `class_order`.

For a genuinely nominal problem — product category, say — omit `class_order`
and use the label metrics below.

## Nominal metrics

`accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`. The last three
need an averaging strategy, set by `config.performance.average`:

```python
config.performance.average = "macro"  # default
```

**Macro weights every class equally**, so a rarely predicted *decline* counts
as much as a common *accept*. `"weighted"` averages by support, which flatters
a model that does well on the majority class and badly on the rare one —
usually the opposite of what you want to gate on.

`metric="auto"` resolves to `balanced_accuracy`, since plain accuracy rewards
a model that never predicts the rare class at all.

!!! note "roc_auc is binary-only"
    Its multiclass form needs a full `(n_rows, n_classes)` probability matrix,
    which the `y_pred` contract does not carry. It is refused rather than
    quietly approximated.

## Fairness needs a favourable outcome

Demographic parity counts a *selected* class. With two classes that is
obvious; with three it is a judgement, and the library will not make it for
you.

`favourable_classes` decides it, defaulting to the last entry of `class_order`
(and logging that it inferred). With neither, `DisparateImpactCheck` reports
`NOT_APPLICABLE` rather than picking one.

The choice genuinely changes the answer:

```python
context.favourable_classes = ["accept"]  # were they approved?
context.favourable_classes = ["accept", "refer"]  # were they spared a decline?
```

On a real book these give different verdicts. Which framing is right depends
on what a referral actually means for an applicant — a referral usually
approved is a very different outcome from one usually declined. That is a
business question, which is why it is a parameter.

## Ordinal robustness

With `class_order` set, `AdversarialRobustnessCheck` reports the mean rank
*distance* a prediction moves under perturbation alongside the flip rate.

Two models can flip at an identical rate while one only ever wobbles between
adjacent classes and the other swings from accept straight to decline. Read
`max_observed_rank_shift`: `1.0` means the model never moved further than an
adjacent class.

```python
config.security.adversarial_max_rank_shift = 0.10
```

Worked end to end in
[notebook 02](../examples/02_multiclass_ordinal_sklearn.ipynb).
