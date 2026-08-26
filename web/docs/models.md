# Any model

Nothing here imports a deep-learning framework. Rather than requiring an
object of a particular shape, the gate accepts plain functions, and the
boundary is deliberately narrow: **DataFrame in, array out**.

Your function owns tensor conversion, device placement, batching and auth.

## Why an adapter is needed

The scikit-learn conventions are not universal:

| Model | `.predict()` | `.predict_proba()` | Takes a DataFrame |
|---|---|---|---|
| scikit-learn, LightGBM, XGBoost sklearn API | yes | `(n, 2)` | yes |
| Keras | yes | no — sigmoid gives `(n, 1)` | yes |
| PyTorch `nn.Module` | **no** — callable | no | **no** — wants a tensor |
| XGBoost native `Booster` | yes | no | **no** — wants a `DMatrix` |
| Remote endpoint | no object at all | | |

## The three hooks

| Field | Signature | Unlocks |
|---|---|---|
| `predict_fn` | `fn(DataFrame) -> array` | everything; takes precedence over `model` |
| `predict_proba_fn` | `fn(DataFrame) -> array` | `CounterfactualFlipCheck` |
| `gradient_fn` | `fn(DataFrame) -> (n_rows, n_features)` | a real targeted adversarial attack |

=== "PyTorch"

    ```python
    context = StructuredGateContext(
        X=X_val,
        y_true=y_val,
        y_pred=y_pred,
        task="binary",
        predict_fn=lambda df: (proba(df) >= 0.5).astype(int),
        predict_proba_fn=proba,
        gradient_fn=input_gradients,
    )
    ```

=== "XGBoost Booster"

    ```python
    import xgboost as xgb

    context = StructuredGateContext(
        X=X_val,
        y_true=y_val,
        y_pred=booster.predict(xgb.DMatrix(X_val)),
        task="binary",
        predict_fn=lambda df: (booster.predict(xgb.DMatrix(df)) >= 0.5).astype(int),
        predict_proba_fn=lambda df: booster.predict(xgb.DMatrix(df)),
    )
    ```

=== "Remote endpoint"

    ```python
    def score(df):
        return np.asarray(requests.post(URL, json=df.to_dict("records")).json()["scores"])


    # No `model=` at all.
    context = StructuredGateContext(
        X=X_val,
        y_true=y_val,
        y_pred=score(X_val),
        task="binary",
        predict_fn=lambda df: (score(df) >= 0.5).astype(int),
        predict_proba_fn=score,
    )
    ```

`model` is optional — a remote endpoint has no object to pass. A bare callable
also works as `model=`, so the two routes are interchangeable.

## Probability shapes are normalised

A Keras sigmoid returns `(n, 1)`, scikit-learn returns `(n, 2)`, and a custom
model might return `(n,)`. All three mean the same thing and reduce to one
positive-class vector, so you never have to know which the library expects.

A genuinely multiclass `(n, k)` output is **refused** rather than silently
sliced — taking column 1 of a three-class model yields a real number that
means nothing.

## Gradients make robustness real

`AdversarialRobustnessCheck` chooses its attack direction in this order:

1. **true per-row gradients** — if `gradient_fn` is supplied
2. **linear coefficients** — for models exposing `coef_`
3. **random noise** — otherwise

The gradient path is a sign-of-gradient (FGSM-style) step at full epsilon,
tried in both directions. On a small network at the default `epsilon=0.02` it
finds a 4.5% flip rate where random noise finds **zero**.

That changes how you read a clean result. "No weakness found with a
scattergun" is much weaker evidence than "no weakness found by an attack
pointed straight at the boundary". If your model is differentiable, supplying
`gradient_fn` is what makes the check worth trusting.

!!! note "Tree models"
    A booster has neither gradients nor `coef_`, so the random path is the best
    available. Read its robustness flag accordingly.

## Pass `model=` when you have it

Even alongside `predict_fn`. Checks that can introspect the model do —
`ShapSubgroupCheck` computes exact contributions via `TreeExplainer` rather
than sampling a black box.

See [notebook 04](examples/04_any_framework_classification.ipynb) and
[notebook 05](examples/05_boosters_and_cli.ipynb).
