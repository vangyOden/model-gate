"""Any model, not just scikit-learn-shaped ones.

The library used to reach into `context.model` from five separate call
sites, each assuming a scikit-learn convention: `.predict()`,
`.predict_proba()` returning exactly two columns, or a `.coef_` attribute.
These tests pin the framework-neutral routes that replaced that.
"""

import json

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
from bdp_model_gate.exceptions import GateConfigurationError, GateValidationError
from bdp_model_gate.model import ModelAdapter
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.fairness import CounterfactualFlipCheck
from bdp_model_gate.structured.security import AdversarialRobustnessCheck


@pytest.fixture
def frame():
    rng = np.random.default_rng(5)
    n = 240
    X = pd.DataFrame(
        {
            "income": rng.normal(50_000, 12_000, n),
            "age": rng.integers(21, 65, n).astype(float),
            "gender": rng.integers(0, 2, n).astype(float),
        }
    )
    y_true = (X["income"] > X["income"].median()).astype(int).to_numpy()
    protected = pd.DataFrame({"gender": X["gender"].astype(int).to_numpy()})
    return X, y_true, protected


def _logit(X):
    z = (X["income"].to_numpy() - 50_000) / 12_000
    return 1.0 / (1.0 + np.exp(-z))


# --- the three ways to supply a model ---------------------------------------


class SklearnStyle:
    def predict(self, X):
        return (_logit(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = _logit(X)
        return np.column_stack([1 - p, p])


def test_sklearn_style_object_still_works(frame):
    X, y_true, _ = frame
    adapter = ModelAdapter(model=SklearnStyle())
    assert adapter.can_predict and adapter.can_predict_proba
    assert adapter.predict(X).shape == (len(X),)
    assert adapter.describe() == "SklearnStyle.predict"


def test_bare_callable_model(frame):
    X, _, _ = frame
    adapter = ModelAdapter(model=lambda df: (_logit(df) >= 0.5).astype(int))
    assert adapter.can_predict
    assert adapter.predict(X).shape == (len(X),)


def test_predict_fn_needs_no_model_object(frame):
    """A remote scoring endpoint has no model object at all."""
    X, y_true, _ = frame
    context = StructuredGateContext(
        X=X,
        y_true=y_true,
        y_pred=_logit(X),
        predict_fn=lambda df: (_logit(df) >= 0.5).astype(int),
    )
    report = ModelGate(checks=default_structured_checks(GateConfig(), include_plugins=False)).run(
        context
    )
    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}
    assert not any(r.flag == "CHECK_ERROR" for r in report.results)


def test_predict_fn_takes_precedence_over_model(frame):
    X, _, _ = frame
    adapter = ModelAdapter(model=SklearnStyle(), predict_fn=lambda df: np.zeros(len(df)))
    assert adapter.describe() == "predict_fn"
    assert np.all(adapter.predict(X) == 0)


# --- probability shape normalisation ----------------------------------------


@pytest.mark.parametrize(
    "shape_name,fn",
    [
        ("sklearn (n,2)", lambda df: np.column_stack([1 - _logit(df), _logit(df)])),
        ("keras (n,1)", lambda df: _logit(df).reshape(-1, 1)),
        ("bare (n,)", lambda df: _logit(df)),
    ],
)
def test_binary_probability_shapes_all_normalise(frame, shape_name, fn):
    """A Keras sigmoid emits (n, 1); scikit-learn emits (n, 2). Both mean the
    same thing and must not require the caller to know which we expect."""
    X, _, _ = frame
    proba = ModelAdapter(predict_proba_fn=fn).predict_positive_proba(X)
    assert proba.shape == (len(X),)
    np.testing.assert_allclose(proba, _logit(X), rtol=1e-9)


def test_multiclass_probabilities_are_refused_not_silently_sliced(frame):
    X, _, _ = frame
    adapter = ModelAdapter(predict_proba_fn=lambda df: np.tile([0.2, 0.3, 0.5], (len(df), 1)))
    with pytest.raises(GateConfigurationError, match="not a binary classifier"):
        adapter.predict_positive_proba(X)


def test_counterfactual_check_works_via_predict_proba_fn(frame):
    """Previously this check required a model with .predict_proba(), so it was
    NOT_APPLICABLE for every Keras or PyTorch model."""
    X, y_true, protected = frame
    context = StructuredGateContext(
        X=X,
        y_true=y_true,
        y_pred=_logit(X),
        protected_df=protected,
        predict_fn=lambda df: (_logit(df) >= 0.5).astype(int),
        predict_proba_fn=lambda df: _logit(df).reshape(-1, 1),  # Keras-shaped
    )
    results = CounterfactualFlipCheck().run(context)
    assert results
    assert all(r.flag != "NOT_APPLICABLE" for r in results)


def test_counterfactual_still_degrades_without_probabilities(frame):
    X, y_true, protected = frame
    context = StructuredGateContext(
        X=X,
        y_true=y_true,
        y_pred=_logit(X),
        protected_df=protected,
        predict_fn=lambda df: (_logit(df) >= 0.5).astype(int),
    )
    result = CounterfactualFlipCheck().run(context)[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "predict_proba_fn" in result.detail


# --- a torch-shaped model (no torch dependency) -----------------------------


class TorchLike:
    """Mimics what makes a PyTorch nn.Module unusable directly: it is callable
    rather than having .predict(), it wants an array rather than a DataFrame,
    and it returns a column vector."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def __call__(self, array):
        if isinstance(array, pd.DataFrame):
            raise TypeError("expected an array, not a DataFrame")
        return (np.asarray(array) @ self.weights).reshape(-1, 1)


def test_torch_shaped_model_via_predict_fn(frame):
    X, y_true, protected = frame
    net = TorchLike([1 / 50_000, 0.0, 0.0])

    # The caller owns the conversion; the library never sees an array-only API.
    context = StructuredGateContext(
        X=X,
        y_true=y_true.astype(float),
        y_pred=net(X.to_numpy()).ravel(),
        protected_df=protected,
        task="regression",
        predict_fn=lambda df: net(df.to_numpy()),
    )
    adapter = ModelAdapter.from_context(context)
    # The (n, 1) column vector is flattened for us.
    assert adapter.predict(X).shape == (len(X),)

    config = GateConfig()
    config.performance.metric = "mae"
    config.performance.max_error = 10.0
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(context)
    assert not any(r.flag == "CHECK_ERROR" for r in report.results)


# --- gradient_fn ------------------------------------------------------------


def test_gradient_fn_drives_a_targeted_attack(frame):
    X, y_true, _ = frame
    weights = np.array([1 / 50_000, 0.0, 0.0])
    net = TorchLike(weights)

    context = StructuredGateContext(
        X=X,
        y_true=y_true.astype(float),
        y_pred=net(X.to_numpy()).ravel(),
        task="regression",
        predict_fn=lambda df: net(df.to_numpy()),
        gradient_fn=lambda df: np.tile(weights, (len(df), 1)),
    )
    result = AdversarialRobustnessCheck().run(context)[0]
    assert result.metadata["method"] == "gradient-fn"


def test_gradient_fn_preferred_over_coef(frame):
    """A real gradient beats a linear approximation, so it wins when both
    are available."""
    X, y_true, _ = frame

    class LinearWithCoef:
        coef_ = np.array([1.0, 0.0, 0.0])

        def predict(self, df):
            return df["income"].to_numpy() / 50_000

    context = StructuredGateContext(
        model=LinearWithCoef(),
        X=X,
        y_true=y_true.astype(float),
        y_pred=X["income"].to_numpy() / 50_000,
        task="regression",
        gradient_fn=lambda df: np.tile([1.0, 0.0, 0.0], (len(df), 1)),
    )
    assert AdversarialRobustnessCheck().run(context)[0].metadata["method"] == "gradient-fn"

    without = StructuredGateContext(
        model=LinearWithCoef(),
        X=X,
        y_true=y_true.astype(float),
        y_pred=X["income"].to_numpy() / 50_000,
        task="regression",
    )
    assert AdversarialRobustnessCheck().run(without)[0].metadata["method"] == "gradient-directed"


def test_misshaped_gradients_are_refused(frame):
    X, _, _ = frame
    adapter = ModelAdapter(model=SklearnStyle(), gradient_fn=lambda df: np.zeros((len(df), 99)))
    with pytest.raises(GateConfigurationError, match="one gradient per"):
        adapter.gradients(X)


def test_non_callable_hooks_rejected(frame):
    X, y_true, _ = frame
    with pytest.raises(GateValidationError, match="must be callable"):
        ModelGate(checks=[]).run(
            StructuredGateContext(X=X, y_true=y_true, y_pred=y_true, predict_fn="not-a-function")
        )


# --- CLI --model-loader -----------------------------------------------------


def test_cli_model_loader(tmp_path, monkeypatch):
    """The CLI must gate a model joblib cannot unpickle, without this package
    importing any framework."""
    pytest.importorskip("sklearn")
    from bdp_model_gate.cli import main

    loader_module = tmp_path / "fake_serving.py"
    loader_module.write_text(
        "import numpy as np\n"
        "def load_scorer():\n"
        "    # Stands in for a torch checkpoint: a plain closure, unpicklable.\n"
        "    return lambda df: (df['income'].to_numpy() > 50_000).astype(int)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({"income": rng.normal(50_000, 12_000, n)})
    df["label"] = (df["income"] > 50_000).astype(int)
    data = tmp_path / "validation.csv"
    df.to_csv(data, index=False)
    out = tmp_path / "report.json"

    exit_code = main(
        [
            "--model-loader",
            "fake_serving:load_scorer",
            "--data",
            str(data),
            "--target-col",
            "label",
            "--task",
            "binary",
            "--metric",
            "accuracy",
            "--min-score",
            "0.5",
            "--output",
            str(out),
        ]
    )
    # The verdict itself is beside the point here — this closure is a hard
    # step at 50,000, so adversarial robustness rightly flags it. What matters
    # is that an unpicklable model was loaded, called, and scored.
    assert exit_code in (0, 1, 2)
    report = json.loads(out.read_text())
    assert report["model_metric"] == "accuracy"
    assert report["model_score"] == 1.0
    flags = [r for rs in report["results_by_category"].values() for r in rs]
    assert not any(r["flag"] == "CHECK_ERROR" for r in flags)


@pytest.mark.parametrize(
    "spec,message",
    [
        ("no_colon_here", "package.module:factory"),
        ("definitely_not_a_module:fn", "could not import"),
    ],
)
def test_cli_model_loader_errors(tmp_path, spec, message):
    from bdp_model_gate.cli import main

    df = pd.DataFrame({"income": [1.0, 2.0], "label": [0, 1]})
    data = tmp_path / "v.csv"
    df.to_csv(data, index=False)

    exit_code = main(
        [
            "--model-loader",
            spec,
            "--data",
            str(data),
            "--target-col",
            "label",
            "--output",
            str(tmp_path / "r.json"),
        ]
    )
    assert exit_code == 1


def test_model_and_model_loader_are_mutually_exclusive():
    from bdp_model_gate.cli import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--model", "m.joblib", "--model-loader", "a:b", "--data", "d.csv", "--target-col", "y"]
        )
