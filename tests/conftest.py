"""Shared fixtures and suite-wide guards.

The autouse guard below exists because of a specific bug: `ShapSubgroupCheck`
crashed on `RandomForestClassifier` for a whole release, and the test that
should have caught it asserted only `len(results) > 0` — which passes, because
`ModelGate` converts an exception into a blocking `CHECK_ERROR` *result*. A
`CHECK_ERROR` is always either a bug or an explicit expectation, never noise,
so the suite now fails on any unexpected one.
"""

import numpy as np
import pandas as pd
import pytest


class DummyModel:
    """A tiny deterministic 'model' so most tests don't depend on scikit-learn."""

    def predict(self, X: pd.DataFrame):
        return (X["income"] > X["income"].median()).astype(int).values

    def predict_proba(self, X: pd.DataFrame):
        p1 = (X["income"] > X["income"].median()).astype(float).values
        return np.column_stack([1 - p1, p1])


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "income": rng.normal(50000, 15000, n),
            "age": rng.integers(18, 70, n),
            "credit_score": rng.normal(650, 50, n),
        }
    )
    protected_df = pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "region": rng.choice(["Lagos", "Abuja", "Kano"], n),
        }
    )
    model = DummyModel()
    y_pred = model.predict_proba(X)[:, 1]
    y_true = (X["income"] > X["income"].quantile(0.4)).astype(int).values
    return model, X, y_true, y_pred, protected_df


@pytest.fixture
def small_valid_context(synthetic_data):
    from bdp_model_gate import StructuredGateContext

    model, X, y_true, y_pred, protected_df = synthetic_data
    return StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )


# --------------------------------------------------------------------------
# Suite-wide guard: no unexpected CHECK_ERROR
# --------------------------------------------------------------------------

_SEEN_REPORTS: list = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "expect_check_error: this test deliberately produces a CHECK_ERROR result",
    )


@pytest.fixture(autouse=True)
def _fail_on_unexpected_check_error(request, monkeypatch):
    """Fails any test whose gate run produced a CHECK_ERROR it did not ask for.

    Wraps `ModelGate.run` so every report produced anywhere in the test —
    including deep inside a helper — is inspected. Opt out with
    `@pytest.mark.expect_check_error` when the error *is* the thing under test.
    """
    from bdp_model_gate.core.gate import ModelGate

    produced: list = []
    original = ModelGate.run

    def recording_run(self, context):
        report = original(self, context)
        produced.append(report)
        return report

    monkeypatch.setattr(ModelGate, "run", recording_run)
    yield

    if request.node.get_closest_marker("expect_check_error"):
        return

    offenders = [
        (r.check_name, r.detail)
        for report in produced
        for r in report.results
        if r.flag == "CHECK_ERROR"
    ]
    if offenders:
        lines = "\n".join(f"  {name}: {detail}" for name, detail in offenders)
        pytest.fail(
            "gate run produced CHECK_ERROR result(s) this test did not expect.\n"
            "A CHECK_ERROR means a check raised — mark the test with\n"
            "@pytest.mark.expect_check_error if that is intended.\n" + lines,
            pytrace=False,
        )


# --------------------------------------------------------------------------
# Hostile fixtures — the shapes that break naive implementations
# --------------------------------------------------------------------------


@pytest.fixture
def wide_scale_frame():
    """Feature magnitudes spanning seven orders of magnitude.

    A perturbation or threshold derived from a single global scale is
    dominated by the largest column here — the bug that made the adversarial
    check report a relative shift of ~1448 and block a stable linear model.
    """
    rng = np.random.default_rng(3)
    n = 300
    return pd.DataFrame(
        {
            "risk_score": rng.gamma(4, 1.5, n),  # single digits
            "tenure_years": rng.integers(0, 40, n).astype(float),
            "sum_insured_ngn": rng.lognormal(15.4, 0.4, n),  # millions
        }
    )


@pytest.fixture
def tiny_group_protected():
    """One large group beside a three-row one — small groups must be reported,
    not scored, or they produce wild ratios that read as findings."""
    return pd.DataFrame({"region": ["Lagos"] * 297 + ["Jigawa"] * 3})


@pytest.fixture
def severe_imbalance():
    """A 99.5 / 0.5 split, where accuracy flatters a model that never predicts
    the rare class."""
    rng = np.random.default_rng(9)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, 5, replace=False)] = 1
    return y
