"""Regression tests for the two fairness-check correctness bugs fixed in 0.2.1.

Both were silent: the gate reported a clean result while measuring nothing
useful, which is the worst failure mode for a governance tool.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import FairnessConfig, StructuredGateContext
from bdp_model_gate.structured.fairness import DisparateImpactCheck, ShapSubgroupCheck


class _ThresholdModel:
    """Predicts on a single 'score' column, so the fixtures below can dictate
    exactly what the selection rates are."""

    def predict(self, X):
        return (X["score"].to_numpy() >= 0.5).astype(int)


def _maximally_unfair(n=400):
    """Every 'M' selected, every 'F' rejected — demographic parity difference
    of 1.0, the worst value the metric can take."""
    gender = np.where(np.arange(n) % 2 == 0, "M", "F")
    proba = np.where(gender == "M", 0.95, 0.05)
    return (
        _ThresholdModel(),
        pd.DataFrame({"score": proba}),
        np.resize([0, 1], n),
        proba,
        pd.DataFrame({"gender": gender}),
    )


# --- DisparateImpactCheck: probabilities used to read as perfectly fair ------


def test_probability_predictions_are_binarised_not_silently_zeroed():
    """The bug: demographic parity counts predictions equal to 1, and a
    probability never is, so every group's selection rate was 0 and the
    difference was always exactly 0.000 -> reported OK."""
    pytest.importorskip("fairlearn")
    model, X, y_true, proba, protected = _maximally_unfair()

    context = StructuredGateContext(
        model=model, X=X, y_true=y_true, y_pred=proba, protected_df=protected
    )
    result = DisparateImpactCheck().run(context)[0]

    assert result.flag == "DISPARITY_RISK"
    assert result.metadata["demographic_parity_diff"] == pytest.approx(1.0)


def test_hard_labels_give_the_same_answer_as_probabilities():
    """Binarising must not change the verdict for a caller who already passes
    hard labels — the two paths have to agree."""
    pytest.importorskip("fairlearn")
    model, X, y_true, proba, protected = _maximally_unfair()

    from_proba = DisparateImpactCheck().run(
        StructuredGateContext(model=model, X=X, y_true=y_true, y_pred=proba, protected_df=protected)
    )[0]
    from_labels = DisparateImpactCheck().run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_true,
            y_pred=(proba >= 0.5).astype(int),
            protected_df=protected,
        )
    )[0]

    assert from_proba.metadata["demographic_parity_diff"] == pytest.approx(
        from_labels.metadata["demographic_parity_diff"]
    )


def test_decision_threshold_shifts_the_parity_measurement():
    pytest.importorskip("fairlearn")
    model, X, y_true, proba, protected = _maximally_unfair()
    context = StructuredGateContext(
        model=model, X=X, y_true=y_true, y_pred=proba, protected_df=protected
    )

    # Above every score, so nobody is selected and the groups cannot differ.
    lenient = DisparateImpactCheck(FairnessConfig(decision_threshold=0.99)).run(context)[0]
    assert lenient.metadata["demographic_parity_diff"] == pytest.approx(0.0)
    assert lenient.metadata["decision_threshold"] == 0.99

    strict = DisparateImpactCheck(FairnessConfig(decision_threshold=0.5)).run(context)[0]
    assert strict.metadata["demographic_parity_diff"] == pytest.approx(1.0)


# --- ShapSubgroupCheck: 3-D shap output used to raise ------------------------


def test_positive_class_reduction_handles_both_shap_shapes():
    """shap returns (rows, features) for some models and
    (rows, features, classes) for others — RandomForestClassifier among them,
    and which you get changed across shap versions."""
    two_d = np.arange(12, dtype=float).reshape(6, 2)
    assert ShapSubgroupCheck._positive_class_values(two_d) is two_d

    three_d = np.stack([np.zeros((6, 2)), np.ones((6, 2))], axis=-1)
    reduced = ShapSubgroupCheck._positive_class_values(three_d)
    assert reduced.shape == (6, 2)
    assert np.all(reduced == 1.0)  # took the positive class, not class 0


def test_multiclass_shap_output_is_not_applicable_rather_than_a_guess():
    four_class = np.zeros((6, 2, 4))
    assert ShapSubgroupCheck._positive_class_values(four_class) is None


def test_shap_subgroup_runs_on_a_random_forest():
    """The concrete regression: RandomForestClassifier produced a 3-D array
    and the check died with 'Must pass 2-d input'."""
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] + rng.normal(0, 0.3, n) > 0).astype(int).to_numpy()
    protected = pd.DataFrame({"group": rng.choice(["x", "y"], n)})
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y,
        y_pred=model.predict_proba(X)[:, 1],
        protected_df=protected,
    )
    results = ShapSubgroupCheck().run(context)

    assert results
    assert all(r.flag != "CHECK_ERROR" for r in results)


def test_gate_does_not_swallow_a_shap_failure_as_a_pass():
    """Guards the reason the original bug went unnoticed: routed through
    ModelGate, a raising check becomes a blocking CHECK_ERROR, so asserting
    only that results exist is not enough — assert the flag."""
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    from bdp_model_gate import ModelGate

    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int).to_numpy()
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y)

    report = ModelGate(checks=[ShapSubgroupCheck()]).run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y,
            y_pred=model.predict_proba(X)[:, 1],
            protected_df=pd.DataFrame({"group": rng.choice(["x", "y"], n)}),
        )
    )
    assert not any(r.flag == "CHECK_ERROR" for r in report.results)
