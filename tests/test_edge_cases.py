"""Edge cases beyond the happy path: models without predict_proba, all
non-numeric features, no protected attributes, check-level exceptions, etc."""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.core.base import BaseCheck, CheckResult


class PredictOnlyModel:
    """A model exposing only .predict(), no .predict_proba() — common for
    some sklearn estimators and most non-probabilistic models."""

    def predict(self, X: pd.DataFrame):
        return (X["income"] > X["income"].median()).astype(int).values


def test_model_without_predict_proba_degrades_gracefully(synthetic_data):
    _, X, y_true, _, protected_df = synthetic_data
    model = PredictOnlyModel()
    y_pred = model.predict(X)
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    report = ModelGate().run(context)

    counterfactual_results = [r for r in report.results if r.check_name == "counterfactual_flip"]
    assert all(r.flag == "NOT_APPLICABLE" for r in counterfactual_results)
    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}


def test_all_categorical_features_skip_proxy_and_robustness_checks(synthetic_data):
    model, _, y_true, y_pred, protected_df = synthetic_data
    X_categorical = pd.DataFrame(
        {
            "tier": np.random.choice(["gold", "silver", "bronze"], len(y_true)),
            "region_code": np.random.choice(["NG-LA", "NG-AB"], len(y_true)),
        }
    )

    class CategoricalModel:
        def predict(self, X):
            return (X["tier"] == "gold").astype(int).values

    context = StructuredGateContext(
        model=CategoricalModel(),
        X=X_categorical,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    report = ModelGate().run(context)

    robustness = [r for r in report.results if r.check_name == "adversarial_robustness"]
    assert all(r.flag == "NOT_APPLICABLE" for r in robustness)


@pytest.mark.expect_check_error
def test_check_exception_is_isolated_and_reported(small_valid_context):
    class ExplodingCheck(BaseCheck):
        name = "exploding_check"
        category = "security"
        blocking = True

        def run(self, context):
            raise RuntimeError("boom")

    gate = ModelGate(checks=[ExplodingCheck()])
    report = gate.run(small_valid_context)

    assert report.gate_status == "BLOCKED"
    error_results = [r for r in report.results if r.flag == "CHECK_ERROR"]
    assert len(error_results) == 1
    assert "boom" in error_results[0].detail


def test_custom_check_can_be_added(small_valid_context):
    class AlwaysOkCheck(BaseCheck):
        name = "always_ok"
        category = "compliance"
        blocking = True

        def run(self, context):
            return [CheckResult(self.name, self.category, "OK", "fine", self.blocking)]

    gate = ModelGate(checks=[AlwaysOkCheck()])
    report = gate.run(small_valid_context)
    assert report.gate_status == "PASS"
    assert report.results[0].check_name == "always_ok"


def test_adversarial_check_is_deterministic_across_runs(synthetic_data):
    """A governance verdict has to be reproducible: the same model and data
    must always yield the same flip rate, or a model can pass one CI run and
    block the next with nothing changed."""
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    _, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(model=PredictOnlyModel(), X=X, y_true=y_true, y_pred=y_pred)

    rates = [AdversarialRobustnessCheck().run(context)[0].metadata["flip_rate"] for _ in range(5)]
    assert len(set(rates)) == 1


def test_adversarial_check_random_state_is_honoured(synthetic_data):
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    _, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(model=PredictOnlyModel(), X=X, y_true=y_true, y_pred=y_pred)

    a = AdversarialRobustnessCheck(random_state=1).run(context)[0]
    b = AdversarialRobustnessCheck(random_state=1).run(context)[0]
    assert a.metadata["flip_rate"] == b.metadata["flip_rate"]


def test_adversarial_coefficients_ignored_when_not_aligned_to_columns(synthetic_data):
    """coef_ is laid out over every column the model was fitted on. If it
    doesn't line up with X's columns we can't index it safely, so the check
    must fall back to random perturbation rather than mis-attribute weights."""
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    _, X, y_true, y_pred, _ = synthetic_data

    class MisalignedCoefModel(PredictOnlyModel):
        # fitted on a transformed feature space — 7 coefficients, 3 columns
        coef_ = np.arange(7, dtype=float)

    context = StructuredGateContext(model=MisalignedCoefModel(), X=X, y_true=y_true, y_pred=y_pred)
    result = AdversarialRobustnessCheck().run(context)[0]
    assert result.metadata["method"] == "random"


def test_adversarial_coefficients_align_by_column_position(synthetic_data):
    """With a mixed numeric/categorical X, the direction must be indexed by
    position in X.columns — not by position among the numeric columns."""
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    _, X, y_true, y_pred, _ = synthetic_data
    X_mixed = X.copy()
    X_mixed.insert(0, "branch", "lagos")  # non-numeric column ahead of the numerics

    class LinearishModel(PredictOnlyModel):
        coef_ = np.array([0.0, 1.0, 0.5, 0.25])  # one weight per column of X_mixed

    context = StructuredGateContext(model=LinearishModel(), X=X_mixed, y_true=y_true, y_pred=y_pred)
    result = AdversarialRobustnessCheck().run(context)[0]
    assert result.metadata["method"] == "gradient-directed"


def test_multiclass_coefficients_fall_back_to_random(synthetic_data):
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    _, X, y_true, y_pred, _ = synthetic_data

    class MulticlassModel(PredictOnlyModel):
        coef_ = np.ones((3, 3))  # one row per class — no single ascent direction

    context = StructuredGateContext(model=MulticlassModel(), X=X, y_true=y_true, y_pred=y_pred)
    result = AdversarialRobustnessCheck().run(context)[0]
    assert result.metadata["method"] == "random"


def test_unknown_modality_is_rejected(synthetic_data):
    """ModelGate dispatches validation on context.modality — an unknown one
    must fail loudly rather than be silently validated as structured."""
    from bdp_model_gate.exceptions import GateValidationError

    _, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(model=PredictOnlyModel(), X=X, y_true=y_true, y_pred=y_pred)
    context.modality = "text"

    with pytest.raises(GateValidationError, match="no input validator registered"):
        ModelGate(checks=[]).run(context)
