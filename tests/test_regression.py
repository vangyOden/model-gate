"""Regression-task support: task resolution, metrics, thresholds, and the
four regression fairness notions."""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import (
    FairnessConfig,
    GateConfig,
    ModelGate,
    PerformanceConfig,
    StructuredGateContext,
)
from bdp_model_gate.exceptions import GateConfigurationError, GateValidationError
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.regression_fairness import (
    CalibrationParityCheck,
    ErrorParityCheck,
    GroupMeanGapCheck,
    LossRatioParityCheck,
)
from bdp_model_gate.task import BINARY, MULTICLASS, REGRESSION, infer_task, resolve_task


class PremiumModel:
    """A stand-in motor pricing model: premium scales with the risk score."""

    def predict(self, X):
        return (X["risk_score"].to_numpy() * 1000.0) + 15000.0


@pytest.fixture
def premium_data():
    """A synthetic book of motor policies with a deliberate margin skew:
    one region is charged more relative to its own expected loss."""
    rng = np.random.default_rng(7)
    n = 600
    region = rng.choice(["Lagos", "Kano"], n)
    risk_score = rng.gamma(shape=4.0, scale=1.5, size=n)

    X = pd.DataFrame({"risk_score": risk_score, "vehicle_age": rng.integers(0, 20, n)})
    model = PremiumModel()
    premium = model.predict(X)

    # Expected loss tracks risk, but Kano's is lower than its premium implies,
    # so Kano carries the fatter margin without a higher headline premium.
    expected_loss = premium / np.where(region == "Kano", 1.45, 1.05)
    realised = expected_loss * rng.uniform(0.9, 1.1, n)

    protected_df = pd.DataFrame({"region": region})
    return model, X, realised, premium, protected_df, expected_loss


def _ctx(premium_data, **overrides):
    model, X, y_true, y_pred, protected_df, expected_loss = premium_data
    kwargs = dict(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
        expected_loss=expected_loss,
        task=REGRESSION,
    )
    kwargs.update(overrides)
    return StructuredGateContext(**kwargs)


# --- task resolution --------------------------------------------------------


@pytest.mark.parametrize(
    "y_true,expected",
    [
        (np.array([0, 1, 1, 0]), BINARY),
        (np.array(["yes", "no", "yes"]), BINARY),
        (np.array(["accept", "refer", "decline", "accept"]), MULTICLASS),
        (np.array([0, 1, 2, 3, 1, 2]), MULTICLASS),
        (np.array([1200.5, 980.0, 44000.25, 310.75, 7.5]), REGRESSION),
    ],
)
def test_task_inference(y_true, expected):
    assert infer_task(y_true) == expected


def test_explicit_task_overrides_inference(premium_data):
    """A count target is genuinely ambiguous, which is why the explicit
    setting has to win."""
    counts = np.array([0, 1, 2, 3, 0, 1] * 20)
    assert infer_task(counts) == MULTICLASS

    _, X, _, y_pred, _, _ = premium_data
    context = StructuredGateContext(
        model=PremiumModel(),
        X=X.head(120),
        y_true=counts,
        y_pred=y_pred[:120],
        task=REGRESSION,
    )
    assert resolve_task(context) == REGRESSION


def test_unknown_task_is_rejected(premium_data):
    with pytest.raises(GateConfigurationError, match="context.task must be one of"):
        ModelGate().run(_ctx(premium_data, task="ordinal"))


def test_missing_labels_warns_and_assumes_binary(premium_data, caplog):
    _, X, _, _, _, _ = premium_data
    context = StructuredGateContext(model=PremiumModel(), X=X, y_true=None, y_pred=None)
    with caplog.at_level("WARNING", logger="bdp_model_gate.task"):
        assert resolve_task(context) == BINARY
    assert "cannot infer without y_true" in caplog.text


# --- metrics and thresholds -------------------------------------------------


@pytest.mark.parametrize("name", ["rmse", "mae", "mape", "r2"])
def test_regression_metrics_run(premium_data, name):
    threshold = {"r2": PerformanceConfig(metric="r2", min_score=-1e9)}.get(
        name, PerformanceConfig(metric=name, max_error=1e12)
    )
    result = PerformanceThresholdCheck(threshold).run(_ctx(premium_data))[0]
    assert result.metadata["metric"] == name
    assert result.flag == "OK"


def test_error_metric_uses_max_error_not_min_score(premium_data):
    config = PerformanceConfig(metric="rmse", max_error=1.0)  # unreachably tight
    result = PerformanceThresholdCheck(config).run(_ctx(premium_data))[0]

    assert result.flag == "PERFORMANCE_RISK"
    assert result.metadata["threshold_field"] == "max_error"
    assert result.metadata["greater_is_better"] is False
    assert "max 1.0" in result.detail


def test_error_metric_without_max_error_is_a_configuration_error(premium_data):
    """Silently passing would be the dangerous outcome: the whole point of
    the gate is the comparison."""
    check = PerformanceThresholdCheck(PerformanceConfig(metric="mae"))
    with pytest.raises(GateConfigurationError, match="performance.max_error"):
        check.run(_ctx(premium_data))


def test_classification_metric_rejected_for_regression(premium_data):
    check = PerformanceThresholdCheck(PerformanceConfig(metric="roc_auc", min_score=0.8))
    with pytest.raises(GateConfigurationError, match="does not apply to a regression task"):
        check.run(_ctx(premium_data))


def test_auto_picks_r2_for_regression(premium_data):
    result = PerformanceThresholdCheck(PerformanceConfig(min_score=-1e9)).run(_ctx(premium_data))[0]
    assert result.metadata["metric"] == "r2"
    assert result.metadata["threshold_field"] == "min_score"


def test_poisson_deviance_rejects_non_positive_predictions():
    from bdp_model_gate.metrics import resolve_metric

    fn = resolve_metric("poisson_deviance", REGRESSION).fn
    with pytest.raises(GateConfigurationError, match="strictly positive"):
        fn(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    assert fn(np.array([0.0, 2.0]), np.array([1.0, 2.0])) > 0  # zero actuals are fine


# --- regression fairness ----------------------------------------------------


def test_loss_ratio_parity_catches_a_margin_skew(premium_data):
    """The headline premium gap is modest, but one region is charged a much
    higher margin over its own expected loss — the actuarial question."""
    result = LossRatioParityCheck().run(_ctx(premium_data))[0]

    assert result.flag == "LOSS_RATIO_RISK"
    assert result.metadata["highest_margin_group"] == "Kano"
    ratios = result.metadata["group_loss_ratio"]
    assert ratios["Kano"] > ratios["Lagos"]


def test_loss_ratio_parity_needs_expected_loss(premium_data):
    """It must not silently fall back to comparing raw prices, which answers
    a different question under the same name."""
    result = LossRatioParityCheck().run(_ctx(premium_data, expected_loss=None))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "expected_loss" in result.detail


def test_group_mean_gap_reports_the_raw_level_difference(premium_data):
    result = GroupMeanGapCheck().run(_ctx(premium_data))[0]
    assert result.metadata["protected_attr"] == "region"
    assert set(result.metadata["group_means"]) == {"Lagos", "Kano"}


def test_error_parity_flags_a_worse_served_group(premium_data):
    model, X, y_true, y_pred, protected, expected_loss = premium_data
    # Make predictions much worse for Kano only.
    degraded = y_pred.copy()
    kano = (protected["region"] == "Kano").to_numpy()
    degraded[kano] = degraded[kano] * 3.0

    result = ErrorParityCheck().run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_true,
            y_pred=degraded,
            protected_df=protected,
            task=REGRESSION,
        )
    )[0]
    assert result.flag == "ERROR_PARITY_RISK"
    assert result.metadata["worst_served_group"] == "Kano"


def test_calibration_parity_flags_systematic_over_prediction(premium_data):
    model, X, y_true, y_pred, protected, _ = premium_data
    biased = y_pred.copy()
    lagos = (protected["region"] == "Lagos").to_numpy()
    biased[lagos] = biased[lagos] + y_true.mean() * 0.5

    result = CalibrationParityCheck().run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_true,
            y_pred=biased,
            protected_df=protected,
            task=REGRESSION,
        )
    )[0]
    assert result.flag == "CALIBRATION_RISK"
    assert result.metadata["most_over_predicted"] == "Lagos"


def test_small_groups_are_reported_not_scored(premium_data):
    model, X, y_true, y_pred, _, expected_loss = premium_data
    # One group of 3 alongside a large one.
    region = np.array(["Lagos"] * (len(X) - 3) + ["Jigawa"] * 3)
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=pd.DataFrame({"region": region}),
        task=REGRESSION,
    )
    results = GroupMeanGapCheck(FairnessConfig(min_group_size=30)).run(context)
    assert results[0].flag == "NOT_APPLICABLE"
    assert "min_group_size" in results[0].detail


def test_regression_fairness_needs_protected_df(premium_data):
    for check in (
        GroupMeanGapCheck(),
        ErrorParityCheck(),
        CalibrationParityCheck(),
        LossRatioParityCheck(),
    ):
        result = check.run(_ctx(premium_data, protected_df=None))[0]
        assert result.flag == "NOT_APPLICABLE"


# --- routing and end-to-end -------------------------------------------------


def test_classification_checks_are_not_applicable_for_regression(premium_data):
    report = ModelGate(
        checks=default_structured_checks(
            GateConfig(performance=PerformanceConfig(metric="r2", min_score=-1e9)),
            include_plugins=False,
        )
    ).run(_ctx(premium_data))

    by_name = {r.check_name: r for r in report.results}
    for name in ("disparate_impact", "counterfactual_flip"):
        assert by_name[name].flag == "NOT_APPLICABLE"
        assert "regression" in by_name[name].detail
    # And the regression suite did run.
    assert by_name["loss_ratio_parity"].flag != "NOT_APPLICABLE"
    assert report.task == REGRESSION


def test_adversarial_uses_relative_shift_not_flip_rate(premium_data):
    """A flip rate on a continuous output is ~1.0, which would block every
    regression model permanently."""
    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    result = AdversarialRobustnessCheck().run(_ctx(premium_data))[0]
    assert "relative_shift" in result.metadata
    assert "flip_rate" not in result.metadata
    assert result.metadata["task"] == REGRESSION
    assert result.flag == "OK"  # a linear premium model is not over-sensitive


def test_report_records_the_task(premium_data):
    report = ModelGate(
        checks=default_structured_checks(
            GateConfig(performance=PerformanceConfig(metric="r2", min_score=-1e9)),
            include_plugins=False,
        )
    ).run(_ctx(premium_data))
    assert report.to_dict()["task"] == REGRESSION
    assert REGRESSION in report.summary()


# --- validation -------------------------------------------------------------


def test_non_numeric_target_rejected_for_regression(premium_data):
    _, X, _, y_pred, _, _ = premium_data
    context = StructuredGateContext(
        model=PremiumModel(),
        X=X,
        y_true=np.array(["a", "b"] * (len(X) // 2)),
        y_pred=y_pred,
        task=REGRESSION,
    )
    with pytest.raises(GateValidationError, match="must be numeric for a regression task"):
        ModelGate().run(context)


def test_nan_target_rejected_for_regression(premium_data):
    _, X, y_true, y_pred, _, _ = premium_data
    broken = y_true.copy()
    broken[0] = np.nan
    context = StructuredGateContext(
        model=PremiumModel(), X=X, y_true=broken, y_pred=y_pred, task=REGRESSION
    )
    with pytest.raises(GateValidationError, match="NaN or infinite"):
        ModelGate().run(context)


def test_expected_loss_must_be_row_aligned(premium_data):
    with pytest.raises(GateValidationError, match="row-aligned"):
        ModelGate().run(_ctx(premium_data, expected_loss=np.array([1.0, 2.0, 3.0])))


def test_negative_expected_loss_rejected(premium_data):
    _, X, _, _, _, expected_loss = premium_data
    bad = expected_loss.copy()
    bad[0] = -5.0
    with pytest.raises(GateValidationError, match="cannot be below zero"):
        ModelGate().run(_ctx(premium_data, expected_loss=bad))


def test_shap_degrades_instead_of_blocking_on_an_opaque_model(premium_data):
    """A model exposing only .predict() satisfies this library's validation,
    but shap's generic Explainer wants a callable or an estimator it knows.
    That must not turn a non-blocking fairness check into a blocking
    CHECK_ERROR that stops a deploy."""
    pytest.importorskip("shap")
    from bdp_model_gate.structured.fairness import ShapSubgroupCheck

    report = ModelGate(checks=[ShapSubgroupCheck()]).run(_ctx(premium_data))
    assert all(r.flag != "CHECK_ERROR" for r in report.results)
    assert report.gate_status != "BLOCKED"


def test_perturbation_scale_is_per_feature_not_global():
    """With one feature in the millions and another in single digits, a single
    global perturbation scale is dominated by the large column and shoves the
    small one by orders of magnitude — which reported a relative shift of
    ~1448 and blocked a perfectly stable linear model."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(
        {
            "risk_score": rng.gamma(4, 1.5, n),  # single digits
            "sum_insured": rng.lognormal(15.2, 0.4, n),  # millions
        }
    )
    y = (X["risk_score"] * 9000 + X["sum_insured"] * 0.03).to_numpy()
    model = LinearRegression().fit(X, y)

    context = StructuredGateContext(
        model=model, X=X, y_true=y, y_pred=model.predict(X), task=REGRESSION
    )
    result = AdversarialRobustnessCheck().run(context)[0]

    assert result.metadata["method"] == "gradient-directed"
    # A 2% input nudge on a linear model must not move the output by more
    # than a few percent, let alone several hundred times.
    assert result.metadata["relative_shift"] < 0.1
    assert result.flag == "OK"
