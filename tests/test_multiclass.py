"""Multiclass and ordinal support.

The running example is underwriting — decline / refer / accept — which is
ordinal: a decline-vs-accept error is worse than refer-vs-accept, and plain
multiclass metrics cannot see that difference.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import GateConfig, ModelGate, PerformanceConfig, StructuredGateContext
from bdp_model_gate.classes import resolve_favourable, to_ranks
from bdp_model_gate.exceptions import GateConfigurationError, GateValidationError
from bdp_model_gate.metrics import ordinal_mae, quadratic_kappa, resolve_metric, to_class_labels
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.fairness import CounterfactualFlipCheck, DisparateImpactCheck
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.security import AdversarialRobustnessCheck
from bdp_model_gate.task import MULTICLASS

ORDER = ["decline", "refer", "accept"]


@pytest.fixture
def underwriting():
    """A book of applications where one region is declined far more often."""
    rng = np.random.default_rng(13)
    n = 600
    region = np.where(np.arange(n) % 2 == 0, "Lagos", "Kano")
    score = rng.uniform(0, 1, n) - np.where(region == "Kano", 0.35, 0.0)

    def decide(values):
        return np.where(values > 0.55, "accept", np.where(values > 0.2, "refer", "decline"))

    X = pd.DataFrame({"score": score, "tenure": rng.integers(0, 30, n).astype(float)})
    y_true = decide(score + rng.normal(0, 0.05, n))
    y_pred = decide(score)
    protected = pd.DataFrame({"region": region})
    return X, y_true, y_pred, protected


def _ctx(underwriting, **overrides):
    X, y_true, y_pred, protected = underwriting
    kwargs = dict(
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected,
        predict_fn=lambda df: np.where(
            df["score"] > 0.55, "accept", np.where(df["score"] > 0.2, "refer", "decline")
        ),
        task=MULTICLASS,
        class_order=ORDER,
    )
    kwargs.update(overrides)
    return StructuredGateContext(**kwargs)


# --- ordinal metrics --------------------------------------------------------


def test_ordinal_metrics_penalise_distance_not_just_error():
    """The whole point: two mistakes of equal count but different severity
    must not score the same."""
    truth = ["accept", "accept", "refer", "decline"]
    one_step = ["refer", "accept", "refer", "decline"]
    two_steps = ["decline", "accept", "refer", "decline"]

    assert ordinal_mae(truth, one_step, ORDER) < ordinal_mae(truth, two_steps, ORDER)
    assert quadratic_kappa(truth, one_step, ORDER) > quadratic_kappa(truth, two_steps, ORDER)

    # Plain accuracy sees one error either way — which is what these replace.
    assert sum(a != b for a, b in zip(truth, one_step)) == sum(
        a != b for a, b in zip(truth, two_steps)
    )


def test_quadratic_kappa_bounds():
    truth = ["decline", "refer", "accept"] * 10
    assert quadratic_kappa(truth, truth, ORDER) == pytest.approx(1.0)
    inverted = ["accept", "refer", "decline"] * 10
    assert quadratic_kappa(truth, inverted, ORDER) < 0  # worse than chance


def test_ordinal_metric_requires_class_order(underwriting):
    check = PerformanceThresholdCheck(PerformanceConfig(metric="ordinal_mae", max_error=1.0))
    with pytest.raises(GateConfigurationError, match="needs context.class_order"):
        check.run(_ctx(underwriting, class_order=None))


def test_ordinal_metrics_run_through_the_gate(underwriting):
    for name, config in (
        ("ordinal_mae", PerformanceConfig(metric="ordinal_mae", max_error=1.0)),
        ("quadratic_kappa", PerformanceConfig(metric="quadratic_kappa", min_score=-1.0)),
    ):
        result = PerformanceThresholdCheck(config).run(_ctx(underwriting))[0]
        assert result.metadata["metric"] == name
        assert result.flag == "OK"


def test_to_ranks_maps_favourability(underwriting):
    np.testing.assert_array_equal(to_ranks(["decline", "refer", "accept"], ORDER), [0, 1, 2])


# --- averaged metrics -------------------------------------------------------


@pytest.mark.parametrize("name", ["accuracy", "balanced_accuracy", "f1", "precision", "recall"])
def test_label_metrics_work_for_multiclass(underwriting, name):
    pytest.importorskip("sklearn")
    result = PerformanceThresholdCheck(PerformanceConfig(metric=name, min_score=0.0)).run(
        _ctx(underwriting)
    )[0]
    assert result.metadata["metric"] == name
    assert result.flag == "OK"


def test_average_strategy_is_configurable(underwriting):
    pytest.importorskip("sklearn")
    macro = PerformanceThresholdCheck(
        PerformanceConfig(metric="f1", min_score=0.0, average="macro")
    ).run(_ctx(underwriting))[0]
    weighted = PerformanceThresholdCheck(
        PerformanceConfig(metric="f1", min_score=0.0, average="weighted")
    ).run(_ctx(underwriting))[0]
    assert macro.metadata["value"] != weighted.metadata["value"]


def test_ranking_metrics_rejected_for_multiclass(underwriting):
    """roc_auc's multiclass form needs a full probability matrix, which the
    y_pred contract does not carry — so it is refused, not approximated."""
    check = PerformanceThresholdCheck(PerformanceConfig(metric="roc_auc", min_score=0.5))
    with pytest.raises(GateConfigurationError, match="does not apply to a multiclass task"):
        check.run(_ctx(underwriting))


def test_auto_picks_balanced_accuracy_for_multiclass():
    pytest.importorskip("sklearn")
    assert resolve_metric("auto", MULTICLASS).name == "balanced_accuracy"


def test_probability_matrix_reduced_by_argmax():
    matrix = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
    np.testing.assert_array_equal(to_class_labels(matrix, ORDER), ["accept", "decline", "refer"])


# --- favourable outcomes ----------------------------------------------------


def test_favourable_defaults_to_best_class():
    assert resolve_favourable(None, ORDER, MULTICLASS) == ["accept"]


def test_favourable_can_be_explicit():
    assert resolve_favourable(["accept", "refer"], ORDER, MULTICLASS) == ["accept", "refer"]


def test_unknown_favourable_class_rejected():
    with pytest.raises(GateConfigurationError, match="not in"):
        resolve_favourable(["approve"], ORDER, MULTICLASS)


def test_disparate_impact_catches_the_underwriting_skew(underwriting):
    pytest.importorskip("fairlearn")
    result = DisparateImpactCheck().run(_ctx(underwriting))[0]
    assert result.flag == "DISPARITY_RISK"
    assert "favourable: accept" in result.detail


def test_disparate_impact_declines_to_guess_a_favourable_class(underwriting):
    """Which outcome is favourable is a judgement the data cannot supply."""
    pytest.importorskip("fairlearn")
    result = DisparateImpactCheck().run(_ctx(underwriting, class_order=None))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "favourable" in result.detail


def test_favourable_set_changes_what_is_measured(underwriting):
    """Choosing the favourable set is a real modelling decision, not a
    formality: "was accepted" and "was not declined" are different questions
    and give different answers on the same data. That is precisely why the
    check refuses to infer it when there is no ordering to infer from."""
    pytest.importorskip("fairlearn")
    accepted = DisparateImpactCheck().run(_ctx(underwriting))[0]
    not_declined = DisparateImpactCheck().run(
        _ctx(underwriting, favourable_classes=["accept", "refer"])
    )[0]

    assert (
        accepted.metadata["demographic_parity_diff"]
        != not_declined.metadata["demographic_parity_diff"]
    )
    assert "favourable: accept" in accepted.detail
    assert "favourable: accept, refer" in not_declined.detail
    # Both are genuine disparities here — widening the set does not flatter
    # the model, because Lagos leads on accept *and* refer.
    assert accepted.flag == "DISPARITY_RISK"
    assert not_declined.flag == "DISPARITY_RISK"


# --- counterfactual ---------------------------------------------------------


def test_counterfactual_runs_for_multiclass(underwriting):
    """It was binary-only before 0.4.0."""
    X, y_true, y_pred, protected = underwriting
    X_with_attr = X.assign(region=(protected["region"] == "Kano").astype(float))

    def proba(df):
        base = df["score"].to_numpy()
        accept = np.clip(base, 0.01, 0.98)
        decline = np.clip(1 - base, 0.01, 0.98)
        refer = np.clip(1 - accept - decline + 0.5, 0.01, 0.98)
        stacked = np.column_stack([accept, decline, refer])  # sorted: accept, decline, refer
        return stacked / stacked.sum(axis=1, keepdims=True)

    context = StructuredGateContext(
        X=X_with_attr,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=pd.DataFrame({"region": X_with_attr["region"].to_numpy()}),
        predict_fn=lambda df: np.full(len(df), "refer"),
        predict_proba_fn=proba,
        task=MULTICLASS,
        class_order=ORDER,
    )
    results = CounterfactualFlipCheck().run(context)
    assert results
    assert all(r.flag != "CHECK_ERROR" for r in results)


def test_counterfactual_needs_class_order_for_multiclass(underwriting):
    result = CounterfactualFlipCheck().run(_ctx(underwriting, class_order=None))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "class_order" in result.detail


# --- ordinal robustness -----------------------------------------------------


def test_robustness_reports_rank_shift_for_ordinal_problems(underwriting):
    result = AdversarialRobustnessCheck().run(_ctx(underwriting))[0]
    assert "mean_rank_shift" in result.metadata
    assert result.metadata["n_classes"] == 3
    assert "ordinal rank shift" in result.detail


def test_rank_shift_absent_without_class_order(underwriting):
    result = AdversarialRobustnessCheck().run(_ctx(underwriting, class_order=None))[0]
    assert "mean_rank_shift" not in result.metadata
    assert "flip_rate" in result.metadata


def test_two_step_swings_score_worse_than_one_step_slips(underwriting):
    """Two models that flip at the same rate under perturbation, but by
    different distances, must not receive the same verdict — a bare flip
    rate cannot tell an accept->refer wobble from accept->decline."""
    from bdp_model_gate import SecurityConfig

    X, y_true, _, protected = underwriting

    def stepper(low_class):
        # Both models share a decision boundary at 0.5, so they flip on the
        # same rows; only how far the prediction moves differs.
        return lambda df: np.where(df["score"].to_numpy() > 0.5, "accept", low_class)

    # A large epsilon so plenty of rows cross the boundary and the two are
    # comparable on flip rate.
    config = SecurityConfig(adversarial_epsilon=0.5)
    common = dict(
        X=X,
        y_true=y_true,
        protected_df=protected,
        task=MULTICLASS,
        class_order=ORDER,
    )
    mild = AdversarialRobustnessCheck(config).run(
        StructuredGateContext(y_pred=stepper("refer")(X), predict_fn=stepper("refer"), **common)
    )[0]
    severe = AdversarialRobustnessCheck(config).run(
        StructuredGateContext(y_pred=stepper("decline")(X), predict_fn=stepper("decline"), **common)
    )[0]

    assert mild.metadata["flip_rate"] == pytest.approx(severe.metadata["flip_rate"])
    assert severe.metadata["mean_rank_shift"] > mild.metadata["mean_rank_shift"]
    assert mild.metadata["max_observed_rank_shift"] == 1.0
    assert severe.metadata["max_observed_rank_shift"] == 2.0


# --- validation and routing -------------------------------------------------


def test_class_order_must_cover_every_label(underwriting):
    with pytest.raises(GateValidationError, match="missing from context.class_order"):
        ModelGate(checks=[]).run(_ctx(underwriting, class_order=["decline", "accept"]))


def test_duplicate_class_order_rejected(underwriting):
    with pytest.raises(GateValidationError, match="duplicate"):
        ModelGate(checks=[]).run(_ctx(underwriting, class_order=["refer", "refer", "accept"]))


def test_regression_checks_not_applicable_for_multiclass(underwriting):
    config = GateConfig()
    config.performance.metric = "quadratic_kappa"
    config.performance.min_score = -1.0
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        _ctx(underwriting)
    )
    by_name = {r.check_name: r for r in report.results}
    for name in ("loss_ratio_parity", "group_mean_gap", "error_parity", "calibration_parity"):
        assert by_name[name].flag == "NOT_APPLICABLE"
    assert report.task == MULTICLASS
    assert not any(r.flag == "CHECK_ERROR" for r in report.results)
