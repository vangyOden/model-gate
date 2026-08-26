"""Tests for configurable performance metrics — selection, fallback
visibility, thresholding, and the deprecated min_accuracy/model_auc aliases."""

import numpy as np
import pytest

from bdp_model_gate import GateConfig, ModelGate, PerformanceConfig, StructuredGateContext
from bdp_model_gate.core.report import GateReport
from bdp_model_gate.exceptions import GateConfigurationError
from bdp_model_gate.metrics import (
    BUILTIN_METRICS,
    resolve_metric,
    to_hard_labels,
    validate_metric,
)
from bdp_model_gate.structured.performance import PerformanceThresholdCheck


def _perf_context(synthetic_data, **kwargs):
    model, X, y_true, y_pred, _ = synthetic_data
    return StructuredGateContext(model=model, X=X, y_true=y_true, y_pred=y_pred, **kwargs)


# --- metric selection -------------------------------------------------------


CLASSIFICATION_METRICS = sorted(m for m, s in BUILTIN_METRICS.items() if "binary" in s.tasks)


@pytest.mark.parametrize("name", CLASSIFICATION_METRICS)
def test_every_builtin_classification_metric_runs(synthetic_data, name):
    pytest.importorskip("sklearn", reason="most builtin metrics need scikit-learn")
    config = PerformanceConfig(metric=name, min_score=0.0)
    results = PerformanceThresholdCheck(config).run(_perf_context(synthetic_data))

    assert len(results) == 1
    assert results[0].metadata["metric"] == name
    assert results[0].flag == "OK"
    assert name in results[0].detail


def test_custom_callable_metric_is_used_and_named(synthetic_data):
    def half_of_everything(y_true, y_pred):
        return 0.5

    config = PerformanceConfig(metric=half_of_everything, min_score=0.8)
    results = PerformanceThresholdCheck(config).run(_perf_context(synthetic_data))

    assert results[0].metadata["metric"] == "half_of_everything"
    assert results[0].metadata["value"] == 0.5
    assert results[0].flag == "PERFORMANCE_RISK"


def test_unknown_metric_name_fails_at_construction():
    with pytest.raises(GateConfigurationError, match="unknown performance.metric"):
        PerformanceThresholdCheck(PerformanceConfig(metric="fscore"))


def test_non_string_non_callable_metric_rejected():
    with pytest.raises(GateConfigurationError, match="metric name or a callable"):
        validate_metric(3.14)


def test_auto_resolves_to_roc_auc_when_sklearn_present():
    pytest.importorskip("sklearn")
    resolved = resolve_metric("auto")
    assert resolved.name == "roc_auc"
    assert resolved.is_fallback is False


def test_auto_falls_back_to_accuracy_and_says_so(monkeypatch, synthetic_data, caplog):
    """The whole point of the fallback being explicit: it must be logged,
    flagged in metadata, and named in the human-readable detail string."""
    monkeypatch.setattr("bdp_model_gate.metrics._load_sklearn_metric", lambda spec: None)

    config = PerformanceConfig(metric="auto", min_score=0.0)
    with caplog.at_level("WARNING", logger="bdp_model_gate.metrics"):
        results = PerformanceThresholdCheck(config).run(_perf_context(synthetic_data))

    assert results[0].metadata["metric"] == "accuracy"
    assert results[0].metadata["metric_is_fallback"] is True
    assert "fell back" in results[0].detail
    assert "scikit-learn not installed" in caplog.text


def test_explicit_metric_errors_rather_than_substituting(monkeypatch, synthetic_data):
    """An explicitly requested metric is never silently swapped — if it
    can't run, the gate blocks instead of scoring against something else."""
    monkeypatch.setattr("bdp_model_gate.metrics._load_sklearn_metric", lambda spec: None)

    check = PerformanceThresholdCheck(PerformanceConfig(metric="roc_auc"))
    with pytest.raises(GateConfigurationError, match="requires scikit-learn"):
        check.run(_perf_context(synthetic_data))


@pytest.mark.expect_check_error
def test_unavailable_metric_blocks_the_gate(monkeypatch, synthetic_data):
    monkeypatch.setattr("bdp_model_gate.metrics._load_sklearn_metric", lambda spec: None)

    checks = [PerformanceThresholdCheck(PerformanceConfig(metric="f1"))]
    report = ModelGate(checks=checks).run(_perf_context(synthetic_data))

    assert report.gate_status == "BLOCKED"
    assert report.flags[0].flag == "CHECK_ERROR"


# --- thresholding -----------------------------------------------------------


def test_continuous_predictions_binarized_for_label_metrics():
    assert list(to_hard_labels(np.array([0.1, 0.7, 0.5]), 0.5)) == [0, 1, 1]
    assert list(to_hard_labels(np.array([0.1, 0.7, 0.5]), 0.75)) == [0, 0, 0]


def test_hard_labels_pass_through_untouched():
    labels = np.array([0, 1, 1, 0])
    assert list(to_hard_labels(labels, 0.9)) == [0, 1, 1, 0]


def test_decision_threshold_changes_accuracy_score(synthetic_data):
    pytest.importorskip("sklearn")
    # The shared fixture's model emits hard 0/1 probabilities, which
    # to_hard_labels passes through untouched — use genuinely continuous
    # scores so the threshold has something to bite on.
    model, X, y_true, _, _ = synthetic_data
    scores = X["income"].rank(pct=True).to_numpy()
    context = StructuredGateContext(model=model, X=X, y_true=y_true, y_pred=scores)

    lenient = PerformanceThresholdCheck(
        PerformanceConfig(metric="accuracy", min_score=0.0, decision_threshold=0.5)
    ).run(context)
    strict = PerformanceThresholdCheck(
        PerformanceConfig(metric="accuracy", min_score=0.0, decision_threshold=0.99)
    ).run(context)

    assert lenient[0].metadata["value"] != strict[0].metadata["value"]


# --- report plumbing --------------------------------------------------------


def test_report_carries_the_configured_metric(synthetic_data):
    pytest.importorskip("sklearn")
    checks = [PerformanceThresholdCheck(PerformanceConfig(metric="accuracy", min_score=0.0))]
    report = ModelGate(checks=checks).run(_perf_context(synthetic_data))

    assert report.model_metric == "accuracy"
    assert report.model_score is not None
    assert "accuracy" in report.summary()
    assert report.to_dict()["model_metric"] == "accuracy"


def test_report_metric_is_none_without_a_performance_check(synthetic_data):
    from bdp_model_gate.structured.security import PIILeakageCheck

    report = ModelGate(checks=[PIILeakageCheck()]).run(_perf_context(synthetic_data))
    assert report.model_metric is None
    assert report.model_score is None


def test_model_auc_only_populated_when_metric_really_is_auc():
    auc_report = GateReport(results=[], model_metric="roc_auc", model_score=0.9)
    f1_report = GateReport(results=[], model_metric="f1", model_score=0.9)

    with pytest.warns(DeprecationWarning):
        assert auc_report.model_auc == 0.9
    with pytest.warns(DeprecationWarning):
        assert f1_report.model_auc is None

    assert auc_report.to_dict()["model_auc"] == 0.9
    assert f1_report.to_dict()["model_auc"] is None


# --- deprecated alias -------------------------------------------------------


def test_min_accuracy_alias_still_sets_min_score():
    config = GateConfig()
    with pytest.warns(DeprecationWarning, match="min_accuracy is deprecated"):
        config.performance.min_accuracy = 0.95
    assert config.performance.min_score == 0.95
    with pytest.warns(DeprecationWarning):
        assert config.performance.min_accuracy == 0.95
