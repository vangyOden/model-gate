"""Smoke tests for the structured gate pipeline. Run with: pytest -q"""

from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext, run_structured_gate


def test_gate_runs_end_to_end(small_valid_context):
    report = ModelGate().run(small_valid_context)

    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}
    assert len(report.results) > 0
    assert isinstance(report.to_dict(), dict)
    assert "gate_status" in report.to_json()
    assert report.total_duration_ms >= 0
    assert all(r.duration_ms is not None for r in report.results)


def test_gate_without_optional_inputs_reports_not_applicable(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(model=model, X=X, y_true=y_true, y_pred=y_pred)
    report = ModelGate().run(context)

    fairness_results = report.by_category("fairness")
    assert all(
        r.flag == "NOT_APPLICABLE"
        for r in fairness_results
        if r.check_name in {"proxy_correlation", "disparate_impact"}
    )


def test_compliance_blocks_on_missing_model_card_fields(synthetic_data):
    model, X, y_true, y_pred, protected_df = synthetic_data
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        model_card={"use_case": "claims_decisioning"},  # missing required fields, high-risk
    )
    report = ModelGate().run(context)

    compliance_flags = [r for r in report.by_category("compliance") if not r.is_ok]
    assert len(compliance_flags) > 0
    assert report.gate_status == "BLOCKED"


def test_pii_leakage_flags_email_column(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    X_with_pii = X.copy()
    X_with_pii["notes"] = ["contact me at test.user@example.com"] * len(X_with_pii)
    context = StructuredGateContext(model=model, X=X_with_pii, y_true=y_true, y_pred=y_pred)
    report = ModelGate().run(context)

    pii_flags = [
        r for r in report.by_category("security") if r.check_name == "pii_leakage" and not r.is_ok
    ]
    assert len(pii_flags) > 0
    assert report.gate_status == "BLOCKED"


def test_run_structured_gate_convenience_function(synthetic_data):
    model, X, y_true, y_pred, protected_df = synthetic_data
    report = run_structured_gate(model, X, y_true, y_pred, protected_df=protected_df)
    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}


def test_custom_config_thresholds(synthetic_data):
    model, X, y_true, y_pred, protected_df = synthetic_data
    strict_config = GateConfig()
    strict_config.performance.min_score = 0.999  # near-impossible on this synthetic data
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    from bdp_model_gate.structured import default_structured_checks

    checks = default_structured_checks(strict_config, include_plugins=False)
    gate = ModelGate(checks=checks)
    report = gate.run(context)
    perf_flags = [r for r in report.by_category("performance") if not r.is_ok]
    assert len(perf_flags) > 0
