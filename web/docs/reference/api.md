# API reference

Generated from the source, so it cannot drift from the code.

## Core

::: bdp_model_gate.core.context.StructuredGateContext
::: bdp_model_gate.core.gate.ModelGate
::: bdp_model_gate.core.report.GateReport
::: bdp_model_gate.core.base.CheckResult
::: bdp_model_gate.core.base.BaseCheck

## Configuration

::: bdp_model_gate.config

## Tasks and classes

::: bdp_model_gate.task
    options:
      members: [resolve_task, infer_task, validate_task, supports]

::: bdp_model_gate.classes
    options:
      members: [resolve_favourable, to_ranks, validate_class_order, favourable_mask]

## Metrics

::: bdp_model_gate.metrics
    options:
      members: [resolve_metric, validate_metric, MetricSpec, ResolvedMetric, to_hard_labels, to_class_labels, ordinal_mae, quadratic_kappa]

## Model adapter

::: bdp_model_gate.model.ModelAdapter

## Checks

### Fairness

::: bdp_model_gate.structured.fairness

### Fairness — regression

::: bdp_model_gate.structured.regression_fairness

### Performance

::: bdp_model_gate.structured.performance

### Compliance

::: bdp_model_gate.structured.compliance

### Security

::: bdp_model_gate.structured.security

## Registry and errors

::: bdp_model_gate.registry
    options:
      members: [discover_plugin_checks]

::: bdp_model_gate.exceptions
