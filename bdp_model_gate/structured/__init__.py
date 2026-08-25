from __future__ import annotations

from ..config import GateConfig
from .compliance import ComplianceMappingCheck
from .fairness import (
    CounterfactualFlipCheck,
    DisparateImpactCheck,
    ProxyCorrelationCheck,
    ShapSubgroupCheck,
)
from .performance import PerformanceThresholdCheck
from .security import (
    AdversarialRobustnessCheck,
    PIILeakageCheck,
    PromptInjectionCheck,
)


def default_structured_checks(config: GateConfig | None = None, include_plugins: bool = True):
    """The full default check suite for structured-data models.

    If `include_plugins` is True (default), also appends any checks
    registered via the `bdp_model_gate.checks` entry-point group — see
    `bdp_model_gate.registry`.
    """
    config = config or GateConfig()
    checks = [
        ProxyCorrelationCheck(config.fairness),
        DisparateImpactCheck(config.fairness),
        ShapSubgroupCheck(config.fairness),
        CounterfactualFlipCheck(config.fairness),
        PerformanceThresholdCheck(config.performance),
        ComplianceMappingCheck(config.compliance),
        AdversarialRobustnessCheck(config.security),
        PIILeakageCheck(config.security),
        PromptInjectionCheck(config.security),
    ]
    if include_plugins:
        from ..registry import discover_plugin_checks

        checks.extend(discover_plugin_checks())
    return checks


__all__ = [
    "ProxyCorrelationCheck",
    "DisparateImpactCheck",
    "ShapSubgroupCheck",
    "CounterfactualFlipCheck",
    "PerformanceThresholdCheck",
    "ComplianceMappingCheck",
    "AdversarialRobustnessCheck",
    "PIILeakageCheck",
    "PromptInjectionCheck",
    "default_structured_checks",
]
