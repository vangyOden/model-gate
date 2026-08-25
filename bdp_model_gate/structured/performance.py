"""Performance/cost thresholds that must pass before promotion."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._logging import get_logger
from ..config import PerformanceConfig
from ..core.base import BaseCheck, CheckResult
from ..metrics import ResolvedMetric, resolve_metric, to_hard_labels, validate_metric

logger = get_logger("performance")


class PerformanceThresholdCheck(BaseCheck):
    """Hard gate on model score, p95 latency, and cost-per-inference.

    The score metric is whatever `PerformanceConfig.metric` names — see
    `bdp_model_gate.metrics`. Which metric actually ran is recorded in the
    result's detail string and metadata, so a report always states what
    `min_score` was compared against.

    latencies_ms and cost_per_inference are optional on the context — if
    neither is supplied, only the score is checked; if the score inputs are
    also unavailable the check reports NOT_APPLICABLE rather than failing.
    """

    name = "performance_thresholds"
    category = "performance"
    blocking = True

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()
        # Fail at construction time on a typo'd metric name, rather than
        # partway through a gate run. Dependency availability is checked
        # lazily in _score(), so building the suite never needs sklearn.
        validate_metric(self.config.metric)

    def _score(self, y_true: Any, y_pred: Any) -> tuple[ResolvedMetric, float]:
        """Scores the model with the configured metric.

        Raises GateConfigurationError if an explicitly requested metric
        isn't available; ModelGate turns that into a blocking CHECK_ERROR
        so the pipeline stops rather than proceeding on a substituted score.
        """
        metric = resolve_metric(self.config.metric)
        y_pred_eval = (
            to_hard_labels(y_pred, self.config.decision_threshold)
            if metric.needs_hard_labels
            else y_pred
        )
        return metric, float(metric.fn(y_true, y_pred_eval))

    def _score_result(self, context) -> CheckResult:
        metric, score = self._score(context.y_true, context.y_pred)

        notes = []
        if metric.is_fallback:
            notes.append("fell back from the preferred metric — scikit-learn not installed")
        if metric.used_fallback_impl:
            notes.append("computed without scikit-learn")
        suffix = f" [{'; '.join(notes)}]" if notes else ""

        logger.debug(
            "scored with metric=%s value=%.4f threshold=%s fallback=%s",
            metric.name,
            score,
            self.config.min_score,
            metric.is_fallback,
        )

        return CheckResult(
            self.name,
            self.category,
            "OK" if score >= self.config.min_score else "PERFORMANCE_RISK",
            detail=f"{metric.name}={score:.4f} (min {self.config.min_score}){suffix}",
            blocking=self.blocking,
            metadata={
                "metric_kind": "score",
                "metric": metric.name,
                "value": round(score, 4),
                "threshold": self.config.min_score,
                "metric_is_fallback": metric.is_fallback,
            },
        )

    def run(self, context) -> list[CheckResult]:
        results = []

        if context.y_true is not None and context.y_pred is not None:
            results.append(self._score_result(context))

        if context.latencies_ms is not None:
            p95 = float(np.percentile(context.latencies_ms, 95))
            flag = "OK" if p95 <= self.config.max_latency_ms_p95 else "PERFORMANCE_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"p95 latency={p95:.2f}ms (max {self.config.max_latency_ms_p95}ms)",
                    blocking=self.blocking,
                    metadata={
                        "metric_kind": "latency",
                        "metric": "latency_p95_ms",
                        "value": round(p95, 2),
                        "threshold": self.config.max_latency_ms_p95,
                    },
                )
            )

        if context.cost_per_inference is not None:
            cost = context.cost_per_inference
            flag = "OK" if cost <= self.config.max_cost_per_inference else "PERFORMANCE_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"cost/inference={cost:.5f} (max {self.config.max_cost_per_inference})",
                    blocking=self.blocking,
                    metadata={
                        "metric_kind": "cost",
                        "metric": "cost_per_inference",
                        "value": round(cost, 5),
                        "threshold": self.config.max_cost_per_inference,
                    },
                )
            )

        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no performance benchmark data supplied",
                self.blocking,
            )
        ]
