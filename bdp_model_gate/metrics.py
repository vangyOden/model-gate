"""Metric resolution for the performance gate.

The performance gate scores a model with whichever metric the caller
configured (`PerformanceConfig.metric`). This module owns the mapping from
that config value to a callable, and — importantly — makes the choice
*explicit* in the report rather than silently depending on which optional
dependencies happen to be installed.

Three kinds of value are accepted:

    "auto"        try the metrics in AUTO_PREFERENCE in order, using the
                  first one whose dependencies are available. A fallback is
                  logged at WARNING level and named in the check's output,
                  so it never happens invisibly.
    "<name>"      any key of BUILTIN_METRICS. If its dependencies are
                  missing, that's a GateConfigurationError — an explicit
                  request is never silently substituted.
    callable      any `fn(y_true, y_pred) -> float`. Called with y_pred
                  exactly as supplied (no thresholding), since only the
                  caller knows what their metric expects.

Metrics differ in what they want from `y_pred`: ranking metrics like
`roc_auc` need continuous scores, while `accuracy`/`f1`/`precision`/
`recall` need hard class labels. `needs_hard_labels` records which, and
the check binarizes at `PerformanceConfig.decision_threshold` when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union  # Union: runtime alias, can't use PEP 604 on 3.9

import numpy as np

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("metrics")

MetricFn = Callable[[Any, Any], float]
MetricSetting = Union[str, MetricFn]


@dataclass(frozen=True)
class MetricSpec:
    """How to obtain one named metric, and what it expects from y_pred."""

    name: str
    sklearn_fn: str
    needs_hard_labels: bool
    #: Pure-numpy equivalent, used only when scikit-learn isn't installed.
    #: None means this metric genuinely requires scikit-learn.
    fallback: MetricFn | None = None


def _accuracy_numpy(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


BUILTIN_METRICS: dict[str, MetricSpec] = {
    "roc_auc": MetricSpec("roc_auc", "roc_auc_score", needs_hard_labels=False),
    "average_precision": MetricSpec(
        "average_precision", "average_precision_score", needs_hard_labels=False
    ),
    "accuracy": MetricSpec(
        "accuracy", "accuracy_score", needs_hard_labels=True, fallback=_accuracy_numpy
    ),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy", "balanced_accuracy_score", needs_hard_labels=True
    ),
    "f1": MetricSpec("f1", "f1_score", needs_hard_labels=True),
    "precision": MetricSpec("precision", "precision_score", needs_hard_labels=True),
    "recall": MetricSpec("recall", "recall_score", needs_hard_labels=True),
}

#: Order tried by metric="auto". roc_auc is threshold-independent and the
#: better default for the imbalanced problems this gate typically sees;
#: accuracy is the base-install fallback since it needs no extra deps.
AUTO_PREFERENCE = ("roc_auc", "accuracy")

AUTO = "auto"


@dataclass(frozen=True)
class ResolvedMetric:
    """A metric ready to call, plus how it was arrived at."""

    name: str
    fn: MetricFn
    needs_hard_labels: bool
    #: True when metric="auto" could not use its first preference. The check
    #: surfaces this in its detail string so a reader of the report knows
    #: the score isn't the metric they'd expect by default.
    is_fallback: bool = False
    #: Set when the metric ran without scikit-learn.
    used_fallback_impl: bool = False


def validate_metric(metric: MetricSetting) -> None:
    """Cheap, import-free check that `metric` is a usable setting.

    Called when the check is constructed so a typo'd metric name fails at
    configuration time rather than midway through a gate run. Whether the
    metric's dependencies are actually installed is deliberately *not*
    checked here — see `resolve_metric`.
    """
    if callable(metric):
        return
    if not isinstance(metric, str):
        raise GateConfigurationError(
            f"performance.metric must be a metric name or a callable, got {type(metric).__name__}"
        )
    if metric == AUTO or metric in BUILTIN_METRICS:
        return
    valid = ", ".join([AUTO, *sorted(BUILTIN_METRICS)])
    raise GateConfigurationError(f"unknown performance.metric {metric!r} — valid options: {valid}")


def _load_sklearn_metric(spec: MetricSpec) -> MetricFn | None:
    try:
        from sklearn import metrics as sk_metrics
    except ImportError:
        return None
    return getattr(sk_metrics, spec.sklearn_fn, None)


def resolve_metric(metric: MetricSetting) -> ResolvedMetric:
    """Turns a config value into a callable metric.

    Raises GateConfigurationError if an explicitly named metric can't be
    satisfied — the gate reports that as a blocking CHECK_ERROR rather than
    scoring the model with something the caller didn't ask for.
    """
    validate_metric(metric)

    if callable(metric):
        name = getattr(metric, "__name__", None) or type(metric).__name__
        return ResolvedMetric(name=name, fn=metric, needs_hard_labels=False)

    if metric == AUTO:
        return _resolve_auto()

    spec = BUILTIN_METRICS[metric]
    fn = _load_sklearn_metric(spec)
    if fn is not None:
        return ResolvedMetric(spec.name, fn, spec.needs_hard_labels)
    if spec.fallback is not None:
        logger.debug(
            "scikit-learn not installed — scoring %r with the built-in numpy implementation",
            spec.name,
        )
        return ResolvedMetric(
            spec.name, spec.fallback, spec.needs_hard_labels, used_fallback_impl=True
        )
    raise GateConfigurationError(
        f"performance.metric={metric!r} requires scikit-learn — install it with "
        "`pip install bdp-model-gate[structured]`, or set performance.metric to "
        f"one of: {', '.join(sorted(m for m, s in BUILTIN_METRICS.items() if s.fallback))}"
    )


def _resolve_auto() -> ResolvedMetric:
    preferred = AUTO_PREFERENCE[0]
    for position, name in enumerate(AUTO_PREFERENCE):
        spec = BUILTIN_METRICS[name]
        fn = _load_sklearn_metric(spec)
        if fn is not None:
            return ResolvedMetric(spec.name, fn, spec.needs_hard_labels, is_fallback=position > 0)
        if spec.fallback is not None:
            logger.warning(
                "performance.metric='auto': %r is unavailable (scikit-learn not installed) — "
                "scoring with %r instead. Set performance.metric explicitly to silence this, "
                "and remember min_score is interpreted against %r, not %r.",
                preferred,
                spec.name,
                spec.name,
                preferred,
            )
            return ResolvedMetric(
                spec.name,
                spec.fallback,
                spec.needs_hard_labels,
                is_fallback=position > 0,
                used_fallback_impl=True,
            )
    raise GateConfigurationError(  # pragma: no cover — accuracy always has a numpy fallback
        "no metric in AUTO_PREFERENCE could be resolved"
    )


def to_hard_labels(y_pred: Any, threshold: float) -> Any:
    """Binarizes continuous scores for a metric that needs class labels.

    Values already restricted to {0, 1} are passed through untouched, so
    callers who supply hard labels aren't affected by `decision_threshold`.
    """
    arr = np.asarray(y_pred)
    if arr.dtype.kind not in "fc":
        return arr
    if np.all(np.isin(arr, (0, 1))):
        return arr.astype(int)
    logger.debug("binarizing continuous y_pred at decision_threshold=%s", threshold)
    return (arr >= threshold).astype(int)


__all__ = [
    "AUTO",
    "AUTO_PREFERENCE",
    "BUILTIN_METRICS",
    "MetricFn",
    "MetricSetting",
    "MetricSpec",
    "ResolvedMetric",
    "resolve_metric",
    "to_hard_labels",
    "validate_metric",
]
