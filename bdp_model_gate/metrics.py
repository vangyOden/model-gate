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
from .task import BINARY, CLASSIFICATION_TASKS, REGRESSION

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
    #: False for error metrics (RMSE, MAE, MAPE, deviance), where a *lower*
    #: value is better. These are gated with `max_error`, not `min_score`.
    greater_is_better: bool = True
    #: Which prediction tasks this metric can score.
    tasks: tuple[str, ...] = CLASSIFICATION_TASKS


def _accuracy_numpy(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _as_floats(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


# The regression metrics are short enough to implement directly, which keeps
# them available on a core install and sidesteps scikit-learn's churn around
# `mean_squared_error(squared=False)` / `root_mean_squared_error`.


def _rmse_numpy(y_true: Any, y_pred: Any) -> float:
    t, p = _as_floats(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def _mae_numpy(y_true: Any, y_pred: Any) -> float:
    t, p = _as_floats(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def _r2_numpy(y_true: Any, y_pred: Any) -> float:
    t, p = _as_floats(y_true, y_pred)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot == 0.0:
        # A constant target has no variance to explain. Perfect prediction is
        # 1.0; anything else is undefined rather than arbitrarily bad.
        return 1.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def _mape_numpy(y_true: Any, y_pred: Any) -> float:
    """Mean absolute percentage error, skipping zero actuals.

    MAPE is the natural metric for skewed money targets like claims severity,
    but it is undefined where the actual is 0. Those rows are excluded and the
    exclusion is logged, rather than returning inf for the whole batch.
    """
    t, p = _as_floats(y_true, y_pred)
    nonzero = t != 0
    n_skipped = int((~nonzero).sum())
    if n_skipped:
        logger.warning(
            "mape: skipped %d row(s) with a zero actual — MAPE is undefined there. "
            "Consider 'mae' or 'poisson_deviance' for targets with true zeros.",
            n_skipped,
        )
    if not nonzero.any():
        raise GateConfigurationError(
            "every y_true value is zero, so MAPE is undefined for this dataset — "
            "use 'mae', 'rmse' or 'poisson_deviance' instead"
        )
    return float(np.mean(np.abs((t[nonzero] - p[nonzero]) / t[nonzero])))


def _poisson_deviance_numpy(y_true: Any, y_pred: Any) -> float:
    """Mean Poisson deviance — the right error measure for count targets such
    as claims frequency, where RMSE understates the cost of over-dispersion."""
    t, p = _as_floats(y_true, y_pred)
    if np.any(p <= 0):
        raise GateConfigurationError(
            "poisson_deviance requires strictly positive predictions "
            "(it takes their log); got a prediction <= 0"
        )
    if np.any(t < 0):
        raise GateConfigurationError("poisson_deviance requires non-negative y_true")
    # x*log(x/mu) -> 0 as x -> 0, so the zero-actual rows contribute only the
    # (mu - x) term. np.where alone would still evaluate log(0), hence the mask.
    safe_t = np.where(t > 0, t, 1.0)
    term = np.where(t > 0, t * np.log(safe_t / p), 0.0)
    return float(np.mean(2.0 * (term - (t - p))))


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
    # Regression. All have numpy implementations, so they work on a core
    # install; scikit-learn is used when present for the ones it defines.
    "rmse": MetricSpec(
        "rmse",
        "",
        needs_hard_labels=False,
        fallback=_rmse_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
    ),
    "mae": MetricSpec(
        "mae",
        "mean_absolute_error",
        needs_hard_labels=False,
        fallback=_mae_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
    ),
    "mape": MetricSpec(
        "mape",
        "",
        needs_hard_labels=False,
        fallback=_mape_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
    ),
    "poisson_deviance": MetricSpec(
        "poisson_deviance",
        "",
        needs_hard_labels=False,
        fallback=_poisson_deviance_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
    ),
    "r2": MetricSpec(
        "r2",
        "r2_score",
        needs_hard_labels=False,
        fallback=_r2_numpy,
        greater_is_better=True,
        tasks=(REGRESSION,),
    ),
}

#: Order tried by metric="auto", per task. For classification, roc_auc is
#: threshold-independent and the better default for the imbalanced problems
#: this gate typically sees, with accuracy as the base-install fallback. For
#: regression, r2 is scale-free — an RMSE default would mean nothing without
#: knowing whether the target is premiums in naira or claim counts.
AUTO_PREFERENCE_BY_TASK: dict[str, tuple[str, ...]] = {
    BINARY: ("roc_auc", "accuracy"),
    REGRESSION: ("r2",),
}

#: Backwards-compatible alias for the binary preference order.
AUTO_PREFERENCE = AUTO_PREFERENCE_BY_TASK[BINARY]

AUTO = "auto"


@dataclass(frozen=True)
class ResolvedMetric:
    """A metric ready to call, plus how it was arrived at."""

    name: str
    fn: MetricFn
    needs_hard_labels: bool
    #: False for error metrics — gated with `max_error` rather than `min_score`.
    greater_is_better: bool = True
    #: True when metric="auto" could not use its first preference. The check
    #: surfaces this in its detail string so a reader of the report knows
    #: the score isn't the metric they'd expect by default.
    is_fallback: bool = False
    #: Set when the metric ran without scikit-learn.
    used_fallback_impl: bool = False


def validate_metric(metric: MetricSetting, task: str | None = None) -> None:
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
    if metric == AUTO:
        return
    if metric not in BUILTIN_METRICS:
        valid = ", ".join([AUTO, *sorted(BUILTIN_METRICS)])
        raise GateConfigurationError(
            f"unknown performance.metric {metric!r} — valid options: {valid}"
        )
    if task is not None and task not in BUILTIN_METRICS[metric].tasks:
        applicable = ", ".join(sorted(m for m, sp in BUILTIN_METRICS.items() if task in sp.tasks))
        raise GateConfigurationError(
            f"performance.metric={metric!r} does not apply to a {task} task — "
            f"metrics available for {task}: {applicable}"
        )


def _load_sklearn_metric(spec: MetricSpec) -> MetricFn | None:
    try:
        from sklearn import metrics as sk_metrics
    except ImportError:
        return None
    return getattr(sk_metrics, spec.sklearn_fn, None)


def resolve_metric(metric: MetricSetting, task: str = BINARY) -> ResolvedMetric:
    """Turns a config value into a callable metric.

    Raises GateConfigurationError if an explicitly named metric can't be
    satisfied — the gate reports that as a blocking CHECK_ERROR rather than
    scoring the model with something the caller didn't ask for.
    """
    validate_metric(metric, task)

    if callable(metric):
        name = getattr(metric, "__name__", None) or type(metric).__name__
        # A custom callable's direction is unknowable, so it is treated as
        # greater-is-better and gated with min_score. Negate inside your own
        # function, or name a built-in error metric, if that is wrong.
        return ResolvedMetric(name=name, fn=metric, needs_hard_labels=False)

    if metric == AUTO:
        return _resolve_auto(task)

    spec = BUILTIN_METRICS[metric]
    fn = _load_sklearn_metric(spec)
    if fn is not None:
        return ResolvedMetric(spec.name, fn, spec.needs_hard_labels, spec.greater_is_better)
    if spec.fallback is not None:
        logger.debug(
            "scikit-learn not installed — scoring %r with the built-in numpy implementation",
            spec.name,
        )
        return ResolvedMetric(
            spec.name,
            spec.fallback,
            spec.needs_hard_labels,
            spec.greater_is_better,
            used_fallback_impl=True,
        )
    raise GateConfigurationError(
        f"performance.metric={metric!r} requires scikit-learn — install it with "
        "`pip install bdp-model-gate[structured]`, or set performance.metric to "
        f"one of: {', '.join(sorted(m for m, s in BUILTIN_METRICS.items() if s.fallback))}"
    )


def _resolve_auto(task: str = BINARY) -> ResolvedMetric:
    preference = AUTO_PREFERENCE_BY_TASK.get(task)
    if not preference:
        raise GateConfigurationError(
            f'performance.metric="auto" has no default for a {task} task — name a metric explicitly'
        )
    preferred = preference[0]
    for position, name in enumerate(preference):
        spec = BUILTIN_METRICS[name]
        fn = _load_sklearn_metric(spec)
        if fn is not None:
            return ResolvedMetric(
                spec.name,
                fn,
                spec.needs_hard_labels,
                spec.greater_is_better,
                is_fallback=position > 0,
            )
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
                spec.greater_is_better,
                is_fallback=position > 0,
                used_fallback_impl=True,
            )
    raise GateConfigurationError(  # pragma: no cover — accuracy always has a numpy fallback
        f"no metric in the {task} preference order could be resolved"
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
    "AUTO_PREFERENCE_BY_TASK",
    "BUILTIN_METRICS",
    "MetricFn",
    "MetricSetting",
    "MetricSpec",
    "ResolvedMetric",
    "resolve_metric",
    "to_hard_labels",
    "validate_metric",
]
