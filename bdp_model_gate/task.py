"""Prediction-task identification.

Nearly half the check suite is task-agnostic — PII scanning, prompt
injection, model-card compliance and proxy correlation care about features
and documentation, not about what the model predicts. The rest is not:
demographic parity needs a favourable class, a "prediction flip" means
nothing for a continuous output, and ROC AUC cannot score a premium.

Every check therefore declares the tasks it supports via
`BaseCheck.supported_tasks`, and reports NOT_APPLICABLE for the others
rather than producing a confident, meaningless number.

`StructuredGateContext.task` names the task. It defaults to "auto", which
infers from `y_true` and **logs what it inferred** — inference is genuinely
ambiguous (a claims-frequency target of 0/1/2/3 is indistinguishable from a
four-class problem by shape alone), so the guess is never silent. State
`task` explicitly for anything you intend to gate on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("task")

AUTO = "auto"
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"

#: Every concrete task. Checks that work regardless of task use this as
#: their `supported_tasks`, which is also the default on BaseCheck so that
#: third-party plugins written before 0.3.0 keep running everywhere.
ALL_TASKS = (BINARY, MULTICLASS, REGRESSION)
CLASSIFICATION_TASKS = (BINARY, MULTICLASS)

VALID_SETTINGS = (AUTO, *ALL_TASKS)

#: An integer-valued target with at most this many distinct values is read as
#: multiclass; more than this is read as regression. Count targets (claims
#: frequency) sit right on this boundary, which is why the inference is
#: logged and `task` can be set explicitly.
MAX_INFERRED_CLASSES = 20


def validate_task(task: Any) -> None:
    """Rejects an unusable `task` setting before any check runs."""
    if not isinstance(task, str) or task not in VALID_SETTINGS:
        raise GateConfigurationError(
            f"context.task must be one of {', '.join(VALID_SETTINGS)} — got {task!r}"
        )


def infer_task(y_true: Any) -> str:
    """Best-effort task inference from the ground-truth labels.

    Deliberately conservative and explainable rather than clever: strings and
    booleans are classification, two distinct values are binary, integral
    targets with few distinct values are multiclass, everything else is
    regression.
    """
    arr = np.asarray(y_true)

    if arr.dtype.kind in "OSUb":  # object, string, unicode, bool
        n_unique = len(np.unique(arr))
        return BINARY if n_unique <= 2 else MULTICLASS

    finite = arr[np.isfinite(arr)] if arr.dtype.kind in "fc" else arr
    uniques = np.unique(finite)
    n_unique = len(uniques)

    if n_unique <= 2:
        return BINARY

    is_integral = bool(np.all(np.equal(np.mod(uniques, 1), 0)))
    if is_integral and n_unique <= MAX_INFERRED_CLASSES:
        return MULTICLASS
    return REGRESSION


def resolve_task(context: Any) -> str:
    """Returns the concrete task for a context, inferring it if set to "auto".

    Raises GateConfigurationError for an unknown setting. When "auto" is
    asked to infer from a context with no `y_true`, it warns and falls back
    to "binary" — the assumption every release before 0.3.0 made implicitly.
    """
    task = getattr(context, "task", AUTO)
    validate_task(task)

    if task != AUTO:
        return task

    y_true = getattr(context, "y_true", None)
    if y_true is None:
        # Nothing to infer from. Every task-specific check needs y_true too,
        # so only the task-agnostic ones (PII, prompt injection, model card,
        # proxy correlation) can run and the value barely matters — but say
        # so rather than let a silent assumption sit in the report.
        logger.warning(
            'context.task="auto" cannot infer without y_true — assuming %r, which is '
            "what releases before 0.3.0 always assumed. Only task-agnostic checks can "
            "run without labels. Set task explicitly to silence this.",
            BINARY,
        )
        return BINARY

    inferred = infer_task(y_true)
    logger.info(
        'context.task="auto" inferred task=%r from y_true. Set task explicitly if that '
        "is wrong — a count target (e.g. claims frequency) is indistinguishable from a "
        "multiclass one by shape alone.",
        inferred,
    )
    return inferred


def supports(check: Any, task: str) -> bool:
    """Whether `check` declares support for `task`. Checks predating
    `supported_tasks` are treated as task-agnostic."""
    return task in getattr(check, "supported_tasks", ALL_TASKS)


__all__ = [
    "ALL_TASKS",
    "AUTO",
    "BINARY",
    "CLASSIFICATION_TASKS",
    "MAX_INFERRED_CLASSES",
    "MULTICLASS",
    "REGRESSION",
    "VALID_SETTINGS",
    "infer_task",
    "resolve_task",
    "supports",
    "validate_task",
]
