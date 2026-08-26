"""Eager input validation for StructuredGateContext.

Runs once, before any check executes. A check-writer can assume the
context it receives is well-formed: X is a non-empty DataFrame, y_true/
y_pred/X are aligned in length, the model exposes .predict(), and any
optional inputs that are present are internally consistent. If something's
wrong, GateValidationError is raised with a message that names the field
and the problem, rather than letting a check fail with a confusing
downstream exception (e.g. a shape mismatch surfacing inside SHAP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..exceptions import GateValidationError
from ..task import REGRESSION, resolve_task, validate_task

if TYPE_CHECKING:
    from .context import StructuredGateContext


def validate_structured_context(context: StructuredGateContext) -> None:
    _validate_task(context)
    _validate_model(context)
    _validate_features(context)
    _validate_labels(context)
    _validate_expected_loss(context)
    _validate_protected_df(context)
    _validate_model_card(context)
    _validate_performance_inputs(context)
    _validate_generate_fn(context)


def _validate_model(context: StructuredGateContext) -> None:
    if context.model is None:
        raise GateValidationError("context.model is required and cannot be None")
    if not hasattr(context.model, "predict"):
        raise GateValidationError(
            "context.model must expose a .predict() method; "
            f"got {type(context.model).__name__} which does not"
        )


def _validate_features(context: StructuredGateContext) -> None:
    if not isinstance(context.X, pd.DataFrame):
        raise GateValidationError(
            f"context.X must be a pandas DataFrame, got {type(context.X).__name__}"
        )
    if context.X.empty:
        raise GateValidationError("context.X is empty — no rows to evaluate")
    if context.X.shape[1] == 0:
        raise GateValidationError("context.X has no columns")


def _validate_labels(context: StructuredGateContext) -> None:
    n_rows = len(context.X)
    for label_name, label_value in (("y_true", context.y_true), ("y_pred", context.y_pred)):
        if label_value is None:
            continue
        length = len(label_value)
        if length != n_rows:
            raise GateValidationError(
                f"context.{label_name} has length {length}, but context.X has "
                f"{n_rows} rows — they must be aligned"
            )

    if context.y_true is None:
        return

    task = resolve_task(context)
    unique_labels = pd.unique(np.asarray(context.y_true))

    if len(unique_labels) < 2:
        detail = (
            "a constant target has no variance to explain, so every regression metric is degenerate"
            if task == REGRESSION
            else "most checks (AUC, disparate impact) need at least two classes present"
        )
        raise GateValidationError(
            f"context.y_true has only one unique value ({unique_labels!r}) — {detail}"
        )

    if task == REGRESSION:
        for name, values in (("y_true", context.y_true), ("y_pred", context.y_pred)):
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.dtype.kind not in "iuf":
                raise GateValidationError(
                    f"context.{name} must be numeric for a regression task, got dtype "
                    f"{arr.dtype} — set context.task explicitly if this is really "
                    "a classification problem"
                )
            if not np.all(np.isfinite(arr.astype(float))):
                raise GateValidationError(
                    f"context.{name} contains NaN or infinite values, which every "
                    "regression metric would propagate"
                )


def _validate_task(context: StructuredGateContext) -> None:
    validate_task(getattr(context, "task", "auto"))


def _validate_expected_loss(context: StructuredGateContext) -> None:
    expected_loss = getattr(context, "expected_loss", None)
    if expected_loss is None:
        return
    arr = np.asarray(expected_loss)
    if arr.dtype.kind not in "iuf":
        raise GateValidationError(f"context.expected_loss must be numeric, got dtype {arr.dtype}")
    if len(arr) != len(context.X):
        raise GateValidationError(
            f"context.expected_loss has length {len(arr)}, but context.X has "
            f"{len(context.X)} rows — they must be row-aligned"
        )
    if np.any(np.asarray(arr, dtype=float) < 0):
        raise GateValidationError(
            "context.expected_loss contains negative values — an expected loss cannot be below zero"
        )


def _validate_protected_df(context: StructuredGateContext) -> None:
    if context.protected_df is None:
        return
    if not isinstance(context.protected_df, pd.DataFrame):
        raise GateValidationError(
            f"context.protected_df must be a pandas DataFrame, got "
            f"{type(context.protected_df).__name__}"
        )
    if len(context.protected_df) != len(context.X):
        raise GateValidationError(
            f"context.protected_df has {len(context.protected_df)} rows, but "
            f"context.X has {len(context.X)} rows — they must be row-aligned"
        )
    all_nan_cols = [c for c in context.protected_df.columns if context.protected_df[c].isna().all()]
    if all_nan_cols:
        raise GateValidationError(
            f"context.protected_df column(s) {all_nan_cols} are entirely NaN — "
            "fairness checks cannot group on an empty attribute"
        )


def _validate_model_card(context: StructuredGateContext) -> None:
    if context.model_card is None:
        return
    if not isinstance(context.model_card, dict):
        raise GateValidationError(
            f"context.model_card must be a dict, got {type(context.model_card).__name__}"
        )


def _validate_performance_inputs(context: StructuredGateContext) -> None:
    if context.latencies_ms is None:
        return
    if len(context.latencies_ms) == 0:
        raise GateValidationError("context.latencies_ms is an empty sequence")
    if any(v < 0 for v in context.latencies_ms):
        raise GateValidationError("context.latencies_ms contains negative values")


def _validate_generate_fn(context: StructuredGateContext) -> None:
    if context.generate_fn is None:
        return
    if not callable(context.generate_fn):
        raise GateValidationError("context.generate_fn must be callable")
