"""Execution context objects — the bundle of data/model a gate run needs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass
class StructuredGateContext:
    """Everything a structured-data governance check needs to run.

    Only `model`, `X`, `y_true`, and `y_pred` are required. Every other
    field is optional — omitting one simply causes the checks that depend
    on it to report NOT_APPLICABLE rather than raise or fail the gate.
    All inputs are validated eagerly by `ModelGate.run()` before any check
    executes; see `bdp_model_gate.core.validation`.

    Attributes:
        model: A fitted model exposing `.predict()` (and ideally `.predict_proba()`).
        X: Feature dataframe used for validation/inference.
        y_true: Ground-truth labels for the validation set.
        y_pred: Model predictions on X (probabilities or hard labels, per your metric).
        protected_df: Dataframe of protected attributes (gender, region, etc.),
            row-aligned to X. Needed for the fairness checks.
        latencies_ms: Per-request inference latencies from a benchmark run, for
            the performance gate.
        cost_per_inference: Estimated or measured cost per inference, for the
            performance gate.
        model_card: Dict describing the model (legal_basis, use_case, etc.),
            for the compliance gate.
        generate_fn: callable(str) -> str, the entry point of any generative
            component sitting alongside the structured model, for prompt
            injection testing.
    """

    model: Any
    X: pd.DataFrame
    y_true: Sequence[Any] | None
    y_pred: Sequence[Any] | None
    protected_df: pd.DataFrame | None = None
    latencies_ms: Sequence[float] | None = None
    cost_per_inference: float | None = None
    model_card: dict | None = None
    generate_fn: Callable[[str], str] | None = None
    modality: str = "structured"
