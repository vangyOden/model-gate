"""Tests for eager input validation — every case here should fail fast with
a clear GateValidationError, before any check executes."""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.exceptions import GateValidationError


def test_empty_dataframe_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    empty_X = X.iloc[0:0]
    context = StructuredGateContext(model=model, X=empty_X, y_true=[], y_pred=[])
    with pytest.raises(GateValidationError, match="empty"):
        ModelGate().run(context)


def test_non_dataframe_X_rejected(synthetic_data):
    model, _, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(
        model=model, X=[[1, 2], [3, 4]], y_true=y_true[:2], y_pred=y_pred[:2]
    )
    with pytest.raises(GateValidationError, match="DataFrame"):
        ModelGate().run(context)


def test_mismatched_label_length_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(model=model, X=X, y_true=y_true[:10], y_pred=y_pred)
    with pytest.raises(GateValidationError, match="length"):
        ModelGate().run(context)


def test_single_class_y_true_rejected(synthetic_data):
    model, X, _, y_pred, _ = synthetic_data
    all_ones = np.ones(len(X), dtype=int)
    context = StructuredGateContext(model=model, X=X, y_true=all_ones, y_pred=y_pred)
    with pytest.raises(GateValidationError, match="one unique value"):
        ModelGate().run(context)


def test_model_without_predict_rejected(synthetic_data):
    _, X, y_true, y_pred, _ = synthetic_data

    class NotAModel:
        pass

    context = StructuredGateContext(model=NotAModel(), X=X, y_true=y_true, y_pred=y_pred)
    with pytest.raises(GateValidationError, match="predict"):
        ModelGate().run(context)


def test_all_nan_protected_column_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    protected_df = pd.DataFrame({"gender": [np.nan] * len(X)})
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    with pytest.raises(GateValidationError, match="entirely NaN"):
        ModelGate().run(context)


def test_mismatched_protected_df_length_rejected(synthetic_data):
    model, X, y_true, y_pred, protected_df = synthetic_data
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df.iloc[:10],
    )
    with pytest.raises(GateValidationError, match="row-aligned"):
        ModelGate().run(context)


def test_non_dict_model_card_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(
        model=model, X=X, y_true=y_true, y_pred=y_pred, model_card="not a dict"
    )
    with pytest.raises(GateValidationError, match="dict"):
        ModelGate().run(context)


def test_non_callable_generate_fn_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        generate_fn="not callable",
    )
    with pytest.raises(GateValidationError, match="callable"):
        ModelGate().run(context)


def test_negative_latencies_rejected(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        latencies_ms=[10, -5, 20],
    )
    with pytest.raises(GateValidationError, match="negative"):
        ModelGate().run(context)
