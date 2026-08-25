"""Shared fixtures for the BDP Model Gate test suite."""

import numpy as np
import pandas as pd
import pytest


class DummyModel:
    """A tiny deterministic 'model' so most tests don't depend on scikit-learn."""

    def predict(self, X: pd.DataFrame):
        return (X["income"] > X["income"].median()).astype(int).values

    def predict_proba(self, X: pd.DataFrame):
        p1 = (X["income"] > X["income"].median()).astype(float).values
        return np.column_stack([1 - p1, p1])


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "income": rng.normal(50000, 15000, n),
            "age": rng.integers(18, 70, n),
            "credit_score": rng.normal(650, 50, n),
        }
    )
    protected_df = pd.DataFrame(
        {
            "gender": rng.choice(["M", "F"], n),
            "region": rng.choice(["Lagos", "Abuja", "Kano"], n),
        }
    )
    model = DummyModel()
    y_pred = model.predict_proba(X)[:, 1]
    y_true = (X["income"] > X["income"].quantile(0.4)).astype(int).values
    return model, X, y_true, y_pred, protected_df


@pytest.fixture
def small_valid_context(synthetic_data):
    from bdp_model_gate import StructuredGateContext

    model, X, y_true, y_pred, protected_df = synthetic_data
    return StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
