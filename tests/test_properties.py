"""Property-based tests for the numeric core.

Example tests cover the inputs someone imagined. `hypothesis` generates the
ones nobody did — the empty group, the constant column, the value that is
exactly the threshold, the array of identical floats. That is where the
degenerate-case bugs live.

Scope is deliberately the pure numeric functions and the adapter's shape
handling. Fitting real models under hypothesis would be slow and would test
scikit-learn rather than this library.
"""

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import array_shapes, arrays  # noqa: E402

from bdp_model_gate.classes import resolve_favourable, to_ranks
from bdp_model_gate.exceptions import GateConfigurationError
from bdp_model_gate.metrics import (
    ordinal_mae,
    quadratic_kappa,
    resolve_metric,
    to_class_labels,
    to_hard_labels,
)
from bdp_model_gate.model import ModelAdapter

# Money and rates, not adversarial float edge cases — the library documents
# that it needs finite numeric input and validates for it.
finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)


def _pairs(strategy=finite, min_size=2, max_size=60):
    """Two equal-length arrays, as every metric takes."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=strategy),
            arrays(np.float64, n, elements=strategy),
        )
    )


# --- regression metrics ------------------------------------------------------


@given(_pairs())
@settings(max_examples=150, deadline=None)
def test_error_metrics_are_non_negative_and_zero_only_on_a_perfect_fit(pair):
    y_true, y_pred = pair
    for name in ("rmse", "mae"):
        fn = resolve_metric(name, "regression").fn
        value = fn(y_true, y_pred)
        assert value >= 0.0
        assert np.isfinite(value)
        assert fn(y_true, y_true) == pytest.approx(0.0, abs=1e-9)


@given(_pairs())
@settings(max_examples=150, deadline=None)
def test_error_metrics_are_symmetric_in_their_arguments(pair):
    """|a - b| does not care which way round it is given."""
    y_true, y_pred = pair
    for name in ("rmse", "mae"):
        fn = resolve_metric(name, "regression").fn
        assert fn(y_true, y_pred) == pytest.approx(fn(y_pred, y_true), rel=1e-9)


@given(_pairs())
@settings(max_examples=100, deadline=None)
def test_rmse_is_never_below_mae(pair):
    """A consequence of Jensen's inequality that must hold for every input."""
    y_true, y_pred = pair
    rmse = resolve_metric("rmse", "regression").fn(y_true, y_pred)
    mae = resolve_metric("mae", "regression").fn(y_true, y_pred)
    assert rmse >= mae - 1e-9


@given(_pairs())
@settings(max_examples=100, deadline=None)
def test_r2_is_at_most_one_and_exactly_one_on_a_perfect_fit(pair):
    y_true, _ = pair
    r2 = resolve_metric("r2", "regression").fn
    assume(np.std(y_true) > 1e-6)  # r2 is undefined on a constant target
    assert r2(y_true, y_true) == pytest.approx(1.0)
    assert r2(y_true, np.asarray(y_true) + 1.0) <= 1.0 + 1e-9


@given(_pairs(strategy=positive))
@settings(max_examples=100, deadline=None)
def test_poisson_deviance_is_non_negative_and_zero_on_a_perfect_fit(pair):
    y_true, y_pred = pair
    fn = resolve_metric("poisson_deviance", "regression").fn
    assert fn(y_true, y_true) == pytest.approx(0.0, abs=1e-6)
    assert fn(y_true, y_pred) >= -1e-9


# --- thresholding ------------------------------------------------------------


@given(
    arrays(
        np.float64,
        array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=50),
        elements=st.floats(0, 1, allow_nan=False),
    ),
    st.floats(0, 1, allow_nan=False),
)
@settings(max_examples=150, deadline=None)
def test_to_hard_labels_only_ever_emits_zero_or_one(values, threshold):
    out = np.asarray(to_hard_labels(values, threshold))
    assert set(np.unique(out)).issubset({0, 1})
    assert len(out) == len(values)


@given(arrays(np.int64, 20, elements=st.integers(0, 1)), st.floats(0, 1, allow_nan=False))
@settings(max_examples=60, deadline=None)
def test_already_binary_predictions_pass_through_any_threshold(values, threshold):
    """decision_threshold must not affect a caller who supplies hard labels."""
    np.testing.assert_array_equal(np.asarray(to_hard_labels(values, threshold)), values)


@given(st.integers(2, 6), st.integers(1, 30))
@settings(max_examples=60, deadline=None)
def test_to_class_labels_argmax_picks_a_label_from_the_ordering(n_classes, n_rows):
    order = [f"c{i}" for i in range(n_classes)]
    rng = np.random.default_rng(n_classes * 100 + n_rows)
    matrix = rng.random((n_rows, n_classes))
    out = to_class_labels(matrix, order)
    assert len(out) == n_rows
    assert set(out).issubset(set(order))
    # And it really is the argmax.
    np.testing.assert_array_equal(out, [order[i] for i in matrix.argmax(axis=1)])


# --- ordinal metrics ---------------------------------------------------------


@given(st.integers(2, 6), st.integers(2, 40))
@settings(max_examples=80, deadline=None)
def test_ordinal_metrics_bounds_and_perfect_agreement(n_classes, n_rows):
    order = [f"c{i}" for i in range(n_classes)]
    rng = np.random.default_rng(n_classes * 7 + n_rows)
    truth = rng.choice(order, n_rows)
    pred = rng.choice(order, n_rows)

    mae = ordinal_mae(truth, pred, order)
    assert 0.0 <= mae <= n_classes - 1
    assert ordinal_mae(truth, truth, order) == pytest.approx(0.0)

    kappa = quadratic_kappa(truth, pred, order)
    assert kappa <= 1.0 + 1e-9
    assert quadratic_kappa(truth, truth, order) == pytest.approx(1.0)


@given(st.integers(2, 6), st.integers(2, 40))
@settings(max_examples=60, deadline=None)
def test_ranks_are_within_the_ordering(n_classes, n_rows):
    order = [f"c{i}" for i in range(n_classes)]
    rng = np.random.default_rng(n_classes + n_rows)
    values = rng.choice(order, n_rows)
    ranks = to_ranks(values, order)
    assert ranks.min() >= 0
    assert ranks.max() <= n_classes - 1


@given(st.lists(st.text(min_size=1, max_size=4), min_size=2, max_size=6, unique=True))
@settings(max_examples=60, deadline=None)
def test_favourable_defaults_to_the_last_class(order):
    assert resolve_favourable(None, order, "multiclass") == [order[-1]]


# --- adapter shape handling --------------------------------------------------


@given(st.integers(1, 40))
@settings(max_examples=60, deadline=None)
def test_every_binary_probability_shape_reduces_to_one_column(n_rows):
    rng = np.random.default_rng(n_rows)
    p = rng.random(n_rows)
    frame = pd.DataFrame({"x": np.arange(n_rows, dtype=float)})

    for build in (
        lambda df: p,  # bare (n,)
        lambda df: p.reshape(-1, 1),  # Keras (n, 1)
        lambda df: np.column_stack([1 - p, p]),  # sklearn (n, 2)
    ):
        out = ModelAdapter(predict_proba_fn=build).predict_positive_proba(frame)
        assert out.shape == (n_rows,)
        np.testing.assert_allclose(out, p, rtol=1e-9)


@given(st.integers(3, 8), st.integers(1, 20))
@settings(max_examples=40, deadline=None)
def test_multiclass_probabilities_are_always_refused(n_classes, n_rows):
    """Never silently sliced — column 1 of a k-class output means nothing."""
    frame = pd.DataFrame({"x": np.arange(n_rows, dtype=float)})
    adapter = ModelAdapter(
        predict_proba_fn=lambda df: np.full((len(df), n_classes), 1.0 / n_classes)
    )
    with pytest.raises(GateConfigurationError):
        adapter.predict_positive_proba(frame)


@given(st.integers(1, 40))
@settings(max_examples=40, deadline=None)
def test_predict_flattens_a_trailing_singleton_axis(n_rows):
    frame = pd.DataFrame({"x": np.arange(n_rows, dtype=float)})
    adapter = ModelAdapter(predict_fn=lambda df: np.zeros((len(df), 1)))
    assert adapter.predict(frame).shape == (n_rows,)
