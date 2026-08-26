"""Framework-neutral access to whatever is being gated.

Before 0.3.1 every check reached into `context.model` and made its own
assumption: that it exposed `.predict()`, or `.predict_proba()` returning
exactly two columns, or a `.coef_` attribute. Those are scikit-learn
conventions. A PyTorch `nn.Module` has none of them, a Keras model has
`.predict()` but no `.predict_proba()`, a raw XGBoost `Booster` wants a
DMatrix, and a remote scoring endpoint is not an object at all.

`ModelAdapter` is the single place that knows how to call a model. Checks
ask it for predictions and never touch `context.model` directly, so adding
a new kind of model means teaching one class rather than five call sites.

Two ways to supply a model:

    model=estimator          anything with .predict() — scikit-learn,
                             Keras, LightGBM, XGBoost's sklearn API, or
                             your own class. Used as-is.

    predict_fn=callable      any `fn(DataFrame) -> array`. The boundary is
                             deliberately "DataFrame in, array out": your
                             function owns tensor conversion, device
                             placement, batching and auth, so this library
                             never imports a deep-learning framework and
                             never guesses at a dtype.

`predict_proba_fn` and `gradient_fn` are optional and unlock the checks
that need more than a point prediction.

This adapter is internal for now; a public, subclassable `ModelAdapter` is
planned for 1.0.0. Until then the extension point is a plain callable,
which a lambda or `functools.partial` covers.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("model")

PredictFn = Callable[[pd.DataFrame], Any]


class ModelAdapter:
    """Normalises any supported model into predict / predict_proba / gradients."""

    def __init__(
        self,
        model: Any = None,
        predict_fn: PredictFn | None = None,
        predict_proba_fn: PredictFn | None = None,
        gradient_fn: PredictFn | None = None,
    ):
        self.model = model
        self._predict_fn = predict_fn
        self._predict_proba_fn = predict_proba_fn
        self._gradient_fn = gradient_fn

    # ------------------------------------------------------------------ build

    @classmethod
    def from_context(cls, context: Any) -> ModelAdapter:
        return cls(
            model=getattr(context, "model", None),
            predict_fn=getattr(context, "predict_fn", None),
            predict_proba_fn=getattr(context, "predict_proba_fn", None),
            gradient_fn=getattr(context, "gradient_fn", None),
        )

    # ------------------------------------------------------------ capabilities

    @property
    def can_predict(self) -> bool:
        return (
            self._predict_fn is not None or hasattr(self.model, "predict") or callable(self.model)
        )

    @property
    def can_predict_proba(self) -> bool:
        return self._predict_proba_fn is not None or hasattr(self.model, "predict_proba")

    @property
    def can_gradient(self) -> bool:
        return self._gradient_fn is not None

    def describe(self) -> str:
        """How predictions are obtained, for logs and result metadata."""
        if self._predict_fn is not None:
            return "predict_fn"
        if hasattr(self.model, "predict"):
            return f"{type(self.model).__name__}.predict"
        if callable(self.model):
            return f"{type(self.model).__name__}.__call__"
        return "none"

    # --------------------------------------------------------------- predicting

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point predictions as a 1-D array."""
        if self._predict_fn is not None:
            raw = self._predict_fn(X)
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(X)
        elif callable(self.model):
            # A bare callable — a torch module wrapped by the caller, or a
            # plain function. Accepted so `model=` and `predict_fn=` are
            # interchangeable rather than a trap.
            raw = self.model(X)
        else:
            raise GateConfigurationError(
                "no way to obtain predictions: context.model has no .predict() and is "
                "not callable, and no predict_fn was supplied. Pass "
                "predict_fn=lambda df: ... for a model this library cannot call directly."
            )
        return _as_1d(np.asarray(raw), "predict")

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class, as a 1-D array.

        Normalises the shapes different frameworks emit for binary
        classification: scikit-learn's `(n, 2)`, Keras's `(n, 1)` from a
        sigmoid, and a bare `(n,)` vector are all reduced to one column.
        """
        if self._predict_proba_fn is not None:
            raw = np.asarray(self._predict_proba_fn(X))
        elif hasattr(self.model, "predict_proba"):
            raw = np.asarray(self.model.predict_proba(X))
        else:
            raise GateConfigurationError(
                "no way to obtain probabilities: context.model has no .predict_proba() "
                "and no predict_proba_fn was supplied"
            )

        if raw.ndim == 1:
            return raw.astype(float)
        if raw.ndim == 2:
            if raw.shape[1] == 1:  # Keras-style sigmoid output
                return raw[:, 0].astype(float)
            if raw.shape[1] == 2:  # scikit-learn-style [P(neg), P(pos)]
                return raw[:, 1].astype(float)
            raise GateConfigurationError(
                f"predict_proba returned {raw.shape[1]} columns, so this is not a "
                "binary classifier — there is no single positive class to take"
            )
        raise GateConfigurationError(
            f"predict_proba returned an array of {raw.ndim} dimensions; expected 1 or 2"
        )

    def predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Full (n_rows, n_classes) probability matrix, for multiclass.

        Unlike `predict_positive_proba` this keeps every column, because a
        multiclass check needs to pick out one or more favourable classes.
        """
        if self._predict_proba_fn is not None:
            raw = np.asarray(self._predict_proba_fn(X), dtype=float)
        elif hasattr(self.model, "predict_proba"):
            raw = np.asarray(self.model.predict_proba(X), dtype=float)
        else:
            raise GateConfigurationError(
                "no way to obtain probabilities: context.model has no .predict_proba() "
                "and no predict_proba_fn was supplied"
            )
        if raw.ndim != 2:
            raise GateConfigurationError(
                f"predict_proba returned {raw.ndim} dimensions; a multiclass check needs "
                "an (n_rows, n_classes) matrix"
            )
        return raw

    def gradients(self, X: pd.DataFrame) -> np.ndarray | None:
        """Per-row, per-feature gradients of the output, if available.

        Returns an (n_rows, n_features) array aligned to `X.columns`, or None
        when no `gradient_fn` was supplied. A mis-shaped result is refused
        rather than broadcast into a meaningless perturbation.
        """
        if self._gradient_fn is None:
            return None
        grads = np.asarray(self._gradient_fn(X), dtype=float)
        if grads.shape != X.shape:
            raise GateConfigurationError(
                f"gradient_fn returned shape {grads.shape}, but X is {X.shape} — it must "
                "return one gradient per (row, feature), aligned to X.columns"
            )
        return grads


def _as_1d(arr: np.ndarray, what: str) -> np.ndarray:
    """Flattens a trailing singleton axis, which neural nets commonly emit."""
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 2:
        logger.debug(
            "%s returned %d columns; using it as-is. Wrap it in predict_fn if a single "
            "column was intended.",
            what,
            arr.shape[1],
        )
        return arr
    raise GateConfigurationError(
        f"{what} returned an array of {arr.ndim} dimensions; expected 1-D predictions"
    )


__all__ = ["ModelAdapter", "PredictFn"]
