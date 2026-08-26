"""Per-feature and outcome-level fairness checks for structured data models."""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import FairnessConfig
from ..core.base import BaseCheck, CheckResult
from ..metrics import to_hard_labels
from ..model import ModelAdapter
from ..task import ALL_TASKS, BINARY, CLASSIFICATION_TASKS

logger = get_logger("fairness")


class ProxyCorrelationCheck(BaseCheck):
    """Flags numeric input features that correlate strongly with a protected
    attribute — even when that attribute itself is excluded from the model."""

    name = "proxy_correlation"
    category = "fairness"
    blocking = False
    supported_tasks = ALL_TASKS  # compares features to attributes, not predictions

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    def run(self, context) -> list[CheckResult]:
        X, protected_df = context.X, context.protected_df
        if protected_df is None or protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]

        results = []
        for feature in X.columns:
            if X[feature].dtype.kind not in "if":  # numeric only
                continue
            for attr in protected_df.columns:
                if protected_df[attr].nunique() >= 10:
                    continue  # treat as continuous — correlation ratio not meaningful
                groups = protected_df[attr]
                overall_mean = X[feature].mean()
                ss_between = sum(
                    len(X[feature][groups == g])
                    * (X[feature][groups == g].mean() - overall_mean) ** 2
                    for g in groups.unique()
                )
                ss_total = ((X[feature] - overall_mean) ** 2).sum()
                eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                if eta_sq > self.config.proxy_corr_threshold:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "PROXY_RISK",
                            detail=f"{feature} correlates with {attr} (eta^2={eta_sq:.3f})",
                            blocking=self.blocking,
                            metadata={
                                "feature": feature,
                                "protected_attr": attr,
                                "proxy_strength": round(eta_sq, 3),
                            },
                        )
                    )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "OK",
                "no proxy correlations above threshold",
                self.blocking,
            )
        ]


class DisparateImpactCheck(BaseCheck):
    """Outcome-level disparity check per protected attribute (demographic parity).

    Demographic parity compares *selection rates* — the share of each group
    predicted positive — so it needs hard class labels. Continuous
    predictions are binarised at `config.decision_threshold` before being
    handed to fairlearn; predictions already in {0, 1} pass through
    untouched. Without that step a probability `y_pred` yields a selection
    rate of 0 in every group and a parity difference of exactly 0.0, which
    reads as "perfectly fair" no matter how skewed the model is.
    """

    name = "disparate_impact"
    category = "fairness"
    blocking = False
    # Demographic parity counts a selected class; there is none for a
    # continuous target. Regression uses the regression_fairness suite.
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        try:
            from fairlearn.metrics import demographic_parity_difference
        except ImportError:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "fairlearn not installed — pip install bdp-model-gate[structured]",
                    self.blocking,
                )
            ]

        y_pred = to_hard_labels(context.y_pred, self.config.decision_threshold)
        if not np.array_equal(np.asarray(context.y_pred), y_pred):
            logger.debug(
                "binarised continuous y_pred at decision_threshold=%s for demographic parity",
                self.config.decision_threshold,
            )

        results = []
        for attr in context.protected_df.columns:
            dpd = demographic_parity_difference(
                context.y_true,
                y_pred,
                sensitive_features=context.protected_df[attr],
            )
            flag = "DISPARITY_RISK" if abs(dpd) > self.config.disparity_threshold else "OK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"{attr}: demographic parity diff={dpd:.3f}",
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "demographic_parity_diff": round(dpd, 3),
                        "decision_threshold": self.config.decision_threshold,
                    },
                )
            )
        return results


class ShapSubgroupCheck(BaseCheck):
    """For each feature, checks whether its SHAP contribution differs
    meaningfully across protected-attribute groups — catches features that
    look fair on average but drive outcomes differently for a subgroup."""

    name = "shap_subgroup_gap"
    category = "fairness"
    blocking = False
    supported_tasks = ALL_TASKS  # SHAP contributions are defined for any output

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    @staticmethod
    def _build_explainer(shap_module, model, X, adapter=None):
        """TreeExplainer is dramatically faster and exact for tree-based
        models; fall back to the generic (permutation/kernel) Explainer for
        everything else."""
        tree_module_markers = (
            "sklearn.ensemble",
            "sklearn.tree",
            "xgboost",
            "lightgbm",
            "catboost",
        )
        model_module = type(model).__module__
        is_tree_model = any(marker in model_module for marker in tree_module_markers)
        if is_tree_model:
            try:
                return shap_module.TreeExplainer(model)
            except Exception:
                pass  # fall through to generic explainer if TreeExplainer can't handle this model
        if model is not None:
            try:
                return shap_module.Explainer(model, X)
            except (TypeError, ValueError):
                pass  # not an estimator shap recognises — fall through
        # shap's generic Explainer wants a callable. Hand it the adapter's
        # predict — the documented black-box pattern — which works for a
        # predict_fn-only context where there is no model object at all.
        if adapter is None:
            adapter = ModelAdapter(model=model)
        return shap_module.Explainer(adapter.predict, X)

    @staticmethod
    def _positive_class_values(values):
        """Normalises SHAP output to one contribution per (row, feature).

        shap returns a 2-D array for regressors and for some binary
        classifiers, but a 3-D (rows, features, classes) array for others —
        `RandomForestClassifier` among them, and which shape you get changed
        across shap versions. Reduce the binary case to the positive class;
        return None for genuine multiclass, which the caller reports as
        NOT_APPLICABLE rather than guessing at a class.
        """
        arr = np.asarray(values)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] == 2:
            return arr[:, :, 1]
        return None

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        try:
            import pandas as pd
            import shap
        except ImportError:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "shap not installed — pip install bdp-model-gate[structured]",
                    self.blocking,
                )
            ]

        try:
            explainer = self._build_explainer(
                shap, context.model, context.X, ModelAdapter.from_context(context)
            )
            shap_values = explainer(context.X)
        except Exception as exc:
            # A non-blocking fairness check must not block a deploy because
            # shap could not introspect the model. ModelGate would otherwise
            # convert the exception into a blocking CHECK_ERROR.
            logger.warning("shap could not explain this model: %r", exc)
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"shap could not explain this model ({type(exc).__name__}: {exc}) — "
                    "subgroup SHAP gaps were not evaluated",
                    self.blocking,
                )
            ]
        values = self._positive_class_values(shap_values.values)
        if values is None:
            n_classes = np.asarray(shap_values.values).shape[-1]
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"multiclass SHAP output ({n_classes} classes) — this check compares "
                    "contributions for a single positive class and has no meaningful "
                    "reduction across more than two",
                    self.blocking,
                )
            ]
        shap_df = pd.DataFrame(values, columns=context.X.columns)

        results = []
        for attr in context.protected_df.columns:
            for feature in context.X.columns:
                group_means = shap_df[feature].groupby(context.protected_df[attr].values).mean()
                gap = group_means.max() - group_means.min()
                if abs(gap) > self.config.shap_gap_threshold:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "SUBGROUP_IMPACT_RISK",
                            detail=f"{feature} SHAP contribution gap across {attr}={gap:.3f}",
                            blocking=self.blocking,
                            metadata={
                                "feature": feature,
                                "protected_attr": attr,
                                "shap_gap": round(float(gap), 3),
                            },
                        )
                    )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "OK",
                "no SHAP subgroup gaps above threshold",
                self.blocking,
            )
        ]


class CounterfactualFlipCheck(BaseCheck):
    """Flips protected-attribute values (when they're model inputs) and
    measures average prediction shift. Only meaningful if a protected
    attribute is actually included as a feature."""

    name = "counterfactual_flip"
    category = "fairness"
    blocking = False
    # Measures a shift in P(positive class); regression's analogue is the
    # mean prediction shift, which GroupMeanGapCheck already covers.
    supported_tasks = (BINARY,)

    def __init__(self, config: FairnessConfig | None = None, n_samples: int = 200):
        self.config = config or FairnessConfig()
        self.n_samples = n_samples

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        adapter = ModelAdapter.from_context(context)
        if not adapter.can_predict_proba:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no probability output available — this check needs either a model "
                    "with .predict_proba() or context.predict_proba_fn",
                    self.blocking,
                )
            ]

        X = context.X
        results = []
        for attr in context.protected_df.columns:
            if attr not in X.columns:
                continue  # attribute excluded from model inputs — nothing to flip
            sample = X.sample(min(self.n_samples, len(X)), random_state=42).copy()
            base_preds = adapter.predict_positive_proba(sample)
            for val in context.protected_df[attr].unique():
                flipped = sample.copy()
                flipped[attr] = val
                flipped_preds = adapter.predict_positive_proba(flipped)
                shift = float(np.mean(np.abs(flipped_preds - base_preds)))
                flag = (
                    "COUNTERFACTUAL_RISK"
                    if shift > self.config.counterfactual_shift_threshold
                    else "OK"
                )
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        flag,
                        detail=f"flipping {attr} to {val!r} shifts predictions by {shift:.4f} on average",
                        blocking=self.blocking,
                        metadata={
                            "protected_attr": attr,
                            "flipped_to": str(val),
                            "avg_prediction_shift": round(shift, 4),
                        },
                    )
                )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attributes present as model inputs",
                self.blocking,
            )
        ]
