"""Per-feature and outcome-level fairness checks for structured data models."""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..classes import favourable_mask, resolve_favourable
from ..config import FairnessConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..metrics import to_class_labels, to_hard_labels
from ..model import ModelAdapter
from ..task import ALL_TASKS, CLASSIFICATION_TASKS, MULTICLASS, resolve_task

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

    For multiclass, "predicted positive" means predicted into
    `context.favourable_classes` — for underwriting, typically `["accept"]`.
    That set defaults to the most favourable entry of `context.class_order`
    when one is given; with neither, the check reports NOT_APPLICABLE rather
    than picking a class arbitrarily, because which outcome counts as
    favourable is a judgement the data cannot supply.

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

        task = resolve_task(context)
        class_order = getattr(context, "class_order", None)

        if task == MULTICLASS:
            favourable = resolve_favourable(
                getattr(context, "favourable_classes", None), class_order, task
            )
            if favourable is None:
                return [
                    CheckResult(
                        self.name,
                        self.category,
                        "NOT_APPLICABLE",
                        "multiclass parity needs to know which outcome counts as "
                        "favourable — set context.favourable_classes (e.g. ['accept']) "
                        "or context.class_order",
                        self.blocking,
                    )
                ]
            labels = to_class_labels(context.y_pred, class_order)
            # Collapse to a binary "got the good outcome" indicator, which is
            # what a selection rate means once there are more than two classes.
            y_pred = favourable_mask(labels, favourable).astype(int)
            y_true_eval = favourable_mask(
                to_class_labels(context.y_true, class_order), favourable
            ).astype(int)
            favourable_note = f" [favourable: {', '.join(map(str, favourable))}]"
        else:
            y_pred = to_hard_labels(context.y_pred, self.config.decision_threshold)
            y_true_eval = context.y_true
            favourable_note = ""
            if not np.array_equal(np.asarray(context.y_pred), y_pred):
                logger.debug(
                    "binarised continuous y_pred at decision_threshold=%s for demographic parity",
                    self.config.decision_threshold,
                )

        results = []
        for attr in context.protected_df.columns:
            dpd = demographic_parity_difference(
                y_true_eval,
                y_pred,
                sensitive_features=context.protected_df[attr],
            )
            flag = "DISPARITY_RISK" if abs(dpd) > self.config.disparity_threshold else "OK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"{attr}: demographic parity diff={dpd:.3f}{favourable_note}",
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
    def _positive_class_values(values, class_index=None):
        """Normalises SHAP output to one contribution per (row, feature).

        shap returns a 2-D array for regressors and some binary classifiers,
        but a 3-D (rows, features, classes) array for others —
        `RandomForestClassifier` among them, and which shape you get changed
        across shap versions.

        Binary reduces to the positive class. Multiclass reduces to
        `class_index`, the column of the favourable outcome, so the check
        answers "does this feature push some groups away from being
        accepted?" rather than averaging across unrelated classes. Without
        a class index there is no defensible reduction, so it returns None
        and the caller reports NOT_APPLICABLE.
        """
        arr = np.asarray(values)
        if arr.ndim == 2:
            return arr
        if arr.ndim != 3:
            return None
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if class_index is not None and 0 <= class_index < arr.shape[-1]:
            return arr[:, :, class_index]
        return None

    @staticmethod
    def _favourable_class_index(context):
        """Column of the favourable class in a multiclass SHAP array.

        shap orders its class axis by the model's sorted class labels, which
        is what `class_order` is matched against here.
        """
        class_order = getattr(context, "class_order", None)
        if class_order is None:
            return None
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if not favourable:
            return None
        # shap's class axis follows the model's sorted classes, not the
        # favourability ordering the caller supplied.
        by_model_order = sorted(class_order, key=str)
        try:
            return by_model_order.index(favourable[0])
        except ValueError:  # pragma: no cover — resolve_favourable validates membership
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
        values = self._positive_class_values(
            shap_values.values, self._favourable_class_index(context)
        )
        if values is None:
            n_classes = np.asarray(shap_values.values).shape[-1]
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"multiclass SHAP output ({n_classes} classes) and no favourable "
                    "class to reduce to — set context.class_order or "
                    "context.favourable_classes so contributions can be compared for "
                    "one outcome",
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
    # Measures a shift in P(favourable outcome). Regression's analogue is
    # the mean prediction shift, which GroupMeanGapCheck already covers.
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None, n_samples: int = 200):
        self.config = config or FairnessConfig()
        self.n_samples = n_samples

    @staticmethod
    def _favourable_proba(adapter, frame, context):
        """Probability of the favourable outcome, for binary or multiclass."""
        if resolve_task(context) != MULTICLASS:
            return adapter.predict_positive_proba(frame)
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if class_order is None or not favourable:
            # run() screens for this, but the helper must not depend on that
            # to stay correct if it is ever called from elsewhere.
            raise GateConfigurationError(
                "multiclass counterfactuals need context.class_order and a favourable "
                "class to measure the shift in"
            )
        matrix = adapter.predict_proba_matrix(frame)
        by_model_order = sorted(class_order, key=str)
        columns = [by_model_order.index(c) for c in favourable]
        return matrix[:, columns].sum(axis=1)

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
        if resolve_task(context) == MULTICLASS and getattr(context, "class_order", None) is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "multiclass counterfactuals need context.class_order to identify "
                    "the favourable outcome to measure a shift in",
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
            base_preds = self._favourable_proba(adapter, sample, context)
            for val in context.protected_df[attr].unique():
                flipped = sample.copy()
                flipped[attr] = val
                flipped_preds = self._favourable_proba(adapter, flipped, context)
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
