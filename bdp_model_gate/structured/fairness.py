"""Per-feature and outcome-level fairness checks for structured data models."""

from __future__ import annotations

import numpy as np

from ..config import FairnessConfig
from ..core.base import BaseCheck, CheckResult


class ProxyCorrelationCheck(BaseCheck):
    """Flags numeric input features that correlate strongly with a protected
    attribute — even when that attribute itself is excluded from the model."""

    name = "proxy_correlation"
    category = "fairness"
    blocking = False

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
    """Outcome-level disparity check per protected attribute (demographic parity)."""

    name = "disparate_impact"
    category = "fairness"
    blocking = False

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

        results = []
        for attr in context.protected_df.columns:
            dpd = demographic_parity_difference(
                context.y_true,
                context.y_pred,
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
                    metadata={"protected_attr": attr, "demographic_parity_diff": round(dpd, 3)},
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

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    @staticmethod
    def _build_explainer(shap_module, model, X):
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
        return shap_module.Explainer(model, X)

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

        explainer = self._build_explainer(shap, context.model, context.X)
        shap_values = explainer(context.X)
        shap_df = pd.DataFrame(shap_values.values, columns=context.X.columns)

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
        if not hasattr(context.model, "predict_proba"):
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "model has no predict_proba — counterfactual check needs probability outputs",
                    self.blocking,
                )
            ]

        X = context.X
        results = []
        for attr in context.protected_df.columns:
            if attr not in X.columns:
                continue  # attribute excluded from model inputs — nothing to flip
            sample = X.sample(min(self.n_samples, len(X)), random_state=42).copy()
            base_preds = context.model.predict_proba(sample)[:, 1]
            for val in context.protected_df[attr].unique():
                flipped = sample.copy()
                flipped[attr] = val
                flipped_preds = context.model.predict_proba(flipped)[:, 1]
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
