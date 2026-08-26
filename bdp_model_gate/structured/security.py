"""Adversarial robustness, PII leakage, and prompt-injection checks."""

from __future__ import annotations

import re

import numpy as np

from .._logging import get_logger
from ..classes import to_ranks
from ..config import SecurityConfig
from ..core.base import BaseCheck, CheckResult
from ..model import ModelAdapter
from ..task import REGRESSION, resolve_task

logger = get_logger("security")

#: Below this magnitude a prediction is treated as ~0 for the purposes of a
#: relative-shift denominator, and the batch mean is used instead.
_REL_SHIFT_FLOOR = 1e-9


class AdversarialRobustnessCheck(BaseCheck):
    """Black-box robustness check: perturbs numeric features by a small
    relative amount and measures how much the prediction moves.

    For classification that is the **class flip rate** — how often the
    predicted label changes — and a high rate means a fragile decision
    boundary.

    For **ordinal** multiclass — where `context.class_order` is set — the
    flip rate is reported alongside the mean *rank distance* moved, because
    accept -> decline is a two-step error while accept -> refer is one. A
    model that only ever slips by one rank is materially safer than one that
    swings across the scale, and a bare flip rate cannot tell them apart.

    For regression there is no such thing as a flip: every perturbation moves
    a continuous output, so a flip rate would be ~1.0 and every model would
    be permanently BLOCKED. Sensitivity is measured instead as the mean
    *relative* change in prediction, gated with
    `SecurityConfig.adversarial_max_relative_shift`. A model whose output
    moves 30% when an input moves 2% is over-sensitive regardless of task.
    """

    name = "adversarial_robustness"
    category = "security"
    blocking = True

    def __init__(
        self,
        config: SecurityConfig | None = None,
        n_samples: int = 200,
        random_state: int = 42,
    ):
        self.config = config or SecurityConfig()
        self.n_samples = n_samples
        # Seeded so the same model and data always produce the same flip
        # rate. An unseeded gate can land on either side of the threshold
        # between runs, which makes a CI verdict irreproducible.
        self.random_state = random_state

    @staticmethod
    def _linear_coefficients(model, feature_names):
        """If the model exposes linear coefficients (LogisticRegression,
        LinearRegression, SGDClassifier, etc.), use them to perturb each
        sample along its steepest-ascent direction — a much stronger test
        than isotropic random noise. Returns None for anything else, and
        the check falls back to random perturbation.

        The returned vector is indexed by position in `feature_names` (all
        of X's columns), not by position among the numeric ones — `coef_`
        is laid out over every column the model was fitted on.
        """
        coef = getattr(model, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef)
        if coef.ndim > 1 and coef.shape[0] > 1:
            return None  # multiclass: no single steepest-ascent direction
        coef = coef.reshape(-1)
        if coef.shape[0] != len(feature_names):
            return None  # model was fitted on transformed features — can't align
        norm = np.linalg.norm(coef)
        return coef / norm if norm > 0 else None

    def run(self, context) -> list[CheckResult]:
        X = context.X
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no numeric features to perturb",
                    self.blocking,
                )
            ]

        task = resolve_task(context)
        adapter = ModelAdapter.from_context(context)
        sample = X.sample(min(self.n_samples, len(X)), random_state=self.random_state).copy()
        base_preds = adapter.predict(sample)

        # Preference order for the attack direction, strongest first:
        #   1. true per-row gradients, if the model can supply them
        #   2. linear coefficients, for models exposing coef_
        #   3. isotropic random noise
        per_row_gradients = adapter.gradients(sample) if adapter.can_gradient else None
        direction = (
            None
            if per_row_gradients is not None
            else self._linear_coefficients(context.model, X.columns)
        )
        if per_row_gradients is not None:
            method = "gradient-fn"
        elif direction is not None:
            method = "gradient-directed"
        else:
            method = "random"

        flips = np.zeros(len(sample), dtype=bool)
        # Worst-case relative movement seen for each row across all
        # perturbations, used for the regression verdict.
        rel_shift = np.zeros(len(sample), dtype=float)

        # Ordinal rank tracking applies only to a classification problem
        # whose classes the caller has actually ordered.
        ordinal_classes = (
            list(getattr(context, "class_order", None) or ()) if task != REGRESSION else []
        )
        rank_shift = np.zeros(len(sample), dtype=float)
        base_ranks = to_ranks(base_preds, ordinal_classes) if ordinal_classes else None

        def record(new_preds) -> None:
            flips[:] |= new_preds != base_preds
            if base_ranks is not None:
                moved = np.abs(to_ranks(new_preds, ordinal_classes) - base_ranks)
                np.maximum(rank_shift, moved, out=rank_shift)
            if task == REGRESSION:
                base = np.asarray(base_preds, dtype=float)
                moved = np.abs(np.asarray(new_preds, dtype=float) - base)
                # Relative to the row's own prediction, falling back to the
                # batch mean where a prediction is ~0 and a ratio would blow up.
                scale_ref = np.where(
                    np.abs(base) > _REL_SHIFT_FLOOR,
                    np.abs(base),
                    max(float(np.mean(np.abs(base))), _REL_SHIFT_FLOOR),
                )
                np.maximum(rel_shift, moved / scale_ref, out=rel_shift)

        if per_row_gradients is not None:
            # A real targeted attack: step every numeric feature along its own
            # per-row gradient, normalised per row so the step size stays
            # comparable to the coefficient and random paths.
            perturbed = sample.copy()
            norms = np.linalg.norm(per_row_gradients, axis=1, keepdims=True)
            unit = np.divide(
                per_row_gradients, norms, out=np.zeros_like(per_row_gradients), where=norms > 0
            )
            for col in numeric_cols:
                col_scale = perturbed[col].abs() * self.config.adversarial_epsilon
                perturbed[col] = perturbed[col] + unit[:, X.columns.get_loc(col)] * col_scale
            record(adapter.predict(perturbed))
        elif direction is not None:
            # Perturb every numeric feature at once, along the direction that
            # most increases the linear decision score — a targeted attack
            # rather than one feature at a time.
            perturbed = sample.copy()
            for col in numeric_cols:
                # Scale each feature's step to that feature's own magnitude.
                # A single scale derived from the mean across all columns is
                # dominated by the largest one, so a sum-insured column in the
                # millions would shove a 0-10 risk score by thousands — not the
                # "small relative perturbation" this check is meant to apply.
                col_scale = float(perturbed[col].abs().mean()) * self.config.adversarial_epsilon
                perturbed[col] = perturbed[col] + direction[X.columns.get_loc(col)] * col_scale
            record(adapter.predict(perturbed))
        else:
            rng = np.random.default_rng(self.random_state)
            for col in numeric_cols:
                perturbed = sample.copy()
                noise = (
                    perturbed[col]
                    * self.config.adversarial_epsilon
                    * rng.choice([-1, 1], size=len(perturbed))
                )
                perturbed[col] = perturbed[col] + noise
                record(adapter.predict(perturbed))

        if task == REGRESSION:
            shift = float(np.mean(rel_shift))
            threshold = self.config.adversarial_max_relative_shift
            flag = "OK" if shift <= threshold else "ROBUSTNESS_RISK"
            return [
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"mean relative prediction shift under {method} perturbation="
                        f"{shift:.4f} (max {threshold}); inputs moved by "
                        f"epsilon={self.config.adversarial_epsilon}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "relative_shift": round(shift, 4),
                        "threshold": threshold,
                        "method": method,
                        "task": task,
                        "epsilon": self.config.adversarial_epsilon,
                    },
                )
            ]

        flip_rate = float(flips.mean())
        over_flip_rate = flip_rate > self.config.adversarial_flip_rate_threshold
        metadata = {
            "flip_rate": round(flip_rate, 4),
            "threshold": self.config.adversarial_flip_rate_threshold,
            "method": method,
            "task": task,
        }
        detail = (
            f"flip rate under {method} perturbation={flip_rate:.4f} "
            f"(max {self.config.adversarial_flip_rate_threshold})"
        )

        over_rank_shift = False
        if base_ranks is not None:
            mean_rank_shift = float(np.mean(rank_shift))
            over_rank_shift = mean_rank_shift > self.config.adversarial_max_rank_shift
            metadata.update(
                {
                    "mean_rank_shift": round(mean_rank_shift, 4),
                    "max_observed_rank_shift": round(float(np.max(rank_shift)), 4),
                    "rank_shift_threshold": self.config.adversarial_max_rank_shift,
                    "n_classes": len(ordinal_classes),
                }
            )
            detail += (
                f"; mean ordinal rank shift={mean_rank_shift:.4f} "
                f"(max {self.config.adversarial_max_rank_shift}), worst "
                f"{float(np.max(rank_shift)):.0f} step(s)"
            )

        flag = "ROBUSTNESS_RISK" if (over_flip_rate or over_rank_shift) else "OK"
        return [
            CheckResult(
                self.name,
                self.category,
                flag,
                detail=detail,
                blocking=self.blocking,
                metadata=metadata,
            )
        ]


class PIILeakageCheck(BaseCheck):
    """Scans string/object columns of structured data for PII patterns that
    shouldn't be present as raw features (should've been hashed/tokenized
    upstream)."""

    name = "pii_leakage"
    category = "security"
    blocking = True

    def __init__(self, config: SecurityConfig | None = None, sample_size: int = 500):
        self.config = config or SecurityConfig()
        self.sample_size = sample_size

    def run(self, context) -> list[CheckResult]:
        X = context.X
        text_cols = X.select_dtypes(include=["object", "string"]).columns
        if len(text_cols) == 0:
            return [
                CheckResult(
                    self.name, self.category, "OK", "no string columns to scan", self.blocking
                )
            ]

        results = []
        for col in text_cols:
            sample_vals = X[col].dropna().astype(str).head(self.sample_size)
            for pii_type, pattern in self.config.pii_patterns.items():
                compiled = re.compile(pattern)
                hits = int(sample_vals.map(lambda v, c=compiled: bool(c.search(v))).sum())
                if hits > 0:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "PII_LEAKAGE_RISK",
                            detail=f"column '{col}' has {hits} value(s) matching {pii_type} pattern",
                            blocking=self.blocking,
                            metadata={"column": col, "pii_type": pii_type, "hits_in_sample": hits},
                        )
                    )

        return results or [
            CheckResult(self.name, self.category, "OK", "no PII patterns detected", self.blocking)
        ]


class PromptInjectionCheck(BaseCheck):
    """Only relevant if the model has a generative component (e.g. an
    LLM-based feature explainer, chatbot, or report generator sitting
    alongside the structured model). Requires `context.generate_fn`.
    Flags any response that appears to comply with an injection attempt
    rather than refuse."""

    name = "prompt_injection"
    category = "security"
    blocking = True

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.refusal_markers = [
            "cannot",
            "can't",
            "not able to",
            "won't",
            "against my",
            "unable to",
        ]

    def run(self, context) -> list[CheckResult]:
        if context.generate_fn is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no generative component supplied",
                    self.blocking,
                )
            ]

        results = []
        for prompt in self.config.jailbreak_prompts:
            response = context.generate_fn(prompt)
            refused = any(marker in response.lower() for marker in self.refusal_markers)
            flag = "OK" if refused else "INJECTION_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}",
                    blocking=self.blocking,
                    metadata={"prompt": prompt, "refused": refused},
                )
            )
        return results
