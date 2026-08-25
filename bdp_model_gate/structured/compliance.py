"""NDPA/NDPR-style compliance mapping tied to the model card."""

from __future__ import annotations

from ..config import ComplianceConfig
from ..core.base import BaseCheck, CheckResult


class ComplianceMappingCheck(BaseCheck):
    """Validates model card completeness, DPIA trigger, and explainability
    requirement. Expects `context.model_card` — a dict that can include:

        legal_basis (str)
        data_minimization_justification (str)
        training_data_source (str)
        use_case (str) — matched against config.high_risk_use_cases
        dpia_completed (bool)
        influences_decision_about_person (bool) — defaults to True if use_case
            matches a high-risk use case
        explainability_method (str)
    """

    name = "compliance_mapping"
    category = "compliance"
    blocking = True

    def __init__(self, config: ComplianceConfig | None = None):
        self.config = config or ComplianceConfig()

    def run(self, context) -> list[CheckResult]:
        model_card = context.model_card
        if not model_card:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no model_card supplied",
                    self.blocking,
                )
            ]

        results = []

        for field_name in self.config.required_model_card_fields:
            present = bool(model_card.get(field_name))
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "OK" if present else "COMPLIANCE_RISK",
                    detail=(
                        f"model_card.{field_name} present"
                        if present
                        else f"model_card.{field_name} missing — required under NDPA/NDPR"
                    ),
                    blocking=self.blocking,
                    metadata={"check": f"model_card.{field_name}"},
                )
            )

        use_case = (model_card.get("use_case") or "").lower()
        is_high_risk = any(hr in use_case for hr in self.config.high_risk_use_cases)
        dpia_done = bool(model_card.get("dpia_completed"))
        dpia_ok = (not is_high_risk) or dpia_done
        results.append(
            CheckResult(
                self.name,
                self.category,
                "OK" if dpia_ok else "COMPLIANCE_RISK",
                detail=(
                    "high-risk use case requires a completed DPIA"
                    if is_high_risk and not dpia_done
                    else ("DPIA completed" if dpia_done else "not high-risk — DPIA not required")
                ),
                blocking=self.blocking,
                metadata={"check": "dpia_trigger", "is_high_risk": is_high_risk},
            )
        )

        influences_person = bool(model_card.get("influences_decision_about_person", is_high_risk))
        explainability_doc = bool(model_card.get("explainability_method"))
        explain_ok = (not influences_person) or explainability_doc
        results.append(
            CheckResult(
                self.name,
                self.category,
                "OK" if explain_ok else "COMPLIANCE_RISK",
                detail=(
                    "explainability method documented"
                    if explainability_doc
                    else (
                        "required — model affects a person's outcome, no method documented"
                        if influences_person
                        else "not required for this use case"
                    )
                ),
                blocking=self.blocking,
                metadata={
                    "check": "explainability_requirement",
                    "influences_person": influences_person,
                },
            )
        )

        return results
