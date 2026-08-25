"""Unstructured-data governance checks (text, image, audio).

Reserved for a future release of BDP Model Gate. The structured pipeline's
architecture — BaseCheck, GateConfig, ModelGate, GateReport — is designed to
extend to this modality without a breaking change: an UnstructuredGateContext
plus a matching set of checks (e.g. toxicity scanning, PII leakage in free
text, jailbreak resistance for the model itself rather than a side-car
generative component) is the intended shape.

Nothing here is implemented yet — importing this module is safe, but
instantiating its context or check-suite factory raises NotImplementedError
with guidance rather than failing silently or half-working.
"""

from typing import Any


class UnstructuredGateContext:
    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Unstructured data support (text/image/audio) is planned but not yet "
            "implemented. Use bdp_model_gate.StructuredGateContext for structured "
            "data models today."
        )


def default_unstructured_checks(*args: Any, **kwargs: Any):
    raise NotImplementedError(
        "Unstructured checks are not yet implemented. "
        "Track this package's changelog for when this lands."
    )


__all__ = ["UnstructuredGateContext", "default_unstructured_checks"]
