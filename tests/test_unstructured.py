"""The unstructured module is a reserved-but-unimplemented interface —
these tests just lock in that it fails loudly and clearly, not silently."""

import pytest

from bdp_model_gate.unstructured import UnstructuredGateContext, default_unstructured_checks


def test_unstructured_context_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet"):
        UnstructuredGateContext()


def test_default_unstructured_checks_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet"):
        default_unstructured_checks()
