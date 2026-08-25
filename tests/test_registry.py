"""Tests for third-party check discovery via entry points."""

from bdp_model_gate.core.base import BaseCheck
from bdp_model_gate.registry import discover_plugin_checks


def test_discover_plugin_checks_returns_list_without_crashing():
    # No plugins are installed in the test environment — this should return
    # an empty list, not raise, since most consumers won't have any registered.
    checks = discover_plugin_checks()
    assert isinstance(checks, list)
    assert all(isinstance(c, BaseCheck) for c in checks)
