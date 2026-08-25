"""Tests for plugin discovery, using monkeypatched entry points so the test
suite doesn't need a real installed plugin package."""

from types import SimpleNamespace

from bdp_model_gate.core.base import BaseCheck, CheckResult
from bdp_model_gate.registry import discover_plugin_checks


class _WellBehavedPlugin(BaseCheck):
    name = "well_behaved_plugin"
    category = "compliance"
    blocking = False

    def run(self, context):
        return [CheckResult(self.name, self.category, "OK", "fine", self.blocking)]


class _NotACheck:
    """Deliberately not a BaseCheck subclass, to exercise the reject path."""


def _fake_entry_point(name, target):
    return SimpleNamespace(name=name, load=lambda: target)


def test_discover_plugin_checks_loads_valid_plugin(monkeypatch):
    fake_eps = [_fake_entry_point("good_plugin", _WellBehavedPlugin)]
    monkeypatch.setattr("bdp_model_gate.registry.entry_points", lambda group=None: fake_eps)

    checks = discover_plugin_checks()
    assert len(checks) == 1
    assert isinstance(checks[0], _WellBehavedPlugin)


def test_discover_plugin_checks_skips_invalid_plugin(monkeypatch):
    fake_eps = [_fake_entry_point("bad_plugin", _NotACheck)]
    monkeypatch.setattr("bdp_model_gate.registry.entry_points", lambda group=None: fake_eps)

    checks = discover_plugin_checks()
    assert checks == []


def test_discover_plugin_checks_skips_plugin_that_fails_to_load(monkeypatch):
    def broken_load():
        raise RuntimeError("plugin import exploded")

    fake_eps = [SimpleNamespace(name="broken_plugin", load=broken_load)]
    monkeypatch.setattr("bdp_model_gate.registry.entry_points", lambda group=None: fake_eps)

    checks = discover_plugin_checks()
    assert checks == []


def test_discover_plugin_checks_handles_pre_310_api(monkeypatch):
    """Simulates the Python < 3.10 entry_points() API, which takes no
    kwargs and returns a dict-like object instead."""
    fake_eps_dict = {
        "bdp_model_gate.checks": [_fake_entry_point("good_plugin", _WellBehavedPlugin)]
    }

    def fake_entry_points(group=None):
        if group is not None:
            raise TypeError("entry_points() takes no keyword arguments")
        return fake_eps_dict

    monkeypatch.setattr("bdp_model_gate.registry.entry_points", fake_entry_points)

    checks = discover_plugin_checks()
    assert len(checks) == 1
    assert isinstance(checks[0], _WellBehavedPlugin)
