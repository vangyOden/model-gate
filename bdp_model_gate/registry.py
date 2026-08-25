"""Plugin discovery for third-party checks.

A downstream package can register additional checks without forking this
library by declaring an entry point in the `bdp_model_gate.checks` group:

    # in the plugin's pyproject.toml
    [project.entry-points."bdp_model_gate.checks"]
    my_check = "my_package.checks:MyCustomCheck"

Each entry point must resolve to a BaseCheck subclass (not an instance).
`discover_plugin_checks()` instantiates each with no arguments — plugin
checks that need configuration should read it from their own defaults or
from environment/config files, since the shared GateConfig only has
sections for the built-in categories.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from ._logging import get_logger
from .core.base import BaseCheck
from .exceptions import GateConfigurationError

logger = get_logger("registry")

ENTRY_POINT_GROUP = "bdp_model_gate.checks"


def discover_plugin_checks() -> list[BaseCheck]:
    """Finds and instantiates every check registered under the
    `bdp_model_gate.checks` entry-point group. A plugin that fails to load
    is logged and skipped rather than crashing the whole gate — a
    misbehaving third-party check shouldn't block your own checks from running.
    """
    checks: list[BaseCheck] = []
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python < 3.10: entry_points() takes no kwargs and returns a
        # dict-like object keyed by group instead of an EntryPoints
        # collection with .select()/group filtering built in.
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            check_cls: type[BaseCheck] = ep.load()
            if not (isinstance(check_cls, type) and issubclass(check_cls, BaseCheck)):
                raise GateConfigurationError(
                    f"entry point '{ep.name}' does not resolve to a BaseCheck subclass"
                )
            checks.append(check_cls())
            logger.debug(
                "loaded plugin check '%s' from entry point '%s'", check_cls.__name__, ep.name
            )
        except Exception as exc:
            logger.warning("skipping plugin check '%s': %r", ep.name, exc)

    return checks
