"""Exceptions raised by BDP Model Gate.

GateConfigurationError is raised for problems with how the gate itself was
set up (bad config values, unregistered checks). GateValidationError is
raised for problems with the inputs handed to a run (shape mismatches,
wrong types) — both are raised eagerly, before any check executes, so a
misconfigured run fails fast with a clear message instead of an opaque
traceback from deep inside a check.
"""


class BDPModelGateError(Exception):
    """Base class for all BDP Model Gate errors."""


class GateConfigurationError(BDPModelGateError):
    """Raised when the gate, its config, or a check is set up incorrectly."""


class GateValidationError(BDPModelGateError):
    """Raised when the context passed to a gate run is invalid."""
