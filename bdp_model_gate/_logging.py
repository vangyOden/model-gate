"""Internal logging configuration.

BDP Model Gate uses the standard `logging` module rather than print(), so
it composes cleanly with a host application's or CI system's own logging
setup. The library never calls `logging.basicConfig()` itself — that's
left to the caller (or to `configure_logging()` below, which the CLI uses).
"""

from __future__ import annotations

import logging

LOGGER_NAME = "bdp_model_gate"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def configure_logging(verbose: bool = False) -> None:
    """Convenience setup for CLI / script usage. Not called on library import."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
