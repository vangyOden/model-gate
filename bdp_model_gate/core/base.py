"""Core interfaces shared by every governance check, regardless of data modality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..task import ALL_TASKS


@dataclass
class CheckResult:
    """The outcome of a single governance check.

    `flag` is "OK", "NOT_APPLICABLE" (check skipped — e.g. optional input
    missing or optional dependency not installed), "CHECK_ERROR" (the check
    raised an exception), or a check-specific risk string such as
    "PROXY_RISK" or "PII_LEAKAGE_RISK".
    """

    check_name: str
    category: str
    flag: str
    detail: str = ""
    blocking: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    @property
    def is_ok(self) -> bool:
        return self.flag in ("OK", "NOT_APPLICABLE")


class BaseCheck:
    """Interface every governance check implements.

    Subclasses set `name`, `category`, `blocking` and optionally
    `supported_tasks` as class attributes, and implement `run(context)`. `blocking=True` means a failing flag from
    this check should block promotion outright; `blocking=False` routes a
    failure to human review instead (used for checks that need judgment,
    like fairness flags that may be false positives).
    """

    name: str = "base_check"
    category: str = "fairness"
    blocking: bool = True
    #: Prediction tasks this check can meaningfully run against. The gate
    #: reports NOT_APPLICABLE for anything else rather than letting the check
    #: produce a confident but meaningless number. Defaults to every task, so
    #: third-party plugins written before 0.3.0 keep running unchanged.
    supported_tasks: tuple[str, ...] = ALL_TASKS

    def run(self, context: Any) -> list[CheckResult]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")
