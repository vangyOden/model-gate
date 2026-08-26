"""Class ordering and favourable outcomes for multiclass problems.

Binary classification carries two conventions for free: class 1 is the
positive class, and there is nothing to order. Multiclass has neither, and
guessing at them is how a governance tool produces a confident wrong answer.

Two pieces of problem structure make the multiclass checks meaningful:

    class_order         The classes in ascending order of favourability,
                        e.g. ["decline", "refer", "accept"]. Supplying it
                        marks the problem as **ordinal**, which unlocks
                        metrics that know a decline-vs-accept error costs
                        more than a refer-vs-accept one. Omit it for a
                        genuinely nominal problem (product category, say),
                        where no ordering exists.

    favourable_classes  Which outcomes count as a positive result for
                        demographic parity. Defaults to the single most
                        favourable class when `class_order` is given, and
                        is logged when inferred.

Both live on the context rather than in `GateConfig`, because they describe
the problem rather than a threshold you might tune per run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("classes")


def validate_class_order(class_order: Any, y_true: Any = None) -> None:
    """Rejects an unusable `class_order` before any check runs."""
    if class_order is None:
        return
    if isinstance(class_order, (str, bytes)) or not isinstance(class_order, Sequence):
        raise GateConfigurationError(
            "context.class_order must be a sequence of class labels in ascending "
            f"order of favourability, got {type(class_order).__name__}"
        )
    ordered = list(class_order)
    if len(ordered) < 2:
        raise GateConfigurationError(
            f"context.class_order needs at least two classes, got {ordered!r}"
        )
    if len(set(map(_key, ordered))) != len(ordered):
        raise GateConfigurationError(f"context.class_order has duplicate labels: {ordered!r}")

    if y_true is not None:
        known = {_key(c) for c in ordered}
        unseen = {_key(v) for v in np.unique(np.asarray(y_true))} - known
        if unseen:
            raise GateConfigurationError(
                f"y_true contains labels missing from context.class_order: "
                f"{sorted(map(str, unseen))} — class_order must list every class"
            )


def _key(label: Any) -> Any:
    """Numpy scalars and Python scalars must compare equal as dict keys."""
    return label.item() if isinstance(label, np.generic) else label


def rank_map(class_order: Sequence[Any]) -> dict[Any, int]:
    """Maps each class label to its position, 0 = least favourable."""
    return {_key(label): position for position, label in enumerate(class_order)}


def to_ranks(values: Any, class_order: Sequence[Any]) -> np.ndarray:
    """Converts class labels into ordinal ranks.

    Rank distance is what makes an ordinal metric ordinal: predicting
    "decline" for an "accept" case is two steps wrong, "refer" is one.
    """
    ranks = rank_map(class_order)
    arr = np.asarray(values)
    try:
        return np.array([ranks[_key(v)] for v in arr], dtype=float)
    except KeyError as exc:
        raise GateConfigurationError(
            f"label {exc.args[0]!r} is not in context.class_order {list(class_order)!r}"
        ) from exc


def resolve_favourable(
    favourable_classes: Any = None,
    class_order: Sequence[Any] | None = None,
    task: str = "binary",
) -> list[Any] | None:
    """Determines which classes count as a favourable outcome.

    Returns None when it cannot be determined, which callers report as
    NOT_APPLICABLE rather than guessing — for a nominal multiclass problem
    there is no basis for picking one.
    """
    if favourable_classes is not None:
        listed = list(favourable_classes)
        if not listed:
            raise GateConfigurationError(
                "context.favourable_classes is empty — omit it, or name at least one class"
            )
        if class_order is not None:
            known = {_key(c) for c in class_order}
            unknown = [c for c in listed if _key(c) not in known]
            if unknown:
                raise GateConfigurationError(
                    f"context.favourable_classes {unknown!r} are not in "
                    f"context.class_order {list(class_order)!r}"
                )
        return listed

    if task == "binary":
        return [1]

    if class_order is not None:
        best = class_order[-1]
        logger.info(
            "context.favourable_classes not set — treating %r, the last entry in "
            "class_order, as the favourable outcome. Set it explicitly if that is wrong.",
            best,
        )
        return [best]

    return None


def favourable_mask(values: Any, favourable: Sequence[Any]) -> np.ndarray:
    """Boolean mask of rows whose label is a favourable outcome."""
    wanted = {_key(c) for c in favourable}
    return np.array([_key(v) in wanted for v in np.asarray(values)], dtype=bool)


__all__ = [
    "favourable_mask",
    "rank_map",
    "resolve_favourable",
    "to_ranks",
    "validate_class_order",
]
