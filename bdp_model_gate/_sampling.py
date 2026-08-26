"""Order-independent subsampling.

Several checks score a subsample rather than the whole validation set, for
speed. `DataFrame.sample(random_state=...)` is reproducible for a *fixed*
frame, but it selects by **position** — so the same data in a different row
order yields a different subsample, and therefore a possibly different
verdict.

For a governance gate that is a defect of the same kind as an unseeded RNG:
sorting a CSV should not change whether a model ships. `stable_sample`
selects by row **content** instead, so permuting the input cannot change
which rows are chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Odd 64-bit constant (golden-ratio derived) used to mix the seed into the
#: row digests, so `random_state` still varies the selection.
_MIX = 0x9E3779B97F4A7C15
_UINT64_MASK = (1 << 64) - 1


def stable_sample(frame: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Returns `n` rows chosen deterministically from `frame`'s contents.

    Selecting the same rows regardless of input order makes every check that
    uses it order-invariant. Identical rows hash identically, so a tie among
    duplicates is harmless — the predictions are the same either way.
    """
    if n >= len(frame):
        return frame.copy()

    digests = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    seed_mix = np.uint64(((random_state + 1) * _MIX) & _UINT64_MASK)
    mixed = digests ^ seed_mix

    chosen = np.argsort(mixed, kind="stable")[:n]
    return frame.iloc[chosen].copy()


__all__ = ["stable_sample"]
