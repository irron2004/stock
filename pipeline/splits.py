from __future__ import annotations

from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


def time_series_purged_splits(
    dates: Iterable[pd.Timestamp], n_splits: int = 5, embargo: int = 5
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    unique_dates = np.array(sorted(pd.unique(dates)))
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    fold_size = max(1, len(unique_dates) // (n_splits + 1))
    for fold in range(1, n_splits + 1):
        val_start = fold * fold_size
        val_end = min(len(unique_dates), val_start + fold_size)
        val_slice = unique_dates[val_start:val_end]
        embargo_start = max(0, val_start - embargo)
        embargo_end = min(len(unique_dates), val_end + embargo)
        train_slice = np.concatenate(
            [unique_dates[:embargo_start], unique_dates[embargo_end:]]
        )
        yield val_slice, train_slice
