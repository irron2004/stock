from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def build_supervised(
    prices_with_feats: pd.DataFrame,
    horizon: int = 5,
    target_kind: str = "regression",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = prices_with_feats.copy()
    close = data["close"]
    future = np.log(close.groupby(level=0).shift(-horizon)) - np.log(close)
    future.name = "y"

    if target_kind == "direction":
        y = (future > 0).astype(int)
    else:
        y = future

    drop_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "ret",
        "hl_range",
        "cl_pos_in_range",
        "rv_5",
        "rv_21",
    }
    feature_cols = [col for col in data.columns if col not in drop_cols]
    X = data[feature_cols]
    mask = X.notnull().all(axis=1) & y.notnull()
    filtered_index = data.index[mask]
    return X.loc[mask], y.loc[mask], filtered_index.get_level_values(1)
