from __future__ import annotations

from typing import Optional

import pandas as pd


def compute_regime_from_inputs(regime_df: pd.DataFrame) -> pd.DataFrame:
    regime = regime_df.copy()
    if {"y10", "y2"}.issubset(regime.columns):
        regime["term_spread"] = regime["y10"] - regime["y2"]
    if {"baa", "aaa"}.issubset(regime.columns):
        regime["cred_spread"] = regime["baa"] - regime["aaa"]
    if {"ivol", "rv_21"}.issubset(regime.columns):
        regime["vrp"] = (regime["ivol"] ** 2) - (regime["rv_21"] ** 2)
    return regime


def asof_join_regime(features: pd.DataFrame, regime: Optional[pd.DataFrame]) -> pd.DataFrame:
    if regime is None or regime.empty:
        return features
    sorted_regime = regime.sort_index()
    index = features.index
    if isinstance(index, pd.MultiIndex):
        dates = index.get_level_values(1)
    else:
        dates = index
    joined = sorted_regime.reindex(dates, method="ffill")
    joined.index = index
    return pd.concat([features, joined], axis=1)
