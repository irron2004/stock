from __future__ import annotations

import pandas as pd


def ensure_sorted(df: pd.DataFrame, asset_col: str = "asset", date_col: str = "date") -> pd.DataFrame:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    return data.sort_values([asset_col, date_col])


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{a}_{b}" for a, b in df.columns]
    return df


def zscore_groupwise(series: pd.Series, group_index) -> pd.Series:
    grouped = series.groupby(group_index)
    mu = grouped.transform("mean")
    sigma = grouped.transform("std") + 1e-9
    return (series - mu) / sigma
