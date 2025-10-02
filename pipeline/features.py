from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .utils import zscore_groupwise


def add_basic_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    data["ret"] = (
        np.log(data["close"]) - np.log(data.groupby(level=0)["close"].shift(1))
    ).replace([np.inf, -np.inf], np.nan)
    data["hl_range"] = data["high"] - data["low"]
    data["cl_pos_in_range"] = (data["close"] - data["low"]) / (
        data["high"] - data["low"] + 1e-9
    )
    if "amount" in data.columns:
        data["amount_z"] = zscore_groupwise(data["amount"], data.index.get_level_values(0))
    data["volume_z"] = zscore_groupwise(data["volume"], data.index.get_level_values(0))
    data["rv_5"] = (
        data.groupby(level=0)["ret"].rolling(window=5, min_periods=3).std().values
    )
    data["rv_21"] = (
        data.groupby(level=0)["ret"].rolling(window=21, min_periods=5).std().values
    )
    return data


def _group_roll_agg(
    df: pd.DataFrame, cols: Iterable[str], window: int, stats: Iterable[str] = ("mean", "std", "min", "max")
) -> pd.DataFrame:
    cols = list(cols)
    out_frames = []
    grouped = df.groupby(level=0)
    min_periods = max(3, window // 5)

    for stat in stats:
        rolled = (
            grouped[cols]
            .rolling(window=window, min_periods=min_periods)
            .agg(stat)
            .reset_index(level=0, drop=True)
        )
        out_frames.append(rolled.add_suffix(f"_w{window}_{stat}"))

    if "max" in stats and "min" in stats:
        rmax = (
            grouped[cols]
            .rolling(window=window, min_periods=min_periods)
            .agg("max")
            .reset_index(level=0, drop=True)
        )
        rmin = (
            grouped[cols]
            .rolling(window=window, min_periods=min_periods)
            .agg("min")
            .reset_index(level=0, drop=True)
        )
        out_frames.append((rmax - rmin).add_suffix(f"_w{window}_range"))

    current = df[cols].add_suffix(f"_w{window}_last")
    out_frames.append(current)

    combined = pd.concat(out_frames, axis=1)

    for col in cols:
        mean_col = f"{col}_w{window}_mean"
        std_col = f"{col}_w{window}_std"
        last_col = f"{col}_w{window}_last"
        if mean_col in combined.columns:
            combined[f"{col}_w{window}_last_minus_mean"] = combined[last_col] - combined[mean_col]
        if std_col in combined.columns:
            combined[f"{col}_w{window}_last_z"] = (
                combined[last_col] - combined.get(mean_col, 0.0)
            ) / (combined[std_col] + 1e-9)

    return combined


def _group_ewm(df: pd.DataFrame, cols: Iterable[str], span: int) -> pd.DataFrame:
    cols = list(cols)
    out_frames = []
    grouped = df.groupby(level=0)
    for col in cols:
        series = grouped[col].apply(lambda x: x.ewm(span=span, adjust=False).mean())
        series.name = f"{col}_ema{span}"
        out_frames.append(series)
    return pd.concat(out_frames, axis=1)


def build_amex_features(
    df: pd.DataFrame,
    base_cols: List[str],
    windows: List[int] | None = None,
    ema_spans: List[int] | None = None,
) -> pd.DataFrame:
    windows = windows or [21, 63, 126]
    ema_spans = ema_spans or [12, 26, 55]
    frames = []
    for window in windows:
        frames.append(_group_roll_agg(df, base_cols, window=window))
    for span in ema_spans:
        frames.append(_group_ewm(df, base_cols, span=span))
    return pd.concat(frames, axis=1)
