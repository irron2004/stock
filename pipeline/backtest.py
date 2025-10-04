from __future__ import annotations

from typing import Dict

import math
import pandas as pd


def long_short_backtest(
    scores: pd.Series,
    future_returns: pd.Series,
    top_k: float = 0.2,
    cost_bps: float = 10.0,
) -> Dict[str, float]:
    df = pd.DataFrame({"score": scores, "fret": future_returns}).dropna()
    positions = []
    for date, cross_section in df.groupby(level=1):
        n_assets = len(cross_section)
        k = max(1, int(n_assets * top_k))
        ordered = cross_section["score"].sort_values(ascending=False)
        long_assets = ordered.index[:k]
        short_assets = ordered.index[-k:]
        pos = pd.Series(0.0, index=cross_section.index)
        pos.loc[long_assets] = 1.0 / k
        pos.loc[short_assets] = -1.0 / k
        positions.append(pos)
    if not positions:
        return {
            "mean_daily_net": 0.0,
            "sharpe_net_annual": 0.0,
            "turnover_mean": 0.0,
        }

    pos = pd.concat(positions, axis=0).sort_index()

    prev_pos = pos.groupby(level=0).shift(1).fillna(0.0)
    turnover = (pos - prev_pos).abs().groupby(level=1).sum()
    costs = turnover * (cost_bps / 1e4)

    pnl_gross = (pos * df["fret"]).groupby(level=1).sum()
    pnl_net = pnl_gross - costs
    sharpe = 0.0
    if pnl_net.std() > 0:
        sharpe = pnl_net.mean() / pnl_net.std() * math.sqrt(252)
    return {
        "mean_daily_net": float(pnl_net.mean()),
        "sharpe_net_annual": float(sharpe),
        "turnover_mean": float(turnover.mean()),
    }
