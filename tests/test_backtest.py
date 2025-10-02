import pandas as pd

from pipeline.backtest import long_short_backtest


def test_long_short_backtest_handles_empty_positions():
    idx_scores = pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2020-01-01"))], names=["asset", "date"]
    )
    idx_returns = pd.MultiIndex.from_tuples(
        [("B", pd.Timestamp("2020-01-02"))], names=["asset", "date"]
    )

    scores = pd.Series(float("nan"), index=idx_scores)
    future_returns = pd.Series(float("nan"), index=idx_returns)

    result = long_short_backtest(scores, future_returns)

    assert result == {
        "mean_daily_net": 0.0,
        "sharpe_net_annual": 0.0,
        "turnover_mean": 0.0,
    }
