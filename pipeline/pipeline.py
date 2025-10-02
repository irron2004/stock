from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import mean_squared_error

from .backtest import long_short_backtest
from .config import Config
from .datasets import build_supervised
from .features import add_basic_price_features, build_amex_features
from .modeling import fit_lgbm_regressor, spearman_ic
from .regime import asof_join_regime, compute_regime_from_inputs
from .splits import time_series_purged_splits
from .ssl import (
    aggregate_embeddings_over_time,
    build_sequence_tensor,
    train_mae1d,
)
from .utils import ensure_sorted


def run_pipeline(
    prices: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
    target_kind: str = "regression",
) -> Dict[str, object]:
    cfg = cfg or Config()

    sorted_prices = ensure_sorted(prices.reset_index(), "asset", "date").set_index(
        ["asset", "date"]
    )
    enriched_prices = add_basic_price_features(sorted_prices)

    base_cols = ["ret", "hl_range", "cl_pos_in_range", "rv_21", "volume_z"]
    amex_features = build_amex_features(
        enriched_prices, base_cols, windows=cfg.windows, ema_spans=cfg.ema_spans
    )

    sequences, keys = build_sequence_tensor(
        enriched_prices, embed_cols=cfg.embed_cols, lookback=cfg.lookback
    )
    if len(keys) == 0:
        raise ValueError("No sequences available. Check lookback or input data length.")
    ssl_model, embeddings = train_mae1d(
        sequences,
        epochs=cfg.ssl_epochs,
        mask_ratio=cfg.mask_ratio,
        mask_len=cfg.mask_len,
    )
    embedding_frame = aggregate_embeddings_over_time(embeddings, suffix="emb")
    embedding_frame.index = pd.MultiIndex.from_tuples(keys, names=["asset", "date"])

    feature_table = amex_features.join(embedding_frame, how="inner")

    if regime_df is not None and not regime_df.empty:
        regime_features = compute_regime_from_inputs(regime_df)
        feature_table = asof_join_regime(feature_table, regime_features)

    prices_for_labels = enriched_prices[["close"]].join(feature_table, how="right")
    X, y, dates = build_supervised(
        prices_for_labels, horizon=cfg.horizon, target_kind=target_kind
    )

    metrics: List[Dict[str, float]] = []
    models = []
    for val_dates, train_dates in time_series_purged_splits(
        dates, n_splits=cfg.n_splits, embargo=cfg.embargo
    ):
        train_mask = X.index.get_level_values(1).isin(train_dates)
        val_mask = X.index.get_level_values(1).isin(val_dates)

        X_train, y_train = X[train_mask], y[train_mask]
        X_valid, y_valid = X[val_mask], y[val_mask]
        if len(X_train) == 0 or len(X_valid) == 0:
            continue

        model = fit_lgbm_regressor(X_train, y_train, X_valid, y_valid)
        models.append(model)

        y_pred = pd.Series(model.predict(X_valid), index=X_valid.index)
        rmse = math.sqrt(mean_squared_error(y_valid, y_pred))
        ic = spearman_ic(y_valid, y_pred)
        ls = long_short_backtest(
            y_pred,
            y_valid,
            top_k=cfg.top_k,
            cost_bps=cfg.cost_bps,
        )
        metrics.append({"rmse": rmse, "rankIC": ic, **ls})

    metrics_frame = pd.DataFrame(metrics)
    return {
        "models": models,
        "ssl_model": ssl_model,
        "metrics_cv": metrics_frame,
        "X_columns": X.columns.tolist(),
        "config": cfg,
    }
