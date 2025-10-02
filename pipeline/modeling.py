from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def fit_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 100,
):
    if lgb is None:
        raise ImportError("lightgbm must be installed to use fit_lgbm_regressor")
    base_params = dict(
        objective="regression",
        learning_rate=0.05,
        n_estimators=5000,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        base_params.update(params)
    model = lgb.LGBMRegressor(**base_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        verbose=False,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def spearman_ic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    correlation = spearmanr(y_true, y_pred).correlation
    if np.isnan(correlation):
        ranked_true = pd.Series(y_true).rank()
        ranked_pred = pd.Series(y_pred).rank()
        correlation = ranked_true.corr(ranked_pred)
    return float(correlation)
