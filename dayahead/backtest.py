"""Rolling-origin backtest for day-ahead CAISO load.

Each month is forecast by a model trained on every day that was fully known by
the first issue time of that month (days <= month_start - 2). Every method is
scored on the same hours: those where actual load and CAISO's DAM forecast exist.

Periods:
  dev  = 2024-01 .. 2024-12   (for choosing features / settings)
  test = 2025-01 .. 2026-09   (touched only for the final numbers)
"""
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

PERIODS = {"dev": ("2024-01-01", "2025-01-01"), "test": ("2025-01-01", "2026-10-01")}
TRAIN_START = "2021-03-15"   # GFS archived forecasts start 2021-03 (JMA fills earlier temps)

META = {"load", "day", "issue_utc", "caiso_dam", "caiso_2da", "batteries", "solar"}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"MAE": np.mean(np.abs(e)), "MAPE": 100 * np.mean(np.abs(e) / y),
            "RMSE": np.sqrt(np.mean(e ** 2)), "bias": np.mean(e)}


def eval_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (df.day >= start) & (df.day < end) & df.load.notna() & df.caiso_dam.notna()


def rolling(df: pd.DataFrame, fit_predict: Callable, period: str = "dev",
            train_start: str = TRAIN_START) -> pd.Series:
    """fit_predict(train_df, test_df) -> predictions for test_df."""
    start, end = PERIODS[period]
    mask = eval_mask(df, start, end)
    preds = pd.Series(np.nan, index=df.index)
    for m in pd.date_range(start, end, freq="MS", inclusive="left"):
        test = df[mask & (df.day >= m) & (df.day < m + pd.offsets.MonthBegin(1))]
        train = df[(df.day >= train_start) & (df.day <= m - pd.Timedelta(days=2)) & df.load.notna()]
        preds[test.index] = fit_predict(train, test)
    return preds[mask]


def score(df: pd.DataFrame, preds: dict[str, pd.Series], period: str = "dev") -> pd.DataFrame:
    mask = eval_mask(df, *PERIODS[period])
    y = df.load[mask]
    rows = {}
    for name, p in preds.items():
        p = p.reindex(y.index)
        ok = p.notna()
        rows[name] = {**metrics(y[ok].to_numpy(), p[ok].to_numpy()), "coverage": ok.mean()}
    return pd.DataFrame(rows).T.sort_values("MAE")


# ---- methods -------------------------------------------------------------

def column(name: str) -> Callable:
    return lambda train, test: test[name].to_numpy()


def feature_cols(df: pd.DataFrame, drop: set = frozenset()) -> list[str]:
    return [c for c in df.columns if c not in META and c not in drop and not c.startswith("load_")
            and c != "t_years" and df[c].dtype != object]


LGB_PARAMS = dict(objective="l2", learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=8)


def lgbm(features: list[str], anchor: str | None = "d1_morning", mode: str = "ratio",
         rounds: int = 1500, params: dict | None = None, seed: int = 0,
         weight_halflife: float | None = None) -> Callable:
    """Gradient boosting on y / anchor ('ratio'), y - anchor ('diff'), or raw MW (anchor=None).

    weight_halflife (days) down-weights old training rows, so recent behaviour
    (more rooftop solar, new demand) counts more.
    """
    params = {**LGB_PARAMS, **(params or {}), "seed": seed}

    def encode(d, y):
        if anchor is None:
            return y
        a = d[anchor].to_numpy()
        return y / a if mode == "ratio" else y - a

    def decode(d, z):
        if anchor is None:
            return z
        a = d[anchor].to_numpy()
        return z * a if mode == "ratio" else z + a

    def fit_predict(train, test):
        if anchor is not None:
            train = train[train[anchor].notna()]
        w = None
        if weight_halflife:
            age = (train.day.max() - train.day).dt.days.to_numpy()
            w = 0.5 ** (age / weight_halflife)
        ds = lgb.Dataset(train[features], encode(train, train.load.to_numpy()), weight=w)
        model = lgb.train(params, ds, num_boost_round=rounds)
        return decode(test, model.predict(test[features]))

    return fit_predict


def per_hour(make: Callable) -> Callable:
    """Fit a separate model for each local hour of the day."""
    def fit_predict(train, test):
        out = np.full(len(test), np.nan)
        for hr in range(24):
            tr, te = train[train.hour == hr], test.hour.to_numpy() == hr
            if te.any():
                out[te] = make()(tr, test[te])
        return out
    return fit_predict
