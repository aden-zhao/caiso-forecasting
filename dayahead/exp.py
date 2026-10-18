"""Every configuration tried, by name, with cached results.

    python exp.py dev a_from2020 mlp_reg_e150 ...   # run what's missing, then compare with CAISO

Predictions are cached in results/<period>_<name>.parquet. Each run happens in
its own process (see run_one.py), three at a time.

Entries above "current features" were scored on dev before the thermal-lag
features existed (commit 7f717a7). Rerunning them now uses today's features,
so their numbers would shift a little.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

import backtest as bt
import features

RES = Path(__file__).parent / "results"
FROM_2020 = "2020-01-10"      # earliest start, using JMA temperatures before GFS begins


@dataclass
class Exp:
    make: Callable                                  # full table -> fit_predict(train, test)
    table: dict = field(default_factory=dict)       # arguments for features.build
    train_start: str = bt.TRAIN_START


def lgbm(*extra_features: str, **kw) -> Callable:
    """LightGBM on the default features plus extra_features; kw go to backtest.lgbm."""
    return lambda df: bt.lgbm(bt.feature_cols(df) + list(extra_features), **kw)


def per_hour_lgbm(*extra_features: str, **kw) -> Callable:
    return lambda df: bt.per_hour(lambda: bt.lgbm(bt.feature_cols(df) + list(extra_features), **kw))


def mlp(**kw) -> Callable:
    def make(df):
        import nn   # imports torch, which must not share a process with LightGBM training
        return nn.mlp(df, **kw)
    return make


RATIO = dict(anchor="d1_morning", mode="ratio")      # predict load / D-1 morning level
FF5 = {"feature_fraction": 0.5}
CAISO_INPUTS = ("caiso_2da", "caiso_dam")

EXPS = {
    # target: what the trees predict relative to (anchor)
    "raw":            Exp(lgbm(anchor=None)),
    "d1m_ratio":      Exp(lgbm(**RATIO)),
    "sh_ratio":       Exp(lgbm(anchor="sh_mean", mode="ratio")),
    "sh_diff":        Exp(lgbm(anchor="sh_mean", mode="diff")),
    "sh_ratio_hl365": Exp(lgbm(anchor="sh_mean", mode="ratio", weight_halflife=365)),
    # weather: without JMA, with ECMWF, fresher than legit, or perfect
    "sh_ratio_nojma": Exp(lgbm(anchor="sh_mean", mode="ratio"), {"extra": ()}),
    "sh_ratio_ecmwf": Exp(lgbm(anchor="sh_mean", mode="ratio"), {"extra": ("jma_seamless", "ecmwf_ifs025")}),
    "d1m_ecmwf":      Exp(lgbm(**RATIO), {"extra": ("jma_seamless", "ecmwf_ifs025")}),
    "d1m_day1":       Exp(lgbm(**RATIO), {"policy": "day1"}),
    "d1m_obs":        Exp(lgbm(**RATIO), {"main": "obs"}),
    # model form and tuning
    "d1m_linear":     Exp(lgbm(**RATIO, params={"linear_tree": True, "linear_lambda": 1.0})),
    "d1m_trend":      Exp(lgbm("t_years", **RATIO)),
    "a_l1":           Exp(lgbm("t_years", **RATIO, params={"objective": "l1"})),
    "a_small":        Exp(lgbm("t_years", **RATIO, params={"num_leaves": 15, "min_data_in_leaf": 100},
                               rounds=3000)),
    "a_slow":         Exp(lgbm("t_years", **RATIO, params={"learning_rate": 0.01}, rounds=4000)),
    "a_ff5":          Exp(lgbm("t_years", **RATIO, params=FF5)),
    # using CAISO's own forecasts as inputs
    "on2da":          Exp(lgbm("caiso_2da", anchor="caiso_2da", mode="ratio")),
    "on2da_diff":     Exp(lgbm("caiso_2da", anchor="caiso_2da", mode="diff")),
    "ondam_ratio":    Exp(lgbm(*CAISO_INPUTS, anchor="caiso_dam", mode="ratio")),

    # ---- current features (with thermal lags) ----
    "a_thermal":      Exp(lgbm("t_years", **RATIO, params=FF5)),
    "a_from2020":     Exp(lgbm("t_years", **RATIO, params=FF5), train_start=FROM_2020),
    "a_perhour":      Exp(per_hour_lgbm("t_years", **RATIO, rounds=600,
                                        params={**FF5, "num_leaves": 15, "min_data_in_leaf": 30})),
    "ondam_diff":     Exp(lgbm(*CAISO_INPUTS, anchor="caiso_dam", mode="diff")),
    "mlp":            Exp(mlp()),
    "mlp_e100":       Exp(mlp(epochs=100)),
    "mlp_wide":       Exp(mlp(epochs=150, width=512)),
    "mlp_2020":       Exp(mlp(), train_start=FROM_2020),
    "mlp_s10":        Exp(mlp(seeds=10), train_start=FROM_2020),
    "mlp_reg":        Exp(mlp(dropout=0.2, wd=1e-3), train_start=FROM_2020),
    "mlp_e150":       Exp(mlp(epochs=150), train_start=FROM_2020),
    "mlp_w128":       Exp(mlp(width=128), train_start=FROM_2020),
    "mlp_reg_e150":   Exp(mlp(dropout=0.2, wd=1e-3, epochs=150), train_start=FROM_2020),
}


@lru_cache
def table(main: str = "gfs_seamless", extra: tuple = ("jma_seamless",), policy: str = "legit") -> pd.DataFrame:
    return features.build(main, extra, policy)


def result_path(name: str, period: str) -> Path:
    return RES / f"{period}_{name}.parquet"


def run(name: str, period: str) -> pd.Series:
    """Predictions for one experiment, from the cache or by running the backtest."""
    path = result_path(name, period)
    if path.exists():
        return pd.read_parquet(path)["pred"]
    e = EXPS[name]
    df = table(**e.table)
    pred = bt.rolling(df, e.make(df), period, e.train_start)
    RES.mkdir(exist_ok=True)
    pred.rename("pred").to_frame().to_parquet(path)
    return pred


def run_all(period: str, names: list[str]) -> dict[str, pd.Series]:
    """Run the uncached experiments in separate processes, then load them all."""
    todo = [n for n in names if not result_path(n, period).exists()]
    runner = str(Path(__file__).parent / "run_one.py")
    with ThreadPoolExecutor(3) as pool:
        codes = list(pool.map(lambda n: subprocess.call([sys.executable, "-W", "ignore", runner, n, period]),
                              todo))
    failed = [n for n, c in zip(todo, codes) if c]
    if failed:
        raise RuntimeError(f"failed: {failed}")
    return {n: run(n, period) for n in names}


def main(period: str, *names: str) -> None:
    df = table()
    preds = {"CAISO DAM": bt.rolling(df, bt.column("caiso_dam"), period), **run_all(period, list(names))}
    print(bt.score(df, preds, period).round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
