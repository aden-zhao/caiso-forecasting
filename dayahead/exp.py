"""The models behind the final forecasts, by name, with cached results.

    python exp.py dev trees mlp_day ...   # run what's missing, then compare with CAISO

Predictions are cached in results/<period>_<name>.parquet. Each run happens in
its own process (see run_one.py), three at a time.

The ~60 configurations tried along the way (targets, weather sources, tuning,
the feature ablation) are at git tag v1-full-features.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    make: Callable                      # full table -> fit_predict(train, test)
    train_start: str = bt.TRAIN_START


def mlp(**kw) -> Callable:
    def make(df):
        import nn   # imports torch, which must not share a process with LightGBM training
        return nn.mlp(df, **kw)
    return make


RATIO = dict(anchor="d1_morning", mode="ratio")      # predict load / D-1 morning level
FF5 = {"feature_fraction": 0.5}

EXPS = {
    # A: our independent forecast is the mean of these three
    "trees":         Exp(lambda df: bt.lgbm(bt.feature_cols(df), **RATIO, params=FF5), FROM_2020),
    "trees_by_hour": Exp(lambda df: bt.per_hour(lambda: bt.lgbm(
                         bt.feature_cols(df), **RATIO, rounds=600,
                         params={**FF5, "num_leaves": 15, "min_data_in_leaf": 30}))),
    "mlp_day":       Exp(mlp(dropout=0.2, wd=1e-3, epochs=150), FROM_2020),
    # B adds CAISO's forecast and this model of CAISO's errors (predicts actual - CAISO)
    "caiso_correction": Exp(lambda df: bt.lgbm(bt.feature_cols(df) + ["caiso_2da", "caiso_dam"],
                                               anchor="caiso_dam", mode="diff")),
}


@lru_cache
def table() -> pd.DataFrame:
    return features.build()


def result_path(name: str, period: str) -> Path:
    return RES / f"{period}_{name}.parquet"


def run(name: str, period: str) -> pd.Series:
    """Predictions for one experiment, from the cache or by running the backtest."""
    path = result_path(name, period)
    if path.exists():
        return pd.read_parquet(path)["pred"]
    e = EXPS[name]
    df = table()
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
