"""Named experiments, run in parallel and cached.

    python exp.py dev base sh_ratio ...     # run (or load cached) and compare with CAISO
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd

import backtest as bt
import features

RES = Path(__file__).parent / "results"


@lru_cache
def table(main: str = "gfs_seamless", extra: tuple = ("jma_seamless",), policy: str = "legit") -> pd.DataFrame:
    df = features.build(main, extra, policy)
    df["t_years"] = (df.day - pd.Timestamp("2021-01-01")).dt.days / 365.25
    return df


def drop(df, *names):
    return [c for c in bt.feature_cols(df) if not any(c == n or c.startswith(n) for n in names)]


# name -> (table args, function(df) -> fit_predict)
EXPS = {
    "d1m_ratio": ({}, lambda df: bt.lgbm(bt.feature_cols(df), "d1_morning", "ratio")),
    "sh_ratio":  ({}, lambda df: bt.lgbm(bt.feature_cols(df), "sh_mean", "ratio")),
    "sh_diff":   ({}, lambda df: bt.lgbm(bt.feature_cols(df), "sh_mean", "diff")),
    "raw":       ({}, lambda df: bt.lgbm(bt.feature_cols(df), None)),
    "sh_ratio_hl365": ({}, lambda df: bt.lgbm(bt.feature_cols(df), "sh_mean", "ratio", weight_halflife=365)),
    "sh_ratio_nojma": ({"extra": ()}, lambda df: bt.lgbm(bt.feature_cols(df), "sh_mean", "ratio")),
    "sh_ratio_ecmwf": ({"extra": ("jma_seamless", "ecmwf_ifs025")},
                       lambda df: bt.lgbm(bt.feature_cols(df), "sh_mean", "ratio")),
    "d1m_obs": ({"main": "obs"}, lambda df: bt.lgbm(bt.feature_cols(df), "d1_morning", "ratio")),
    "d1m_day1": ({"policy": "day1"}, lambda df: bt.lgbm(bt.feature_cols(df), "d1_morning", "ratio")),
    "d1m_ecmwf": ({"extra": ("jma_seamless", "ecmwf_ifs025")},
                  lambda df: bt.lgbm(bt.feature_cols(df), "d1_morning", "ratio")),
    # stack on CAISO's own 2-day-ahead forecast (published on D-2, so known at issue)
    "on2da": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["caiso_2da"], "caiso_2da", "ratio")),
    "on2da_diff": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["caiso_2da"], "caiso_2da", "diff")),
    "d1m_linear": ({}, lambda df: bt.lgbm(bt.feature_cols(df), "d1_morning", "ratio",
                                          params={"linear_tree": True, "linear_lambda": 1.0})),
    "d1m_trend": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio")),
    # track B: improve CAISO's day-ahead forecast (public by ~09:30, before 10:00 bids)
    "ondam_diff": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["caiso_2da", "caiso_dam"], "caiso_dam", "diff")),
    "ondam_ratio": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["caiso_2da", "caiso_dam"], "caiso_dam", "ratio")),
    # track A tuning
    "a_l1": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio",
                                    params={"objective": "l1"})),
    "a_small": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio",
                                       params={"num_leaves": 15, "min_data_in_leaf": 100}, rounds=3000)),
    "a_ff5": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio",
                                     params={"feature_fraction": 0.5})),
    "a_slow": ({}, lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio",
                                      params={"learning_rate": 0.01}, rounds=4000)),
}


FF5 = {"feature_fraction": 0.5}
A_BASE = lambda df: bt.lgbm(bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio", params=FF5)
EXPS.update({
    "a_thermal": ({}, A_BASE),   # same as a_ff5, after adding thermal-lag features
    "a_perhour": ({}, lambda df: bt.per_hour(lambda: bt.lgbm(
        bt.feature_cols(df) + ["t_years"], "d1_morning", "ratio",
        params={**FF5, "num_leaves": 15, "min_data_in_leaf": 30}, rounds=600))),
    "a_from2020": ({}, A_BASE),
})
TRAIN_START = {"a_from2020": "2020-01-10"}


def _mlp(df, **kw):
    import nn
    return nn.mlp(df, **kw)


EXPS.update({
    "mlp": ({}, lambda df: _mlp(df)),
    "mlp_e100": ({}, lambda df: _mlp(df, epochs=100)),
    "mlp_wide": ({}, lambda df: _mlp(df, epochs=150, width=512)),
    "mlp_2020": ({}, lambda df: _mlp(df)),
})
MLP_VARIANTS = {
    "mlp_s10": dict(seeds=10),
    "mlp_reg": dict(dropout=0.2, wd=1e-3),
    "mlp_e150": dict(epochs=150),
    "mlp_w128": dict(width=128),
    "mlp_reg_e150": dict(dropout=0.2, wd=1e-3, epochs=150),
}
for _n, _kw in MLP_VARIANTS.items():
    EXPS[_n] = ({}, lambda df, _kw=_kw: _mlp(df, **_kw))
for _n in ["mlp_2020", *MLP_VARIANTS]:
    TRAIN_START[_n] = "2020-01-10"


def run(name: str, period: str) -> pd.Series:
    path = RES / f"{period}_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)["pred"]
    args, make = EXPS[name]
    df = table(**args)
    pred = bt.rolling(df, make(df), period, TRAIN_START.get(name, bt.TRAIN_START))
    RES.mkdir(exist_ok=True)
    pred.rename("pred").to_frame().to_parquet(path)
    return pred


def main(period: str, *names: str) -> None:
    # one process per experiment (see run_one.py), three at a time
    todo = [n for n in names if not (RES / f"{period}_{n}.parquet").exists()]
    with ThreadPoolExecutor(3) as pool:
        codes = list(pool.map(lambda n: subprocess.call(
            [sys.executable, "-W", "ignore", str(Path(__file__).parent / "run_one.py"), n, period]), todo))
    assert not any(codes), dict(zip(todo, codes))
    preds = {n: run(n, period) for n in names}
    df = table()
    preds = {"CAISO DAM": bt.rolling(df, bt.column("caiso_dam"), period), **preds}
    print(bt.score(df, preds, period).round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
