"""Bottom-up vs direct: is it better to forecast each region and add them up?

    python regional.py            # runs every model on 2024 (dev), 3 seeds each, then compares

CAISO's forecast covers demand without battery charging or pumped hydro, but the
regional actuals on OASIS include them. So we build battery-free regional demand:
  gap(t)      = sum of the four regions' OASIS load - Outlook demand   (charging + pumping)
  demand_r(t) = OASIS load_r(t) - share_r * gap(t)
Each region's share is how much its OASIS load runs above its own day-ahead
forecast as the gap grows (a regression over the previous 12 months, refit
monthly, so nothing is estimated from the future). The shares sum to 1, so the
four regional demands add up exactly to Outlook demand, the target CAISO forecasts.

Models (all the same LightGBM as exp "trees"):
  direct     one model for system demand (= exp "trees")
  bu_state   one model per region, weather averaged over all of CAISO, summed
  bu_local   one model per region, weather averaged over that region's own cities, summed
  bu_mlp     the whole-day MLP (exp "mlp_day") per region, summed; compare with exp "mlp_day"
"""
import subprocess
import sys

if sys.argv[1:3] == ["one", "bu_mlp"]:
    import torch  # noqa: F401  load torch before LightGBM, or it segfaults
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

import backtest as bt
import data
import exp
import features

REGIONS = {   # region -> the cities whose weather it uses in bu_local
    "PGE-TAC":  ("san_jose", "oakland", "sacramento", "fresno", "bakersfield"),
    "SCE-TAC":  ("los_angeles", "riverside", "santa_ana", "palm_springs"),
    "SDGE-TAC": ("san_diego",),
    "VEA-TAC":  ("palm_springs",),        # rural Nevada desert; Palm Springs is the closest stand-in
}
SEEDS = (0, 1, 2)
RES = exp.RES / "regional"


def shares() -> pd.DataFrame:
    """Each region's share of the gap, refit at the start of every month on the previous 12 months."""
    h = data.hourly()
    act = data.oasis("ACTUAL").reindex(h.index)[list(REGIONS)]
    dam = data.oasis("DAM").reindex(h.index)[list(REGIONS)]
    gap = act.sum(axis=1, min_count=len(REGIONS)) - h.load
    out = {}
    for m in pd.date_range(h.day.min(), h.day.max(), freq="MS"):
        win = (h.day >= m - pd.DateOffset(years=1)) & (h.day < m)
        if win.sum() < 24 * 60:                     # first months: use the first year instead
            win = h.day < h.day.min() + pd.DateOffset(years=1)
        ok = win & gap.notna() & act.notna().all(axis=1) & dam.notna().all(axis=1)
        X = np.c_[np.ones(ok.sum()), gap[ok]]
        slopes = pd.Series({r: np.linalg.lstsq(X, (act[r] - dam[r])[ok], rcond=None)[0][1] for r in REGIONS})
        out[m] = slopes / slopes.sum()
    return pd.DataFrame(out).T


@lru_cache
def regional_demand() -> pd.DataFrame:
    h = data.hourly()
    act = data.oasis("ACTUAL").reindex(h.index)[list(REGIONS)]
    gap = act.sum(axis=1, min_count=len(REGIONS)) - h.load
    month = h.day.dt.to_period("M").dt.to_timestamp()
    sh = shares().reindex(month.to_numpy())
    sh.index = h.index
    return act.sub(sh.mul(gap, axis=0))


def table_for(target: pd.Series, cities: tuple | None) -> pd.DataFrame:
    """The usual feature table with the target, and the load features built from it, swapped in."""
    df = features.build(cities)
    h = data.hourly().copy()
    h["target"] = target
    lf = features.load_features(h, "target")
    df[lf.columns] = lf
    df["load"] = target
    return df


def run(variant: str, region: str, seed: int) -> pd.Series:
    path = RES / f"{variant}_{region}_s{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)["pred"]
    if variant == "direct":
        df = table_for(data.hourly().load, None)
    else:
        df = table_for(regional_demand()[region], REGIONS[region] if variant == "bu_local" else None)
    if variant == "bu_mlp":
        import nn
        fit = nn.mlp(df, dropout=0.2, wd=1e-3, epochs=150)    # same settings as exp "mlp_day"
    else:
        fit = bt.lgbm(bt.feature_cols(df), **exp.RATIO, params=exp.FF5, seed=seed)
    pred = bt.rolling(df, fit, "dev", exp.FROM_2020)
    RES.mkdir(parents=True, exist_ok=True)
    pred.rename("pred").to_frame().to_parquet(path)
    return pred


def jobs() -> list[tuple[str, str, int]]:
    return [("direct", "ALL", s) for s in SEEDS] + [
        (v, r, s) for v in ("bu_state", "bu_local") for r in REGIONS for s in SEEDS] + [
        ("bu_mlp", r, 0) for r in REGIONS]     # the MLP already averages 5 seeds inside


def main() -> None:
    todo = [j for j in jobs() if not (RES / f"{j[0]}_{j[1]}_s{j[2]}.parquet").exists()]
    with ThreadPoolExecutor(3) as pool:      # one process per run, three at a time
        codes = list(pool.map(lambda j: subprocess.call(
            [sys.executable, "-W", "ignore", __file__, "one", j[0], j[1], str(j[2])]), todo))
    assert not any(codes), codes

    y = data.hourly().load
    rows = {}
    for s in SEEDS:
        preds = {"direct": run("direct", "ALL", s)}
        for v in ("bu_state", "bu_local"):
            preds[v] = sum(run(v, r, s) for r in REGIONS)
        preds["avg(direct, bu_local)"] = (preds["direct"] + preds["bu_local"]) / 2
        # score every variant on the same hours (one 2024 hour lacks regional data)
        idx = preds["direct"].index.intersection(preds["bu_local"].dropna().index)
        rows[s] = {k: (p[idx] - y[idx]).abs().mean() for k, p in preds.items()}
    table = pd.DataFrame(rows)
    table["mean"] = table.mean(axis=1)
    print("2024 MAE (MW) by seed, CAISO = 519\n" + table.round(1).to_string())

    mlp_direct = exp.run("mlp_day", "dev")
    mlp_bu = sum(run("bu_mlp", r, 0) for r in REGIONS)
    idx = mlp_direct.index.intersection(mlp_bu.dropna().index)
    for name, p in [("mlp direct", mlp_direct), ("mlp bottom-up", mlp_bu),
                    ("avg of both", (mlp_direct + mlp_bu) / 2)]:
        print(f"{name:14s} {(p[idx] - y[idx]).abs().mean():6.1f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["one"]:
        bt.LGB_PARAMS["num_threads"] = 3
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
