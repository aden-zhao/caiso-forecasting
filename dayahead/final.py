"""Final comparison with CAISO, configs frozen on the 2024 dev year.

    python final.py dev|test

A: independent forecast (no CAISO inputs): mean of MLP, LightGBM, per-hour LightGBM.
B: market participant at ~09:45 who also sees CAISO's day-ahead forecast:
   mean of CAISO DAM, A, and LightGBM correcting CAISO DAM.
Uncertainty: 95% interval on the MAE difference vs CAISO from a bootstrap over days.
"""
import sys

import numpy as np
import pandas as pd

import exp
import backtest as bt

A_PARTS = ["mlp_reg_e150", "a_from2020", "a_perhour"]
B_EXTRA = "ondam_diff"


def bootstrap_gap(y, p, ref, days, n=2000, seed=0):
    """95% CI of MAE(p) - MAE(ref), resampling whole days."""
    d = pd.DataFrame({"g": (p - y).abs() - (ref - y).abs(), "day": days.to_numpy()}).groupby("day").g
    s, c = d.sum().to_numpy(), d.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    stats = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return np.percentile(stats, [2.5, 97.5])


def main(period: str) -> None:
    df = exp.table()
    P = {n: exp.run(n, period) for n in A_PARTS + [B_EXTRA]}
    A = sum(P[n] for n in A_PARTS) / len(A_PARTS)
    c = df.caiso_dam.reindex(A.index)
    B = (c + A + P[B_EXTRA]) / 3
    y, days = df.load.reindex(A.index), df.day.reindex(A.index)
    preds = {"CAISO DAM": c, "A: independent": A, "B: with CAISO": B,
             **{f"  part {n}": P[n] for n in A_PARTS + [B_EXTRA]}}
    print(bt.score(df, preds, period).round(2).to_string())
    for name, p in [("A", A), ("B", B)]:
        lo, hi = bootstrap_gap(y, p, c, days)
        print(f"{name} - CAISO MAE: {(p - y).abs().mean() - (c - y).abs().mean():+.1f} MW "
              f"(95% CI {lo:+.1f} .. {hi:+.1f})")
    # by year, and on each day's peak hour (what matters most for reliability and prices)
    e = pd.DataFrame({k: (v - y).abs() for k, v in [("CAISO", c), ("A", A), ("B", B)]})
    print(e.groupby(days.dt.year).mean().round(0).to_string())
    peak = y.groupby(days).idxmax()
    print("daily-peak-hour MAE:", e.loc[peak].mean().round(0).to_dict())
    # share of days each forecast beats CAISO
    dm = e.groupby(days).mean()
    print("days beating CAISO:", {k: f"{(dm[k] < dm.CAISO).mean():.0%}" for k in ["A", "B"]})


if __name__ == "__main__":
    main(*sys.argv[1:])
