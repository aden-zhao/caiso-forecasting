"""Whole-day neural forecaster: one row per operating day, 24 outputs.

Inputs per day D (all known at issue):
  - load profiles of D-2 .. D-8 and the known morning of D-1, divided by the
    D-1 morning level
  - hourly weather forecasts for D (and D-1), per city and weighted
  - calendar
Output: the 24 local hours of D divided by the same level. A small MLP, an
ensemble over seeds, L1 loss.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from data import LAST_KNOWN_HOUR
from features import day_hour

# Results depend on the CPU thread count (it changes floating-point summation order),
# so it is fixed. With it fixed, runs are bit-for-bit reproducible.
THREADS = 3
HOURLY = ["w_temperature_2m", "t_los_angeles", "t_riverside", "t_santa_ana", "t_san_diego",
          "t_san_jose", "t_oakland", "t_sacramento", "t_fresno", "t_bakersfield", "t_palm_springs",
          "jma_w_cloud_cover", "jma_solar_proxy"]
DAILY = ["dow", "holiday", "holiday_adj", "lag2_offday", "lag7_holiday", "doy_sin", "doy_cos", "t_years"]


def day_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    m = day_hour(df[col], df).reindex(columns=range(24))
    return m.interpolate(axis=1, limit_direction="both")   # fills the spring-forward gap


def day_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """X (days x features), Y (days x 24, scaled), scale (days)."""
    L = day_matrix(df, "load")
    days = L.index
    morning = L.loc[:, :LAST_KNOWN_HOUR]
    scale = morning.mean(axis=1).shift(1, freq="D").reindex(days)     # D-1 morning level
    parts = []
    for k in range(2, 9):
        parts.append(L.shift(k, freq="D").reindex(days).div(scale, axis=0).add_prefix(f"L{k}_"))
    parts.append(morning.shift(1, freq="D").reindex(days).div(scale, axis=0).add_prefix("L1_"))
    for c in HOURLY:
        W = day_matrix(df, c)
        parts.append(W.add_prefix(c + "_"))
        if c == "w_temperature_2m":
            parts.append(W.shift(1, freq="D").reindex(days).add_prefix(c + "_d1_"))
    first = df.groupby("day")[DAILY].first().reindex(days)
    parts.append(pd.get_dummies(first.pop("dow"), prefix="dow").astype(float))
    parts.append(first)
    X = pd.concat(parts, axis=1).reindex(days)
    X.columns = X.columns.astype(str)
    Y = L.div(scale, axis=0)
    return X, Y, scale


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int = 256, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(width, width), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(width, 24))

    def forward(self, x):
        return self.net(x)


def fit_one(X, Y, seed, epochs=300, lr=1e-3, wd=1e-4, width=256, dropout=0.1):
    """Train one MLP with AdamW and a cosine schedule, batch size 64, L1 loss."""
    torch.manual_seed(seed)
    model = MLP(X.shape[1], width, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    n = len(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            loss = (model(X[idx]) - Y[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def mlp(df: pd.DataFrame, seeds: int = 5, **kw):
    """fit_predict for backtest.rolling: an ensemble of `seeds` MLPs (kw go to fit_one).

    df is the full table. day_table only reads inputs known at each day's issue,
    and test days' targets are never used for fitting.
    """
    torch.set_num_threads(THREADS)
    X, Y, scale = day_table(df)

    def fit_predict(train, test):
        tr = X.index.isin(np.unique(train.day)) & X.notna().all(axis=1) & Y.notna().all(axis=1)
        te = X.index.isin(np.unique(test.day))
        # standardise inputs and centre targets with training-day statistics
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        Xs = ((X - mu) / sd).fillna(0).to_numpy(dtype=np.float32)
        ymu = Y[tr].mean().to_numpy()
        models = [fit_one(Xs[tr], Y[tr].to_numpy() - ymu, s, **kw) for s in range(seeds)]
        with torch.no_grad():
            preds = np.mean([m(torch.tensor(Xs[te])).numpy() for m in models], axis=0)
        P = pd.DataFrame(preds + ymu, index=X.index[te], columns=range(24)).mul(scale[te], axis=0)
        # back from day x hour to the test rows (both fall-back hours get the same value)
        return P.stack().reindex(pd.MultiIndex.from_arrays([test.day, test.hour])).to_numpy()

    return fit_predict
