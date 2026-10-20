"""Bottom-up vs direct: is it better to forecast each region and add them up?

    python regional.py        # 2024 (dev) only

Regional actuals only exist on OASIS's battery-inclusive basis, so both
approaches forecast the same battery-inclusive target: the sum of the four
regions (PG&E + SCE + SDG&E + VEA). Not comparable to CAISO's forecast.
  direct:    one LightGBM on the regional sum
  bottom-up: one LightGBM per region (same settings), predictions added up
Each model sees the same weather and calendar features; only its load history
and target differ.
"""
import sys

import pandas as pd

import backtest as bt
import data
import exp
import features

REGIONS = ["PGE-TAC", "SCE-TAC", "SDGE-TAC", "VEA-TAC"]


def table_for(target: pd.Series) -> pd.DataFrame:
    """The usual feature table, with load features and target rebuilt from `target`."""
    df = exp.table().copy()
    h = data.hourly().copy()
    h["target"] = target
    lf = features.load_features(h, "target")
    df[lf.columns] = lf
    df["load"] = target
    return df


def forecast(target: pd.Series) -> pd.Series:
    df = table_for(target)
    model = exp.EXPS["a_from2020"]
    return bt.rolling(df, model.make(df), "dev", model.train_start)


def main() -> None:
    act = data.oasis("ACTUAL").reindex(data.hourly().index)
    total = act[REGIONS].sum(axis=1, min_count=len(REGIONS))
    parts = {r: forecast(act[r]) for r in REGIONS}
    bottom_up = sum(parts.values())
    direct = forecast(total)
    idx = bottom_up.index.intersection(direct.index)
    y = total[idx]
    rows = {"direct": direct[idx], "bottom-up": bottom_up[idx]}
    for name, p in rows.items():
        print(f"{name:10s} MAE {(p - y).abs().mean():7.1f} MW   MAPE {100 * ((p - y).abs() / y).mean():.2f}%")
    for r, p in parts.items():
        print(f"  {r:9s} MAE {(p[idx] - act[r][idx]).abs().mean():7.1f} MW")
    e_parts = pd.DataFrame({r: parts[r][idx] - act[r][idx] for r in REGIONS})
    print("correlation of regional errors:\n", e_parts.corr().round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
