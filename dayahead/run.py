"""Score baselines and models on a period. Usage: python run.py [dev|test]"""
import sys
from pathlib import Path

import pandas as pd

import backtest as bt
import features

OUT = Path(__file__).parent / "results"


def main(period: str = "dev") -> None:
    df = features.build()
    feats = bt.feature_cols(df)
    methods = {
        "CAISO DAM": bt.column("caiso_dam"),
        "naive: same hour D-7": bt.column("lag7"),
        "naive: same hour D-2": bt.column("lag2"),
        "lgbm raw": bt.lgbm(feats, target="raw"),
        "lgbm ratio": bt.lgbm(feats, target="ratio"),
    }
    if "caiso_2da" in df:
        methods["CAISO 2DA"] = bt.column("caiso_2da")
    preds = {}
    for name, fn in methods.items():
        preds[name] = bt.rolling(df, fn, period)
        print(f"done {name}", flush=True)
    table = bt.score(df, preds, period)
    print(table.round(2).to_string())
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(preds).assign(load=df.load).to_parquet(OUT / f"preds_{period}.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:])
