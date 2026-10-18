"""Leakage test: corrupting load after a cutoff must not change any feature of a
day whose issue time is before the cutoff. Run: python test_leakage.py"""
import numpy as np
import pandas as pd

import data
import features
from data import LAST_KNOWN_HOUR


def main() -> None:
    h = data.hourly().copy()
    base = features.load_features(h)
    rng = np.random.default_rng(0)
    for cutoff in pd.to_datetime(["2023-05-17 16:00", "2024-11-03 17:00", "2025-03-09 10:00"], utc=True):
        # load intervals starting after LAST_KNOWN_HOUR on the issue day are unknown at issue
        corrupt = h.copy()
        unknown = corrupt.index >= cutoff - pd.Timedelta(hours=1)
        corrupt.loc[unknown, "load"] = rng.uniform(1e4, 5e4, unknown.sum())
        f = features.load_features(corrupt)
        known = h.issue_utc <= cutoff
        diff = (f[known] - base[known]).abs().max()
        assert (diff.fillna(0) == 0).all(), diff[diff > 0]
        # and the check has teeth: features of later days do change
        assert (f[~known] - base[~known]).abs().max().max() > 0
        print("ok", cutoff)
    # the last load hour used is LAST_KNOWN_HOUR on D-1, i.e. before the 09:00 issue
    assert LAST_KNOWN_HOUR + 1 <= data.ISSUE_HOUR
    print("all leakage checks passed")


if __name__ == "__main__":
    main()
