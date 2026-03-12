"""
Baseline: Same-hour-last-week (naive) load forecast with walk-forward MAE evaluation.

Prediction rule:
    load[t] = load[t - 168]   (exactly 1 week ago)

Adapted for:
    caiso_dataset_20200101_to_20241231.csv
"""

import pandas as pd
import numpy as np

# Config
CSV_PATH = "caiso_dataset_20200101_to_20241231.csv"
REGION = "caiso"
TIMESTAMP_COL = "time_utc"
TARGET_COL = "load_mw"

LAG_HOURS = 168           # 1 week
TEST_WINDOW_HOURS = 24
INITIAL_TRAIN_HOURS = 168 # Need 1 week of history before first prediction

# Load data
df = pd.read_csv(CSV_PATH)

# Keep CAISO only
if "region" in df.columns:
    df = df[df["region"].astype(str).str.lower() == REGION].copy()

# Parse and sort timestamps
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Keep only rows with target present
df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)

load = df[TARGET_COL].to_numpy(dtype=float)
n = len(load)

# Walk-forward: start after we have LAG_HOURS of history
predictions = []
actuals = []
fold_results = []

start_idx = LAG_HOURS  # First index we can predict

for test_start in range(start_idx, n, TEST_WINDOW_HOURS):
    test_end = min(test_start + TEST_WINDOW_HOURS, n)

    for i in range(test_start, test_end):
        pred = load[i - LAG_HOURS]  # Load from exactly 168 hours ago
        predictions.append(pred)
        actuals.append(load[i])

    # MAE for this fold
    fold_len = test_end - test_start
    fold_preds = np.array(predictions[-fold_len:])
    fold_actuals = np.array(actuals[-fold_len:])
    fold_mae = np.mean(np.abs(fold_preds - fold_actuals))

    fold_results.append({
        "fold": len(fold_results) + 1,
        "test_start": df.iloc[test_start][TIMESTAMP_COL],
        "test_end": df.iloc[test_end - 1][TIMESTAMP_COL],
        "n_hours": fold_len,
        "MAE": fold_mae,
    })

    print(
        f"Fold {len(fold_results)}: "
        f"{df.iloc[test_start][TIMESTAMP_COL].date()} → "
        f"{df.iloc[test_end - 1][TIMESTAMP_COL].date()}  "
        f"MAE = {fold_mae:.2f} MW"
    )

# Overall MAE
predictions = np.array(predictions)
actuals = np.array(actuals)
overall_mae = np.mean(np.abs(predictions - actuals))

print("\n" + "=" * 50)
print(f"Overall MAE (same-hour-last-week baseline): {overall_mae:.2f} MW")
print(f"Total predictions: {len(predictions)}")
print(f"Mean actual load: {np.mean(actuals):.2f} MW")
print(f"MAE as % of mean load: {100 * overall_mae / np.mean(actuals):.2f}%")