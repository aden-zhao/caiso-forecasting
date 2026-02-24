"""
Baseline: Same-hour-last-week (naive) load forecast with walk-forward MAE evaluation.

Prediction rule: load[t] = load[t - 168]  (exactly 1 week ago)
"""

import pandas as pd
import numpy as np

# Config
LAG_HOURS = 168  # 1 week
TEST_WINDOW_HOURS = 24
INITIAL_TRAIN_HOURS = 168  # Need 1 week of history before first prediction

# Load data
df = pd.read_csv("unified_load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
load = df["load_mw"].values
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
    fold_preds = np.array(predictions[- (test_end - test_start):])
    fold_actuals = np.array(actuals[- (test_end - test_start):])
    fold_mae = np.mean(np.abs(fold_preds - fold_actuals))
    fold_results.append({
        "fold": len(fold_results) + 1,
        "test_start": df.iloc[test_start]["timestamp"],
        "test_end": df.iloc[test_end - 1]["timestamp"],
        "n_hours": len(fold_preds),
        "MAE": fold_mae,
    })
    print(f"Fold {len(fold_results)}: {df.iloc[test_start]['timestamp'].date()} → {df.iloc[test_end-1]['timestamp'].date()}  MAE = {fold_mae:.2f} MW")

# Overall MAE
predictions = np.array(predictions)
actuals = np.array(actuals)
overall_mae = np.mean(np.abs(predictions - actuals))

print("\n" + "=" * 50)
print(f"Overall MAE (same-hour-last-week baseline): {overall_mae:.2f} MW")
print(f"Total predictions: {len(predictions)}")
print(f"Mean actual load: {np.mean(actuals):.2f} MW")
print(f"MAE as % of mean load: {100 * overall_mae / np.mean(actuals):.2f}%")
