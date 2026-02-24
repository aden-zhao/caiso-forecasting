"""
Linear regression load forecast with walk-forward MAE evaluation.
Uses all dataset features (time, weather, cyclical) + load from 1 week ago.

Requires: pip install scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Config (match baseline)
LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336  # 14 days minimum for fitting regression

# Load data
df = pd.read_csv("unified_load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Create lag feature
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

# Features: all except target, timestamp, and constant columns (market, region)
EXCLUDE = ["timestamp", "load_mw", "market", "region"]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Drop rows without lag (first 168 rows)
df_model = df.loc[LAG_HOURS:].copy()

X = df_model[feature_cols]
y = df_model["load_mw"]

# Verify: show features and target
assert "load_mw" not in X.columns, "load_mw must not be a feature (it is the target)"
print("Target (y): load_mw")
print("Features (X):", list(X.columns))
print("Feature count:", len(feature_cols))
print()

# Walk-forward evaluation
predictions = []
actuals = []
fold_results = []
n = len(df_model)

for test_start in range(MIN_TRAIN_SIZE, n, TEST_WINDOW_HOURS):
    test_end = min(test_start + TEST_WINDOW_HOURS, n)
    train_end = test_start

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    predictions.extend(preds)
    actuals.extend(y_test.values)

    fold_mae = np.mean(np.abs(preds - y_test.values))
    ts_start = df_model.iloc[test_start]["timestamp"].date()
    ts_end = df_model.iloc[test_end - 1]["timestamp"].date()
    fold_results.append({"fold": len(fold_results) + 1, "MAE": fold_mae})
    print(f"Fold {len(fold_results)}: {ts_start} → {ts_end}  MAE = {fold_mae:.2f} MW")

# Overall metrics
predictions = np.array(predictions)
actuals = np.array(actuals)
overall_mae = np.mean(np.abs(predictions - actuals))

print("\n" + "=" * 50)
print(f"Overall MAE (linear regression): {overall_mae:.2f} MW")
print(f"Total predictions: {len(predictions)}")
print(f"Mean actual load: {np.mean(actuals):.2f} MW")
print(f"MAE as % of mean load: {100 * overall_mae / np.mean(actuals):.2f}%")
