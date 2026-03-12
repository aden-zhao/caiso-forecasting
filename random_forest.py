"""
Random Forest load forecast with walk-forward MAE evaluation.
Uses all dataset features (weather + engineered cyclical time features + load from 1 week ago).

Requires: pip install scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Config
CSV_PATH = "caiso_dataset_20200101_to_20241231.csv"
REGION = "caiso"
TIMESTAMP_COL = "time_utc"
TARGET_COL = "load_mw"

LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336  # 14 days minimum for fitting model

# Load data
df = pd.read_csv(CSV_PATH)

# Keep CAISO only
if "region" in df.columns:
    df = df[df["region"].astype(str).str.lower() == REGION].copy()

# Parse and sort timestamps
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Keep only rows with target present
df = df[df[TARGET_COL].notna()].copy()

# --------------------------
# Time-based feature engineering
# --------------------------
df["hour"] = df[TIMESTAMP_COL].dt.hour
df["day_of_week"] = df[TIMESTAMP_COL].dt.dayofweek
df["month"] = df[TIMESTAMP_COL].dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag feature
df["load_lag_168"] = df[TARGET_COL].shift(LAG_HOURS)

# Features: all except target, timestamp, identifiers, and raw calendar columns
EXCLUDE = [
    TIMESTAMP_COL,
    "time_key",
    "region",
    TARGET_COL,
    "hour",
    "day_of_week",
    "month",
]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Drop rows with missing lag / feature values
df_model = df.dropna(subset=feature_cols + [TARGET_COL]).copy()

X = df_model[feature_cols]
y = df_model[TARGET_COL]

# Verify
assert TARGET_COL not in X.columns, f"{TARGET_COL} must not be a feature (it is the target)"
print(f"Target (y): {TARGET_COL}")
print("Features (X):", list(X.columns))
print("Feature count:", len(feature_cols))
print()

# Walk-forward evaluation
predictions = []
actuals = []
fold_results = []
n = len(df_model)

fold = 1558
for test_start in range(MIN_TRAIN_SIZE, n, TEST_WINDOW_HOURS):
    if fold > 1600:
        break
    fold += 1
    test_end = min(test_start + TEST_WINDOW_HOURS, n)
    X_train = X.iloc[:test_start]
    y_train = y.iloc[:test_start]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    predictions.extend(preds)
    actuals.extend(y_test.values)

    fold_mae = np.mean(np.abs(preds - y_test.values))
    ts_start = df_model.iloc[test_start][TIMESTAMP_COL].date()
    ts_end = df_model.iloc[test_end - 1][TIMESTAMP_COL].date()
    fold_results.append({"fold": len(fold_results) + 1, "MAE": fold_mae})
    print(f"Fold {len(fold_results)}: {ts_start} → {ts_end}  MAE = {fold_mae:.2f} MW")
    break

# Overall metrics
predictions = np.array(predictions)
actuals = np.array(actuals)
overall_mae = np.mean(np.abs(predictions - actuals))

print("\n" + "=" * 50)
print(f"Overall MAE (random forest): {overall_mae:.2f} MW")
print(f"Total predictions: {len(predictions)}")
print(f"Mean actual load: {np.mean(actuals):.2f} MW")
print(f"MAE as % of mean load: {100 * overall_mae / np.mean(actuals):.2f}%")