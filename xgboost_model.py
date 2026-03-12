"""
XGBoost load forecast with walk-forward MAE evaluation.
Adapted for the new CAISO dataset.

Uses:
- weather features from the dataset
- engineered cyclical time features
- load from 1 week ago

Requires: pip install xgboost
"""

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc

# Config
DATA_PATH = "caiso_dataset_20200101_to_20241231.csv"
LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336  # 14 days minimum for fitting model

# Load data
df = pd.read_csv(DATA_PATH)
df["time_utc"] = pd.to_datetime(df["time_utc"])
df = df.sort_values("time_utc").reset_index(drop=True)

# Keep CAISO rows only, in case the dataset ever contains multiple regions
if "region" in df.columns:
    df = df[df["region"].astype(str).str.lower() == "caiso"].copy()

# Time features
df["hour"] = df["time_utc"].dt.hour
df["dayofweek"] = df["time_utc"].dt.dayofweek
df["month"] = df["time_utc"].dt.month
df["dayofyear"] = df["time_utc"].dt.dayofyear

# Cyclical encodings
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)

# Lag feature
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

# Features: exclude identifiers, timestamps, target, and raw calendar columns
EXCLUDE = [
    "time_utc",
    "time_key",
    "region",
    "load_mw",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Drop rows with missing values (lag row loss + any weather nulls)
df_model = df.dropna(subset=feature_cols + ["load_mw"]).copy()

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

    X_train = X.iloc[:test_start]
    y_train = y.iloc[:test_start]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    predictions.extend(preds)
    actuals.extend(y_test.values)

    fold_mae = np.mean(np.abs(preds - y_test.values))
    ts_start = df_model.iloc[test_start]["time_utc"]
    ts_end = df_model.iloc[test_end - 1]["time_utc"]
    fold_results.append({"fold": len(fold_results) + 1, "MAE": fold_mae})
    print(f"Fold {len(fold_results)}: {ts_start} → {ts_end}  MAE = {fold_mae:.2f} MW")

# Overall metrics
predictions = np.array(predictions)
actuals = np.array(actuals)
overall_mae = np.mean(np.abs(predictions - actuals))

print("\n" + "=" * 50)
print(f"Overall MAE (xgboost): {overall_mae:.2f} MW")
print(f"Total predictions: {len(predictions)}")
print(f"Mean actual load: {np.mean(actuals):.2f} MW")
print(f"MAE as % of mean load: {100 * overall_mae / np.mean(actuals):.2f}%")
