"""
Ridge and Lasso regression load forecast with walk-forward MAE evaluation
using the CAISO dataset.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Config
LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336
ALPHA = 1.0  # Regularization strength

# Load data
df = pd.read_csv("caiso_dataset_20200101_to_20241231.csv")
df["time_utc"] = pd.to_datetime(df["time_utc"])
df = df.sort_values("time_utc").reset_index(drop=True)

# Keep CAISO only, just in case
df = df[df["region"].astype(str).str.lower() == "caiso"].copy()

# --------------------------
# Time-based feature engineering
# --------------------------
df["hour"] = df["time_utc"].dt.hour
df["day_of_week"] = df["time_utc"].dt.dayofweek
df["month"] = df["time_utc"].dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag feature
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

# Features
EXCLUDE = [
    "time_utc",
    "time_key",
    "region",
    "load_mw",
    "hour",
    "day_of_week",
    "month",
]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Drop rows with missing lag / feature values
df_model = df.dropna(subset=feature_cols + ["load_mw"]).copy()

X = df_model[feature_cols]
y = df_model["load_mw"]

assert "load_mw" not in X.columns

print("Target (y): load_mw")
print("Features (X):", list(X.columns))
print("Feature count:", len(feature_cols))
print()

# ==========================
# WALK FORWARD FUNCTION
# ==========================

def walk_forward(model_name, model_builder):
    predictions = []
    actuals = []
    n = len(df_model)

    print(f"\nRunning {model_name}...")

    for test_start in range(MIN_TRAIN_SIZE, n, TEST_WINDOW_HOURS):
        test_end = min(test_start + TEST_WINDOW_HOURS, n)
        train_end = test_start

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = model_builder()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        predictions.extend(preds)
        actuals.extend(y_test.values)

        fold_mae = np.mean(np.abs(preds - y_test.values))
        ts_start = df_model.iloc[test_start]["time_utc"].date()
        ts_end = df_model.iloc[test_end - 1]["time_utc"].date()
        print(f"Fold {(test_start - MIN_TRAIN_SIZE) // TEST_WINDOW_HOURS + 1}: {ts_start} → {ts_end}  MAE = {fold_mae:.2f} MW")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    print("\n" + "=" * 50)
    print(f"{model_name} Overall MAE: {mae:.2f} MW")
    print(f"Total predictions: {len(predictions)}")
    print(f"Mean actual load: {np.mean(actuals):.2f} MW")
    print(f"MAE as % of mean load: {100 * mae / np.mean(actuals):.2f}%")

    return mae


# ==========================
# Ridge Regression
# ==========================

ridge_mae = walk_forward(
    "Ridge",
    lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA))
    ])
)

# ==========================
# Lasso Regression
# ==========================

lasso_mae = walk_forward(
    "Lasso",
    lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=ALPHA, max_iter=10000))
    ])
)

print("\nComparison Complete.")