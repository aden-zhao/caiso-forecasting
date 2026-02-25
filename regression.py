"""
Ridge and Lasso regression load forecast with walk-forward MAE evaluation.
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
df = pd.read_csv("unified_load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Create lag feature
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

# Features
EXCLUDE = ["timestamp", "load_mw", "market", "region"]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

df_model = df.loc[LAG_HOURS:].copy()

X = df_model[feature_cols]
y = df_model["load_mw"]

assert "load_mw" not in X.columns

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

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    print(f"{model_name} Overall MAE: {mae:.2f} MW")
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