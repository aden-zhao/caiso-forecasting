"""
Feature importance analysis for load forecasting model.
1. Standardized coefficients - comparable magnitude when features are scaled
2. Permutation importance - how much MAE worsens when feature is shuffled
3. Ablation - how much MAE worsens when feature is removed
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Config (match linear_regression.py)
LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336

# Load data
ROOT = Path(__file__).resolve().parents[1]   # hourahead/
df = pd.read_csv(ROOT / "data" / "unified_load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

EXCLUDE = ["timestamp", "load_mw", "market", "region"]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

df_model = df.loc[LAG_HOURS:].copy()
X_full = df_model[feature_cols]
y_full = df_model["load_mw"]
n = len(df_model)


def run_walk_forward_mae(X: pd.DataFrame, y: pd.Series):
    """Run walk-forward eval with scaling, return MAE."""
    predictions = []
    actuals = []
    scaler = StandardScaler()

    for test_start in range(MIN_TRAIN_SIZE, n, TEST_WINDOW_HOURS):
        test_end = min(test_start + TEST_WINDOW_HOURS, n)
        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

        model = LinearRegression()
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        predictions.extend(preds)
        actuals.extend(y_test.values)

    return mean_absolute_error(actuals, predictions)


# ---- 1. Standardized coefficients (train on full period, last fold's train) ----
print("=" * 60)
print("1. STANDARDIZED COEFFICIENTS (higher |coef| = more impact)")
print("=" * 60)

# Use last fold's train set for a single fit (most data)
train_end = (n // TEST_WINDOW_HOURS) * TEST_WINDOW_HOURS
if train_end < MIN_TRAIN_SIZE:
    train_end = MIN_TRAIN_SIZE

X_train = X_full.iloc[:train_end]
y_train = y_full.iloc[:train_end]
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Standardized coefs are comparable; also multiply by std to get "importance" in target units
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef": model.coef_,
    "abs_coef": np.abs(model.coef_),
}).sort_values("abs_coef", ascending=False)

for _, row in coef_df.iterrows():
    sign = "+" if row["coef"] >= 0 else ""
    print(f"  {row['feature']:30s}  {sign}{row['coef']:8.2f}  (|coef|={row['abs_coef']:.2f})")

# ---- 2. Permutation importance ----
print("\n" + "=" * 60)
print("2. PERMUTATION IMPORTANCE (MAE increase when feature shuffled)")
print("=" * 60)

base_mae = run_walk_forward_mae(X_full, y_full)
perm_results = []

for feat in feature_cols:
    X_perm = X_full.copy()
    np.random.seed(42)
    X_perm[feat] = np.random.permutation(X_perm[feat].values)
    mae_perm = run_walk_forward_mae(X_perm, y_full)
    delta = mae_perm - base_mae
    perm_results.append({"feature": feat, "MAE_shuffled": mae_perm, "MAE_increase": delta})

perm_df = pd.DataFrame(perm_results).sort_values("MAE_increase", ascending=False)
for _, row in perm_df.iterrows():
    print(f"  {row['feature']:30s}  +{row['MAE_increase']:7.2f} MW  (MAE: {row['MAE_shuffled']:.2f})")

# ---- 3. Ablation (leave-one-out) ----
print("\n" + "=" * 60)
print("3. ABLATION (MAE when feature REMOVED)")
print("=" * 60)

ablation_results = []

for feat in feature_cols:
    cols = [c for c in feature_cols if c != feat]
    X_abl = X_full[cols]
    mae_abl = run_walk_forward_mae(X_abl, y_full)
    delta = mae_abl - base_mae
    ablation_results.append({"feature": feat, "MAE_no_feat": mae_abl, "MAE_increase": delta})

abl_df = pd.DataFrame(ablation_results).sort_values("MAE_increase", ascending=False)
for _, row in abl_df.iterrows():
    print(f"  {row['feature']:30s}  +{row['MAE_increase']:7.2f} MW  (MAE without: {row['MAE_no_feat']:.2f})")

print("\n" + "=" * 60)
print(f"Baseline MAE (all features): {base_mae:.2f} MW")
print("=" * 60)
