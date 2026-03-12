"""
Tree-based load forecasting models with hyperparameter tuning.

Models:
- Linear Regression
- Random Forest
- XGBoost

Evaluation:
Walk-forward validation (same structure as baseline)
Metric: MAE

Requires:
pip install scikit-learn xgboost
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# ============================
# Config
# ============================

LAG_HOURS = 168
TEST_WINDOW_HOURS = 24
MIN_TRAIN_SIZE = 336


# ============================
# Load dataset
# ============================

df = pd.read_csv("unified_load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# lag feature
df["load_lag_168"] = df["load_mw"].shift(LAG_HOURS)

EXCLUDE = ["timestamp", "load_mw", "market", "region"]
feature_cols = [c for c in df.columns if c not in EXCLUDE]

df_model = df.loc[LAG_HOURS:].copy()

X = df_model[feature_cols]
y = df_model["load_mw"]

print("Features:", feature_cols)
print()


# ============================
# Hyperparameter grids
# ============================

rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

xgb_grid = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1]
}


# ============================
# Walk-forward evaluation
# ============================

def walk_forward(model_type):

    predictions = []
    actuals = []
    n = len(df_model)

    print(f"\nRunning {model_type}...\n")

    for test_start in range(MIN_TRAIN_SIZE, n, TEST_WINDOW_HOURS):

        test_end = min(test_start + TEST_WINDOW_HOURS, n)
        train_end = test_start

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # --------------------
        # Model selection
        # --------------------

        if model_type == "linear":

            model = LinearRegression()
            model.fit(X_train, y_train)

        elif model_type == "rf":

            base_model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

            grid = GridSearchCV(
                base_model,
                rf_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            model = grid.best_estimator_

            print("Best RF params:", grid.best_params_)

        elif model_type == "xgb":

            base_model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42
            )

            grid = GridSearchCV(
                base_model,
                xgb_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            model = grid.best_estimator_

            print("Best XGB params:", grid.best_params_)

        # --------------------
        # Prediction
        # --------------------

        preds = model.predict(X_test)

        predictions.extend(preds)
        actuals.extend(y_test.values)

        fold_mae = mean_absolute_error(y_test, preds)

        ts_start = df_model.iloc[test_start]["timestamp"].date()
        ts_end = df_model.iloc[test_end - 1]["timestamp"].date()

        print(
            f"{model_type.upper()} Fold {len(predictions)//24}: "
            f"{ts_start} → {ts_end} MAE={fold_mae:.2f}"
        )

    overall_mae = mean_absolute_error(actuals, predictions)

    print("\n" + "=" * 60)
    print(f"{model_type.upper()} OVERALL MAE: {overall_mae:.2f}")
    print(f"Mean Load: {np.mean(actuals):.2f}")
    print(f"MAE %: {100*overall_mae/np.mean(actuals):.2f}%")

    return overall_mae


# ============================
# Run models
# ============================

mae_lr = walk_forward("linear")
mae_rf = walk_forward("rf")
mae_xgb = walk_forward("xgb")

print("\nFINAL COMPARISON")
print("---------------------------")
print(f"Linear Regression MAE : {mae_lr:.2f}")
print(f"Random Forest MAE     : {mae_rf:.2f}")
print(f"XGBoost MAE           : {mae_xgb:.2f}")