"""
CAISO-only classical-model run with the SAME dataset and split as the
deep-learning models (ST-CALNet, LSTM, Bi-LSTM) so the absolute numbers
can be put side-by-side in the presentation.

Mirrors st_calnet_caiso.py:
  * Dataset:  caiso_dataset_20200101_to_20241231.csv (CAISO only)
  * Split:    70 / 15 / 15 chronological (TRAIN / VAL / TEST)
  * Target:   load_mw
  * Features: all numeric weather columns + cyclical time features
              (hour/dow/month sin+cos) + autoregressive lags of load_mw
              at t-1 .. t-24 (matching SEQ_LEN=24 in ST-CALNet) and t-168
              (one week back). The lag block is the regression-friendly
              equivalent of the sequence ST-CALNet's encoder sees.
  * Tuning:   small random search on the val slice, refit on train+val,
              evaluate ONCE on the held-out test slice.

Outputs (forecast_outputs_caiso_compare/):
  - metrics_caiso_compare.csv
  - predictions_caiso.csv
  - timeseries_caiso.png
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Config — locked to ST-CALNet / LSTM split
# ---------------------------------------------------------------------------
CSV_PATH = Path("caiso_dataset_20200101_to_20241231.csv")
REGION = "caiso"
TARGET_COL = "load_mw"
TIMESTAMP_COL = "time_utc"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # TEST_FRAC = 0.15 (remainder)

LAG_HOURS = list(range(1, 25)) + [168]
N_RANDOM_SEARCH = 12
RANDOM_SEED = 42

OUT_DIR = Path("forecast_outputs_caiso_compare")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0)
    return df


def load_caiso(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    df = df[df["region"].str.lower() == REGION].copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df = df[df[TARGET_COL].notna()].copy()

    df = add_cyclical_time_features(df)

    for k in LAG_HOURS:
        df[f"load_lag_{k}"] = df[TARGET_COL].shift(k)

    df = df.dropna().reset_index(drop=True)

    exclude = {"region", "time_key", TIMESTAMP_COL, TARGET_COL}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in exclude and df[c].nunique(dropna=False) > 1]

    return df, feature_cols


def chrono_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
        "R2": r2,
        "mean_actual": float(np.mean(y_true)),
        "MAE_pct_of_mean": float(mae / np.mean(y_true) * 100) if np.mean(y_true) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
RF_SPACE = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 0.7],
}

XGB_SPACE = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}


def sample_params(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def tune_rf(X_tr, y_tr, X_val, y_val, rng):
    seen, best = set(), (float("inf"), None)
    for _ in range(N_RANDOM_SEARCH):
        for _ in range(20):
            params = sample_params(RF_SPACE, rng)
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                break
        m = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        m.fit(X_tr, y_tr)
        score = mean_absolute_error(y_val, m.predict(X_val))
        if score < best[0]:
            best = (score, params)
    return best


def tune_xgb(X_tr, y_tr, X_val, y_val, rng):
    seen, best = set(), (float("inf"), None, None)
    for _ in range(N_RANDOM_SEARCH):
        for _ in range(20):
            params = sample_params(XGB_SPACE, rng)
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                break
        m = XGBRegressor(
            **params,
            n_estimators=2000,
            early_stopping_rounds=30,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        n_best = int(m.best_iteration) + 1 if m.best_iteration is not None else 2000
        score = mean_absolute_error(y_val, m.predict(X_val))
        if score < best[0]:
            best = (score, params, n_best)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {CSV_PATH} ...")
    df, feature_cols = load_caiso(CSV_PATH)
    print(f"CAISO rows after lag drop: {len(df):,}   features: {len(feature_cols)}")

    train, val, test = chrono_split(df)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"test window: {test[TIMESTAMP_COL].iloc[0]}  ->  {test[TIMESTAMP_COL].iloc[-1]}")

    X_tr, y_tr = train[feature_cols], train[TARGET_COL].to_numpy()
    X_val, y_val = val[feature_cols], val[TARGET_COL].to_numpy()
    X_te, y_te = test[feature_cols], test[TARGET_COL].to_numpy()
    X_trval = pd.concat([X_tr, X_val], ignore_index=True)
    y_trval = np.concatenate([y_tr, y_val])

    rng = random.Random(RANDOM_SEED)
    metrics_rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    best_params: dict = {}

    # Linear Regression
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_trval, y_trval)
    preds["Linear Regression"] = np.asarray(lr.predict(X_te), dtype=float)
    best_params["Linear Regression"] = {"params": {}, "fit_seconds": round(time.time() - t0, 1)}

    # Random Forest
    print("Tuning Random Forest ...", flush=True)
    t0 = time.time()
    rf_val_mae, rf_params = tune_rf(X_tr, y_tr, X_val, y_val, rng)
    rf = RandomForestRegressor(**rf_params, n_jobs=-1, random_state=42)
    rf.fit(X_trval, y_trval)
    preds["Random Forest"] = np.asarray(rf.predict(X_te), dtype=float)
    best_params["Random Forest"] = {
        "params": rf_params, "val_mae": rf_val_mae, "fit_seconds": round(time.time() - t0, 1),
    }
    print(f"  best val MAE={rf_val_mae:.2f}  params={rf_params}")

    # XGBoost
    print("Tuning XGBoost ...", flush=True)
    t0 = time.time()
    xgb_val_mae, xgb_params, xgb_n_est = tune_xgb(X_tr, y_tr, X_val, y_val, rng)
    xgb = XGBRegressor(
        **xgb_params, n_estimators=xgb_n_est,
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_trval, y_trval)
    preds["XGBoost"] = np.asarray(xgb.predict(X_te), dtype=float)
    best_params["XGBoost"] = {
        "params": {**xgb_params, "n_estimators": xgb_n_est},
        "val_mae": xgb_val_mae, "fit_seconds": round(time.time() - t0, 1),
    }
    print(f"  best val MAE={xgb_val_mae:.2f}  n_est={xgb_n_est}  params={xgb_params}")

    # Test-set metrics
    for name, y_pred in preds.items():
        m = compute_metrics(y_te, y_pred)
        metrics_rows.append({"model": name, **m})
        print(f"{name:<18} TEST  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"MAPE={m['MAPE_pct']:.2f}%  R²={m['R2']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_caiso_compare.csv", index=False)
    with (OUT_DIR / "best_params.json").open("w") as f:
        json.dump(best_params, f, indent=2, default=str)

    pred_df = pd.DataFrame({TIMESTAMP_COL: test[TIMESTAMP_COL].to_numpy(), "actual": y_te})
    for name, y_pred in preds.items():
        pred_df[name] = y_pred
    pred_df.to_csv(OUT_DIR / "predictions_caiso.csv", index=False)

    sample = min(len(test), 24 * 14)
    plt.figure(figsize=(13, 5))
    plt.plot(test[TIMESTAMP_COL].iloc[:sample], y_te[:sample], label="Actual", color="black", linewidth=1.6)
    for name, y_pred in preds.items():
        plt.plot(test[TIMESTAMP_COL].iloc[:sample], y_pred[:sample], label=name, alpha=0.85, linewidth=1.1)
    plt.title("CAISO – Classical Models, first 2 test weeks (matched DL split)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "timeseries_caiso.png", dpi=200)
    plt.close()

    print(f"\nSaved metrics to: {(OUT_DIR / 'metrics_caiso_compare.csv').resolve()}")


if __name__ == "__main__":
    main()
