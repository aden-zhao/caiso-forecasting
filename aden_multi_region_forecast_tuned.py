"""
Per-region load forecasting with hyperparameter tuning.

Same dataset / features as `aden_multi_region_forecast.py`, but:
  * Chronological 3-way split per region: train 64% / val 16% / test 20%.
  * Random search on the validation slice:
      - Random Forest: ~12 candidates over n_estimators / depth / leaf / max_features.
      - XGBoost:       ~12 candidates over depth / lr / subsample / colsample / regs,
                       n_estimators chosen by early stopping on the val set.
  * Best model per region is refit on train+val with the chosen hyperparameters
    and evaluated *once* on the held-out test slice.
  * Linear Regression has no hyperparameters to tune; included as the reference
    baseline for comparability with the untuned run.

Outputs (forecast_outputs_tuned/):
    metrics_by_region_model.csv
    best_params.json
    predictions_<region>.csv
    mae_by_region.png, rmse_by_region.png, mape_by_region.png,
    mae_pct_by_region.png, r2_by_region.png
    timeseries_<region>.png, scatter_<region>.png, residuals_<region>.png
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
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("filtered.csv")
OUT_DIR = Path("forecast_outputs_tuned")
OUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "load_mw"
TIMESTAMP_COL = "time_utc"
REGION_COL = "region"
NON_FEATURE_COLS = {TIMESTAMP_COL, "time_key", REGION_COL, TARGET_COL}

TRAIN_FRAC = 0.64
VAL_FRAC = 0.16
TEST_FRAC = 0.20
N_RANDOM_SEARCH = 12  # candidates per model per region
RANDOM_SEED = 42

RF_SPACE = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [10, 15, 20, 25, None],
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE_pct": mape,
        "R2": float(r2),
        "mean_actual": float(np.mean(y_true)),
        "MAE_pct_of_mean": float(mae / np.mean(y_true) * 100) if np.mean(y_true) else float("nan"),
    }


def split_region_3way(df: pd.DataFrame, region: str):
    sub = df[df[REGION_COL].astype(str).str.lower() == region].copy()
    sub[TIMESTAMP_COL] = pd.to_datetime(sub[TIMESTAMP_COL])
    sub = sub.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    sub = sub[sub[TARGET_COL].notna()]
    feature_cols = [c for c in sub.columns if c not in NON_FEATURE_COLS]
    sub = sub.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

    n = len(sub)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = sub.iloc[:n_train].reset_index(drop=True)
    val = sub.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = sub.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test, feature_cols


def sample_params(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def tune_rf(X_train, y_train, X_val, y_val, rng: random.Random):
    seen = set()
    best = (float("inf"), None)
    for _ in range(N_RANDOM_SEARCH):
        for _ in range(20):
            params = sample_params(RF_SPACE, rng)
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                break
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        score = mean_absolute_error(y_val, model.predict(X_val))
        if score < best[0]:
            best = (score, params)
    return best


def tune_xgb(X_train, y_train, X_val, y_val, rng: random.Random):
    seen = set()
    best = (float("inf"), None, None)  # (val_mae, params, best_n_estimators)
    for _ in range(N_RANDOM_SEARCH):
        for _ in range(20):
            params = sample_params(XGB_SPACE, rng)
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                break
        model = XGBRegressor(
            **params,
            n_estimators=2000,
            early_stopping_rounds=30,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        n_best = int(model.best_iteration) + 1 if model.best_iteration is not None else 2000
        score = mean_absolute_error(y_val, model.predict(X_val))
        if score < best[0]:
            best = (score, params, n_best)
    return best


# ---------------------------------------------------------------------------
# Plotting (same shapes as untuned script)
# ---------------------------------------------------------------------------
def plot_metric_by_region(metrics_df, metric, ylabel, path, log_y=False):
    pivot = metrics_df.pivot(index="region", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 6), edgecolor="black", logy=log_y)
    ax.set_title(f"{ylabel} by Region and Model (tuned)")
    ax.set_ylabel(ylabel + (" (log scale)" if log_y else ""))
    ax.set_xlabel("Region")
    ax.grid(axis="y", alpha=0.3, which="both" if log_y else "major")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_timeseries(region, test_df, preds, path):
    sample = min(len(test_df), 24 * 14)
    times = test_df[TIMESTAMP_COL].iloc[:sample]
    actual = test_df[TARGET_COL].iloc[:sample].to_numpy()
    plt.figure(figsize=(13, 5))
    plt.plot(times, actual, label="Actual", color="black", linewidth=1.6)
    for name, y_pred in preds.items():
        plt.plot(times, y_pred[:sample], label=name, alpha=0.85, linewidth=1.1)
    plt.title(f"{region.upper()} – Actual vs. Predicted Load (tuned, first 2 test weeks)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scatter(region, actual, preds, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 5), sharex=True, sharey=True)
    if len(preds) == 1:
        axes = [axes]
    lo, hi = float(actual.min()), float(actual.max())
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ax.scatter(actual, y_pred, alpha=0.25, s=8)
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual Load (MW)")
        ax.set_ylabel("Predicted Load (MW)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{region.upper()} – Predicted vs. Actual (tuned)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_residuals(region, actual, preds, path):
    plt.figure(figsize=(10, 5))
    for name, y_pred in preds.items():
        plt.hist(y_pred - actual, bins=60, alpha=0.45, label=name, edgecolor="black", linewidth=0.3)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"{region.upper()} – Residuals (predicted − actual, tuned)")
    plt.xlabel("Residual (MW)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    regions = sorted(df[REGION_COL].astype(str).str.lower().unique())
    print(f"Regions found: {regions}")

    metrics_rows: list[dict] = []
    best_params_all: dict = {}

    for region in regions:
        train, val, test, feature_cols = split_region_3way(df, region)
        if len(train) < 500 or len(val) < 100 or len(test) < 50:
            print(f"\n[{region}] skipping – too few rows (train={len(train)}, val={len(val)}, test={len(test)}).")
            continue

        print(f"\n=== {region.upper()} ===")
        print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_cols)}")
        print(f"test span: {test[TIMESTAMP_COL].iloc[0]} -> {test[TIMESTAMP_COL].iloc[-1]}")

        train_mean = float(train[TARGET_COL].mean())
        test_mean = float(test[TARGET_COL].mean())
        if train_mean > 0 and abs(test_mean - train_mean) / train_mean > 0.30:
            print(
                f"  WARNING: large train/test load drift "
                f"(train mean={train_mean:.1f} MW, test mean={test_mean:.1f} MW)."
            )

        X_train, y_train = train[feature_cols], train[TARGET_COL].to_numpy()
        X_val, y_val = val[feature_cols], val[TARGET_COL].to_numpy()
        X_test, y_test = test[feature_cols], test[TARGET_COL].to_numpy()
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = np.concatenate([y_train, y_val])

        rng = random.Random(RANDOM_SEED + abs(hash(region)) % 1000)
        preds_by_model: dict[str, np.ndarray] = {}
        region_best: dict[str, dict] = {}

        # --- Linear Regression (no tuning) ---
        t0 = time.time()
        lr = LinearRegression()
        lr.fit(X_trainval, y_trainval)
        preds_by_model["Linear Regression"] = np.asarray(lr.predict(X_test), dtype=float)
        region_best["Linear Regression"] = {"params": {}, "val_mae": None, "fit_seconds": round(time.time() - t0, 1)}

        # --- Random Forest tuning ---
        print("  tuning Random Forest ...", flush=True)
        t0 = time.time()
        rf_val_mae, rf_params = tune_rf(X_train, y_train, X_val, y_val, rng)
        rf_final = RandomForestRegressor(**rf_params, n_jobs=-1, random_state=42)
        rf_final.fit(X_trainval, y_trainval)
        preds_by_model["Random Forest"] = np.asarray(rf_final.predict(X_test), dtype=float)
        region_best["Random Forest"] = {
            "params": rf_params,
            "val_mae": float(rf_val_mae),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"    best val MAE={rf_val_mae:.2f}  params={rf_params}")

        # --- XGBoost tuning (early stopping for n_estimators) ---
        print("  tuning XGBoost ...", flush=True)
        t0 = time.time()
        xgb_val_mae, xgb_params, xgb_n_est = tune_xgb(X_train, y_train, X_val, y_val, rng)
        xgb_final = XGBRegressor(
            **xgb_params,
            n_estimators=xgb_n_est,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_final.fit(X_trainval, y_trainval)
        preds_by_model["XGBoost"] = np.asarray(xgb_final.predict(X_test), dtype=float)
        region_best["XGBoost"] = {
            "params": {**xgb_params, "n_estimators": xgb_n_est},
            "val_mae": float(xgb_val_mae),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"    best val MAE={xgb_val_mae:.2f}  n_estimators={xgb_n_est}  params={xgb_params}")

        # --- Test-set metrics ---
        for name, y_pred in preds_by_model.items():
            m = compute_metrics(y_test, y_pred)
            metrics_rows.append({"region": region, "model": name, **m})
            print(f"  {name:<18} TEST  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE_pct']:.2f}%  R²={m['R2']:.3f}")

        best_params_all[region] = region_best

        pred_df = pd.DataFrame({TIMESTAMP_COL: test[TIMESTAMP_COL].to_numpy(), "actual": y_test})
        for name, y_pred in preds_by_model.items():
            pred_df[name] = y_pred
        pred_df.to_csv(OUT_DIR / f"predictions_{region}.csv", index=False)

        plot_timeseries(region, test, preds_by_model, OUT_DIR / f"timeseries_{region}.png")
        plot_scatter(region, y_test, preds_by_model, OUT_DIR / f"scatter_{region}.png")
        plot_residuals(region, y_test, preds_by_model, OUT_DIR / f"residuals_{region}.png")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUT_DIR / "metrics_by_region_model.csv"
    metrics_df.to_csv(metrics_path, index=False)
    with (OUT_DIR / "best_params.json").open("w") as f:
        json.dump(best_params_all, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Tuned metrics summary (sorted by region, then MAE):")
    print(metrics_df.sort_values(["region", "MAE"]).to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.resolve()}")
    print(f"Saved best params to: {(OUT_DIR / 'best_params.json').resolve()}")

    if not metrics_df.empty:
        plot_metric_by_region(metrics_df, "MAE", "MAE (MW)", OUT_DIR / "mae_by_region.png", log_y=True)
        plot_metric_by_region(metrics_df, "RMSE", "RMSE (MW)", OUT_DIR / "rmse_by_region.png", log_y=True)
        plot_metric_by_region(metrics_df, "MAPE_pct", "MAPE (%)", OUT_DIR / "mape_by_region.png")
        plot_metric_by_region(metrics_df, "MAE_pct_of_mean", "MAE / mean load (%)", OUT_DIR / "mae_pct_by_region.png")
        plot_metric_by_region(metrics_df, "R2", "R²", OUT_DIR / "r2_by_region.png")

    print(f"All outputs written under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
