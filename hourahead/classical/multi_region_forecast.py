"""
Per-region load forecasting on Andrew's `filtered.csv`.

For every region in the dataset, fits Linear Regression, Random Forest, and
XGBoost using the features Andrew engineered (weather + calendar + hour
one-hots + load_previous_week), evaluates on a chronological hold-out, and
emits a metrics table + visualizations.

Outputs (written next to this script):
    outputs/multi_region/metrics_by_region_model.csv
    outputs/multi_region/predictions_<region>.csv
    outputs/multi_region/mae_by_region.png
    outputs/multi_region/mape_by_region.png
    outputs/multi_region/rmse_by_region.png
    outputs/multi_region/r2_by_region.png
    outputs/multi_region/timeseries_<region>.png
    outputs/multi_region/scatter_<region>.png
    outputs/multi_region/residuals_<region>.png

Requires: pandas, numpy, scikit-learn, xgboost, matplotlib
"""

from __future__ import annotations

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
ROOT = Path(__file__).resolve().parents[1]   # hourahead/
DATA_PATH = ROOT / "data" / "filtered.csv"   # not in the repo
OUT_DIR = ROOT / "outputs" / "multi_region"
OUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "load_mw"
TIMESTAMP_COL = "time_utc"
REGION_COL = "region"

# Columns that are not features (identifiers / target).
NON_FEATURE_COLS = {TIMESTAMP_COL, "time_key", REGION_COL, TARGET_COL}

TEST_FRACTION = 0.20  # last 20% of each region (chronological) is held out

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # MAPE: skip rows where y_true is zero to avoid division by zero.
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE_pct": mape,
        "R2": float(r2),
        "mean_actual": float(np.mean(y_true)),
        "MAE_pct_of_mean": float(mae / np.mean(y_true) * 100) if np.mean(y_true) else float("nan"),
    }


def split_region(df: pd.DataFrame, region: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return train_df, test_df, and feature column list for a region."""
    sub = df[df[REGION_COL].astype(str).str.lower() == region].copy()
    sub[TIMESTAMP_COL] = pd.to_datetime(sub[TIMESTAMP_COL])
    sub = sub.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    sub = sub[sub[TARGET_COL].notna()]

    feature_cols = [c for c in sub.columns if c not in NON_FEATURE_COLS]
    sub = sub.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

    n_test = int(len(sub) * TEST_FRACTION)
    train = sub.iloc[:-n_test].reset_index(drop=True)
    test = sub.iloc[-n_test:].reset_index(drop=True)
    return train, test, feature_cols


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_metric_by_region(
    metrics_df: pd.DataFrame, metric: str, ylabel: str, path: Path, log_y: bool = False
) -> None:
    pivot = metrics_df.pivot(index="region", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 6), edgecolor="black", logy=log_y)
    ax.set_title(f"{ylabel} by Region and Model")
    ax.set_ylabel(ylabel + (" (log scale)" if log_y else ""))
    ax.set_xlabel("Region")
    ax.grid(axis="y", alpha=0.3, which="both" if log_y else "major")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_timeseries(region: str, test_df: pd.DataFrame, preds: dict[str, np.ndarray], path: Path) -> None:
    sample = min(len(test_df), 24 * 14)  # show ~2 weeks of hourly data
    times = test_df[TIMESTAMP_COL].iloc[:sample]
    actual = test_df[TARGET_COL].iloc[:sample].to_numpy()

    plt.figure(figsize=(13, 5))
    plt.plot(times, actual, label="Actual", color="black", linewidth=1.6)
    for model_name, y_pred in preds.items():
        plt.plot(times, y_pred[:sample], label=model_name, alpha=0.85, linewidth=1.1)
    plt.title(f"{region.upper()} – Actual vs. Predicted Load (first 2 test weeks)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scatter(region: str, actual: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 5), sharex=True, sharey=True)
    if len(preds) == 1:
        axes = [axes]
    lo, hi = float(actual.min()), float(actual.max())
    for ax, (model_name, y_pred) in zip(axes, preds.items()):
        ax.scatter(actual, y_pred, alpha=0.25, s=8)
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Actual Load (MW)")
        ax.set_ylabel("Predicted Load (MW)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{region.upper()} – Predicted vs. Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_residuals(region: str, actual: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model_name, y_pred in preds.items():
        residuals = y_pred - actual
        plt.hist(residuals, bins=60, alpha=0.45, label=model_name, edgecolor="black", linewidth=0.3)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"{region.upper()} – Residuals (predicted − actual)")
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

    for region in regions:
        train, test, feature_cols = split_region(df, region)
        if len(train) < 500 or len(test) < 50:
            print(f"\n[{region}] skipping – too few rows (train={len(train)}, test={len(test)}).")
            continue

        print(f"\n=== {region.upper()} ===")
        print(f"train rows: {len(train):,}  test rows: {len(test):,}  features: {len(feature_cols)}")
        print(f"test span: {test[TIMESTAMP_COL].iloc[0]} -> {test[TIMESTAMP_COL].iloc[-1]}")

        train_mean = float(train[TARGET_COL].mean())
        test_mean = float(test[TARGET_COL].mean())
        if train_mean > 0 and abs(test_mean - train_mean) / train_mean > 0.30:
            print(
                f"  WARNING: large train/test load drift "
                f"(train mean={train_mean:.1f} MW, test mean={test_mean:.1f} MW). "
                "Hold-out metrics will be pessimistic."
            )

        X_train, y_train = train[feature_cols], train[TARGET_COL].to_numpy()
        X_test, y_test = test[feature_cols], test[TARGET_COL].to_numpy()

        preds_by_model: dict[str, np.ndarray] = {}
        for model_name, factory in MODELS.items():
            model = factory()
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)
            preds_by_model[model_name] = y_pred

            m = compute_metrics(y_test, y_pred)
            metrics_rows.append({"region": region, "model": model_name, **m})
            print(
                f"  {model_name:<18} "
                f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
                f"MAPE={m['MAPE_pct']:.2f}%  R²={m['R2']:.3f}"
            )

        # Per-region prediction CSV (handy for downstream analysis).
        pred_df = pd.DataFrame({TIMESTAMP_COL: test[TIMESTAMP_COL].to_numpy(), "actual": y_test})
        for model_name, y_pred in preds_by_model.items():
            pred_df[model_name] = y_pred
        pred_df.to_csv(OUT_DIR / f"predictions_{region}.csv", index=False)

        plot_timeseries(region, test, preds_by_model, OUT_DIR / f"timeseries_{region}.png")
        plot_scatter(region, y_test, preds_by_model, OUT_DIR / f"scatter_{region}.png")
        plot_residuals(region, y_test, preds_by_model, OUT_DIR / f"residuals_{region}.png")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUT_DIR / "metrics_by_region_model.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 70)
    print("Metrics summary (sorted by region, then MAE):")
    print(metrics_df.sort_values(["region", "MAE"]).to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.resolve()}")

    if not metrics_df.empty:
        # Loads span ~100 MW (VEA) to ~25,000 MW (CAISO) -> log scale on absolute metrics.
        plot_metric_by_region(metrics_df, "MAE", "MAE (MW)", OUT_DIR / "mae_by_region.png", log_y=True)
        plot_metric_by_region(metrics_df, "RMSE", "RMSE (MW)", OUT_DIR / "rmse_by_region.png", log_y=True)
        plot_metric_by_region(metrics_df, "MAPE_pct", "MAPE (%)", OUT_DIR / "mape_by_region.png")
        plot_metric_by_region(
            metrics_df,
            "MAE_pct_of_mean",
            "MAE / mean load (%)",
            OUT_DIR / "mae_pct_by_region.png",
        )
        plot_metric_by_region(metrics_df, "R2", "R²", OUT_DIR / "r2_by_region.png")

    print(f"All outputs written under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
