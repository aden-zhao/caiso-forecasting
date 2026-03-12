"""
Compare load forecasting baseline models and graph their performance.

This script uses the current summary metrics collected from experiments and
creates comparison charts for:
- MAE (MW)
- MAE as % of mean load
- Total predictions

It also prints a ranked summary table.

Update the METRICS dictionary below as new model results come in.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Current experiment results
# ============================================================
# Set values to None for models whose results are not available yet.
METRICS = {
    "Baseline (Last Week)": {
        "mae_mw": 1781.47,
        "total_predictions": 43671,
        "mean_actual_load": 25291.40,
        "mae_pct": 7.04,
    },
    "Linear Regression": {
        "mae_mw": 1353.16,
        "total_predictions": 43326,
        "mean_actual_load": 25306.84,
        "mae_pct": 5.35,
    },
    "Lasso": {
        "mae_mw": 1359.07,
        "total_predictions": 43326,
        "mean_actual_load": 25306.84,
        "mae_pct": 5.37,
    },
    "XGBoost": {
        "mae_mw": 894.11,
        "total_predictions": 8400,
        "mean_actual_load": 25265.46,
        "mae_pct": 3.54,
    },
    "Random Forest": {
        "mae_mw": None,
        "total_predictions": None,
        "mean_actual_load": None,
        "mae_pct": None,
    },
    "ST-CALNet": {
        "mae_mw": 595.879,
        "rmse_mw": 824.193,
        "mape_pct": 2.22,
        "r2": 0.975,
        "total_predictions": None,
        "mean_actual_load": None,
        "mae_pct": 2.22,
    },
}

OUTPUT_DIR = Path(".")
BAR_CHART_PATH = OUTPUT_DIR / "model_comparison_mae.png"
PCT_CHART_PATH = OUTPUT_DIR / "model_comparison_mae_pct.png"
PRED_CHART_PATH = OUTPUT_DIR / "model_comparison_predictions.png"
CSV_PATH = OUTPUT_DIR / "model_comparison_summary.csv"


def build_results_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, vals in metrics.items():
        row = {"model": model_name, **vals}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["has_result"] = df["mae_mw"].notna()
    return df


def print_summary(df: pd.DataFrame) -> None:
    available = (
        df[df["has_result"]]
        .sort_values("mae_mw", ascending=True)
        .reset_index(drop=True)
    )
    pending = df[~df["has_result"]]["model"].tolist()

    print("\nModel comparison summary")
    print("=" * 70)
    if not available.empty:
        for i, row in available.iterrows():
            pred_text = (
                str(int(row["total_predictions"]))
                if pd.notna(row["total_predictions"])
                else "N/A"
            )
        mae_pct_text = (
        f"{row['mae_pct']:.2f}%"
        if pd.notna(row["mae_pct"])
        else "N/A"
            )

        print(
            f"{i + 1}. {row['model']}: "
            f"MAE={row['mae_mw']:.2f} MW | "
            f"MAE %={mae_pct_text} | "
            f"Predictions={pred_text}"
        )
    else:
        print("No completed model results found.")

    if pending:
        print("\nPending models:")
        for model in pending:
            print(f"- {model}")

    if len(available) >= 2:
        best = available.iloc[0]
        worst = available.iloc[-1]
        improvement = 100.0 * (worst["mae_mw"] - best["mae_mw"]) / worst["mae_mw"]
        print(
            f"\nBest available model: {best['model']} "
            f"with MAE {best['mae_mw']:.2f} MW."
        )
        print(f"Relative improvement over worst available model: {improvement:.2f}%")


def autolabel(ax, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=9,
        )


def plot_metric(
    df: pd.DataFrame, metric_col: str, ylabel: str, title: str, save_path: Path
) -> None:
    plot_df = df[df[metric_col].notna()].sort_values(metric_col, ascending=True)
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(plot_df["model"], plot_df[metric_col])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    autolabel(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    df = build_results_frame(METRICS)
    df.to_csv(CSV_PATH, index=False)

    print_summary(df)

    print(f"\nSaved summary CSV to: {CSV_PATH.resolve()}")

    plot_metric(
        df,
        metric_col="mae_mw",
        ylabel="MAE (MW)",
        title="Load Forecasting Model Comparison — MAE",
        save_path=BAR_CHART_PATH,
    )
    print(f"Saved MAE chart to: {BAR_CHART_PATH.resolve()}")

    plot_metric(
        df,
        metric_col="mae_pct",
        ylabel="MAE as % of Mean Load",
        title="Load Forecasting Model Comparison — Relative Error",
        save_path=PCT_CHART_PATH,
    )
    print(f"Saved MAE % chart to: {PCT_CHART_PATH.resolve()}")

    plot_metric(
        df,
        metric_col="total_predictions",
        ylabel="Total Predictions",
        title="Load Forecasting Model Comparison — Prediction Count",
        save_path=PRED_CHART_PATH,
    )
    print(f"Saved prediction count chart to: {PRED_CHART_PATH.resolve()}")


if __name__ == "__main__":
    main()
