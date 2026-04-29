"""
Standalone stacked bidirectional LSTM for CAISO hourly load forecasting.

Architecture: BiLSTM (seq_len=168h) → FC head
Ablation relative to ST-CALNet: no Conv1d feature extractor, no self-attention.
Adds bidirectionality and a longer context window (1 week vs 1 day) to test
whether weekly patterns alone can rival the full ST-CALNet architecture.

Outputs (all saved to lstm_caiso_outputs/):
  lstm_caiso.pt                   — model checkpoint + config
  lstm_caiso_test_predictions.csv — per-hour test predictions
  lstm_caiso_training_history.csv — epoch-by-epoch train/val loss
  loss_curve.png                  — training curves
  test_predictions.png            — actual vs predicted (first 14 days of test set)
  scatter.png                     — prediction scatter plot
model_comparison_summary.csv is also updated (project root).
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe for background runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =========================
# Config
# =========================
CSV_PATH = "caiso_dataset_20200101_to_20241231.csv"
REGION = "caiso"
TIMESTAMP_COL = "time_utc"
TARGET_COL = "load_mw"

# 1-week context window: captures daily + weekly seasonality without CNN/attention
SEQ_LEN = 168

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BATCH_SIZE = 128        # larger than ST-CALNet (32) to offset seq_len cost
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 10           # more generous than ST-CALNet's 5
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

HIDDEN_SIZE = 64        # matches ST-CALNet LSTM_HIDDEN for fair comparison
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.20

NUM_WORKERS = 0

# MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

OUT_DIR = Path("lstm_caiso_outputs")
MODEL_OUT = OUT_DIR / "lstm_caiso.pt"


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Dataset utilities — identical to st_calnet_caiso.py for consistency
# =========================
def add_cyclical_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    df["dow_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["dow_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0)
    return df


def load_and_prepare_dataframe(csv_path: str, region: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)

    if "region" in df.columns:
        df = df[df["region"].str.lower() == region.lower()].copy()

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df = df[df[TARGET_COL].notna()].copy()

    df = add_cyclical_time_features(df, TIMESTAMP_COL)

    exclude = {"region", "time_key", TIMESTAMP_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if TARGET_COL not in feature_cols:
        feature_cols.append(TARGET_COL)

    feature_cols = [c for c in feature_cols if df[c].nunique(dropna=False) > 1]

    return df.reset_index(drop=True), feature_cols


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


@dataclass
class MinMaxScalerFrame:
    min_: pd.Series
    max_: pd.Series

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        out = df.copy()
        denom = (self.max_[columns] - self.min_[columns]).replace(0, 1.0)
        out.loc[:, columns] = (out[columns] - self.min_[columns]) / denom
        return out

    def inverse_transform_target(self, values: np.ndarray, target_col: str) -> np.ndarray:
        return values * (self.max_[target_col] - self.min_[target_col]) + self.min_[target_col]


def fit_minmax_scaler(train_df: pd.DataFrame, columns: List[str]) -> MinMaxScalerFrame:
    return MinMaxScalerFrame(min_=train_df[columns].min(), max_=train_df[columns].max())


class SequenceDataset(Dataset):
    """
    Lazy sliding-window dataset. Builds windows on __getitem__ instead of
    pre-allocating them, so seq_len=168 doesn't require ~2 GB of RAM upfront.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
    ) -> None:
        df = df.reset_index(drop=True)
        self.values     = df[feature_cols].to_numpy(dtype=np.float32)
        self.target_idx = feature_cols.index(target_col)
        self.timestamps = df[TIMESTAMP_COL].astype(str).to_numpy()
        self.seq_len    = seq_len
        self._len       = len(self.values) - seq_len

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        x  = self.values[idx : idx + self.seq_len]          # [seq_len, features]
        y  = self.values[idx + self.seq_len, self.target_idx]  # scalar
        ts = self.timestamps[idx + self.seq_len]
        return x, y, ts


# =========================
# Model
# =========================
class LSTMForecaster(nn.Module):
    """
    Stacked bidirectional LSTM → FC head.

    Uses the last-layer final hidden states from both directions as the
    sequence summary, then passes through a two-layer FC head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # inter-layer dropout (PyTorch ignores this for single-layer LSTMs)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_size]
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers * num_directions, B, hidden_size]
        # Concatenate forward (h_n[-2]) and backward (h_n[-1]) last-layer states
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, hidden*2]
        else:
            h = h_n[-1]                                  # [B, hidden]
        return self.head(h).squeeze(-1)                  # [B]


# =========================
# Training / evaluation
# =========================
def make_loader(dataset: SequenceDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "MAPE": mape(y_true, y_pred),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)

    losses = []
    for xb, yb, _ in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        preds = model(xb)
        loss  = criterion(preds, yb)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    scaler: MinMaxScalerFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_all, y_all, ts_all = [], [], []

    for xb, yb, tsb in loader:
        xb = xb.to(DEVICE)
        preds = model(xb).detach().cpu().numpy()
        y_np  = yb.numpy()

        preds_all.append(scaler.inverse_transform_target(preds, target_col))
        y_all.append(scaler.inverse_transform_target(y_np, target_col))
        ts_all.append(np.asarray(tsb))

    return (
        np.concatenate(preds_all),
        np.concatenate(y_all),
        np.concatenate(ts_all),
    )


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"{title}:")
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.3f}")
    print()


# =========================
# Plots (saved to files — no plt.show() so script runs unattended)
# =========================
def plot_loss_curves(train_losses: List[float], val_losses: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title("LSTM Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)


def plot_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_days: int = 14,
) -> None:
    n = min(n_days * 24, len(timestamps))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pd.to_datetime(timestamps[:n]), y_true[:n], label="Actual Load",    linewidth=2)
    ax.plot(pd.to_datetime(timestamps[:n]), y_pred[:n], label="Predicted Load", alpha=0.8)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"LSTM Test Set: Actual vs Predicted (first {n_days} days)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "test_predictions.png", dpi=150)
    plt.close(fig)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual Load (MW)")
    ax.set_ylabel("Predicted Load (MW)")
    ax.set_title("LSTM Prediction Scatter")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter.png", dpi=150)
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    set_seed(RANDOM_SEED)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset: {CSV_PATH}\n")

    df, feature_cols = load_and_prepare_dataframe(CSV_PATH, REGION)
    print(f"Rows after filter: {len(df):,}")
    print(f"Timestamp range:   {df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()}")
    print(f"Feature count:     {len(feature_cols)}")
    print()

    train_df, val_df, test_df = chronological_split(df, TRAIN_FRAC, VAL_FRAC)
    print(f"Train rows: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print()

    scaler      = fit_minmax_scaler(train_df, feature_cols)
    train_scaled = scaler.transform(train_df, feature_cols)
    val_scaled   = scaler.transform(val_df,   feature_cols)
    test_scaled  = scaler.transform(test_df,  feature_cols)

    train_ds = SequenceDataset(train_scaled, feature_cols, TARGET_COL, SEQ_LEN)
    val_ds   = SequenceDataset(val_scaled,   feature_cols, TARGET_COL, SEQ_LEN)
    test_ds  = SequenceDataset(test_scaled,  feature_cols, TARGET_COL, SEQ_LEN)

    print(f"Train sequences: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    print()

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Sequence length:  {SEQ_LEN}h  ({SEQ_LEN // 24} days)")
    print()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Halve LR if val loss plateaus for 4 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    best_state      = None
    best_val_loss   = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print("Starting training...\n")
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer=None)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t_ep
        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state      = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time / 60:.1f} min.\n")

    if best_state is None:
        raise RuntimeError("No best model state captured — training may have failed.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols":     feature_cols,
            "seq_len":          SEQ_LEN,
            "target_col":       TARGET_COL,
            "region":           REGION,
            "train_min":        scaler.min_.to_dict(),
            "train_max":        scaler.max_.to_dict(),
            "config": {
                "hidden_size":   HIDDEN_SIZE,
                "num_layers":    NUM_LAYERS,
                "bidirectional": BIDIRECTIONAL,
                "dropout":       DROPOUT,
                "batch_size":    BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
        },
        MODEL_OUT,
    )
    print(f"Saved best model to: {MODEL_OUT}\n")

    # Save training history
    pd.DataFrame(
        {
            "epoch":      range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss":   val_losses,
        }
    ).to_csv(OUT_DIR / "lstm_caiso_training_history.csv", index=False)

    # Evaluate
    val_pred,  val_true,  _       = predict(model, val_loader,  scaler, TARGET_COL)
    test_pred, test_true, test_ts = predict(model, test_loader, scaler, TARGET_COL)

    val_metrics  = compute_metrics(val_true,  val_pred)
    test_metrics = compute_metrics(test_true, test_pred)

    print_metrics("Validation metrics (MW scale)", val_metrics)
    print_metrics("Test metrics (MW scale)",       test_metrics)

    # Save test predictions
    pd.DataFrame(
        {
            "timestamp":          pd.to_datetime(test_ts),
            "actual_load_mw":     test_true,
            "predicted_load_mw":  test_pred,
            "absolute_error_mw":  np.abs(test_true - test_pred),
        }
    ).to_csv(OUT_DIR / "lstm_caiso_test_predictions.csv", index=False)

    # Plots
    plot_loss_curves(train_losses, val_losses)
    plot_predictions(test_ts, test_true, test_pred)
    plot_scatter(test_true, test_pred)

    # Update model_comparison_summary.csv
    mean_load = float(np.mean(test_true))
    row = {
        "model":             "LSTM (BiLSTM, 2L, seq=168)",
        "mae_mw":            round(test_metrics["MAE"],  3),
        "total_predictions": len(test_true),
        "mean_actual_load":  round(mean_load, 2),
        "mae_pct":           round(100 * test_metrics["MAE"] / mean_load, 2),
        "rmse_mw":           round(test_metrics["RMSE"], 3),
        "mape_pct":          round(test_metrics["MAPE"], 2),
        "r2":                round(test_metrics["R2"],   3),
        "has_result":        True,
    }
    comp_path = Path("model_comparison_summary.csv")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[comp_df["model"] != row["model"]]
        comp_df = pd.concat([comp_df, pd.DataFrame([row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([row])
    comp_df.to_csv(comp_path, index=False)

    print("Artifacts written:")
    print(f"  - {MODEL_OUT}")
    print(f"  - {OUT_DIR}/lstm_caiso_test_predictions.csv")
    print(f"  - {OUT_DIR}/lstm_caiso_training_history.csv")
    print(f"  - {OUT_DIR}/loss_curve.png")
    print(f"  - {OUT_DIR}/test_predictions.png")
    print(f"  - {OUT_DIR}/scatter.png")
    print("  - model_comparison_summary.csv  (updated)")
    print("\nDone.")


if __name__ == "__main__":
    main()
