"""
ST-CALNet-style load forecasting model for CAISO.

Architecture:
    CNN -> LSTM -> Scaled Dot-Product Attention -> Residual + LayerNorm -> LSTM -> Dense

This script adapts the ST-CALNet paper architecture to the uploaded CAISO-style dataset.
It uses:
- chronological train/validation/test split
- min-max scaling fit on train only
- 24-hour sliding windows
- all numeric weather features + engineered cyclical time features + historical load
- next-hour load prediction

Paper-aligned defaults:
- sequence length T = 24
- Conv1d kernel size = 3
- Conv1d filters = 64
- LSTM hidden size = 64
- attention dimension = 64
- batch size = 32
- learning rate = 1e-3
- patience = 5
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

SEQ_LEN = 24
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 5
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

CONV_CHANNELS = 64
CONV_KERNEL_SIZE = 3
LSTM_HIDDEN = 64
ATTN_DIM = 64
DROPOUT = 0.10

NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OUT = "st_calnet_caiso.pt"


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
# Dataset utilities
# =========================
def add_cyclical_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col])
    hour = ts.dt.hour
    dayofweek = ts.dt.dayofweek
    month = ts.dt.month

    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return df


def load_and_prepare_dataframe(csv_path: str, region: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)

    if "region" in df.columns:
        df = df[df["region"].str.lower() == region.lower()].copy()

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Keep only rows where the target exists.
    df = df[df[TARGET_COL].notna()].copy()

    df = add_cyclical_time_features(df, TIMESTAMP_COL)

    # Use all numeric non-constant columns except identifiers.
    exclude = {"region", "time_key", TIMESTAMP_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Ensure the target itself is included as an autoregressive input feature.
    if TARGET_COL not in feature_cols:
        feature_cols.append(TARGET_COL)

    # Drop constant columns if any.
    final_feature_cols = []
    for c in feature_cols:
        if df[c].nunique(dropna=False) > 1:
            final_feature_cols.append(c)

    return df.reset_index(drop=True), final_feature_cols


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


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
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

        values = self.df[feature_cols].to_numpy(dtype=np.float32)
        target = self.df[target_col].to_numpy(dtype=np.float32)
        timestamps = self.df[TIMESTAMP_COL].astype(str).to_numpy()

        self.X = []
        self.y = []
        self.ts = []

        for end_idx in range(seq_len, len(self.df)):
            start_idx = end_idx - seq_len
            self.X.append(values[start_idx:end_idx])
            self.y.append(target[end_idx])
            self.ts.append(timestamps[end_idx])

        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        self.ts = np.asarray(self.ts)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.ts[idx]

# =========================
# Model
# =========================
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, hidden]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # [B, T, T]
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)  # [B, T, H]
        return context, weights


class STCALNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        conv_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        attn_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.attn = TemporalSelfAttention(hidden_dim=hidden_dim, attn_dim=attn_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, features]
        x = x.transpose(1, 2)                # [B, F, T]
        z = self.relu(self.conv1(x))         # [B, C, T]
        z = self.dropout(z)
        z = z.transpose(1, 2)                # [B, T, C]

        h, _ = self.lstm1(z)                 # [B, T, H]
        context, attn_weights = self.attn(h) # [B, T, H], [B, T, T]
        h_refined = self.norm(h + context)   # residual + layer norm

        h2, (hn, _) = self.lstm2(h_refined)
        final_hidden = hn[-1]                # [B, H]

        pred = self.head(final_hidden).squeeze(-1)
        return pred, attn_weights


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
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
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

        preds, _ = model(xb)
        loss = criterion(preds, yb)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    scaler: MinMaxScalerFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_all, y_all, ts_all, attn_lastrow_all = [], [], [], []

    for xb, yb, tsb in loader:
        xb = xb.to(DEVICE)
        preds, attn = model(xb)

        preds_np = preds.detach().cpu().numpy()
        y_np = yb.numpy()

        preds_mw = scaler.inverse_transform_target(preds_np, target_col)
        y_mw = scaler.inverse_transform_target(y_np, target_col)

        # Aggregate each sample's attention "focus on source steps" using last query row.
        # Shape [B, T] corresponding to attention from final encoded step to all history steps.
        attn_lastrow = attn[:, -1, :].detach().cpu().numpy()

        preds_all.append(preds_mw)
        y_all.append(y_mw)
        ts_all.append(tsb)
        attn_lastrow_all.append(attn_lastrow)

    return (
        np.concatenate(preds_all),
        np.concatenate(y_all),
        np.concatenate(ts_all),
        np.concatenate(attn_lastrow_all),
    )


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"{title}:")
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.3f}")
    print()


def save_predictions_csv(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str = "st_calnet_test_predictions.csv",
) -> None:
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "actual_load_mw": y_true,
            "predicted_load_mw": y_pred,
            "absolute_error_mw": np.abs(y_true - y_pred),
        }
    )
    out.to_csv(path, index=False)


def save_attention_csv(attn: np.ndarray, seq_len: int, path: str = "st_calnet_test_attention.csv") -> None:
    cols = [f"t_minus_{seq_len - i}" for i in range(seq_len)]
    pd.DataFrame(attn, columns=cols).to_csv(path, index=False)


def main() -> None:
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading dataset: {CSV_PATH}")
    df, feature_cols = load_and_prepare_dataframe(CSV_PATH, REGION)

    print(f"Rows after region filter + non-null target filter: {len(df):,}")
    print(f"Timestamp range: {df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()}")
    print(f"Feature count: {len(feature_cols)}")
    print("Features:")
    for col in feature_cols:
        print(f"  - {col}")
    print()

    train_df, val_df, test_df = chronological_split(df, TRAIN_FRAC, VAL_FRAC)
    print(f"Train rows: {len(train_df):,}")
    print(f"Val rows:   {len(val_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print()

    scaler = fit_minmax_scaler(train_df, feature_cols)
    train_scaled = scaler.transform(train_df, feature_cols)
    val_scaled = scaler.transform(val_df, feature_cols)
    test_scaled = scaler.transform(test_df, feature_cols)

    train_ds = SequenceDataset(train_scaled, feature_cols, TARGET_COL, SEQ_LEN)
    val_ds = SequenceDataset(val_scaled, feature_cols, TARGET_COL, SEQ_LEN)
    test_ds = SequenceDataset(test_scaled, feature_cols, TARGET_COL, SEQ_LEN)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = STCALNet(
        num_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=CONV_KERNEL_SIZE,
        hidden_dim=LSTM_HIDDEN,
        attn_dim=ATTN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was captured.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
            "target_col": TARGET_COL,
            "region": REGION,
            "train_min": scaler.min_.to_dict(),
            "train_max": scaler.max_.to_dict(),
            "config": {
                "conv_channels": CONV_CHANNELS,
                "conv_kernel_size": CONV_KERNEL_SIZE,
                "lstm_hidden": LSTM_HIDDEN,
                "attn_dim": ATTN_DIM,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
        },
        MODEL_OUT,
    )
    print(f"\nSaved best model to: {MODEL_OUT}\n")

    val_pred, val_true, val_ts, val_attn = predict(model, val_loader, scaler, TARGET_COL)
    test_pred, test_true, test_ts, test_attn = predict(model, test_loader, scaler, TARGET_COL)

    val_metrics = compute_metrics(val_true, val_pred)
    test_metrics = compute_metrics(test_true, test_pred)

    print_metrics("Validation metrics (MW scale)", val_metrics)
    print_metrics("Test metrics (MW scale)", test_metrics)

    save_predictions_csv(test_ts, test_true, test_pred, path="st_calnet_test_predictions.csv")
    save_attention_csv(test_attn, seq_len=SEQ_LEN, path="st_calnet_test_attention.csv")

    avg_attention = test_attn.mean(axis=0)
    attention_df = pd.DataFrame(
        {
            "history_step": [f"t-{SEQ_LEN - i}" for i in range(SEQ_LEN)],
            "mean_attention_weight": avg_attention,
        }
    ).sort_values("mean_attention_weight", ascending=False)

    print("Top 10 attended historical steps on test set:")
    print(attention_df.head(10).to_string(index=False))
    print()

    print("Done.")
    print("Artifacts written:")
    print("  -", MODEL_OUT)
    print("  - st_calnet_test_predictions.csv")
    print("  - st_calnet_test_attention.csv")


if __name__ == "__main__":
    main()
