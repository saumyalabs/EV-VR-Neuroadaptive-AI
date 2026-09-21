"""Train a participant-split, mask-aware Transformer fusion model."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from temporal_training_utils import (
    load_forecasting_data,
    make_loader,
    metrics_for,
    paths,
    positive_weights,
    run_logits,
    save_confusion_plot,
    set_seed,
    split_indices,
    standardize_from_train,
    tune_thresholds,
    write_outputs,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerFusionModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 5),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        key_padding_mask = mask <= 0
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        masked = encoded * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def evaluate_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_X, batch_mask, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X, batch_mask)
            losses.append(float(criterion(logits, batch_y).item()))
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    set_seed(42)
    run_paths = paths()
    run_paths["models"].mkdir(parents=True, exist_ok=True)
    run_paths["results"].mkdir(parents=True, exist_ok=True)

    data = load_forecasting_data(run_paths["data"])
    train_idx, val_idx, test_idx, split = split_indices(data["participant_ids"], data["groups"])
    X, preprocessing_state = standardize_from_train(data["X"], data["mask"], train_idx)
    y = data["y"]
    mask = data["mask"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train/val/test windows: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Test participants: {split['test_participants']}")

    batch_size = 16
    train_loader = make_loader(X, y, mask, train_idx, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X, y, mask, val_idx, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(X, y, mask, test_idx, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerFusionModel(input_size=X.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights(y[train_idx]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs = 60

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch_X, batch_mask, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f}"
        )

    model.load_state_dict(best_state)
    val_logits, val_targets = run_logits(model, val_loader, device)
    test_logits, test_targets = run_logits(model, test_loader, device)
    thresholds = tune_thresholds(sigmoid_np(val_logits), val_targets)
    predictions, metrics, report = metrics_for(sigmoid_np(test_logits), test_targets, thresholds)

    report_path = run_paths["results"] / "transformer_fusion_report.txt"
    metrics_path = run_paths["results"] / "transformer_temporal_metrics.json"
    model_path = run_paths["models"] / "transformer_fusion_model.pth"
    write_outputs(
        model_name="Transformer",
        report_path=report_path,
        metrics_path=metrics_path,
        model_path=model_path,
        model=model,
        metrics=metrics,
        report=report,
        split=split,
        preprocessing_state=preprocessing_state,
    )
    save_confusion_plot(
        predictions,
        test_targets,
        "Transformer Participant-Split Confusion Matrix Analysis",
        run_paths["visualizations"] / "TRANSFORMER_confusion_matrices.png",
    )
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
