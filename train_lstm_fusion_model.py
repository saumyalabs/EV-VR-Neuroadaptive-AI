"""Train a participant-split, mask-aware LSTM fusion model."""

from __future__ import annotations

import copy
from pathlib import Path

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


class FusionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 5),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, lstm_out.size(-1))
        last_valid = lstm_out.gather(1, gather_index).squeeze(1)
        return self.head(last_valid)


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
    model = FusionLSTM(input_size=X.shape[2]).to(device)
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

    report_path = run_paths["results"] / "global_model_report.txt"
    metrics_path = run_paths["results"] / "lstm_temporal_metrics.json"
    model_path = run_paths["models"] / "fusion_lstm_model.pth"
    write_outputs(
        model_name="LSTM",
        report_path=report_path,
        metrics_path=metrics_path,
        model_path=model_path,
        model=model,
        metrics=metrics,
        report=report,
        split=split,
        preprocessing_state=preprocessing_state,
    )
    (run_paths["results"] / "classification_report.txt").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    save_confusion_plot(
        predictions,
        test_targets,
        "LSTM Participant-Split Confusion Matrix Analysis",
        run_paths["visualizations"] / "lstm_confusion_matrices.png",
    )
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
