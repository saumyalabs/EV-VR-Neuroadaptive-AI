"""Shared helpers for participant-split temporal fusion experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


def paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "root": root,
        "data": root / "datasets",
        "models": root / "models",
        "results": root / "results",
        "visualizations": root / "results" / "visualizations",
    }


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_forecasting_data(dataset_path: Path) -> dict[str, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    loaded = np.load(dataset_path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def split_indices(participant_ids: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    participant_group: dict[str, str] = {}
    for participant_id, group in zip(participant_ids, groups, strict=True):
        participant_group.setdefault(str(participant_id), str(group))

    test_ids: set[str] = set()
    for group in sorted(set(participant_group.values())):
        ids = sorted(pid for pid, pid_group in participant_group.items() if pid_group == group)
        n_test = max(1, round(len(ids) * 0.2))
        test_ids.update(ids[-n_test:])

    train_ids = set(participant_group) - test_ids
    train_mask = np.asarray([str(pid) in train_ids for pid in participant_ids])
    train_participants = [str(pid) for pid in participant_ids[train_mask]]
    train_groups = groups[train_mask]

    inner_group_map: dict[str, str] = {}
    for participant_id, group in zip(train_participants, train_groups, strict=True):
        inner_group_map.setdefault(participant_id, str(group))

    validation_ids: set[str] = set()
    for group in sorted(set(inner_group_map.values())):
        ids = sorted(pid for pid, pid_group in inner_group_map.items() if pid_group == group)
        n_val = max(1, round(len(ids) * 0.2))
        validation_ids.update(ids[-n_val:])

    inner_train_ids = set(inner_group_map) - validation_ids
    train_idx = np.where(np.asarray([str(pid) in inner_train_ids for pid in participant_ids]))[0]
    val_idx = np.where(np.asarray([str(pid) in validation_ids for pid in participant_ids]))[0]
    test_idx = np.where(np.asarray([str(pid) in test_ids for pid in participant_ids]))[0]
    return train_idx, val_idx, test_idx, {
        "train_participants": sorted(inner_train_ids),
        "validation_participants": sorted(validation_ids),
        "test_participants": sorted(test_ids),
    }


def standardize_from_train(X: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    train_X = X[train_idx]
    train_mask = mask[train_idx]

    if train_X.ndim < 2:
        raise ValueError("Expected at least 2D array for X")

    valid = train_X[train_mask]
    if valid.size == 0:
        mean = np.zeros(X.shape[-1], dtype=np.float32)
        std = np.ones(X.shape[-1], dtype=np.float32)
    else:
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

    standardized = (X - mean) / std
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    if standardized.ndim == 3:
        standardized[~mask] = 0.0
    elif standardized.ndim == 2:
        standardized[~mask] = 0.0

    return standardized.astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def make_loader(X: np.ndarray, y: np.ndarray, mask: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(mask[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def positive_weights(y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    weights = np.clip(weights, 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_mask, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)
            output = model(batch_X, batch_mask)
            logits.append(output.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
    return np.vstack(logits), np.vstack(targets)


def tune_thresholds(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    thresholds: list[float] = []
    for i in range(targets.shape[1]):
        best_threshold = 0.5
        best_score = -math.inf
        for threshold in np.linspace(0.05, 0.9, 70):
            pred = (probabilities[:, i] >= threshold).astype(int)
            score = 0.50 * f1_score(targets[:, i], pred, zero_division=0)
            score += 0.20 * recall_score(targets[:, i], pred, zero_division=0)
            score += 0.15 * balanced_accuracy_score(targets[:, i], pred)
            score += 0.10 * accuracy_score(targets[:, i], pred)
            score += 0.05 * precision_score(targets[:, i], pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def metrics_for(probabilities: np.ndarray, targets: np.ndarray, thresholds: list[float]) -> tuple[np.ndarray, dict[str, Any], str]:
    predictions = (probabilities >= np.asarray(thresholds)[None, :]).astype(int)
    metrics: dict[str, Any] = {"per_target": {}}
    if targets.ndim == 1:
        targets = targets[:, None]
        predictions = predictions[:, None]

    for i in range(targets.shape[1]):
        y_true = targets[:, i].astype(int)
        y_pred = predictions[:, i].astype(int)
        metrics["per_target"][f"target_{i+1}"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    report = classification_report(targets, predictions, zero_division=0)
    return predictions, metrics, report


def save_confusion_plot(predictions: np.ndarray, targets: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if targets.ndim == 1:
        targets = targets[:, None]
        predictions = predictions[:, None]

    n_targets = targets.shape[1]
    cols = min(3, n_targets)
    rows = math.ceil(n_targets / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(title)

    for idx in range(n_targets):
        ax = axes[idx]
        cm = confusion_matrix(targets[:, idx], predictions[:, idx])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Target {idx + 1}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, int(value), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_targets, len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    model_name: str,
    report_path: Path,
    metrics_path: Path,
    model_path: Path,
    model: torch.nn.Module,
    metrics: dict[str, Any],
    report: str,
    split: dict[str, list[str]],
    preprocessing_state: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(report, encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "metrics": metrics,
                "split": split,
                "preprocessing": preprocessing_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), model_path)
