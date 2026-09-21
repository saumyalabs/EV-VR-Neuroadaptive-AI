"""Train the multi-task EEG + gaze/interaction fusion model."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


MAIN_TARGETS = [
    "ai_used",
    "ai_followed",
    "unsafe_choice",
    "ai_usage_increased",
    "ai_usage_decreased",
]


def paths() -> dict[str, Path]:
    fusion_root = Path(__file__).resolve().parents[1]
    return {
        "dataset": fusion_root / "datasets" / "multitask_fusion_sequences.npz",
        "metadata": fusion_root / "datasets" / "multitask_fusion_metadata.json",
        "model": fusion_root / "models" / "multitask_fusion_model.pth",
        "metrics": fusion_root / "results" / "multitask_fusion_metrics.json",
        "report": fusion_root / "results" / "multitask_fusion_report.txt",
        "predictions": fusion_root / "results" / "multitask_fusion_predictions.csv",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_aware_split(participant_ids: np.ndarray, groups: np.ndarray, fraction: float = 0.2) -> tuple[set[str], set[str]]:
    participant_group: dict[str, str] = {}
    for participant_id, group in zip(participant_ids, groups, strict=True):
        participant_group.setdefault(str(participant_id), str(group))
    test_ids: set[str] = set()
    for group in sorted(set(participant_group.values())):
        group_ids = sorted(pid for pid, value in participant_group.items() if value == group)
        n_test = max(1, round(len(group_ids) * fraction))
        test_ids.update(group_ids[-n_test:])
    return set(participant_group) - test_ids, test_ids


def split_indices(participant_ids: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    train_ids, test_ids = group_aware_split(participant_ids, groups)
    train_mask = np.asarray([pid in train_ids for pid in participant_ids])
    inner_train_ids, validation_ids = group_aware_split(participant_ids[train_mask], groups[train_mask])
    train_idx = np.where(np.asarray([pid in inner_train_ids for pid in participant_ids]))[0]
    val_idx = np.where(np.asarray([pid in validation_ids for pid in participant_ids]))[0]
    test_idx = np.where(np.asarray([pid in test_ids for pid in participant_ids]))[0]
    return train_idx, val_idx, test_idx, {
        "train_participants": sorted(inner_train_ids),
        "validation_participants": sorted(validation_ids),
        "test_participants": sorted(test_ids),
    }


def standardize_modalities(data: dict[str, np.ndarray], train_idx: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out = dict(data)
    state: dict[str, Any] = {}
    mask = data["mask"].astype(bool)
    for key in ["X_gaze", "X_eeg", "X_meta"]:
        train_values = data[key][train_idx]
        train_mask = mask[train_idx]
        valid = train_values[train_mask]
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
        std[std < 1e-6] = 1.0
        scaled = (data[key] - mean) / std
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        scaled[~mask] = 0.0
        out[key] = scaled.astype(np.float32)
        state[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return out, state


def main_positive_weights(y: np.ndarray) -> torch.Tensor:
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    return torch.tensor(np.clip(weights, 1.0, 20.0), dtype=torch.float32)


def make_loader(data: dict[str, np.ndarray], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [
        torch.tensor(data["X_gaze"][indices], dtype=torch.float32),
        torch.tensor(data["X_eeg"][indices], dtype=torch.float32),
        torch.tensor(data["X_meta"][indices], dtype=torch.float32),
        torch.tensor(data["mask"][indices], dtype=torch.float32),
        torch.tensor(data["y_main"][indices], dtype=torch.float32),
        torch.tensor(data["y_current_eeg_token"][indices], dtype=torch.long),
        torch.tensor(data["y_current_scores"][indices], dtype=torch.float32),
        torch.tensor(data["y_next_behavior_token"][indices], dtype=torch.long),
        torch.tensor(data["y_next_eeg_token"][indices], dtype=torch.long),
        torch.tensor(data["y_next_scores"][indices], dtype=torch.float32),
        torch.tensor(data["next_mask"][indices], dtype=torch.float32),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


class MultiTaskFusionModel(nn.Module):
    def __init__(self, gaze_dim: int, eeg_dim: int, meta_dim: int, behavior_classes: int, eeg_token_classes: int) -> None:
        super().__init__()
        self.gaze_encoder = nn.Sequential(nn.Linear(gaze_dim, 48), nn.LayerNorm(48), nn.ReLU(), nn.Dropout(0.25))
        self.eeg_encoder = nn.Sequential(nn.Linear(eeg_dim, 48), nn.LayerNorm(48), nn.ReLU(), nn.Dropout(0.25))
        self.meta_encoder = nn.Sequential(nn.Linear(meta_dim, 16), nn.LayerNorm(16), nn.ReLU(), nn.Dropout(0.15))
        self.fusion = nn.Sequential(nn.Linear(48 + 48 + 16, 80), nn.LayerNorm(80), nn.ReLU(), nn.Dropout(0.30))
        self.temporal = nn.LSTM(input_size=80, hidden_size=64, num_layers=1, batch_first=True)
        self.gaze_temporal = nn.LSTM(input_size=48 + 16, hidden_size=48, num_layers=1, batch_first=True)

        self.behavior_head = nn.Sequential(nn.Dropout(0.20), nn.Linear(64, len(MAIN_TARGETS)))
        self.next_behavior_token_head = nn.Linear(64, behavior_classes)
        self.next_eeg_token_head = nn.Linear(64, eeg_token_classes)
        self.next_score_head = nn.Linear(64, 3)

        self.current_eeg_token_from_gaze_head = nn.Linear(48, eeg_token_classes)
        self.current_score_from_gaze_head = nn.Linear(48, 3)

    @staticmethod
    def last_valid(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
        return sequence.gather(1, gather_index).squeeze(1)

    def forward(self, gaze: torch.Tensor, eeg: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        gaze_h = self.gaze_encoder(gaze)
        eeg_h = self.eeg_encoder(eeg)
        meta_h = self.meta_encoder(meta)
        fused_step = self.fusion(torch.cat([gaze_h, eeg_h, meta_h], dim=-1))
        temporal_out, _ = self.temporal(fused_step)
        fused_last = self.last_valid(temporal_out, mask)

        gaze_temporal_out, _ = self.gaze_temporal(torch.cat([gaze_h, meta_h], dim=-1))
        gaze_last = self.last_valid(gaze_temporal_out, mask)

        return {
            "main_logits": self.behavior_head(fused_last),
            "next_behavior_token_logits": self.next_behavior_token_head(fused_last),
            "next_eeg_token_logits": self.next_eeg_token_head(fused_last),
            "next_scores": self.next_score_head(fused_last),
            "current_eeg_token_from_gaze_logits": self.current_eeg_token_from_gaze_head(gaze_last),
            "current_scores_from_gaze": self.current_score_from_gaze_head(gaze_last),
        }


def batch_to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device) for item in batch)


def compute_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], losses: dict[str, nn.Module]) -> torch.Tensor:
    _, _, _, _, y_main, y_current_eeg_token, y_current_scores, y_next_behavior_token, y_next_eeg_token, y_next_scores, next_mask = batch
    loss = losses["main"](outputs["main_logits"], y_main)
    loss = loss + 0.05 * losses["current_eeg_token"](outputs["current_eeg_token_from_gaze_logits"], y_current_eeg_token)
    loss = loss + 0.05 * losses["current_scores"](outputs["current_scores_from_gaze"], y_current_scores)
    if next_mask.sum() > 0:
        valid = next_mask > 0
        loss = loss + 0.03 * losses["next_behavior_token"](outputs["next_behavior_token_logits"][valid], y_next_behavior_token[valid])
        loss = loss + 0.03 * losses["next_eeg_token"](outputs["next_eeg_token_logits"][valid], y_next_eeg_token[valid])
        loss = loss + 0.05 * losses["next_scores"](outputs["next_scores"][valid], y_next_scores[valid])
    return loss


def evaluate_loss(model: nn.Module, loader: DataLoader, losses: dict[str, nn.Module], device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            values.append(float(compute_loss(outputs, batch, losses).item()))
    return float(np.mean(values)) if values else float("inf")


def collect_main_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            logits.append(outputs["main_logits"].cpu().numpy())
            targets.append(batch[4].cpu().numpy())
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(np.vstack(logits), -40, 40)))
    return probabilities, np.vstack(targets)


def tune_thresholds(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    thresholds: list[float] = []
    for i in range(targets.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        target_rate = float(targets[:, i].mean())
        for threshold in np.linspace(0.05, 0.9, 70):
            pred = (probabilities[:, i] >= threshold).astype(int)
            pred_rate = float(pred.mean())
            rate_penalty = abs(pred_rate - target_rate) / max(target_rate, 0.10)
            score = 0.55 * f1_score(targets[:, i], pred, zero_division=0)
            score += 0.20 * recall_score(targets[:, i], pred, zero_division=0)
            score += 0.15 * balanced_accuracy_score(targets[:, i], pred)
            score += 0.10 * accuracy_score(targets[:, i], pred)
            score -= 0.20 * rate_penalty
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        if targets[:, i].sum() <= 1:
            best_threshold = max(0.05, best_threshold - 0.08)
        thresholds.append(best_threshold)
    return thresholds


def validation_main_score(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    probabilities, targets = collect_main_predictions(model, loader, device)
    thresholds = tune_thresholds(probabilities, targets)
    metrics, _, _ = summarize_main(probabilities, targets, thresholds)
    macro_f1 = float(np.mean([values["f1"] for values in metrics.values()]))
    macro_recall = float(np.mean([values["recall"] for values in metrics.values()]))
    macro_balanced_accuracy = float(np.mean([values["balanced_accuracy"] for values in metrics.values()]))
    return 0.60 * macro_f1 + 0.25 * macro_recall + 0.15 * macro_balanced_accuracy


def summarize_main(probabilities: np.ndarray, targets: np.ndarray, thresholds: list[float]) -> tuple[dict[str, Any], str, np.ndarray]:
    predictions = np.zeros_like(probabilities, dtype=np.int64)
    for i, threshold in enumerate(thresholds):
        predictions[:, i] = (probabilities[:, i] >= threshold).astype(int)

    metrics: dict[str, Any] = {}
    reports: list[str] = []
    for i, target in enumerate(MAIN_TARGETS):
        y_true = targets[:, i].astype(int)
        y_pred = predictions[:, i].astype(int)
        metrics[target] = {
            "threshold": thresholds[i],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "positive_support": int(y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
        }
        reports.append(f"\n=================================\nTARGET {i}: {target}\n=================================\n")
        reports.append(classification_report(y_true, y_pred, zero_division=0))
    return metrics, "\n".join(reports), predictions


def main() -> None:
    set_seed(42)
    p = paths()
    data_npz = np.load(p["dataset"], allow_pickle=True)
    metadata = json.loads(p["metadata"].read_text(encoding="utf-8"))
    data = {key: data_npz[key] for key in data_npz.files}
    data["participant_ids"] = data["participant_ids"].astype(str)
    data["groups"] = data["groups"].astype(str)

    train_idx, val_idx, test_idx, split = split_indices(data["participant_ids"], data["groups"])
    data, preprocessing_state = standardize_modalities(data, train_idx)

    train_loader = make_loader(data, train_idx, batch_size=16, shuffle=True)
    val_loader = make_loader(data, val_idx, batch_size=16, shuffle=False)
    test_loader = make_loader(data, test_idx, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskFusionModel(
        gaze_dim=data["X_gaze"].shape[2],
        eeg_dim=data["X_eeg"].shape[2],
        meta_dim=data["X_meta"].shape[2],
        behavior_classes=len(metadata["behavior_token_mapping"]),
        eeg_token_classes=len(metadata["eeg_token_mapping"]),
    ).to(device)

    losses = {
        "main": nn.BCEWithLogitsLoss(pos_weight=main_positive_weights(data["y_main"][train_idx]).to(device)),
        "current_eeg_token": nn.CrossEntropyLoss(),
        "current_scores": nn.SmoothL1Loss(),
        "next_behavior_token": nn.CrossEntropyLoss(),
        "next_eeg_token": nn.CrossEntropyLoss(),
        "next_scores": nn.SmoothL1Loss(),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_score = -1.0
    best_epoch = 0
    patience = 18
    epochs_without_improvement = 0
    for epoch in range(100):
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad()
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            loss = compute_loss(outputs, batch, losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_loss = evaluate_loss(model, val_loader, losses, device)
        val_score = validation_main_score(model, val_loader, device)
        scheduler.step(val_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch + 1:02d}/100 train_loss={np.mean(epoch_losses):.4f} "
            f"val_loss={val_loss:.4f} val_main_score={val_score:.4f}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_state)
    val_prob, val_targets = collect_main_predictions(model, val_loader, device)
    test_prob, test_targets = collect_main_predictions(model, test_loader, device)
    thresholds = tune_thresholds(val_prob, val_targets)
    metrics, report, predictions = summarize_main(test_prob, test_targets, thresholds)

    payload = {
        "model": "MultiTaskFusionModel",
        "split": split,
        "samples": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "best_validation_loss": best_val_loss,
        "best_validation_main_score": best_val_score,
        "best_epoch": best_epoch,
        "main_targets": metrics,
        "auxiliary_targets": [
            "current_eeg_token_from_gaze",
            "current_cognitive_load_stress_attention_from_gaze",
            "next_behaviour_token",
            "next_eeg_token",
            "next_cognitive_load_stress_attention",
        ],
        "preprocessing": preprocessing_state,
    }
    p["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    p["report"].write_text(
        "Multi-Task Fusion Model Report\n"
        "==============================\n\n"
        f"Train participants: {', '.join(split['train_participants'])}\n"
        f"Validation participants: {', '.join(split['validation_participants'])}\n"
        f"Test participants: {', '.join(split['test_participants'])}\n"
        f"Best epoch: {best_epoch}\n"
        f"Best validation loss: {best_val_loss:.4f}\n"
        f"Best validation main score: {best_val_score:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata, "metrics": payload}, p["model"])

    rows = []
    test_participants = data["participant_ids"][test_idx]
    test_station_ids = data["station_ids"][test_idx]
    for row_index in range(len(test_idx)):
        row = {
            "participant_id": test_participants[row_index],
            "station_id": int(test_station_ids[row_index]),
        }
        for target_index, target in enumerate(MAIN_TARGETS):
            row[f"{target}_true"] = int(test_targets[row_index, target_index])
            row[f"{target}_probability"] = float(test_prob[row_index, target_index])
            row[f"{target}_predicted"] = int(predictions[row_index, target_index])
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(p["predictions"], index=False)

    print("Multi-task fusion training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation main score: {best_val_score:.4f}")
    for target, values in metrics.items():
        print(
            f"{target}: accuracy={values['accuracy']:.3f}, "
            f"recall={values['recall']:.3f}, f1={values['f1']:.3f}"
        )
    print(f"Saved metrics: {p['metrics']}")
    print(f"Saved report: {p['report']}")


if __name__ == "__main__":
    main()
