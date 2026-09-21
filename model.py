"""Train lightweight station-level models and export fusion-ready embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import IDENTIFIER_COLUMNS, TARGET_COLUMNS


def participant_split(participants: list[str], groups: pd.Series, test_fraction: float = 0.2) -> tuple[set[str], set[str]]:
    """Deterministic group-aware participant split."""
    group_by_pid = groups.to_dict()
    test: set[str] = set()
    for group in sorted(set(group_by_pid.values())):
        group_ids = sorted(pid for pid, pid_group in group_by_pid.items() if pid_group == group)
        n_test = max(1, round(len(group_ids) * test_fraction))
        test.update(group_ids[-n_test:])
    train = set(participants) - test
    return train, test


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS)
    blocked.update(
        {
            "assistant_event_count",
            "assistant_reply_count",
            "charge_amount",
            "decision_latency_ms",
            "decision_points",
            "gaze_sample_count",
            "gaze_window_duration_ms",
            "interaction_event_count",
        }
    )
    columns: list[str] = []
    for column in frame.columns:
        if column in blocked:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def fit_preprocess(train_frame: pd.DataFrame, all_frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_values = train_frame[columns].replace([np.inf, -np.inf], np.nan)
    medians = train_values.median(axis=0).fillna(0.0)
    train_matrix = train_values.fillna(medians).to_numpy(dtype=float)
    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds[stds == 0] = 1.0

    all_values = all_frame[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    all_matrix = (all_values.to_numpy(dtype=float) - means) / stds
    train_matrix = (train_matrix - means) / stds
    state = {
        "feature_columns": columns,
        "medians": medians.to_dict(),
        "means": means.tolist(),
        "stds": stds.tolist(),
    }
    return train_matrix, all_matrix, state


def fit_pca(train_matrix: np.ndarray, all_matrix: np.ndarray, embedding_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    centered_train = train_matrix - train_matrix.mean(axis=0, keepdims=True)
    center = train_matrix.mean(axis=0)
    if centered_train.shape[0] < 2 or centered_train.shape[1] == 0:
        embeddings = np.zeros((all_matrix.shape[0], embedding_dim), dtype=float)
        return embeddings, {"center": center.tolist(), "components": [], "embedding_dim": embedding_dim}

    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    component_count = min(embedding_dim, vt.shape[0])
    components = vt[:component_count]
    projected = (all_matrix - center) @ components.T
    if component_count < embedding_dim:
        projected = np.pad(projected, ((0, 0), (0, embedding_dim - component_count)))
    return projected, {
        "center": center.tolist(),
        "components": components.tolist(),
        "embedding_dim": embedding_dim,
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 1500,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    mask = ~np.isnan(y)
    x_train = x[mask]
    y_train = y[mask].astype(float)
    if len(np.unique(y_train)) < 2:
        raise ValueError("target has fewer than two classes")

    weights = np.zeros(x_train.shape[1], dtype=float)
    bias = float(np.log((y_train.mean() + 1e-6) / (1.0 - y_train.mean() + 1e-6)))
    positive_weight = len(y_train) / (2 * max(1, y_train.sum()))
    negative_weight = len(y_train) / (2 * max(1, len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, positive_weight, negative_weight)

    for _ in range(epochs):
        pred = sigmoid(x_train @ weights + bias)
        error = (pred - y_train) * sample_weights
        grad_w = (x_train.T @ error) / len(y_train) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true)
    y = y_true[mask].astype(int)
    p = y_prob[mask]
    if len(y) == 0:
        return {"n": 0}
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_models(dataset_path: Path, output_dir: Path, embedding_dim: int = 128) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path, dtype={"participant_id": str})
    participants = sorted(frame["participant_id"].dropna().unique().tolist())
    group_map = frame.drop_duplicates("participant_id").set_index("participant_id")["group"]
    train_ids, test_ids = participant_split(participants, group_map)
    train_frame = frame[frame["participant_id"].isin(train_ids)].copy()
    test_mask = frame["participant_id"].isin(test_ids).to_numpy()
    train_mask = frame["participant_id"].isin(train_ids).to_numpy()

    columns = feature_columns(frame)
    train_x, all_x, preprocess_state = fit_preprocess(train_frame, frame, columns)
    embeddings, pca_state = fit_pca(train_x, all_x, embedding_dim)

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {
        "train_participants": sorted(train_ids),
        "test_participants": sorted(test_ids),
        "n_rows": int(len(frame)),
        "n_features": int(len(columns)),
        "embedding_dim": int(embedding_dim),
        "targets": {},
    }
    predictions = frame[
        ["participant_id", "group", "station_id", "behaviour_token", *TARGET_COLUMNS]
    ].copy()

    for target in TARGET_COLUMNS:
        y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        y_train = y_all[train_mask]
        try:
            weights, bias = train_logistic(train_x, y_train)
        except ValueError as exc:
            metrics["targets"][target] = {"skipped": str(exc)}
            continue
        probabilities = sigmoid(all_x @ weights + bias)
        predictions[f"{target}_prob"] = probabilities
        metrics["targets"][target] = {
            "train": binary_metrics(y_all[train_mask], probabilities[train_mask]),
            "test": binary_metrics(y_all[test_mask], probabilities[test_mask]),
        }
        models[target] = {"weights": weights.tolist(), "bias": bias}

    embedding_frame = frame[
        ["participant_id", "group", "threat_framing", "ai_available", "station_id", "behaviour_token", *TARGET_COLUMNS]
    ].copy()
    embedding_columns = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{index:03d}" for index in range(embedding_dim)],
        index=embedding_frame.index,
    )
    embedding_frame = pd.concat([embedding_frame, embedding_columns], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "station_embeddings.csv"
    predictions_path = output_dir / "station_predictions.csv"
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model_state.json"
    embedding_frame.to_csv(embedding_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "preprocess": preprocess_state,
                "pca": pca_state,
                "models": models,
                "target_columns": TARGET_COLUMNS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "embedding_path": str(embedding_path),
        "predictions_path": str(predictions_path),
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="model_outputs/station_dataset.csv")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()
    result = train_models(Path(args.dataset), Path(args.output_dir), args.embedding_dim)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote embeddings to {result['embedding_path']}")
    print(f"Wrote predictions to {result['predictions_path']}")
    print(f"Wrote model state to {result['model_path']}")


if __name__ == "__main__":
    main()
