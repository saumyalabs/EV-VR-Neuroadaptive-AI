"""Train a leak-aware station-level fusion model for EEG + gaze/interaction.

The existing temporal scripts are useful experiments, but they mix participant
windows in the train/test split and include behaviour_token as a feature. This
script builds a cleaner benchmark:

- one row per participant_id + station_id
- gaze duplicates aggregated before merging with EEG
- behaviour_token excluded because it encodes target labels
- participant-level group-aware train/validation/test split
- class-balanced candidate models and threshold tuning per target
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = [
    "ai_used",
    "ai_followed",
    "unsafe_choice",
    "ai_usage_increased",
    "ai_usage_decreased",
]

RANDOM_STATE = 42


@dataclass
class TargetResult:
    target: str
    model_name: str
    threshold: float
    metrics: dict[str, Any]
    report: str
    confusion_matrix: list[list[int]]


def project_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    fusion_root = script_path.parents[1]
    model_fusion_root = script_path.parents[2]
    models_root = script_path.parents[3]
    return {
        "models_root": models_root,
        "model_fusion_root": model_fusion_root,
        "fusion_root": fusion_root,
        "datasets": fusion_root / "datasets",
        "results": fusion_root / "results",
        "models": fusion_root / "models",
        "gaze_embeddings": models_root
        / "Eye Gaze + Interaction Model"
        / "model_outputs"
        / "station_embeddings.csv",
        "eeg_embeddings": models_root / "EEG Model" / "eeg_station_embeddings.csv",
    }


def standardize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str).str.zfill(3)
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce").astype("Int64")
    return frame


def aggregate_gaze_embeddings(gaze: pd.DataFrame) -> pd.DataFrame:
    gaze = standardize_keys(gaze)
    for target in TARGET_COLUMNS:
        gaze[target] = pd.to_numeric(gaze[target], errors="coerce").fillna(0).astype(int)

    numeric_embedding_cols = [col for col in gaze.columns if col.startswith("embedding_")]
    aggregations: dict[str, str] = {col: "mean" for col in numeric_embedding_cols}

    for col in ["group", "threat_framing", "ai_available", "behaviour_token"]:
        if col in gaze.columns:
            aggregations[col] = "first"
    for target in TARGET_COLUMNS:
        if target in gaze.columns:
            aggregations[target] = "max"

    return (
        gaze.groupby(["participant_id", "station_id"], as_index=False)
        .agg(aggregations)
        .sort_values(["participant_id", "station_id"])
        .reset_index(drop=True)
    )


def aggregate_eeg_embeddings(eeg: pd.DataFrame) -> pd.DataFrame:
    eeg = standardize_keys(eeg)
    numeric_cols = [
        col
        for col in eeg.columns
        if col.startswith("eeg_embedding_")
        or col
        in {
            "eeg_stress_score",
            "eeg_cognitive_load_score",
            "eeg_fatigue_score",
            "eeg_hesitation_score",
            "eeg_attention_score",
            "eeg_artifact_score",
        }
    ]
    aggregations: dict[str, str] = {col: "mean" for col in numeric_cols}

    for col in ["group", "eeg_quality_flag", "eeg_token"]:
        if col in eeg.columns:
            aggregations[col] = "first"

    return (
        eeg.groupby(["participant_id", "station_id"], as_index=False)
        .agg(aggregations)
        .sort_values(["participant_id", "station_id"])
        .reset_index(drop=True)
    )


def build_clean_fusion_dataset(paths: dict[str, Path]) -> pd.DataFrame:
    gaze_raw = pd.read_csv(paths["gaze_embeddings"], dtype={"participant_id": str})
    eeg_raw = pd.read_csv(paths["eeg_embeddings"], dtype={"participant_id": str})

    gaze = aggregate_gaze_embeddings(gaze_raw)
    eeg = aggregate_eeg_embeddings(eeg_raw)

    fusion = gaze.merge(eeg, on=["participant_id", "station_id"], how="inner", suffixes=("", "_eeg"))
    for target in TARGET_COLUMNS:
        fusion[target] = pd.to_numeric(fusion[target], errors="coerce").fillna(0).astype(int)

    fusion = fusion.sort_values(["participant_id", "station_id"]).reset_index(drop=True)
    return fusion


def group_aware_participant_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[set[str], set[str]]:
    participant_groups = (
        frame.drop_duplicates("participant_id").set_index("participant_id")["group"].to_dict()
    )
    test_ids: set[str] = set()
    for group in sorted(set(participant_groups.values())):
        ids = sorted(pid for pid, pid_group in participant_groups.items() if pid_group == group)
        n_test = max(1, round(len(ids) * test_fraction))
        test_ids.update(ids[-n_test:])
    train_ids = set(participant_groups) - test_ids
    return train_ids, test_ids


def make_preprocessor(frame: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    categorical_cols = [col for col in feature_cols if frame[col].dtype == "object"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def candidate_models() -> dict[str, Any]:
    return {
        "logistic_balanced": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.1,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "logistic_aggressive": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.01,  # Stronger regularization for minority class
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,  # Increased from 4 for better splits
            min_samples_leaf=2,  # Decreased from 3 for more leaves
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,  # Increased from 4
            min_samples_leaf=2,  # Decreased from 3
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "extra_trees_aggressive": ExtraTreesClassifier(
            n_estimators=1000,  # More trees for better ensemble
            max_depth=6,
            min_samples_leaf=1,  # Very granular splits for minority class
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,  # Increased from 100
            learning_rate=0.04,
            max_depth=3,  # Increased from 2 for more complexity
            subsample=0.8,  # Add subsampling for regularization
            random_state=RANDOM_STATE,
        ),
    }


def validation_score(y_true: np.ndarray, y_pred: np.ndarray, target: str = None) -> float:
    """Target-specific scoring for better threshold tuning.
    
    - Rare targets: Prioritize recall (catch positive examples)
    - Over-predictors: Prioritize precision (reduce false positives)
    - Balanced targets: Balance all three metrics
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if target in {'ai_usage_increased', 'ai_usage_decreased'}:
        # Rare targets: heavily weight F1 and balanced accuracy
        return 0.70 * f1 + 0.25 * ba + 0.05 * acc
    elif target == 'unsafe_choice':
        # Over-predictor: weight precision more
        return 0.50 * f1 + 0.35 * ba + 0.15 * acc
    else:
        # Standard targets: balanced approach
        return 0.60 * f1 + 0.30 * ba + 0.10 * acc


def fit_one_target(
    target: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: ColumnTransformer,
) -> tuple[Pipeline, TargetResult, pd.DataFrame]:
    y_train_inner = train_frame[target].to_numpy(dtype=int)
    y_val = validation_frame[target].to_numpy(dtype=int)
    y_test = test_frame[target].to_numpy(dtype=int)

    best: tuple[float, str, float] | None = None
    thresholds = np.linspace(0.05, 0.95, 100)  # Finer-grained search (was 37)

    for model_name, classifier in candidate_models().items():
        if len(np.unique(y_train_inner)) < 2:
            continue
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
        pipeline.fit(train_frame[feature_cols], y_train_inner)
        probabilities = pipeline.predict_proba(validation_frame[feature_cols])[:, 1]
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = validation_score(y_val, predictions, target=target)  # Pass target name
            if best is None or score > best[0]:
                best = (score, model_name, float(threshold))

    if best is None:
        raise RuntimeError(f"Could not train target {target}: training split has one class.")

    _, best_model_name, best_threshold = best
    final_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", candidate_models()[best_model_name]),
        ]
    )
    final_train = pd.concat([train_frame, validation_frame], axis=0)
    final_pipeline.fit(final_train[feature_cols], final_train[target].to_numpy(dtype=int))

    test_probabilities = final_pipeline.predict_proba(test_frame[feature_cols])[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "support": int(len(y_test)),
        "positive_support": int(y_test.sum()),
        "predicted_positive": int(test_predictions.sum()),
    }
    report = classification_report(y_test, test_predictions, zero_division=0)
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist()

    prediction_frame = test_frame[["participant_id", "station_id"]].copy()
    prediction_frame[f"{target}_true"] = y_test
    prediction_frame[f"{target}_probability"] = test_probabilities
    prediction_frame[f"{target}_predicted"] = test_predictions

    result = TargetResult(
        target=target,
        model_name=best_model_name,
        threshold=best_threshold,
        metrics=metrics,
        report=report,
        confusion_matrix=cm,
    )
    return final_pipeline, result, prediction_frame


def main() -> None:
    paths = project_paths()
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["models"].mkdir(parents=True, exist_ok=True)

    fusion = build_clean_fusion_dataset(paths)
    clean_dataset_path = paths["datasets"] / "master_fusion_dataset_clean.csv"
    fusion.to_csv(clean_dataset_path, index=False)

    leakage_cols = {"behaviour_token"}
    exclude_cols = {"participant_id", *TARGET_COLUMNS, *leakage_cols}
    feature_cols = [col for col in fusion.columns if col not in exclude_cols]

    train_ids, test_ids = group_aware_participant_split(fusion, test_fraction=0.2)
    train_full = fusion[fusion["participant_id"].isin(train_ids)].copy()
    test_frame = fusion[fusion["participant_id"].isin(test_ids)].copy()
    train_inner_ids, validation_ids = group_aware_participant_split(train_full, test_fraction=0.2)
    train_frame = train_full[train_full["participant_id"].isin(train_inner_ids)].copy()
    validation_frame = train_full[train_full["participant_id"].isin(validation_ids)].copy()

    preprocessor = make_preprocessor(fusion, feature_cols)
    fitted_models: dict[str, Pipeline] = {}
    target_results: list[TargetResult] = []
    prediction_frames: list[pd.DataFrame] = []

    for target in TARGET_COLUMNS:
        model, result, predictions = fit_one_target(
            target=target,
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            feature_cols=feature_cols,
            preprocessor=preprocessor,
        )
        fitted_models[target] = model
        target_results.append(result)
        prediction_frames.append(predictions)

    predictions_out = test_frame[["participant_id", "station_id", "group", "threat_framing"]].copy()
    for frame in prediction_frames:
        value_cols = [col for col in frame.columns if col not in {"participant_id", "station_id"}]
        predictions_out = predictions_out.merge(
            frame[["participant_id", "station_id", *value_cols]],
            on=["participant_id", "station_id"],
            how="left",
        )

    metrics = {
        "dataset": {
            "clean_rows": int(len(fusion)),
            "participants": int(fusion["participant_id"].nunique()),
            "duplicate_participant_station_keys": int(
                len(fusion) - fusion[["participant_id", "station_id"]].drop_duplicates().shape[0]
            ),
            "feature_count": int(len(feature_cols)),
            "excluded_leakage_columns": sorted(leakage_cols),
            "target_positive_rates": {
                target: float(fusion[target].mean()) for target in TARGET_COLUMNS
            },
        },
        "split": {
            "train_participants": sorted(train_inner_ids),
            "validation_participants": sorted(validation_ids),
            "test_participants": sorted(test_ids),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
        },
        "targets": {
            result.target: {
                "model": result.model_name,
                "threshold": result.threshold,
                "metrics": result.metrics,
                "confusion_matrix": result.confusion_matrix,
            }
            for result in target_results
        },
    }

    metrics_path = paths["results"] / "improved_fusion_metrics.json"
    report_path = paths["results"] / "improved_fusion_report.txt"
    predictions_path = paths["results"] / "improved_fusion_predictions.csv"
    model_path = paths["models"] / "improved_fusion_models.pkl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_out.to_csv(predictions_path, index=False)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "models": fitted_models,
                "feature_columns": feature_cols,
                "target_columns": TARGET_COLUMNS,
                "metrics": metrics,
            },
            handle,
        )

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Improved Leak-Aware Fusion Model Report\n")
        handle.write("=======================================\n\n")
        handle.write(f"Clean dataset: {clean_dataset_path}\n")
        handle.write(f"Rows: {len(fusion)}\n")
        handle.write(f"Participants: {fusion['participant_id'].nunique()}\n")
        handle.write(f"Feature columns: {len(feature_cols)}\n")
        handle.write(f"Excluded leakage columns: {', '.join(sorted(leakage_cols))}\n\n")
        handle.write(f"Train participants: {', '.join(sorted(train_inner_ids))}\n")
        handle.write(f"Validation participants: {', '.join(sorted(validation_ids))}\n")
        handle.write(f"Test participants: {', '.join(sorted(test_ids))}\n\n")
        for result in target_results:
            handle.write("\n=================================\n")
            handle.write(f"TARGET: {result.target}\n")
            handle.write("=================================\n")
            handle.write(f"Chosen model: {result.model_name}\n")
            handle.write(f"Threshold: {result.threshold:.3f}\n")
            handle.write(json.dumps(result.metrics, indent=2))
            handle.write("\n\n")
            handle.write(result.report)
            handle.write("\n")

    print("Improved fusion training complete.")
    print(f"Clean dataset: {clean_dataset_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Model state: {model_path}")


if __name__ == "__main__":
    main()
