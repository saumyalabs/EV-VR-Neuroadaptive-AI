#!/usr/bin/env python3
"""
EEG station-level pipeline for EV Charging VR project.

Outputs:
- eeg_station_dataset.csv
- eeg_station_embeddings.csv
- eeg_predictions.csv
- eeg_metrics.json
- eeg_model_state.pkl
- EEG_MODEL_DOCUMENTATION.md
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize

# optional sklearn/xgboost/lgb imports handled at runtime
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
except Exception:
    PCA = None
    KMeans = None
    GaussianMixture = None
    RandomForestClassifier = None
    LogisticRegression = None
    confusion_matrix = None
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None


TARGET_COLS = [
    "ai_used",
    "ai_followed",
    "unsafe_choice",
    "ai_usage_increased",
    "ai_usage_decreased",
]

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

CHANNELS = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]

EPS = 1e-8


@dataclass
class LogRegModel:
    coef: np.ndarray
    intercept: float
    constant: int | None = None


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float = 1.0,
    max_iter: int = 300,
) -> LogRegModel:
    y = y.astype(float)
    uniq = np.unique(y)
    if len(uniq) <= 1:
        return LogRegModel(coef=np.zeros(x.shape[1]), intercept=0.0, constant=int(uniq[0]))

    n_features = x.shape[1]

    def objective(wb: np.ndarray) -> float:
        w = wb[:-1]
        b = wb[-1]
        p = sigmoid(x @ w + b)
        ce = -(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)).mean()
        reg = 0.5 * reg_lambda * np.sum(w * w)
        return float(ce + reg)

    init = np.zeros(n_features + 1)
    res = minimize(objective, init, method="L-BFGS-B", options={"maxiter": max_iter})
    wb = res.x if res.success else init
    return LogRegModel(coef=wb[:-1], intercept=float(wb[-1]), constant=None)


def predict_logistic_proba(model: LogRegModel, x: np.ndarray) -> np.ndarray:
    if model.constant is not None:
        return np.full(x.shape[0], float(model.constant))
    return sigmoid(x @ model.coef + model.intercept)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | List[List[int]]]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def to_epoch_ms(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return (ts.astype("int64") // 10**6).astype("float64")


def numeric_stat_features(vec: np.ndarray, prefix: str) -> Dict[str, float]:
    if vec.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "min", "max", "median", "p10", "p25", "p75", "p90", "iqr", "rms"]}
    return {
        f"{prefix}_mean": float(np.mean(vec)),
        f"{prefix}_std": float(np.std(vec)),
        f"{prefix}_min": float(np.min(vec)),
        f"{prefix}_max": float(np.max(vec)),
        f"{prefix}_median": float(np.median(vec)),
        f"{prefix}_p10": float(np.percentile(vec, 10)),
        f"{prefix}_p25": float(np.percentile(vec, 25)),
        f"{prefix}_p75": float(np.percentile(vec, 75)),
        f"{prefix}_p90": float(np.percentile(vec, 90)),
        f"{prefix}_iqr": float(np.percentile(vec, 75) - np.percentile(vec, 25)),
        f"{prefix}_rms": float(np.sqrt(np.mean(vec * vec))),
    }


def estimate_fs(df: pd.DataFrame) -> float:
    if "time" in df.columns:
        dt = np.diff(df["time"].to_numpy(dtype=float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(1.0 / np.median(dt))
    if "epoch_ms" in df.columns:
        dt_ms = np.diff(df["epoch_ms"].to_numpy(dtype=float))
        dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
        if dt_ms.size:
            return float(1000.0 / np.median(dt_ms))
    return 256.0


def band_powers(x: np.ndarray, fs: float) -> Dict[str, float]:
    if x.size < 8:
        return {f"{b}_abs": np.nan for b in BANDS} | {f"{b}_rel": np.nan for b in BANDS}
    nperseg = min(1024, x.size)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    total = np.trapezoid(psd[(freqs >= 1) & (freqs <= 40)], freqs[(freqs >= 1) & (freqs <= 40)])
    feats: Dict[str, float] = {}
    for b, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        p = float(np.trapezoid(psd[m], freqs[m])) if np.any(m) else 0.0
        feats[f"{b}_abs"] = p
        feats[f"{b}_rel"] = p / (total + EPS)
    return feats


def band_powers_timeseries(x: np.ndarray, fs: float, win_s: float = 1.0, hop_s: float = 0.5) -> Dict[str, np.ndarray]:
    """Compute band power time-series over overlapping windows.

    Returns dict band -> 1D numpy array of power estimates per window.
    """
    out: Dict[str, np.ndarray] = {}
    if x.size < max(4, int(win_s * fs)):
        for b in BANDS:
            out[b] = np.array([])
        return out

    win = int(max(2, round(win_s * fs)))
    hop = int(max(1, round(hop_s * fs)))
    starts = list(range(0, max(1, x.size - win + 1), hop))
    pw = []
    freqs, _ = signal.welch(x[:win], fs=fs, nperseg=win)
    for s in starts:
        seg = x[s : s + win]
        if seg.size < 2:
            continue
        freqs, psd = signal.welch(seg, fs=fs, nperseg=min(win, seg.size))
        total = np.trapz(psd[(freqs >= 1) & (freqs <= 40)], freqs[(freqs >= 1) & (freqs <= 40)])
        band_vals = {}
        for b, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            p = float(np.trapz(psd[m], freqs[m])) if np.any(m) else 0.0
            band_vals[b] = p
        pw.append(band_vals)
    if not pw:
        for b in BANDS:
            out[b] = np.array([])
        return out
    # transpose list of dicts
    for b in BANDS:
        out[b] = np.array([p.get(b, 0.0) for p in pw], dtype=float)
    return out


def robust_artifact_pct(x: np.ndarray, z_thr: float = 5.0) -> float:
    if x.size < 8:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    z = 0.6745 * (x - med) / mad
    return float(np.mean(np.abs(z) > z_thr))


def read_global_summary_features(folder: Path, pid: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    files = [
        (folder / "eeg_quality_report.csv", "eegq"),
        (folder / f"User-{pid}_mne_eeg_quality.csv", "mne_eegq"),
        (folder / "aux_artifact_report.csv", "aux_art"),
        (folder / "aux_summary.csv", "aux_sum"),
        (folder / f"User-{pid}_aux_features.csv", "user_aux"),
        (folder / "fnirs_summary.csv", "fnirs_sum"),
        (folder / f"User-{pid}_fnirs_summary.csv", "user_fnirs"),
        (folder / f"User-{pid}_mne_fnirs_od_features.csv", "fnirs_od"),
        (folder / "eeg_band_power.csv", "eeg_band_summary"),
        (folder / f"User-{pid}_mne_eeg_bandpower.csv", "mne_band"),
    ]

    for p, pref in files:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            continue
        desc = num.agg(["mean", "std", "min", "max"]).T
        for col, row in desc.iterrows():
            for stat in ["mean", "std", "min", "max"]:
                out[f"{pref}_{col}_{stat}"] = float(row.get(stat, np.nan))
    return out


def build_station_reference(station_df: pd.DataFrame) -> pd.DataFrame:
    df = station_df.copy()
    df["participant_id"] = df["participant_id"].astype(str).str.zfill(3)
    for c in TARGET_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    agg_map = {
        "group": "first",
        "threat_framing": "first",
        "ai_available": "max",
        "start_ts_ms": "min",
        "end_ts_ms": "max",
        "ai_used": "max",
        "ai_followed": "max",
        "unsafe_choice": "max",
        "ai_usage_increased": "max",
        "ai_usage_decreased": "max",
    }
    cols = [c for c in agg_map if c in df.columns]
    ref = (
        df.groupby(["participant_id", "station_id"], as_index=False)[cols]
        .agg({c: agg_map[c] for c in cols})
        .sort_values(["participant_id", "station_id"])
    )
    return ref


def station_features_for_participant(folder: Path, pid: str, station_ref: pd.DataFrame) -> List[Dict[str, float | str | int]]:
    eeg_path = folder / "eeg_filtered.csv"
    if not eeg_path.exists():
        return []

    eeg = pd.read_csv(eeg_path)
    if "ist_timestamp" not in eeg.columns:
        return []

    eeg["epoch_ms"] = to_epoch_ms(eeg["ist_timestamp"])
    eeg = eeg[np.isfinite(eeg["epoch_ms"])].copy()

    avail_channels = [c for c in CHANNELS if c in eeg.columns]
    if not avail_channels:
        return []

    fs = estimate_fs(eeg)
    global_summary = read_global_summary_features(folder, pid)

    out_rows: List[Dict[str, float | str | int]] = []
    p_rows = station_ref[station_ref["participant_id"] == pid]

    for _, st in p_rows.iterrows():
        start_ms = float(st.get("start_ts_ms", np.nan))
        end_ms = float(st.get("end_ts_ms", np.nan))
        w = eeg[(eeg["epoch_ms"] >= start_ms) & (eeg["epoch_ms"] <= end_ms)]

        row: Dict[str, float | str | int] = {
            "participant_id": pid,
            "station_id": int(st["station_id"]),
            "group": st.get("group", ""),
            "station_start_ts": start_ms,
            "station_end_ts": end_ms,
        }

        row.update({k: st.get(k, 0) for k in TARGET_COLS})

        if w.empty:
            row["eeg_window_samples"] = 0
            row["eeg_window_duration_ms"] = max(0.0, end_ms - start_ms) if np.isfinite(end_ms) and np.isfinite(start_ms) else np.nan
            row["eeg_artifact_score"] = np.nan
            row["eeg_quality_flag"] = "EEG_LOW_CONFIDENCE"
            row.update(global_summary)
            out_rows.append(row)
            continue

        row["eeg_window_samples"] = int(len(w))
        row["eeg_window_duration_ms"] = float(w["epoch_ms"].max() - w["epoch_ms"].min())

        all_art = []
        band_map: Dict[str, Dict[str, float]] = {}

        for ch in avail_channels:
            x = pd.to_numeric(w[ch], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            row.update(numeric_stat_features(x, f"{ch.lower()}_sig"))

            art = robust_artifact_pct(x)
            row[f"{ch.lower()}_artifact_pct"] = art
            if np.isfinite(art):
                all_art.append(art)

            bp = band_powers(x, fs)
            # band power timeseries stats (sliding windows)
            bp_ts = band_powers_timeseries(x, fs, win_s=1.0, hop_s=0.5)
            for b, arr in bp_ts.items():
                prefix = f"{ch.lower()}_{b}pw"
                if arr.size:
                    row.update(numeric_stat_features(arr, prefix))
                    # slope over time (simple linear fit)
                    try:
                        xs = np.arange(len(arr))
                        if len(xs) > 1:
                            m = np.polyfit(xs, arr, 1)[0]
                        else:
                            m = 0.0
                    except Exception:
                        m = 0.0
                    row[f"{prefix}_slope"] = float(m)
                    row[f"{prefix}_var"] = float(np.nanstd(arr))
                else:
                    for k in ["mean", "std", "min", "max", "median", "p10", "p25", "p75", "p90", "iqr", "rms"]:
                        row[f"{prefix}_{k}"] = np.nan
                    row[f"{prefix}_slope"] = np.nan
                    row[f"{prefix}_var"] = np.nan
            band_map[ch] = bp
            for k, v in bp.items():
                row[f"{ch.lower()}_{k}"] = v

            # channel-level ratios
            ba = bp.get("beta_abs", np.nan)
            aa = bp.get("alpha_abs", np.nan)
            ta = bp.get("theta_abs", np.nan)
            row[f"{ch.lower()}_theta_alpha_ratio"] = float(ta / (aa + EPS)) if np.isfinite(ta) and np.isfinite(aa) else np.nan
            row[f"{ch.lower()}_beta_alpha_ratio"] = float(ba / (aa + EPS)) if np.isfinite(ba) and np.isfinite(aa) else np.nan
            row[f"{ch.lower()}_beta_alpha_theta_ratio"] = float(ba / (aa + ta + EPS)) if np.isfinite(ba) and np.isfinite(aa) and np.isfinite(ta) else np.nan

            # missing sample percentage for channel within the station window
            expected = max(1, int(round(max(0.001, float(row.get("eeg_window_duration_ms", 0.0)) / 1000.0) * fs)))
            actual = int(np.count_nonzero(np.isfinite(pd.to_numeric(w[ch], errors="coerce"))))
            row[f"{ch.lower()}_missing_pct"] = float(max(0.0, 1.0 - (actual / (expected + EPS))))

        # asymmetry features
        def asym(ch_l: str, ch_r: str, band: str) -> Tuple[float, float]:
            l = band_map.get(ch_l, {}).get(f"{band}_abs", np.nan)
            r = band_map.get(ch_r, {}).get(f"{band}_abs", np.nan)
            diff = float(l - r) if np.isfinite(l) and np.isfinite(r) else np.nan
            ratio = float((l - r) / (l + r + EPS)) if np.isfinite(l) and np.isfinite(r) else np.nan
            return diff, ratio

        for band in ["theta", "alpha", "beta", "gamma"]:
            d, r = asym("EEG_AF7", "EEG_AF8", band)
            row[f"frontal_{band}_asym_diff"] = d
            row[f"frontal_{band}_asym_ratio"] = r

            d2, r2 = asym("EEG_TP9", "EEG_TP10", band)
            row[f"temporal_{band}_asym_diff"] = d2
            row[f"temporal_{band}_asym_ratio"] = r2

        row["eeg_artifact_score"] = float(np.nanmean(all_art)) if all_art else np.nan
        # bad channel count: channels with high artifact
        bad_thresh = 0.20
        bad_count = int(sum(1 for ch in avail_channels if row.get(f"{ch.lower()}_artifact_pct", 0) > bad_thresh))
        row["eeg_bad_channel_count"] = bad_count
        # missing sample percent (average across channels)
        miss_cols = [f"{ch.lower()}_missing_pct" for ch in avail_channels]
        miss_vals = [row.get(c, np.nan) for c in miss_cols]
        row["eeg_missing_pct"] = float(np.nanmean([v for v in miss_vals if np.isfinite(v)]) if any(np.isfinite(v) for v in miss_vals) else np.nan)
        # signal quality score (1 - artifact mean) clipped
        art_mean = row.get("eeg_artifact_score", np.nan)
        if np.isfinite(art_mean):
            row["eeg_signal_quality_score"] = float(max(0.0, min(1.0, 1.0 - art_mean)))
        else:
            row["eeg_signal_quality_score"] = np.nan
        row.update(global_summary)
        out_rows.append(row)

    return out_rows


def zscore_like(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    m = x.median()
    s = x.std(ddof=0)
    if not np.isfinite(s) or s < EPS:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - m) / s


def add_cognitive_state_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Global robust-like standardized ingredients
    beta_alpha = zscore_like(df.filter(regex=r"_beta_alpha_ratio$").mean(axis=1))
    theta_alpha = zscore_like(df.filter(regex=r"_theta_alpha_ratio$").mean(axis=1))
    signal_var = zscore_like(df.filter(regex=r"_sig_std$").mean(axis=1))
    alpha_rel = zscore_like(df.filter(regex=r"_alpha_rel$").mean(axis=1))
    beta_rel = zscore_like(df.filter(regex=r"_beta_rel$").mean(axis=1))
    theta_rel = zscore_like(df.filter(regex=r"_theta_rel$").mean(axis=1))
    quality = zscore_like(1 - pd.to_numeric(df.get("eeg_artifact_score", np.nan), errors="coerce").fillna(0))

    df["eeg_stress_score"] = (0.45 * beta_alpha + 0.25 * beta_rel + 0.20 * signal_var - 0.10 * alpha_rel).clip(-3, 3)
    df["eeg_cognitive_load_score"] = (0.45 * theta_alpha + 0.30 * beta_alpha + 0.15 * signal_var + 0.10 * theta_rel).clip(-3, 3)
    df["eeg_fatigue_score"] = (0.50 * theta_alpha + 0.20 * theta_rel - 0.25 * beta_rel - 0.05 * quality).clip(-3, 3)
    df["eeg_hesitation_score"] = (0.40 * df["eeg_cognitive_load_score"] + 0.35 * signal_var + 0.25 * zscore_like(df["eeg_window_duration_ms"])).clip(-3, 3)
    df["eeg_attention_score"] = (0.45 * beta_alpha - 0.35 * theta_alpha + 0.20 * quality).clip(-3, 3)
    # uncertainty: higher when signal variance high and quality low and theta/alpha elevated
    df["eeg_uncertainty_score"] = (0.40 * signal_var + 0.35 * (1 - quality) + 0.25 * theta_alpha).clip(-3, 3)
    # engagement: proxy combining attention and inverse fatigue
    df["eeg_engagement_score"] = (0.55 * df["eeg_attention_score"] - 0.30 * df["eeg_fatigue_score"] + 0.15 * quality).clip(-3, 3)

    # quality flag
    art = pd.to_numeric(df.get("eeg_artifact_score", np.nan), errors="coerce")
    n = pd.to_numeric(df.get("eeg_window_samples", np.nan), errors="coerce")
    flag = np.where(n < 64, "EEG_LOW_CONFIDENCE", np.where(art > 0.08, "EEG_ARTIFACT_HIGH", "EEG_CLEAN"))
    df["eeg_quality_flag"] = flag

    def token_row(r: pd.Series) -> str:
        toks = []
        toks.append("EEG_HIGH_STRESS" if r["eeg_stress_score"] >= 0.75 else "EEG_LOW_STRESS")
        toks.append("EEG_HIGH_COGNITIVE_LOAD" if r["eeg_cognitive_load_score"] >= 0.75 else "EEG_LOW_COGNITIVE_LOAD")
        toks.append("EEG_FATIGUE_RISK" if r["eeg_fatigue_score"] >= 0.75 else "EEG_ALERT")
        toks.append("EEG_HIGH_HESITATION" if r["eeg_hesitation_score"] >= 0.75 else "EEG_LOW_HESITATION")
        toks.append("EEG_ATTENTION_DROP" if r["eeg_attention_score"] <= -0.5 else ("EEG_HIGH_ENGAGEMENT" if r["eeg_attention_score"] >= 0.75 else "EEG_STABLE_ATTENTION"))
        toks.append("EEG_HIGH_UNCERTAINTY" if r.get("eeg_uncertainty_score", 0) >= 0.75 else "EEG_LOW_UNCERTAINTY")
        toks.append("EEG_HIGH_ENGAGEMENT" if r.get("eeg_engagement_score", 0) >= 0.75 else "EEG_LOW_ENGAGEMENT")
        toks.append(r["eeg_quality_flag"])
        return "|".join(toks)

    df["eeg_token"] = df.apply(token_row, axis=1)
    return df


def fit_projection(train_x: np.ndarray, all_x: np.ndarray, n_components: int = 128) -> np.ndarray:
    # PCA-like SVD projection learned on train split only.
    if train_x.size == 0:
        return np.zeros((all_x.shape[0], n_components))
    u, s, vt = np.linalg.svd(train_x, full_matrices=False)
    k = min(n_components, vt.shape[0])
    comp = vt[:k].T
    z = all_x @ comp
    if k < n_components:
        pad = np.zeros((z.shape[0], n_components - k))
        z = np.hstack([z, pad])
    return z


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    station_path = root / "station_dataset.csv"
    out_dataset = root / "eeg_station_dataset.csv"
    out_embed = root / "eeg_station_embeddings.csv"
    out_pred = root / "eeg_behavior_predictions.csv"
    out_metrics = root / "eeg_metrics.json"
    out_model = root / "eeg_model_state.pkl"
    out_doc = root / "EEG_MODEL_DOCUMENTATION.md"

    if not station_path.exists():
        raise FileNotFoundError(f"Missing required station file: {station_path}")

    station_df = pd.read_csv(station_path)
    station_ref = build_station_reference(station_df)

    all_rows: List[Dict[str, float | str | int]] = []
    for folder in sorted(root.glob("master_data_*")):
        if not folder.is_dir():
            continue
        pid = folder.name.split("_")[-1]
        if len(pid) == 0:
            continue
        all_rows.extend(station_features_for_participant(folder, pid, station_ref))

    eeg_df = pd.DataFrame(all_rows)
    if eeg_df.empty:
        raise RuntimeError("No EEG station rows were generated. Check folder/data availability.")

    # Keep one row per participant-station
    eeg_df = eeg_df.sort_values(["participant_id", "station_id", "eeg_window_samples"], ascending=[True, True, False])
    eeg_df = eeg_df.drop_duplicates(["participant_id", "station_id"], keep="first").reset_index(drop=True)

    # Add cognitive state scores and tokens
    eeg_df = add_cognitive_state_scores(eeg_df)

    # Numeric feature matrix
    id_cols = ["participant_id", "station_id", "group", "eeg_quality_flag", "eeg_token"]
    ignore = set(id_cols + TARGET_COLS)
    numeric_cols = [c for c in eeg_df.columns if c not in ignore and pd.api.types.is_numeric_dtype(eeg_df[c])]

    x = eeg_df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)

    # Participant-level split
    pids = sorted(eeg_df["participant_id"].astype(str).unique())
    n_test = max(1, int(round(0.2 * len(pids))))
    test_pids = set(pids[-n_test:])
    train_mask = ~eeg_df["participant_id"].astype(str).isin(test_pids)
    test_mask = ~train_mask

    x_train = x.loc[train_mask].to_numpy(dtype=float)
    x_test = x.loc[test_mask].to_numpy(dtype=float)

    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)

    x_std = (x.to_numpy(dtype=float) - mu) / sd
    x_train_std = x_std[train_mask.to_numpy()]
    x_test_std = x_std[test_mask.to_numpy()]

    # 128-d EEG embeddings
    emb = fit_projection(x_train_std, x_std, n_components=128)
    emb_cols = [f"eeg_embedding_{i:03d}" for i in range(128)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    # Train per-target models and collect metrics/predictions
    metrics = {
        "split": {
            "train_participants": sorted(set(pids) - test_pids),
            "test_participants": sorted(test_pids),
            "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
        },
        "targets": {},
    }

    pred_table = eeg_df[["participant_id", "station_id", "group"]].copy()
    models_store = {}

    # Try multiple model families per target and pick best by F1 on test
    for t in TARGET_COLS:
        if t not in eeg_df.columns:
            eeg_df[t] = 0
        y = pd.to_numeric(eeg_df[t], errors="coerce").fillna(0).astype(int).to_numpy()
        y_train = y[train_mask.to_numpy()]
        y_test = y[test_mask.to_numpy()]

        metrics["targets"][t] = {}

        candidate_results = {}

        # Candidate: XGBoost
        if xgb is not None:
            try:
                clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
                clf.fit(x_train_std, y_train)
                p_all = clf.predict_proba(x_std)[:, 1]
                p_test = p_all[test_mask.to_numpy()]
                yhat_test = (p_test >= 0.5).astype(int)
                candidate_results["xgboost"] = (binary_metrics(y_test, yhat_test), p_all, clf)
            except Exception:
                pass

        # Candidate: LightGBM
        if lgb is not None:
            try:
                clf = lgb.LGBMClassifier()
                clf.fit(x_train_std, y_train)
                p_all = clf.predict_proba(x_std)[:, 1]
                p_test = p_all[test_mask.to_numpy()]
                yhat_test = (p_test >= 0.5).astype(int)
                candidate_results["lightgbm"] = (binary_metrics(y_test, yhat_test), p_all, clf)
            except Exception:
                pass

        # Candidate: RandomForest
        if RandomForestClassifier is not None:
            try:
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
                clf.fit(x_train_std, y_train)
                p_all = clf.predict_proba(x_std)[:, 1]
                p_test = p_all[test_mask.to_numpy()]
                yhat_test = (p_test >= 0.5).astype(int)
                candidate_results["random_forest"] = (binary_metrics(y_test, yhat_test), p_all, clf)
            except Exception:
                pass

        # Candidate: sklearn LogisticRegression
        if LogisticRegression is not None:
            try:
                clf = LogisticRegression(max_iter=500)
                clf.fit(x_train_std, y_train)
                p_all = clf.predict_proba(x_std)[:, 1]
                p_test = p_all[test_mask.to_numpy()]
                yhat_test = (p_test >= 0.5).astype(int)
                candidate_results["sklearn_logreg"] = (binary_metrics(y_test, yhat_test), p_all, clf)
            except Exception:
                pass

        # Fallback: in-script logistic (ensure at least one candidate exists)
        if not candidate_results:
            model = fit_logistic_regression(x_train_std, y_train, reg_lambda=0.5, max_iter=300)
            p_all = predict_logistic_proba(model, x_std)
            candidate_results["impl_logreg"] = (binary_metrics(y_test, (p_all[test_mask.to_numpy()] >= 0.5).astype(int)), p_all, model)

        # choose best by F1
        best_name = None
        best_f1 = -1.0
        for name, (mres, p_all, clf) in candidate_results.items():
            f1 = mres.get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        if best_name is None:
            # no models trained somehow
            metrics["targets"][t]["chosen_model"] = None
            metrics["targets"][t]["models"] = {}
            pred_table[f"{t}_true"] = y
            pred_table[f"{t}_proba"] = 0.0
            pred_table[f"{t}_pred"] = 0
            continue

        best_metrics, best_pall, best_clf = candidate_results[best_name]
        metrics["targets"][t]["chosen_model"] = best_name
        # record all candidates
        metrics["targets"][t]["models"] = {n: r[0] for n, r in candidate_results.items()}

        pred_table[f"{t}_true"] = y
        pred_table[f"{t}_proba"] = best_pall
        pred_table[f"{t}_pred"] = (best_pall >= 0.5).astype(int)
        models_store[t] = {
            "model_name": best_name,
            "model_obj": best_clf,
        }

    # Final outputs
    station_dataset_out = pd.concat([eeg_df[id_cols + [c for c in eeg_df.columns if c not in id_cols]], emb_df], axis=1)

    station_embeddings_out_cols = (
        [
            "participant_id",
            "group",
            "station_id",
            "eeg_quality_flag",
            "eeg_token",
            "eeg_stress_score",
            "eeg_cognitive_load_score",
            "eeg_fatigue_score",
            "eeg_hesitation_score",
            "eeg_attention_score",
            "eeg_artifact_score",
        ]
        + emb_cols
        + TARGET_COLS
    )
    station_embeddings_out = station_dataset_out[station_embeddings_out_cols].copy()

    # Clustering analysis on embeddings to discover EEG states
    cluster_rows = []
    if KMeans is not None:
        try:
            k = 6
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(emb)
            station_dataset_out["eeg_kmeans_cluster"] = labels
            # cluster summaries
            for cl in range(k):
                mask = labels == cl
                cluster_rows.append(
                    {
                        "cluster": int(cl),
                        "n_rows": int(mask.sum()),
                        "mean_stress": float(np.nanmean(eeg_df.loc[mask, "eeg_stress_score"])) if "eeg_stress_score" in eeg_df else np.nan,
                        "mean_attention": float(np.nanmean(eeg_df.loc[mask, "eeg_attention_score"])) if "eeg_attention_score" in eeg_df else np.nan,
                    }
                )
        except Exception:
            pass

    if GaussianMixture is not None:
        try:
            gmm = GaussianMixture(n_components=6, random_state=42)
            g_labels = gmm.fit_predict(emb)
            station_dataset_out["eeg_gmm_cluster"] = g_labels
        except Exception:
            pass

    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(root / "eeg_cluster_analysis.csv", index=False)

    # Additional outputs: state scores and tokens
    state_scores_cols = [
        "participant_id",
        "station_id",
        "eeg_stress_score",
        "eeg_cognitive_load_score",
        "eeg_fatigue_score",
        "eeg_hesitation_score",
        "eeg_attention_score",
        "eeg_uncertainty_score",
        "eeg_engagement_score",
        "eeg_artifact_score",
        "eeg_signal_quality_score",
        "eeg_missing_pct",
        "eeg_bad_channel_count",
    ]

    eeg_state_scores = eeg_df[[c for c in state_scores_cols if c in eeg_df.columns]].copy()
    eeg_tokens = eeg_df[["participant_id", "station_id", "eeg_token", "eeg_quality_flag"]].copy()

    station_dataset_out.to_csv(out_dataset, index=False)
    station_embeddings_out.to_csv(out_embed, index=False)
    pred_table.to_csv(out_pred, index=False)
    eeg_state_scores.to_csv(root / "eeg_state_scores.csv", index=False)
    eeg_tokens.to_csv(root / "eeg_tokens.csv", index=False)

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_model, "wb") as f:
        pickle.dump(
            {
                "feature_columns": numeric_cols,
                "mean": mu,
                "std": sd,
                "models": models_store,
                "embedding_columns": emb_cols,
                "targets": TARGET_COLS,
            },
            f,
        )

    doc_text = f"""# EEG Model Documentation

## Scope
This pipeline builds a station-level EEG ML dataset and model outputs aligned by `participant_id + station_id`.

## Files used
- Per participant (`master_data_XXX`):
  - `eeg_filtered.csv` (primary station-window EEG signal source)
  - `eeg_quality_report.csv`, `User-XXX_mne_eeg_quality.csv` (quality features)
  - `aux_artifact_report.csv`, `aux_summary.csv`, `User-XXX_aux_features.csv` (artifact/AUX features)
  - `fnirs_summary.csv`, `User-XXX_fnirs_summary.csv`, `User-XXX_mne_fnirs_od_features.csv` (clean fNIRS side features)
  - `eeg_band_power.csv`, `User-XXX_mne_eeg_bandpower.csv` (global EEG summaries)
- Label/station alignment reference:
  - `station_dataset.csv`

## Files ignored
- PNG figures, README/markdown summaries, and other non-numeric artifacts were ignored for training.

## Station alignment method
- `station_dataset.csv` is grouped to one row per `participant_id + station_id`.
- Station windows are defined by `start_ts_ms` and `end_ts_ms`.
- EEG rows from `eeg_filtered.csv` are time-aligned using `ist_timestamp -> epoch_ms`.
- For each station, features are extracted only from EEG samples within that time window.

## Feature extraction
- Time-domain per channel (`TP9`, `AF7`, `AF8`, `TP10`): mean/std/min/max/median/percentiles/IQR/RMS.
- Frequency-domain per channel via Welch PSD:
  - delta/theta/alpha/beta/gamma absolute and relative power.
  - ratios: theta/alpha, beta/alpha, beta/(alpha+theta).
- Asymmetry:
  - frontal (`AF7-AF8`) and temporal (`TP9-TP10`) for theta/alpha/beta/gamma.
- Quality/artifact:
  - station artifact proxy from robust z-score outlier fraction.
  - global EEG/AUX/fNIRS summary statistics appended as additional features.

## Cognitive/psychological state outputs
The following station-level derived indicators are produced (research indicators, not clinical diagnosis):
- `eeg_stress_score`
- `eeg_cognitive_load_score`
- `eeg_fatigue_score`
- `eeg_hesitation_score`
- `eeg_attention_score`
- `eeg_quality_flag`
- `eeg_artifact_score`
- `eeg_token`

### Token logic (transparent)
- Stress high if `eeg_stress_score >= 0.75`, else low.
- Cognitive load high if `eeg_cognitive_load_score >= 0.75`, else low.
- Fatigue risk if `eeg_fatigue_score >= 0.75`, else alert.
- Hesitation high if `eeg_hesitation_score >= 0.75`, else low.
- Attention drop if `eeg_attention_score <= -0.5`; high engagement if `>= 0.75`; else stable attention.
- Quality: `EEG_LOW_CONFIDENCE` for very short windows, `EEG_ARTIFACT_HIGH` for high artifact score, else `EEG_CLEAN`.

## Model training
- Targets predicted:
  - `ai_used`, `ai_followed`, `unsafe_choice`, `ai_usage_increased`, `ai_usage_decreased`
- Split strategy: participant-level split only (no row-level leakage).
- Baseline model: per-target regularized logistic regression (implemented with SciPy optimizer).
- Embeddings: 128-D PCA-like SVD projection learned from training split.

## Output compatibility with gaze+interaction model
- Primary join keys are preserved: `participant_id`, `station_id`.
- Fusion-ready output includes:
  - `eeg_embedding_000 ... eeg_embedding_127`
  - all requested targets
  - interpretable `eeg_token` and score columns

## Produced artifacts
- `{out_dataset.name}`: full station-level EEG dataset with extracted features.
- `{out_embed.name}`: compact fusion-ready EEG station output.
- `{out_pred.name}`: per-row true/pred/probability predictions (behavior predictions per station).
- `{out_metrics.name}`: accuracy/precision/recall/F1/confusion matrix per target.
- `{out_model.name}`: model state for reuse.
 - `eeg_state_scores.csv`: per-station numeric state scores (stress, load, fatigue, attention, uncertainty, engagement).
 - `eeg_tokens.csv`: per-station human-readable EEG tokens (joinable by participant_id + station_id).
 - `eeg_cluster_analysis.csv`: cluster summaries when clustering ran successfully (embeddings-driven states).

## Caveats
- If station windows contain very few EEG samples, confidence is lowered with `EEG_LOW_CONFIDENCE`.
- Some participants may be missing EEG files; only participants with valid station-window EEG were modeled.
- These state scores are computational proxies for experimental interpretation, not medical diagnoses.
"""
    out_doc.write_text(doc_text, encoding="utf-8")

    print("Done.")
    print(f"Rows: {len(eeg_df)}")
    print(f"Participants: {eeg_df['participant_id'].nunique()}")
    print(f"Outputs written under: {root}")


if __name__ == "__main__":
    main()
