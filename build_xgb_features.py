#!/usr/bin/env python3
"""
Build per-participant and cross-subject feature tables from master_data_<id> folders.
- Produces per-participant: master_features_<id>.csv (one row)
- Produces cross-subject: cross_subject_features.csv (one row per participant)
- Produces participant blob table: participant_all_data.csv (one row per participant, one JSON column)

Extraction strategy (generic/simple but robust):
- For each CSV in participant folder: load with pandas
- For numeric columns: compute mean, std, min, max, pct_nan
- If a column called 'channel' or 'ch' exists and other columns are band names, create per-channel_band_mean features
- Prefix features by filename (sanitized)
- Save raw aggregated features; also save a processed XGBoost-ready CSV with median imputation

Requires: pandas, numpy
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/Users/saumya/Downloads/EV-VR Master File")
OUT_CROSS = BASE / "cross_subject_features.csv"
OUT_PARTICIPANT_BLOB = BASE / "participant_all_data.csv"
OUT_PARTICIPANT_BLOB_LABELED = BASE / "participant_all_data_labeled.csv"
OUT_WIDE_LABELED = BASE / "cross_subject_features_xgb_ready_labeled.csv"
MASTER_GLOB = BASE / "master_data_*"

def safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def agg_df(df: pd.DataFrame, prefix: str) -> dict:
    res = {}
    # keep numeric columns
    num = df.select_dtypes(include=[np.number])
    for col in num.columns:
        series = num[col]
        res[f"{prefix}__{col}__mean"] = float(series.mean()) if not series.empty else np.nan
        res[f"{prefix}__{col}__std"] = float(series.std()) if not series.empty else np.nan
        res[f"{prefix}__{col}__min"] = float(series.min()) if not series.empty else np.nan
        res[f"{prefix}__{col}__max"] = float(series.max()) if not series.empty else np.nan
        res[f"{prefix}__{col}__pct_nan"] = float(series.isna().mean()) if not series.empty else 1.0
    # detect channel x band style tables
    cols_lower = [c.lower() for c in df.columns]
    if any("channel" == c for c in cols_lower) or any(c in ("ch", "electrode") for c in cols_lower):
        # try to pivot: find band-like columns (delta/theta/alpha/beta/gamma)
        band_cols = [c for c in df.columns if c.lower() in ("delta","theta","alpha","beta","gamma")]
        ch_col = None
        for c in df.columns:
            if c.lower() == "channel" or c.lower() == "ch" or c.lower() == "electrode":
                ch_col = c
                break
        if ch_col and band_cols:
            for _, row in df.dropna(subset=[ch_col]).iterrows():
                ch = str(row[ch_col])
                for b in band_cols:
                    val = row.get(b, np.nan)
                    key = f"{prefix}__{ch}__{b}__val"
                    # if multiple rows per channel/band exist, aggregate later by mean
                    if key in res:
                        # keep as running mean via list
                        if not isinstance(res[key], list):
                            res[key] = [res[key]]
                        res[key].append(val)
                    else:
                        res[key] = val
            # finalize lists
            for k in list(res.keys()):
                if isinstance(res[k], list):
                    arr = pd.Series(res[k]).dropna()
                    res[k] = float(arr.mean()) if not arr.empty else np.nan
    return res


def process_participant(pdir: Path) -> tuple[str, dict]:
    all_features = {}
    participant = pdir.name.replace("master_data_", "")
    candidate_files = list(pdir.glob("*.csv")) + list(pdir.glob("*.txt")) + list(pdir.glob("*.md"))
    csv_files = []
    for f in candidate_files:
        name = f.name.lower()
        # skip generated outputs so reruns don't recursively ingest features
        if name.startswith("master_features_"):
            continue
        csv_files.append(f)
    for f in csv_files:
        name = safe_name(f.stem)
        try:
            df = pd.read_csv(f)
        except Exception:
            # try reading with low memory and python engine
            try:
                df = pd.read_csv(f, engine="python", error_bad_lines=False)
            except Exception:
                # skip unreadable
                continue
        feats = agg_df(df, name)
        # merge
        for k,v in feats.items():
            # if key collision happens, keep latest with filename prefix; collisions unlikely
            all_features[k] = v
    # include simple file-level meta: number of CSVs read
    all_features["num_files"] = len(csv_files)
    return participant, all_features


def main():
    parts = sorted([p for p in BASE.glob("master_data_*") if p.is_dir()])
    rows = []
    blob_rows = []
    for p in parts:
        pid, feats = process_participant(p)
        if not feats:
            print(f"Warning: no features for {pid}")
            continue
        df = pd.DataFrame([feats], index=[pid])
        out_file = p / f"master_features_{pid}.csv"
        df.to_csv(out_file)
        rows.append(df.assign(participant=pid))
        blob_rows.append({"participant_id": pid, "all_data_json": json.dumps(feats, ensure_ascii=False, sort_keys=True)})
        print(f"Wrote {out_file} ({df.shape[1]} features)")

    if not rows:
        print("No participant features produced.")
        return
    big = pd.concat(rows).reset_index(drop=True).set_index('participant')
    # unify columns (already aligned by concat)
    big.to_csv(OUT_CROSS)
    print(f"Wrote cross-subject features to {OUT_CROSS} ({big.shape[0]} x {big.shape[1]})")

    # XGBoost-ready: simple median impute
    processed = big.copy()
    for col in processed.columns:
        if processed[col].dtype.kind in 'biufc':
            med = processed[col].median()
            processed[col] = processed[col].fillna(med)
    processed_out = BASE / "cross_subject_features_xgb_ready.csv"
    processed.to_csv(processed_out)
    print(f"Wrote XGBoost-ready features to {processed_out}")

    processed_labeled = processed.reset_index().rename(columns={"participant": "participant_id"})
    processed_labeled.to_csv(OUT_WIDE_LABELED, index=False)
    print(f"Wrote labeled XGBoost-ready table to {OUT_WIDE_LABELED} ({processed_labeled.shape[0]} x {processed_labeled.shape[1]})")

    blob_df = pd.DataFrame(blob_rows).sort_values("participant_id")
    blob_df.to_csv(OUT_PARTICIPANT_BLOB, index=False)
    print(f"Wrote participant blob table to {OUT_PARTICIPANT_BLOB} ({blob_df.shape[0]} rows x {blob_df.shape[1]} cols)")

    blob_df.to_csv(OUT_PARTICIPANT_BLOB_LABELED, index=False)
    print(f"Wrote labeled participant blob table to {OUT_PARTICIPANT_BLOB_LABELED} ({blob_df.shape[0]} rows x {blob_df.shape[1]} cols)")

if __name__ == '__main__':
    main()
