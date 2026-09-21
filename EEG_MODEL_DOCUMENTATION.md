# EEG Model Documentation

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
- `eeg_station_dataset.csv`: full station-level EEG dataset with extracted features.
- `eeg_station_embeddings.csv`: compact fusion-ready EEG station output.
- `eeg_predictions.csv`: per-row true/pred/probability predictions.
- `eeg_metrics.json`: accuracy/precision/recall/F1/confusion matrix per target.
- `eeg_model_state.pkl`: model state for reuse.

## Caveats
- If station windows contain very few EEG samples, confidence is lowered with `EEG_LOW_CONFIDENCE`.
- Some participants may be missing EEG files; only participants with valid station-window EEG were modeled.
- These state scores are computational proxies for experimental interpretation, not medical diagnoses.
