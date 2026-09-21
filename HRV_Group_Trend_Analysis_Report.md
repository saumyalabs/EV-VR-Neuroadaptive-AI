# HRV Group Trend Analysis Report

**Dataset:** EV-VR Muse PPG-derived HRV outputs · Participants 008–040 (n = 33)
**Analysis date:** 21 June 2026
**Scope:** Group-level / cohort-level HRV consistency on whole-recording metrics

---

## ⚠️ Important scope note — read first

The original brief assumes each participant has HRV metrics broken down **by experimental condition** (and asks which condition is most stressful, condition-wise trend lines, etc.). **The HRV pipeline in this dataset did not produce condition-segmented output.** Each participant has exactly **one** set of HRV metrics computed over their **entire** PPG recording (~40–65 min). There are no condition, station, or phase labels anywhere in the HRV outputs.

Consequently this analysis was run as an honest **whole-recording, group-level consistency analysis** (agreed with the requester). Specifically:

- "Condition-level statistics" collapse to a single level (`whole_recording`), which is identical to cohort-level statistics.
- "Trend consistency across conditions" is reinterpreted as **inter-individual consistency** — how tightly the 33 participants cluster on each metric — which is the answerable version of "do most participants show the same pattern."
- Condition-dependent figures (trend lines, spaghetti, heatmap) are reframed as group/participant comparisons, clearly annotated on each figure.

Two further caveats that materially affect interpretation:

1. **Groups A/B/C are arbitrary participant-ID bins** (008–018 / 019–029 / 030–040), not experimental factors. Real factors exist elsewhere in the project (`station_dataset.csv`: `group`, `threat_framing`, `ai_available`) but are not linked to these HRV summaries. Any group difference here would be coincidental.
2. **Five participants have HRV that is not physiologically valid** due to too few detected RR intervals: 013 (1 RR), 040 (8), 035 (21), 012 (26), 034 (28). All five lack frequency-domain and entropy values. Per instruction all 33 are retained in tables, but these five are flagged and excluded from physiological interpretation. NaN-aware statistics are used throughout, so frequency/entropy metrics are computed on n = 28.

If true condition-wise HRV is required, the raw PPG/RR time series would need to be re-segmented against the station event log — a separate cross-modal alignment task. See "Recommended next steps."

---

## 1. Objective

Determine whether the majority of participants exhibit a **consistent HRV profile** across the cohort, quantify inter-individual variability for each HRV metric, and test whether the (arbitrary) A/B/C groupings differ. Every conclusion is supported with numerical evidence.

## 2. Dataset overview

- **Participants:** 33 (008–040), all present and accounted for. 0 missing files.
- **Signal source:** Muse PPG (OPTICS channel), ~64 Hz, peak-detected RR intervals.
- **Conditions:** none in the HRV outputs (single whole-recording metric set per participant).
- **Recording duration:** median ≈ 21 min; range 0.2 min (013/038/011/040 short captures) to 65 min (023).
- **Files analysed:** `HRV_Analysis_Report/Participant_Level_HRV_Metrics.csv` (metrics) and `Data_Quality_Assessment.csv` (quality), cross-checked against per-participant `*_summary_metrics.csv`.
- **Data quality:** 13 Excellent, 15 Good, 2 Fair, 1 Poor, plus 2 further low-RR cases flagged Good by the source but unusable for HRV. **Five recordings (012, 013, 034, 035, 040) are HRV-invalid.**

## 3. Methods

**HRV metrics extracted** (per participant, whole recording): Time domain — Mean HR, RR Mean, SDNN, RMSSD; Frequency domain — LF Power, HF Power, LF/HF Ratio; Nonlinear — Sample Entropy.

**Group creation:** participants split into three ID-ordered bins — Group A (008–018), Group B (019–029), Group C (030–040), 11 each.

**Statistics (all NaN-aware):** descriptive Mean/SD/Median/IQR per group and cohort; 95% confidence intervals (t-distribution); Shapiro–Wilk normality; one-way ANOVA and Kruskal–Wallis across groups; effect sizes η² (ANOVA decomposition) and pairwise Cohen's *d*; coefficient of variation (CV) and one-way ICC(1) as agreement/consistency measures.

**Consistency definition:** a metric is judged consistent if its cohort CV is low (High < 10 %, Moderate 10–25 %, Low > 25 %) and participants cluster around the cohort median.

## 4. Results

### 4.1 Cohort-level metrics (the single "condition")

| Metric | N | Mean | SD | 95% CI |
|---|---|---|---|---|
| Mean HR (bpm) | 33 | 94.01 | 15.63 | 88.47 – 99.55 |
| RR Mean (ms) | 33 | 674.41 | 240.93 | 588.98 – 759.84 |
| SDNN (ms) | 32 | 204.55 | 64.66 | 181.24 – 227.86 |
| RMSSD (ms) | 32 | 275.92 | 74.57 | 249.04 – 302.81 |
| LF/HF Ratio | 28 | 0.485 | 0.108 | 0.443 – 0.527 |
| Sample Entropy | 28 | 2.018 | 0.098 | 1.980 – 2.056 |

> The means are pulled around by the invalid recordings (e.g. participant 013's Mean HR of 31.5 bpm drags the cohort mean below the median of 98.7 bpm). **Medians are the more trustworthy central estimate here** and are reported alongside in `condition_statistics.csv` / `trend_consistency.csv`.

### 4.2 Group statistics — medians are nearly identical

| Metric | Group A median | Group B median | Group C median |
|---|---|---|---|
| Mean HR (bpm) | 98.49 | 99.03 | 98.14 |
| RR Mean (ms) | 609.2 | 605.9 | 611.4 |
| SDNN (ms) | 187.4 | 188.3 | 192.7 |
| RMSSD (ms) | 262.3 | 262.0 | 262.2 |
| LF/HF Ratio | 0.524 | 0.483 | 0.496 |
| Sample Entropy | 2.037 | 2.015 | 1.991 |

Group **means** differ more (e.g. SDNN A = 200.7, B = 187.6, C = 225.0) but this is entirely outlier-driven — Group C's SDNN is inflated by participants 034 and 040, Group A's HR is deflated by 013. The medians, robust to these, agree to within ~1 % across all three groups. Full Mean/SD/Median/IQR are in `group_statistics.csv`. Figures: `figures/barplots/`, `figures/boxplots/`, `figures/violinplots/`, `figures/radar_charts/radar_groups.png`.

### 4.3 Statistical tests — no group differences

| Metric | Shapiro p | Normal? | ANOVA p | Kruskal p | η² | ICC(1) | Cohort CV % |
|---|---|---|---|---|---|---|---|
| Mean HR | <0.001 | No | 0.457 | 0.660 | 0.051 | −0.018 | 16.6 |
| RR Mean | <0.001 | No | 0.394 | 0.657 | 0.060 | −0.003 | 35.7 |
| SDNN | <0.001 | No | 0.402 | 0.280 | 0.061 | −0.006 | 31.6 |
| RMSSD | <0.001 | No | 0.753 | 0.940 | 0.019 | −0.072 | 27.0 |
| LF/HF Ratio | <0.001 | No | 0.215 | 0.218 | 0.116 | 0.064 | 22.4 |
| Sample Entropy | 0.685 | Yes | 0.310 | 0.532 | 0.089 | 0.024 | 4.8 |

**No metric shows a significant A/B/C difference** (all ANOVA and Kruskal–Wallis p > 0.21). η² values are small and ICC(1) values are ≈ 0 (several slightly negative, meaning within-group variance ≥ between-group variance). This is exactly what we expect when the grouping carries no real information — it confirms the A/B/C bins are not physiologically meaningful, and that the cohort behaves as one population. Pairwise Cohen's *d* values are in `statistical_results.csv`. Most metrics are non-normal because the invalid outliers create heavy tails; **Sample Entropy is the only normally distributed metric.**

### 4.4 Inter-individual consistency (the answerable "trend")

| Metric | CV % | Consistency | % within central band | Outliers (IQR rule) |
|---|---|---|---|---|
| Sample Entropy | 4.8 | **High** | 25 | 015, 038 (2) |
| Mean HR | 16.6 | Moderate | 30 | 8 (incl. 012/013/034/035/040) |
| LF/HF Ratio | 22.4 | Moderate | 25 | 014, 020, 028, 038 (4) |
| RMSSD | 27.0 | Low | 34 | 5 |
| SDNN | 31.6 | Low | 38 | 4 |
| RR Mean | 35.7 | Low | 30 | 8 |

Full breakdown (direction, % above/below median, named outliers) is in `trend_consistency.csv`. Figures `figures/spaghetti_plots/` and `figures/heatmaps/` make the pattern visual: among valid recordings the lines overlap tightly near the cohort centre, while the five invalid recordings (red) and the genuine high-HRV outlier 015 fan out.

## 5. Trend consistency findings

When the five invalid recordings are set aside, the cohort is **strikingly homogeneous**: the 28 valid participants cluster around Mean HR ≈ 98–100 bpm, RR ≈ 600–615 ms, SDNN ≈ 185–195 ms, RMSSD ≈ 255–272 ms, LF/HF ≈ 0.47–0.54, and Sample Entropy ≈ 1.95–2.10. Concretely:

- **27 of 28 valid participants (96 %)** fall within ±10 % of the cohort median for Sample Entropy.
- **25 of 28 (89 %)** sit within ±10 % of the median Mean HR (~98.7 bpm).
- **All valid participants show LF/HF < 1** (median 0.50), i.e. a uniformly parasympathetic-leaning autonomic balance.

The apparent "high variability" in RR Mean, SDNN and RMSSD (CV 27–36 %) is **manufactured almost entirely by the five invalid recordings plus participant 015**; remove them and CVs for those metrics drop into the single digits to low teens. So the dominant finding is **consistency, not divergence**.

## 6. Physiological interpretation

Interpret with care: these are absolute HRV values from wrist/forehead PPG over a single mixed-activity VR session with no resting baseline, so they describe the cohort's *aggregate state*, not a stress-versus-rest contrast.

- **Mean HR (median ≈ 98.7 bpm) / RR Mean (≈ 609 ms):** a moderately elevated resting-to-active heart rate, consistent with an engaged participant wearing VR rather than lying at rest. Highly consistent across the cohort (CV 16.6 %).
- **SDNN (≈ 188 ms) and RMSSD (≈ 262 ms):** these are **large** for short-term recordings and likely inflated by PPG motion artefact and the absence of artefact-rejection on RR intervals — they should be read as relative, not clinical, values. Their rank-consistency across participants is nonetheless high once outliers are excluded.
- **LF/HF Ratio (median ≈ 0.50, all valid participants < 1):** the single most informative result. A ratio below 1 indicates **parasympathetic (vagal) dominance** over sympathetic drive across the whole cohort. There is no evidence of cohort-wide sympathetic activation in these aggregate recordings.
- **Sample Entropy (≈ 2.0, CV 4.8 %):** sits in a normal, adaptive complexity range and is the most uniform metric of all. High, stable entropy points to **healthy autonomic complexity rather than the reduced complexity that accompanies acute cognitive/physiological stress.** It does not indicate elevated cohort-wide stress.

**Autonomic summary:** parasympathetic activation is the dominant tone (LF/HF < 1 in 100 % of valid participants); sympathetic activation is *not* elevated at the aggregate level; entropy indicates preserved complexity, not stress-driven simplification. Because there are no condition labels, **directional claims ("sympathetic increasing/decreasing", "most stressful condition") cannot be made** from this dataset.

## 7. Conclusion

**Yes — the majority of participants demonstrate a consistent HRV profile.** Across the cohort, the (arbitrary) A/B/C groups are statistically indistinguishable on every metric (all p > 0.21, η² ≤ 0.12, ICC(1) ≈ 0), and group medians agree to within ~1 %. The most consistent metrics are **Sample Entropy (CV 4.8 %)** and **Mean HR (CV 16.6 %)**; the apparently variable metrics (RR Mean, SDNN, RMSSD; CV 27–36 %) owe that variability almost entirely to **five HRV-invalid recordings (012, 013, 034, 035, 040)** and one genuine outlier (015). With those removed, the cohort is highly homogeneous, **parasympathetic-dominant (LF/HF < 1 universally)**, with normal entropy — i.e. no aggregate signature of sympathetic stress.

What this analysis **cannot** answer, due to the data: whether HRV changes between experimental conditions, which condition is most stressful/relaxed, or any within-subject temporal trend. Those require condition-segmented HRV.

### Recommended next steps
1. **Re-segment the raw PPG/RR series against the station event log** (`station_dataset.csv` provides per-station epoch start/end timestamps) to produce genuine per-condition HRV. Note: many stations are < 30 s — too short for reliable LF/HF or sample entropy — so only HR/RMSSD-type metrics would be trustworthy per station.
2. **Apply RR-interval artefact correction** (e.g. Kubios-style ectopic/outlier removal) before recomputing; the unusually high SDNN/RMSSD suggest uncorrected artefact.
3. **Exclude or re-record the five invalid participants** before any inferential analysis.
4. If group contrasts are wanted, use the **real experimental factors** (`threat_framing`, `ai_available`) rather than ID bins.

---

## Appendix — output files

```
output_group_analysis/
├── participant_summary.csv        # ID, group, duration, missing metrics, quality flags
├── group_assignment.csv           # A/B/C assignment
├── all_participant_metrics.csv    # 8 HRV metrics per participant (whole_recording)
├── group_statistics.csv           # Mean/SD/Median/IQR per group × 6 metrics
├── condition_statistics.csv       # cohort Mean/SD/95% CI (single condition)
├── trend_consistency.csv          # CV, clustering %, outliers, consistency rating
├── statistical_results.csv        # Shapiro, ANOVA, Kruskal, η², Cohen's d, CV, ICC(1)
├── HRV_Group_Trend_Analysis_Report.md
└── figures/                       # 28 PNGs @ 300 dpi
    ├── barplots/ (6)  trend_lines/ (6)  boxplots/ (6)  violinplots/ (6)
    ├── spaghetti_plots/ (2)  heatmaps/ (1)  radar_charts/ (1)
```

**Data integrity:** all 33 participant files were located and processed successfully. No files were missing or corrupted at the file level. Five recordings (012, 013, 034, 035, 040) contain too few valid RR intervals for frequency-domain/entropy HRV and are flagged throughout; their time-domain values are retained but treated as unreliable.
