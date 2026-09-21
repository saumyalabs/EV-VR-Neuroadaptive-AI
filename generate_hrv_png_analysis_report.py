#!/usr/bin/env python3
"""Generate a PNG-focused HRV interpretation report for each participant.

The report explains how to analyse the HRV PNGs and provides a short
result + interpretation for each user based on the existing cohort CSVs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path('/Users/saumya/Downloads/HRV outputs')
ANALYSIS_DIR = ROOT / 'HRV_PPG_Analysis_Report'
OUT_FILE = ANALYSIS_DIR / 'HRV_PNG_Analysis_and_Individual_Results.md'
PARTICIPANT_METRICS = ANALYSIS_DIR / 'Participant_Level_HRV_Metrics.csv'
QUALITY = ANALYSIS_DIR / 'Data_Quality_Assessment.csv'
COHORT = ANALYSIS_DIR / 'Cohort_Level_Statistics.csv'


def to_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def fmt(value, nd=2):
    if value is None:
        return 'NA'
    return f'{value:.{nd}f}'


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_data():
    metrics_rows = read_csv(PARTICIPANT_METRICS)
    quality_rows = read_csv(QUALITY)
    quality_by_id = {r['Participant_ID']: r for r in quality_rows}
    data = []
    for row in metrics_rows:
        pid = row['Participant_ID']
        q = quality_by_id.get(pid, {})
        data.append({
            'pid': pid,
            'hr': to_float(row.get('Mean_HR_bpm')),
            'rr_mean': to_float(row.get('RR_Mean_ms')),
            'sdnn': to_float(row.get('SDNN_ms')),
            'rmssd': to_float(row.get('RMSSD_ms')),
            'lfhf': to_float(row.get('LF_HF_Ratio')),
            'entropy': to_float(row.get('Sample_Entropy')),
            'rr_count': to_float(row.get('RR_Count')),
            'duration': to_float(row.get('Recording_Duration_sec')),
            'quality': q.get('Quality_Score') or 'Unknown',
            'flags': q.get('Quality_Flags') or 'None',
            'peak_ratio': q.get('Peak_Ratio') or 'NA',
        })
    return data


def load_cohort():
    rows = read_csv(COHORT)
    return {row['Metric']: row for row in rows}


def interpret(row):
    hr = row['hr']
    rr_count = row['rr_count']
    sdnn = row['sdnn']
    rmssd = row['rmssd']
    lfhf = row['lfhf']
    entropy = row['entropy']
    quality = row['quality']
    flags = row['flags']

    if rr_count is not None and rr_count < 30:
        return 'Very low RR count; do not rely on HRV metrics for strong conclusions.'

    parts = []
    if hr is not None:
        if hr < 70:
            parts.append('low heart rate')
        elif hr <= 105:
            parts.append('mid-range heart rate')
        else:
            parts.append('high heart rate')

    if sdnn is not None and rmssd is not None and lfhf is not None:
        if sdnn >= 300 or rmssd >= 350:
            parts.append('very strong variability')
        elif rmssd >= 220:
            parts.append('strong HRV')
        else:
            parts.append('moderate HRV')

        if lfhf < 1:
            parts.append('parasympathetic-dominant balance')
        elif lfhf > 2:
            parts.append('sympathetic-dominant balance')
        else:
            parts.append('balanced autonomic profile')

    if entropy is not None:
        if entropy < 1.5:
            parts.append('low complexity')
        elif entropy <= 2.3:
            parts.append('normal complexity')
        else:
            parts.append('high complexity')

    if quality in {'Fair', 'Poor'} or 'Very_Low_RR_Count' in flags or 'Poor_Peak_Detection' in flags:
        parts.append('low-confidence quality')
    elif quality == 'Good':
        parts.append('usable with checks')
    else:
        parts.append('reliable')

    return '; '.join(parts).capitalize() + '.'


def result_label(row):
    rr_count = row['rr_count']
    quality = row['quality']
    flags = row['flags']
    if rr_count is not None and rr_count < 30:
        return 'Low-confidence'
    if quality == 'Poor' or 'Poor_Peak_Detection' in flags:
        return 'Poor'
    if quality == 'Fair':
        return 'Fair'
    if quality == 'Good':
        return 'Good'
    return 'Excellent'


def build_report():
    data = load_data()
    cohort = load_cohort()

    lines = []
    lines.append('# HRV PNG Analysis Guide and Individual Results')
    lines.append('')
    lines.append('**Analysis date:** 12 June 2026')
    lines.append('')
    lines.append('This report shows how to analyse each user\'s HRV PNG and gives a short interpretation for every participant based on the cohort CSV outputs.')
    lines.append('')
    lines.append('## How to analyse the HRV PNG for each user')
    lines.append('')
    lines.append('When you open a participant\'s PNG, look at these items in order:')
    lines.append('')
    lines.append('1. **Heart rate / pulse** — tells you the average beat rate.')
    lines.append('2. **RR count** — tells you whether the file is long enough for stable HRV.')
    lines.append('3. **SDNN** — overall HRV variability.')
    lines.append('4. **RMSSD** — short-term parasympathetic variability.')
    lines.append('5. **LF/HF ratio** — rough autonomic balance.')
    lines.append('6. **Sample entropy** — complexity / adaptability of the signal.')
    lines.append('7. **Quality score and flags** — decide whether the file is reliable enough to trust.')
    lines.append('')
    lines.append('### Reading rules')
    lines.append('')
    lines.append('- **Higher HR / pulse** usually means shorter RR intervals and a more activated state.')
    lines.append('- **Lower HR / pulse** usually means longer RR intervals and may reflect rest or recovery.')
    lines.append('- **High SDNN and RMSSD** usually indicate stronger variability and better autonomic flexibility.')
    lines.append('- **LF/HF below 1** usually points to parasympathetic dominance.')
    lines.append('- **Very low RR count** or **poor peak detection** means the result is low confidence.')
    lines.append('- Always compare a user against the cohort median, not just against a single threshold.')
    lines.append('')
    lines.append('## Cohort reference points')
    lines.append('')
    lines.append('| Metric | Cohort median | Range |')
    lines.append('|---|---:|---:|')
    lines.append(f'| Mean HR | {cohort["Mean HR (bpm)"]["median"]} | {cohort["Mean HR (bpm)"]["min"]} to {cohort["Mean HR (bpm)"]["max"]} |')
    lines.append(f'| SDNN | {cohort["SDNN (ms)"]["median"]} | {cohort["SDNN (ms)"]["min"]} to {cohort["SDNN (ms)"]["max"]} |')
    lines.append(f'| RMSSD | {cohort["RMSSD (ms)"]["median"]} | {cohort["RMSSD (ms)"]["min"]} to {cohort["RMSSD (ms)"]["max"]} |')
    lines.append(f'| LF/HF | {cohort["LF/HF Ratio"]["median"]} | {cohort["LF/HF Ratio"]["min"]} to {cohort["LF/HF Ratio"]["max"]} |')
    lines.append(f'| Sample Entropy | {cohort["Sample Entropy"]["median"]} | {cohort["Sample Entropy"]["min"]} to {cohort["Sample Entropy"]["max"]} |')
    lines.append('')
    lines.append('## Individual user results')
    lines.append('')
    lines.append('| Participant | HR / Pulse (bpm) | RR count | SDNN | RMSSD | LF/HF | Quality | Result | Interpretation |')
    lines.append('|---|---:|---:|---:|---:|---:|---|---|---|')

    for row in data:
        lines.append(
            f'| {row["pid"]} | {fmt(row["hr"], 2)} | {fmt(row["rr_count"], 0)} | {fmt(row["sdnn"], 1)} | {fmt(row["rmssd"], 1)} | {fmt(row["lfhf"], 3)} | {row["quality"]} | {result_label(row)} | {interpret(row)} |'
        )

    lines.append('')
    lines.append('## How to use these results')
    lines.append('')
    lines.append('- **Excellent / Good** files are usually safe to report.')
    lines.append('- **Fair** files should be described cautiously.')
    lines.append('- **Poor** or **Low-confidence** files should be treated as likely artifacts or insufficient data.')
    lines.append('- For presentations, show the participant PNG together with the row from this table.')
    lines.append('')
    lines.append('## Related files')
    lines.append('')
    lines.append('- `HRV_PPG_Analysis_Report/Final_HeartRate_Pulse_Report.md`')
    lines.append('- `HRV_PPG_User_Interpretations/Per_User_Interpretation.md`')
    lines.append('- `HRV_PPG_PNG_Report/Combined_HeartRate_Pulse_Analysis.md`')
    lines.append('- `HRV_PPG_User_Interpretations/graphs_png/`')
    lines.append('- `HRV_PPG_PNG_Report/cohort_overview.png`')
    lines.append('- `HRV_PPG_PNG_Report/participant_gallery.png`')

    OUT_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUT_FILE}')


if __name__ == '__main__':
    build_report()
