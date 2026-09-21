import csv
import math
import statistics
from pathlib import Path

base = Path('/Users/saumya/Downloads/HRV outputs/muse_ppg_hrv_output_final')
summary_path = base / 'all_files_summary.csv'


def to_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values_sorted = sorted(values)
    return {
        'N': len(values),
        'mean': sum(values) / len(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': values_sorted[0],
        'max': values_sorted[-1],
    }


with summary_path.open() as f:
    rows = list(csv.DictReader(f))

print('rows', len(rows))

num_cols = [
    'Mean HR', 'SDNN', 'RMSSD', 'pNN50', 'LF Power', 'HF Power',
    'LF/HF Ratio', 'SD1', 'SD2', 'Sample Entropy', 'rr_intervals',
    'peaks', 'optics_rows'
]

for c in num_cols:
    d = describe([to_float(r.get(c, '')) for r in rows])
    if d is None:
        continue
    print(
        f"{c}\tN={d['N']}\tmean={d['mean']:.3f}\tmedian={d['median']:.3f}"
        f"\tstd={d['std']:.3f}\tmin={d['min']:.3f}\tmax={d['max']:.3f}"
    )

valid_freq = sum(1 for r in rows if r.get('LF/HF Ratio', '').strip() != '')
very_low_rr = [r['file'] for r in rows if (to_float(r.get('rr_intervals', '')) or 0) < 30]
short_record = [r['file'] for r in rows if (to_float(r.get('optics_rows', '')) or 0) < 10000]
extreme_hr = []
for r in rows:
    hr = to_float(r.get('Mean HR', ''))
    if hr is not None and (hr < 45 or hr > 120):
        extreme_hr.append(r['file'])

print('\nquality counts')
print('valid_freq', valid_freq)
print('very_low_rr', len(very_low_rr))
print('short_record', len(short_record))
print('extreme_hr', len(extreme_hr))
print('files with very_low_rr:', ', '.join(very_low_rr))
print('files with extreme_hr:', ', '.join(extreme_hr))

chan = {}
for r in rows:
    ch = r.get('ppg_channel', '')
    chan[ch] = chan.get(ch, 0) + 1
print('\nppg channel counts')
for k, v in sorted(chan.items(), key=lambda kv: kv[1], reverse=True):
    print(k, v)

for sid in ['008_muse', '015_muse', '034_muse']:
    d = base / sid

    def loadcol(path, col):
        out = []
        with path.open() as f:
            rd = csv.DictReader(f)
            for row in rd:
                v = to_float(row.get(col, ''))
                if v is not None:
                    out.append(v)
        return out

    hr = loadcol(d / f'{sid}_HeartRate.csv', 'HeartRate_bpm')
    pulse = loadcol(d / f'{sid}_Pulse.csv', 'Pulse_bpm')
    rr = loadcol(d / f'{sid}_RR_intervals.csv', 'RR_ms')

    ppg_f = []
    ppg_peaks = 0
    total = 0
    with (d / f'{sid}_PPG.csv').open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            v = to_float(row.get('PPG_filtered', ''))
            if v is not None:
                ppg_f.append(v)
            total += 1
            if row.get('is_peak', '').strip().lower() == 'true':
                ppg_peaks += 1

    print(f"\n{sid} lens HR={len(hr)} Pulse={len(pulse)} PPG={total} RR={len(rr)}")
    if hr:
        print(' HR bpm mean', round(sum(hr) / len(hr), 3), 'std', round(statistics.stdev(hr), 3) if len(hr) > 1 else 0)
    if rr:
        print(' RR ms mean', round(sum(rr) / len(rr), 3), 'std', round(statistics.stdev(rr), 3) if len(rr) > 1 else 0)
    if ppg_f:
        pf_std = round(statistics.stdev(ppg_f), 6) if len(ppg_f) > 1 else 0
        frac = round(ppg_peaks / total, 6) if total else 0
        print(' PPG filtered std', pf_std, 'peak fraction', frac)
