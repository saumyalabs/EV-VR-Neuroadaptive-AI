from pathlib import Path
import math
import html

import numpy as np
import pandas as pd


# =========================
# Configurable variables
# =========================

INPUT_FOLDER = Path("/Users/saumya/Downloads/VR Muse Decoded Output")
OUTPUT_FOLDER = Path("/Users/saumya/Downloads/EV-VR Muse IST Data/VR_Muse_Analysis_Output")

TIMESTAMP_COLUMN = "ist_timestamp"
TIME_COLUMN = "time"

EEG_FILE = "EEG_data.csv"
AUX_FILE = "AUX_data.csv"
FNIRS_FILE = "fNIRS_data.csv"

EEG_COLUMNS = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]
AUX_COLUMNS = ["AUX_1", "AUX_2", "AUX_3", "AUX_4"]

EEG_BANDS_HZ = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_40": (30.0, 40.0),
}

CHUNK_ROWS = 100_000
PREVIEW_POINTS = 2000


# =========================
# General helpers
# =========================

def numeric_columns(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [col for col in columns if col not in {TIMESTAMP_COLUMN, TIME_COLUMN}]


def safe_read_first_last(path):
    first = pd.read_csv(path, nrows=1)
    last = None
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS):
        if not chunk.empty:
            last = chunk.tail(1)
    return first, last


def estimate_sampling_rate(path):
    try:
        sample = pd.read_csv(path, usecols=[TIME_COLUMN], nrows=50_000)
    except ValueError:
        return np.nan
    times = pd.to_numeric(sample[TIME_COLUMN], errors="coerce").dropna().to_numpy()
    if len(times) < 2:
        return np.nan
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(1.0 / np.median(diffs))


def summarize_signal_file(path, signal_name):
    if not path.exists():
        return {
            "signal": signal_name,
            "file_exists": False,
            "rows": 0,
            "start_ist": "",
            "end_ist": "",
            "duration_seconds": np.nan,
            "estimated_sampling_rate_hz": np.nan,
        }

    rows = 0
    start_ist = ""
    end_ist = ""
    min_time = np.inf
    max_time = -np.inf

    for chunk in pd.read_csv(path, usecols=[TIMESTAMP_COLUMN, TIME_COLUMN], chunksize=CHUNK_ROWS):
        if chunk.empty:
            continue
        rows += len(chunk)
        if not start_ist:
            start_ist = str(chunk[TIMESTAMP_COLUMN].iloc[0])
        end_ist = str(chunk[TIMESTAMP_COLUMN].iloc[-1])
        times = pd.to_numeric(chunk[TIME_COLUMN], errors="coerce")
        if times.notna().any():
            min_time = min(min_time, float(times.min()))
            max_time = max(max_time, float(times.max()))

    duration = max_time - min_time if rows and np.isfinite(min_time) and np.isfinite(max_time) else np.nan
    return {
        "signal": signal_name,
        "file_exists": True,
        "rows": rows,
        "start_ist": start_ist,
        "end_ist": end_ist,
        "duration_seconds": duration,
        "estimated_sampling_rate_hz": estimate_sampling_rate(path),
    }


def channel_statistics(path, columns):
    if not path.exists() or not columns:
        return pd.DataFrame()

    pieces = []
    usecols = [col for col in columns if col in pd.read_csv(path, nrows=0).columns]
    if not usecols:
        return pd.DataFrame()

    # Exact percentiles over full columns can be very memory-heavy for these files.
    # This uses pandas' chunk stats plus a bounded sample for distribution percentiles.
    running = {
        col: {"count": 0, "missing": 0, "sum": 0.0, "sum_sq": 0.0, "min": np.inf, "max": -np.inf, "sample": []}
        for col in usecols
    }

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
        for col in usecols:
            values = pd.to_numeric(chunk[col], errors="coerce")
            nonmissing = values.dropna()
            stats = running[col]
            stats["count"] += int(nonmissing.shape[0])
            stats["missing"] += int(values.isna().sum())
            if nonmissing.empty:
                continue
            arr = nonmissing.to_numpy(dtype=float)
            stats["sum"] += float(arr.sum())
            stats["sum_sq"] += float(np.square(arr).sum())
            stats["min"] = min(stats["min"], float(arr.min()))
            stats["max"] = max(stats["max"], float(arr.max()))
            if len(stats["sample"]) < 200_000:
                take = min(5000, len(arr), 200_000 - len(stats["sample"]))
                stats["sample"].extend(arr[:take].tolist())

    for col, stats in running.items():
        count = stats["count"]
        mean = stats["sum"] / count if count else np.nan
        variance = (stats["sum_sq"] / count - mean * mean) if count else np.nan
        std = math.sqrt(max(variance, 0.0)) if count else np.nan
        sample = np.asarray(stats["sample"], dtype=float)
        percentiles = np.percentile(sample, [1, 5, 50, 95, 99]) if sample.size else [np.nan] * 5
        pieces.append({
            "channel": col,
            "count": count,
            "missing": stats["missing"],
            "mean": mean,
            "std": std,
            "min": stats["min"] if np.isfinite(stats["min"]) else np.nan,
            "p01_sampled": percentiles[0],
            "p05_sampled": percentiles[1],
            "median_sampled": percentiles[2],
            "p95_sampled": percentiles[3],
            "p99_sampled": percentiles[4],
            "max": stats["max"] if np.isfinite(stats["max"]) else np.nan,
        })

    return pd.DataFrame(pieces)


def eeg_band_power(path, columns):
    if not path.exists():
        return pd.DataFrame()
    fs = estimate_sampling_rate(path)
    if not np.isfinite(fs) or fs <= 0:
        return pd.DataFrame()

    available = pd.read_csv(path, nrows=0).columns
    usecols = [col for col in columns if col in available]
    if not usecols:
        return pd.DataFrame()

    totals = {col: {band: 0.0 for band in EEG_BANDS_HZ} for col in usecols}
    total_power = {col: 0.0 for col in usecols}
    window_count = {col: 0 for col in usecols}
    window_size = int(min(max(round(fs * 4), 256), 2048))

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
        for col in usecols:
            values = pd.to_numeric(chunk[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) < window_size:
                continue
            n_windows = len(values) // window_size
            values = values[: n_windows * window_size].reshape(n_windows, window_size)
            values = values - values.mean(axis=1, keepdims=True)
            taper = np.hanning(window_size)
            spectra = np.abs(np.fft.rfft(values * taper, axis=1)) ** 2
            freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
            for band, (low, high) in EEG_BANDS_HZ.items():
                mask = (freqs >= low) & (freqs < high)
                if mask.any():
                    totals[col][band] += float(spectra[:, mask].sum(axis=1).sum())
            total_mask = (freqs >= 1.0) & (freqs < 40.0)
            total_power[col] += float(spectra[:, total_mask].sum(axis=1).sum())
            window_count[col] += n_windows

    rows = []
    for col in usecols:
        row = {"channel": col, "estimated_sampling_rate_hz": fs, "fft_window_samples": window_size, "windows": window_count[col]}
        for band in EEG_BANDS_HZ:
            absolute = totals[col][band] / window_count[col] if window_count[col] else np.nan
            relative = totals[col][band] / total_power[col] if total_power[col] else np.nan
            row[f"{band}_power"] = absolute
            row[f"{band}_relative"] = relative
        rows.append(row)
    return pd.DataFrame(rows)


def collect_preview(path, columns):
    if not path.exists() or not columns:
        return pd.DataFrame()
    usecols = [TIME_COLUMN] + [col for col in columns if col in pd.read_csv(path, nrows=0).columns]
    if len(usecols) <= 1:
        return pd.DataFrame()

    row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    if row_count <= 0:
        return pd.DataFrame()
    stride = max(1, row_count // PREVIEW_POINTS)
    chunks = []
    seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
        mask = ((np.arange(seen, seen + len(chunk)) % stride) == 0)
        chunks.append(chunk.loc[mask])
        seen += len(chunk)
    return pd.concat(chunks, ignore_index=True).head(PREVIEW_POINTS)


def svg_plot(data, columns, title, output_path):
    if data.empty:
        return
    width, height = 1100, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 25, 45, 45
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    t = pd.to_numeric(data[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_t = t[np.isfinite(t)]
    if finite_t.size == 0:
        return
    t_min, t_max = float(finite_t.min()), float(finite_t.max())
    if t_max == t_min:
        t_max = t_min + 1.0

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    all_values = []
    for col in columns:
        if col in data.columns:
            vals = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            all_values.append(vals[np.isfinite(vals)])
    if not all_values:
        return
    joined = np.concatenate([v for v in all_values if v.size])
    if joined.size == 0:
        return
    y_min, y_max = np.percentile(joined, [1, 99])
    if y_max == y_min:
        y_min, y_max = float(joined.min()), float(joined.max())
    if y_max == y_min:
        y_max = y_min + 1.0

    def x_scale(value):
        return margin_left + (value - t_min) / (t_max - t_min) * plot_w

    def y_scale(value):
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    polylines = []
    legend = []
    for index, col in enumerate(columns):
        if col not in data.columns:
            continue
        vals = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        points = []
        for x_val, y_val in zip(t, vals):
            if np.isfinite(x_val) and np.isfinite(y_val):
                y_clipped = min(max(y_val, y_min), y_max)
                points.append(f"{x_scale(x_val):.2f},{y_scale(y_clipped):.2f}")
        color = colors[index % len(colors)]
        if points:
            polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.3" points="{" ".join(points)}" />')
            legend.append(
                f'<text x="{margin_left + 130 * (index % 4)}" y="{height - 22 + 16 * (index // 4)}" '
                f'font-size="12" fill="{color}">{html.escape(col)}</text>'
            )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_left}" y="25" font-size="18" font-family="Arial">{html.escape(title)}</text>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>
<text x="{margin_left}" y="{height - 8}" font-size="12" font-family="Arial">time seconds: {t_min:.2f} to {t_max:.2f}</text>
<text x="8" y="{margin_top + 15}" font-size="12" font-family="Arial">1st-99th pct y-scale</text>
{''.join(polylines)}
{''.join(legend)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def write_readme(user_output, session, inventory):
    lines = [
        f"# {session} Biosignal Analysis",
        "",
        "Generated from decoded Muse CSV files.",
        "",
        "Files:",
        "- `data_inventory.csv`: file presence, row counts, time span, estimated sampling rate.",
        "- `EEG_channel_statistics.csv`, `AUX_channel_statistics.csv`, `fNIRS_channel_statistics.csv`: per-channel descriptive statistics.",
        "- `EEG_band_power.csv`: FFT-based delta/theta/alpha/beta/gamma power estimates for EEG channels.",
        "- `*_preview.csv`: downsampled rows for quick inspection.",
        "- `*_quicklook.svg`: browser-openable quick-look plots.",
        "",
        "Notes:",
        "- No missing EEG/AUX file was fabricated. If a source file is absent, the inventory records it.",
        "- fNIRS values are kept as decoded OPTICS channels; this script does not convert them to HbO/HbR.",
    ]
    user_output.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_session(session_folder):
    session = session_folder.name
    user_output = OUTPUT_FOLDER / session
    user_output.mkdir(parents=True, exist_ok=True)

    paths = {
        "EEG": session_folder / EEG_FILE,
        "AUX": session_folder / AUX_FILE,
        "fNIRS": session_folder / FNIRS_FILE,
    }

    inventory = pd.DataFrame([
        summarize_signal_file(paths["EEG"], "EEG"),
        summarize_signal_file(paths["AUX"], "AUX"),
        summarize_signal_file(paths["fNIRS"], "fNIRS"),
    ])
    inventory.to_csv(user_output / "data_inventory.csv", index=False)

    eeg_stats = channel_statistics(paths["EEG"], EEG_COLUMNS)
    if not eeg_stats.empty:
        eeg_stats.to_csv(user_output / "EEG_channel_statistics.csv", index=False)
        eeg_bands = eeg_band_power(paths["EEG"], EEG_COLUMNS)
        eeg_bands.to_csv(user_output / "EEG_band_power.csv", index=False)
        eeg_preview = collect_preview(paths["EEG"], EEG_COLUMNS)
        eeg_preview.to_csv(user_output / "EEG_preview.csv", index=False)
        svg_plot(eeg_preview, EEG_COLUMNS, f"{session} EEG Quick Look", user_output / "EEG_quicklook.svg")

    aux_stats = channel_statistics(paths["AUX"], AUX_COLUMNS)
    if not aux_stats.empty:
        aux_stats.to_csv(user_output / "AUX_channel_statistics.csv", index=False)
        aux_preview = collect_preview(paths["AUX"], AUX_COLUMNS)
        aux_preview.to_csv(user_output / "AUX_preview.csv", index=False)
        svg_plot(aux_preview, AUX_COLUMNS, f"{session} AUX Quick Look", user_output / "AUX_quicklook.svg")

    fnirs_cols = numeric_columns(paths["fNIRS"]) if paths["fNIRS"].exists() else []
    fnirs_stats = channel_statistics(paths["fNIRS"], fnirs_cols)
    if not fnirs_stats.empty:
        fnirs_stats.to_csv(user_output / "fNIRS_channel_statistics.csv", index=False)
        fnirs_preview = collect_preview(paths["fNIRS"], fnirs_cols)
        fnirs_preview.to_csv(user_output / "fNIRS_preview.csv", index=False)
        svg_plot(fnirs_preview, fnirs_cols, f"{session} fNIRS/OPTICS Quick Look", user_output / "fNIRS_quicklook.svg")

    write_readme(user_output, session, inventory)
    return {
        "session": session,
        "eeg_rows": int(inventory.loc[inventory["signal"] == "EEG", "rows"].iloc[0]),
        "aux_rows": int(inventory.loc[inventory["signal"] == "AUX", "rows"].iloc[0]),
        "fnirs_rows": int(inventory.loc[inventory["signal"] == "fNIRS", "rows"].iloc[0]),
        "output_folder": str(user_output),
    }


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    session_folders = sorted(folder for folder in INPUT_FOLDER.iterdir() if folder.is_dir())
    if not session_folders:
        raise FileNotFoundError(f"No user/session folders found in {INPUT_FOLDER}")

    rows = []
    for index, session_folder in enumerate(session_folders, start=1):
        print(f"[{index}/{len(session_folders)}] Analysing {session_folder.name}", flush=True)
        rows.append(analyze_session(session_folder))

    master = pd.DataFrame(rows)
    master.to_csv(OUTPUT_FOLDER / "all_users_analysis_summary.csv", index=False)
    print(f"Done. Analysis outputs are in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
