"""
CCA Offline Analysis with Sensor Noise Suppression (SNS)
========================================================

This script mirrors `cca_offline.py` but adds a Sensor Noise Suppression (SNS)
step from `meegkit.sns` after the initial bandpass + DC removal.

Pipeline:
- Load EEG + markers using `load_xdf_eeg` from `xdf_loader.py` with
  bandpass 2–45 Hz + DC removal (as in `cca_offline.py`)
- Apply SNS (n_neighbors=7, skip=0) on the continuous EEG
- Parse Start1..Start24 / Stop1..Stop24 markers into trials
- For each trial, cut a segment (default 4s, anchored on Stop marker) and run
  CCA classification using `classifier` from `cca.py`
- Aggregate results into a CSV and a summary figure, with `_sns` suffix
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca import classifier
from meegkit.sns import sns as sns_denoise
from xdf_loader import load_xdf_eeg


def _normalize_markers(markers: Any) -> List[str]:
    """Normalize marker time_series into a list of string labels."""
    labels: List[str] = []
    if markers is None:
        return labels

    if isinstance(markers, list):
        for d in markers:
            if isinstance(d, (list, np.ndarray)) and len(d) > 0:
                labels.append(str(d[0]))
            else:
                labels.append(str(d))
    elif isinstance(markers, np.ndarray):
        if markers.ndim == 2:
            for d in markers:
                labels.append(str(d[0]))
        else:
            for d in markers:
                labels.append(str(d))
    else:
        labels.append(str(markers))
    return labels


def _extract_trials_from_markers(
    markers: Any, marker_ts: Optional[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Parse StartN/StopN markers into ordered trials.

    Returns a list of dicts with:
    - trial_idx (1-based)
    - start_label, stop_label
    - start_time, stop_time  (absolute XDF timestamps)
    """
    if marker_ts is None or markers is None:
        return []

    labels = _normalize_markers(markers)
    if len(labels) != len(marker_ts):
        # Fallback: truncate to shortest
        n = min(len(labels), len(marker_ts))
        labels = labels[:n]
        marker_ts = marker_ts[:n]

    start_events: List[Tuple[int, str, float]] = []
    stop_events: List[Tuple[int, str, float]] = []
    pattern = re.compile(r"^(Start|Stop)(\d+)$", re.IGNORECASE)

    for lab, ts in zip(labels, marker_ts):
        m = pattern.match(lab.strip())
        if not m:
            continue
        kind = m.group(1).lower()
        num = int(m.group(2))
        if kind == "start":
            start_events.append((num, lab, float(ts)))
        else:
            stop_events.append((num, lab, float(ts)))

    if not start_events or not stop_events:
        return []

    # Sort by numeric index (Start1, Start2, ...)
    start_events.sort(key=lambda x: x[0])
    stop_events.sort(key=lambda x: x[0])

    n_trials = min(len(start_events), len(stop_events))
    trials: List[Dict[str, Any]] = []
    for i in range(n_trials):
        _, s_label, s_time = start_events[i]
        _, t_label, t_time = stop_events[i]
        trial_idx = i + 1
        trials.append(
            {
                "trial_idx": trial_idx,
                "start_marker": s_label,
                "stop_marker": t_label,
                "start_time": s_time,
                "stop_time": t_time,
            }
        )
    return trials


def _cut_trial_segment(
    eeg_data: np.ndarray,
    eeg_ts: np.ndarray,
    start_time: float,
    stop_time: float,
    srate: float,
    duration: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Extract a segment for one trial, anchored on the Stop marker:
    take `duration` seconds of data immediately BEFORE stop_time.
    """
    if eeg_ts is None or len(eeg_ts) == 0:
        return None, 0.0

    stop_idx = int(np.searchsorted(eeg_ts, stop_time, side="right"))
    stop_idx = max(1, min(stop_idx, len(eeg_ts)))

    target_len = int(duration * srate)
    start_idx = max(0, stop_idx - target_len)

    used_len = stop_idx - start_idx
    if used_len <= 0:
        return None, 0.0

    segment = eeg_data[start_idx:stop_idx, :]
    return segment, used_len / srate


def run_cca_offline_sns(
    xdf_paths: Iterable[Path],
    duration: float = 4.0,
    display: str = "mobile",
) -> pd.DataFrame:
    """
    Run offline CCA with SNS on all provided XDF files.

    Parameters
    ----------
    xdf_paths : iterable of Path
        Paths to .xdf files.
    duration : float
        Segment length in seconds used for CCA (default: 4.0).
    display : str
        Passed to `classifier` to select frequency ordering (default: 'mobile').

    Returns
    -------
    df_results : pandas.DataFrame
        One row per trial with columns such as:
        subject, trial_idx, true_freq, pred_freq, correct, ...
    """
    records: List[Dict[str, Any]] = []

    for xdf_path in xdf_paths:
        xdf_path = Path(xdf_path)
        subject_id = xdf_path.stem
        print(f"\nProcessing {xdf_path} (subject={subject_id}) [bandpass+dc+sns]")

        try:
            eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
                xdf_path, apply_filter=True, low=2.0, high=45.0, order=3
            )
        except Exception as e:
            print(f"  [WARNING] Skipping {xdf_path} due to load error: {e}")
            continue

        if markers is None or marker_ts is None:
            print(f"  [WARNING] No markers found in {xdf_path}, skipping.")
            continue

        # Apply Sensor Noise Suppression on continuous EEG
        try:
            X = eeg_data[:, :, None]  # (n_samples, n_chans, 1)
            eeg_sns, sns_matrix = sns_denoise(X, n_neighbors=7, skip=0)
            eeg_sns = eeg_sns[:, :, 0]
        except Exception as e:
            print(f"  [WARNING] SNS failed for {xdf_path}: {e}")
            continue

        trials = _extract_trials_from_markers(markers, marker_ts)
        if not trials:
            print(f"  [WARNING] Could not parse Start/Stop markers in {xdf_path}, skipping.")
            continue

        freqs = [7, 8, 9, 11, 7.5, 8.5]
        clf = classifier(srate=int(srate), display=display, duration=duration, name=subject_id)

        for trial in trials:
            trial_idx = int(trial["trial_idx"])
            start_time = float(trial["start_time"])
            stop_time = float(trial["stop_time"])

            seg, used_dur = _cut_trial_segment(
                eeg_data=eeg_sns,
                eeg_ts=eeg_ts,
                start_time=start_time,
                stop_time=stop_time,
                srate=srate,
                duration=duration,
            )
            if seg is None or used_dur <= 0:
                print(f"  [INFO] Trial {trial_idx}: invalid or empty segment, skipping.")
                continue

            true_freq = freqs[(trial_idx - 1) % len(freqs)]

            # classifier expects (n_channels, n_samples)
            eeg_input = seg.T
            try:
                cmd_idx, rhos = clf.get_ssvep_command(eeg_input)
            except Exception as e:
                print(f"  [WARNING] Trial {trial_idx}: CCA failed ({e}), skipping.")
                continue

            pred_class_idx = int(cmd_idx)
            pred_freq = clf.freqs[pred_class_idx]
            correct = bool(pred_freq == true_freq)

            records.append(
                {
                    "subject": subject_id,
                    "trial_idx": trial_idx,
                    "start_marker": trial["start_marker"],
                    "stop_marker": trial["stop_marker"],
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "used_duration": used_dur,
                    "true_freq": true_freq,
                    "pred_class_idx": pred_class_idx,
                    "pred_freq": pred_freq,
                    "correct": correct,
                    "rhos": list(np.asarray(rhos, dtype=float)),
                    "preproc": "bandpass+dc+sns",
                }
            )

    if not records:
        print("No valid trials found across all XDF files (bandpass+dc+sns).")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    return df


def _plot_accuracy_by_freq(df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of accuracy vs true frequency."""
    if df.empty:
        print("No data to plot accuracy by frequency.")
        return

    acc_by_freq = df.groupby("true_freq")["correct"].mean().reset_index()
    acc_by_freq = acc_by_freq.sort_values("true_freq")

    freqs = acc_by_freq["true_freq"].values
    acc = acc_by_freq["correct"].values

    plt.figure(figsize=(8, 5))
    bars = plt.bar(freqs, acc, width=0.4)
    plt.ylim(0, 1.05)
    plt.xlabel("True frequency (Hz)")
    plt.ylabel("Accuracy")
    plt.title("CCA Offline Accuracy by Frequency (SNS)")
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved accuracy figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline CCA analysis on XDF files using xdf_loader, SNS "
            "from meegkit.sns, and cca.classifier"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw",
        help="Directory containing .xdf files (default: raw)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.xdf",
        help="Glob pattern for XDF files (default: *.xdf)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Segment length (seconds) for CCA (default: 4.0)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save CSV and figures (default: results)",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return

    xdf_paths = sorted(raw_dir.glob(args.pattern))
    if not xdf_paths:
        print(f"[ERROR] No XDF files matching pattern '{args.pattern}' in {raw_dir}")
        return

    print(f"Found {len(xdf_paths)} XDF files in {raw_dir}")

    df_results = run_cca_offline_sns(xdf_paths, duration=args.duration, display="mobile")
    if df_results.empty:
        print("No results to save.")
        return

    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = results_dir / "cca_offline_results_sns.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results CSV to {csv_path}")

    # Basic statistics
    overall_acc = df_results["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    acc_by_subject = df_results.groupby("subject")["correct"].mean()
    print("\nAccuracy by subject:")
    for subj, acc in acc_by_subject.items():
        print(f"  {subj}: {acc:.3f} ({acc*100:.1f}%)")

    acc_fig_path = results_dir / "cca_offline_accuracy_by_freq_sns.png"
    _plot_accuracy_by_freq(df_results, acc_fig_path)


if __name__ == "__main__":
    main()

