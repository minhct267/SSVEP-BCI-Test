"""
CCA Offline Analysis with SNS + ASR (Artifact Subspace Reconstruction)
======================================================================

Pipeline: DC removal + bandpass 2–45 Hz + SNS + ASR, then CCA per trial.

Experiment design (from protocol):
- 6 flicker frequencies 7, 8, 9, 11, 7.5, 8.5 Hz; 24 trials (4 cycles).
- Each trial: 5 s rest then 5 s flicker (all 6 flash); subject focuses on one.
- Markers: Start1..Start24, Stop1..Stop24. Rest segments ~5 s between Stop_i and Start_{i+1}.
- CCA uses the last 4 s of each trial (drop first 1 s of flicker for focus latency).

ASR calibration: Use rest segments (between Stop_i and Start_{i+1}) as "clean"
baseline—no flicker, minimal task-related artifact—to meet ASR's requirement
for clean calibration data (>= 30 s recommended). Rest segments are concatenated
and passed to ASR.fit(); then the full stream is transformed with ASR before
trial cutting.
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
from meegkit.asr import ASR
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
    Returns list of dicts with trial_idx, start_marker, stop_marker, start_time, stop_time.
    """
    if marker_ts is None or markers is None:
        return []

    labels = _normalize_markers(markers)
    if len(labels) != len(marker_ts):
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

    start_events.sort(key=lambda x: x[0])
    stop_events.sort(key=lambda x: x[0])

    n_trials = min(len(start_events), len(stop_events))
    trials: List[Dict[str, Any]] = []
    for i in range(n_trials):
        _, s_label, s_time = start_events[i]
        _, t_label, t_time = stop_events[i]
        trials.append({
            "trial_idx": i + 1,
            "start_marker": s_label,
            "stop_marker": t_label,
            "start_time": s_time,
            "stop_time": t_time,
        })
    return trials


def _get_rest_segment_sample_ranges(
    trials: List[Dict[str, Any]],
    eeg_ts: np.ndarray,
    min_rest_duration_sec: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Get sample-index ranges for rest segments: between Stop_i and Start_{i+1}.
    Optionally include segment from stream start to Start_1 if long enough.

    Returns
    -------
    list of (start_idx, end_idx) in sample space (end exclusive).
    """
    if not trials or eeg_ts is None or len(eeg_ts) == 0:
        return []

    n_samples = len(eeg_ts)
    eeg_start = float(eeg_ts[0])
    eeg_end = float(eeg_ts[-1])
    ranges: List[Tuple[int, int]] = []

    # Segment before first trial: [eeg_start, Start_1]
    t_start_first = float(trials[0]["start_time"])
    if t_start_first - eeg_start >= min_rest_duration_sec:
        start_idx = 0
        end_idx = int(np.searchsorted(eeg_ts, t_start_first, side="left"))
        end_idx = min(end_idx, n_samples)
        if end_idx - start_idx >= 1:
            ranges.append((start_idx, end_idx))

    # Rest between trials: [Stop_i, Start_{i+1}]
    for i in range(len(trials) - 1):
        stop_time = float(trials[i]["stop_time"])
        start_next = float(trials[i + 1]["start_time"])
        if start_next - stop_time < min_rest_duration_sec:
            continue
        start_idx = int(np.searchsorted(eeg_ts, stop_time, side="left"))
        end_idx = int(np.searchsorted(eeg_ts, start_next, side="right"))
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        if end_idx - start_idx >= 1:
            ranges.append((start_idx, end_idx))

    return ranges


def _cut_trial_segment(
    eeg_data: np.ndarray,
    eeg_ts: np.ndarray,
    start_time: float,
    stop_time: float,
    srate: float,
    duration: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Extract segment for one trial: `duration` seconds immediately before stop_time.
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


def run_cca_offline_sns_asr(
    xdf_paths: Iterable[Path],
    duration: float = 4.0,
    display: str = "mobile",
    asr_cutoff: float = 5.0,
    asr_calib_min_sec: float = 30.0,
) -> pd.DataFrame:
    """
    Run offline CCA with pipeline: bandpass+DC + SNS + ASR, then CCA.

    ASR is calibrated on rest segments (between Stop_i and Start_{i+1}).
    If total rest duration < asr_calib_min_sec, uses first asr_calib_min_sec
    seconds of the file as fallback.
    """
    records: List[Dict[str, Any]] = []

    for xdf_path in xdf_paths:
        xdf_path = Path(xdf_path)
        subject_id = xdf_path.stem
        print(f"\nProcessing {xdf_path} (subject={subject_id}) [bandpass+dc+sns+asr]")

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

        # SNS on continuous EEG (n_samples, n_channels)
        try:
            X = eeg_data[:, :, None]
            eeg_sns, _ = sns_denoise(X, n_neighbors=6, skip=1)
            eeg_sns = eeg_sns[:, :, 0]
        except Exception as e:
            print(f"  [WARNING] SNS failed for {xdf_path}: {e}")
            continue

        trials = _extract_trials_from_markers(markers, marker_ts)
        if not trials:
            print(f"  [WARNING] Could not parse Start/Stop markers in {xdf_path}, skipping.")
            continue

        # ASR calibration from rest segments
        rest_ranges = _get_rest_segment_sample_ranges(trials, eeg_ts, min_rest_duration_sec=1.0)
        calib_parts = [eeg_sns[s:e, :] for s, e in rest_ranges]
        if calib_parts:
            calib_data = np.concatenate(calib_parts, axis=0)
        else:
            calib_data = np.empty((0, eeg_sns.shape[1]))

        n_calib_samples = calib_data.shape[0]
        n_calib_sec = n_calib_samples / srate if srate > 0 else 0
        min_required = int(asr_calib_min_sec * srate)

        # Default: no ASR (use SNS-only)
        eeg_clean = eeg_sns

        if n_calib_samples < min_required:
            # Fallback: use first asr_calib_min_sec seconds of file
            if eeg_sns.shape[0] >= min_required:
                calib_data = eeg_sns[:min_required, :]
                n_calib_samples = min_required
                n_calib_sec = asr_calib_min_sec
                print(f"  [ASR] Rest segments < {asr_calib_min_sec}s; using first {asr_calib_min_sec:.0f}s for calibration.")
            else:
                print(f"  [WARNING] Insufficient data for ASR calibration (rest={n_calib_sec:.1f}s), skipping ASR for this file.")

        if n_calib_samples >= min_required:
            X_calib = calib_data.T
            try:
                asr = ASR(
                    sfreq=float(srate),
                    cutoff=asr_cutoff,
                    method="euclid",
                    estimator="scm",
                )
                asr.fit(X_calib)
                eeg_clean = asr.transform(eeg_sns.T).T
                print(f"  [ASR] Calibrated on {n_calib_sec:.1f}s rest segments, applied to full stream.")
            except Exception as e:
                print(f"  [WARNING] ASR failed for {xdf_path}: {e}, using SNS-only data.")

        freqs = [7, 8, 9, 11, 7.5, 8.5]
        clf = classifier(srate=int(srate), display=display, duration=duration, name=subject_id)

        for trial in trials:
            trial_idx = int(trial["trial_idx"])
            start_time = float(trial["start_time"])
            stop_time = float(trial["stop_time"])

            seg, used_dur = _cut_trial_segment(
                eeg_data=eeg_clean,
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

            eeg_input = seg.T
            try:
                cmd_idx, rhos = clf.get_ssvep_command(eeg_input)
            except Exception as e:
                print(f"  [WARNING] Trial {trial_idx}: CCA failed ({e}), skipping.")
                continue

            pred_class_idx = int(cmd_idx)
            pred_freq = clf.freqs[pred_class_idx]
            correct = bool(pred_freq == true_freq)

            records.append({
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
                "preproc": "bandpass+dc+sns+asr",
            })

    if not records:
        print("No valid trials found across all XDF files (bandpass+dc+sns+asr).")
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


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
    plt.title("CCA Offline Accuracy by Frequency (SNS+ASR)")
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
            "Run offline CCA with pipeline: bandpass+DC+SNS+ASR (calibration on rest segments), then cca.classifier"
        ),
    )
    parser.add_argument("--raw-dir", type=str, default="raw", help="Directory containing .xdf files (default: raw)")
    parser.add_argument("--pattern", type=str, default="*.xdf", help="Glob pattern for XDF files (default: *.xdf)")
    parser.add_argument("--duration", type=float, default=4.0, help="Segment length (seconds) for CCA (default: 4.0)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSV and figures (default: results)")
    parser.add_argument("--asr-cutoff", type=float, default=5.0, help="ASR cutoff (default: 5, conservative)")
    parser.add_argument("--asr-calib-min-sec", type=float, default=30.0, help="Minimum calibration duration in seconds (default: 30)")

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

    df_results = run_cca_offline_sns_asr(
        xdf_paths,
        duration=args.duration,
        display="mobile",
        asr_cutoff=args.asr_cutoff,
        asr_calib_min_sec=args.asr_calib_min_sec,
    )
    if df_results.empty:
        print("No results to save.")
        return

    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = results_dir / "cca_offline_results_sns_asr.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results CSV to {csv_path}")

    overall_acc = df_results["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    acc_by_subject = df_results.groupby("subject")["correct"].mean()
    print("\nAccuracy by subject:")
    for subj, acc in acc_by_subject.items():
        print(f"  {subj}: {acc:.3f} ({acc*100:.1f}%)")

    acc_fig_path = results_dir / "cca_offline_accuracy_by_freq_sns_asr.png"
    _plot_accuracy_by_freq(df_results, acc_fig_path)


if __name__ == "__main__":
    main()
