"""
RESS Offline Analysis (DC + Bandpass + SNS)
===========================================

Run SSVEP RESS classification offline on all XDF files in a directory.
Pipeline matches cca_offline_sns: DC removal + Bandpass 2–45 Hz + SNS,
then leave-one-trial-out RESS per subject.

Pipeline:
- Load EEG + markers using `load_xdf_eeg` from `xdf_loader.py` with
  bandpass 2–45 Hz + DC removal
- Apply SNS from `meegkit.sns` on the continuous EEG
- Parse Start1..Start24 / Stop1..Stop24 markers into trials
- For each trial: fit 6 RESS filters (one per target frequency) on all other
  trials of the same subject (leave-one-trial-out), transform the test segment,
  score by power at f and 2f, predict by argmax
- Aggregate results into a CSV and a summary figure
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

from meegkit.ress import RESS
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
    - start_marker, stop_marker
    - start_time, stop_time  (absolute XDF timestamps)
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


def _score_ress_component_at_freq(
    component_1d: np.ndarray,
    sfreq: float,
    peak_freq: float,
    n_harmonics: int = 2,
) -> float:
    """
    Score a RESS component time series by power at peak_freq and its harmonics.

    Parameters
    ----------
    component_1d : np.ndarray
        One-dimensional component signal, shape (n_samples,).
    sfreq : float
        Sampling frequency in Hz.
    peak_freq : float
        Target frequency (e.g. 7, 8, 9, 11, 7.5, 8.5).
    n_harmonics : int
        Number of harmonics to include (default 2: f and 2f).

    Returns
    -------
    float
        Sum of power at peak_freq, 2*peak_freq, ... (magnitude squared at nearest FFT bin).
    """
    n = len(component_1d)
    if n == 0:
        return 0.0
    comp = np.asarray(component_1d, dtype=float).ravel()
    comp = comp - np.mean(comp)
    spec = np.fft.rfft(comp)
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    power = np.abs(spec) ** 2
    total = 0.0
    for h in range(1, n_harmonics + 1):
        f = peak_freq * h
        if f >= freqs[-1]:
            break
        idx = np.argmin(np.abs(freqs - f))
        total += power[idx]
    return float(total)


FREQS = [7, 8, 9, 11, 7.5, 8.5]  # mobile order


def run_ress_offline(
    xdf_paths: Iterable[Path],
    duration: float = 4.0,
    neig_freq: float = 0.5,
    peak_width: float = 0.5,
    neig_width: float = 0.5,
    n_keep: int = 1,
    gamma: float = 0.01,
    sns_n_neighbors: int = 6,
    sns_skip: int = 1,
    score_norm: str = "zscore",
) -> pd.DataFrame:
    """
    Run offline RESS (leave-one-trial-out) on all provided XDF files.

    Preprocessing: DC removal + Bandpass 2–45 Hz (xdf_loader) + SNS.
    For each trial, 6 RESS filters are fitted on the other trials of the same
    subject; the test segment is transformed and scored by power at f and 2f.

    Parameters
    ----------
    xdf_paths : iterable of Path
        Paths to .xdf files.
    duration : float
        Segment length in seconds (default: 4.0).
    neig_freq, peak_width, neig_width, n_keep, gamma
        RESS parameters (see meegkit.ress.RESS).
    sns_n_neighbors, sns_skip
        SNS parameters.
    score_norm : str
        How to normalize the 6 frequency scores before argmax: "none" (raw power),
        "zscore" (z-score across 6 scores per trial), or "power_ratio" (score =
        power at f / total component power, then z-score). Default "zscore".

    Returns
    -------
    pd.DataFrame
        One row per trial: subject, trial_idx, true_freq, pred_freq, correct, scores, ...
    """
    records: List[Dict[str, Any]] = []

    for xdf_path in xdf_paths:
        xdf_path = Path(xdf_path)
        subject_id = xdf_path.stem
        print(f"\nProcessing {xdf_path} (subject={subject_id}) [bandpass+dc+sns+RESS]")

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

        try:
            X = eeg_data[:, :, np.newaxis]
            eeg_sns, _ = sns_denoise(X, n_neighbors=sns_n_neighbors, skip=sns_skip)
            eeg_sns = eeg_sns[:, :, 0]
        except Exception as e:
            print(f"  [WARNING] SNS failed for {xdf_path}: {e}")
            continue

        trials = _extract_trials_from_markers(markers, marker_ts)
        if not trials:
            print(f"  [WARNING] Could not parse Start/Stop markers in {xdf_path}, skipping.")
            continue

        segments: List[np.ndarray] = []
        trial_meta: List[Dict[str, Any]] = []
        for trial in trials:
            seg, used_dur = _cut_trial_segment(
                eeg_data=eeg_sns,
                eeg_ts=eeg_ts,
                start_time=float(trial["start_time"]),
                stop_time=float(trial["stop_time"]),
                srate=srate,
                duration=duration,
            )
            if seg is None or used_dur <= 0:
                continue
            segments.append(seg)
            trial_meta.append(
                {
                    "trial_idx": trial["trial_idx"],
                    "start_marker": trial["start_marker"],
                    "stop_marker": trial["stop_marker"],
                    "start_time": trial["start_time"],
                    "stop_time": trial["stop_time"],
                    "used_duration": used_dur,
                    "true_freq": FREQS[(trial["trial_idx"] - 1) % len(FREQS)],
                }
            )

        n_trials_valid = len(segments)
        if n_trials_valid < 2:
            print(
                f"  [WARNING] Need at least 2 valid segments for leave-one-out, got {n_trials_valid}, skipping."
            )
            continue

        # RESS requires all segments to have the same length (n_samples).
        # Use the minimum length and take the last n_samples from each segment
        # (closest to stop marker) so we do not drop trials.
        n_samples = min(seg.shape[0] for seg in segments)
        segments = [seg[-n_samples:, :].copy() for seg in segments]

        sfreq = int(srate)

        for t in range(n_trials_valid):
            X_train = np.stack([segments[i] for i in range(n_trials_valid) if i != t], axis=2)
            if X_train.shape[2] < 1:
                continue

            meta = trial_meta[t]
            test_seg = segments[t]

            scores = np.zeros(len(FREQS))
            for ifreq, f in enumerate(FREQS):
                ress = RESS(
                    sfreq=sfreq,
                    peak_freq=f,
                    neig_freq=neig_freq,
                    peak_width=peak_width,
                    neig_width=neig_width,
                    n_keep=n_keep,
                    gamma=gamma,
                    compute_unmixing=False,
                )
                ress.fit(X_train)
                comp = ress.transform(test_seg)  # (n_samples, n_keep)
                comp_1d = np.squeeze(comp)
                if comp_1d.ndim > 1:
                    comp_1d = comp_1d[:, 0]
                power_at_f = _score_ress_component_at_freq(comp_1d, srate, f, n_harmonics=2)
                if score_norm == "power_ratio":
                    total_power = np.sum(comp_1d.astype(float) ** 2)
                    scores[ifreq] = power_at_f / (total_power + 1e-10)
                else:
                    scores[ifreq] = power_at_f

            # Normalize scores across the 6 frequencies so they are comparable (same scale).
            # "zscore": raw power -> z-score per trial. "power_ratio": already ratio; still z-score.
            # "none": no normalization.
            if score_norm in ("zscore", "power_ratio"):
                scores_std = np.std(scores)
                if scores_std > 1e-10:
                    scores_norm = (scores - np.mean(scores)) / scores_std
                else:
                    scores_norm = scores.copy()
            else:
                scores_norm = scores.copy()

            pred_class_idx = int(np.argmax(scores_norm))
            pred_freq = FREQS[pred_class_idx]
            correct = bool(pred_freq == meta["true_freq"])

            records.append(
                {
                    "subject": subject_id,
                    "trial_idx": meta["trial_idx"],
                    "start_marker": meta["start_marker"],
                    "stop_marker": meta["stop_marker"],
                    "start_time": meta["start_time"],
                    "stop_time": meta["stop_time"],
                    "used_duration": meta["used_duration"],
                    "true_freq": meta["true_freq"],
                    "pred_class_idx": pred_class_idx,
                    "pred_freq": pred_freq,
                    "correct": correct,
                    "scores": list(np.asarray(scores, dtype=float)),
                    "preproc": "bandpass+dc+sns",
                }
            )

    if not records:
        print("No valid trials found across all XDF files (RESS).")
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
    plt.title("RESS Offline Accuracy by Frequency (DC + Bandpass + SNS)")
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
            "Run offline RESS analysis on XDF files: DC + Bandpass + SNS, "
            "then leave-one-trial-out RESS classification (meegkit)."
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
        help="Segment length in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save CSV and figures (default: results)",
    )
    parser.add_argument(
        "--score-norm",
        type=str,
        default="zscore",
        choices=["none", "zscore", "power_ratio"],
        help="Score normalization: none, zscore (default), or power_ratio",
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

    df_results = run_ress_offline(
        xdf_paths, duration=args.duration, score_norm=args.score_norm
    )
    if df_results.empty:
        print("No results to save.")
        return

    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = results_dir / "ress_offline_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results CSV to {csv_path}")

    overall_acc = df_results["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    acc_by_subject = df_results.groupby("subject")["correct"].mean()
    print("\nAccuracy by subject:")
    for subj, acc in acc_by_subject.items():
        print(f"  {subj}: {acc:.3f} ({acc*100:.1f}%)")

    acc_fig_path = results_dir / "ress_offline_accuracy_by_freq.png"
    _plot_accuracy_by_freq(df_results, acc_fig_path)


if __name__ == "__main__":
    main()
