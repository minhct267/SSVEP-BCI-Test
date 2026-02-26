"""
CCA Offline Analysis with ASR
=============================

Run SSVEP CCA classification offline on all XDF files in a directory,
but first clean the continuous EEG with Artifact Subspace Reconstruction
(ASR). This allows you to compare decoding accuracy before vs after ASR.

Pipeline:
- Load EEG + markers using ``load_xdf_eeg`` from ``xdf_loader.py``
  (band-pass 2–45 Hz + DC removal, same as ``cca_offline.py``).
- Parse Start1..Start24 / Stop1..Stop24 markers into trials.
- Use 4-second segments (same as CCA window) to calibrate ASR.
- Apply ASR to the full continuous recording using 1-second windows.
- Extract 4-second segments from the cleaned EEG and run CCA.
- Aggregate results across all subjects into a CSV and a summary figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca import classifier
from cca_offline import _cut_trial_segment, _extract_trials_from_markers
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from xdf_loader import load_xdf_eeg


def run_cca_offline_asr(
    xdf_paths: Iterable[Path],
    duration: float = 4.0,
    display: str = "mobile",
) -> pd.DataFrame:
    """
    Run offline CCA on ASR-cleaned EEG for all provided XDF files.

    Parameters
    ----------
    xdf_paths : iterable of Path
        Paths to .xdf files.
    duration : float
        Segment length in seconds used for CCA (default: 4.0).
    display : str
        Passed to `classifier` to select frequency ordering
        (default: 'mobile').

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
        print(f"\nProcessing {xdf_path} (subject={subject_id})")

        try:
            eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
                xdf_path,
                apply_filter=True,
                low=2.0,
                high=45.0,
                order=3,
            )
        except Exception as e:
            print(f"  [WARNING] Skipping {xdf_path} due to load error: {e}")
            continue

        if markers is None or marker_ts is None:
            print(f"  [WARNING] No markers found in {xdf_path}, skipping.")
            continue

        trials = _extract_trials_from_markers(markers, marker_ts)
        if not trials:
            print(f"  [WARNING] Could not parse Start/Stop markers in {xdf_path}, skipping.")
            continue

        n_samples, n_channels = eeg_data.shape
        print(f"  Loaded EEG: {n_samples} samples, {n_channels} channels, srate={srate:.2f} Hz")

        # ------------------------------------------------------------------
        # Calibrate ASR on all 4 s CCA segments concatenated
        # ------------------------------------------------------------------
        segs_for_calib: List[np.ndarray] = []
        total_calib_dur = 0.0

        for trial in trials:
            start_time = float(trial["start_time"])
            stop_time = float(trial["stop_time"])
            seg, used_dur = _cut_trial_segment(
                eeg_data=eeg_data,
                eeg_ts=eeg_ts,
                start_time=start_time,
                stop_time=stop_time,
                srate=srate,
                duration=duration,
            )
            if seg is None or used_dur <= 0:
                continue
            segs_for_calib.append(seg)
            total_calib_dur += used_dur

        if not segs_for_calib:
            print(
                f"  [WARNING] No valid 4 s segments for ASR calibration in {xdf_path}, "
                "skipping subject."
            )
            continue

        calib_concat = np.concatenate(segs_for_calib, axis=0)  # (n_samples_calib, n_channels)
        calib_data_T = calib_concat.T  # (n_channels, n_times)

        print(
            f"  ASR calibration: {len(segs_for_calib)} segments, "
            f"total {total_calib_dur:.2f} s (shape={calib_data_T.shape})"
        )

        asr = ASR(method="euclid")
        asr.fit(calib_data_T)

        # ------------------------------------------------------------------
        # Apply ASR to continuous data using 1-second non-overlapping windows
        # ------------------------------------------------------------------
        data_T = eeg_data.T  # (n_channels, n_times)
        win_samp = int(round(srate))
        step_samp = win_samp
        win_samp = max(1, win_samp)
        step_samp = max(1, step_samp)

        if n_samples < win_samp:
            print(
                f"  [WARNING] Recording too short for 1 s windows "
                f"(n_samples={n_samples}, win_samp={win_samp}), applying ASR once."
            )
            clean_T = asr.transform(data_T)
        else:
            X = sliding_window(data_T, window=win_samp, step=step_samp)
            # X shape: (n_channels, n_windows, win_samp)
            Y = np.zeros_like(X)
            n_windows = X.shape[1]
            print(
                f"  Running ASR on {n_windows} windows "
                f"(win={win_samp} samples, step={step_samp})"
            )
            for i in range(n_windows):
                Y[:, i, :] = asr.transform(X[:, i, :])

            clean_T = Y.reshape(n_channels, -1)

        n_samples_clean = clean_T.shape[1]
        if n_samples_clean < n_samples:
            print(
                f"  ASR output shorter than input "
                f"({n_samples_clean} < {n_samples}), truncating eeg_ts accordingly."
            )
        eeg_clean = clean_T.T  # (n_samples_clean, n_channels)
        eeg_ts_clean = eeg_ts[:n_samples_clean]

        # ------------------------------------------------------------------
        # CCA on ASR-cleaned data
        # ------------------------------------------------------------------
        freqs = [7, 8, 9, 11, 7.5, 8.5]
        clf = classifier(srate=int(srate), display=display, duration=duration, name=subject_id)

        for trial in trials:
            trial_idx = int(trial["trial_idx"])
            start_time = float(trial["start_time"])
            stop_time = float(trial["stop_time"])

            seg, used_dur = _cut_trial_segment(
                eeg_data=eeg_clean,
                eeg_ts=eeg_ts_clean,
                start_time=start_time,
                stop_time=stop_time,
                srate=srate,
                duration=duration,
            )
            if seg is None or used_dur <= 0:
                print(f"  [INFO] Trial {trial_idx}: invalid or empty segment after ASR, skipping.")
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
                    "pipeline": "asr",
                }
            )

    if not records:
        print("No valid trials found across all XDF files.")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    return df


def _plot_accuracy_by_freq_asr(df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of accuracy vs true frequency for ASR-cleaned data."""
    if df.empty:
        print("No data to plot accuracy by frequency (ASR).")
        return

    acc_by_freq = df.groupby("true_freq")["correct"].mean().reset_index()

    # Order and labels you want on the x-axis
    freq_order = [7.0, 8.0, 9.0, 11.0, 7.5, 8.5]
    x_pos = np.arange(len(freq_order))

    # Map frequency -> accuracy (nếu thiếu thì cho NaN hoặc 0)
    acc_map = {float(f): float(a) for f, a in zip(acc_by_freq["true_freq"], acc_by_freq["correct"])}
    acc = [acc_map.get(f, np.nan) for f in freq_order]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_pos, acc, width=0.6)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Accuracy")
    plt.title("CCA Offline Accuracy by Frequency (ASR-cleaned)")
    plt.xticks(x_pos, [str(f) for f in freq_order])
    plt.grid(axis="y", alpha=0.3, linestyle="--")
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
    print(f"Saved ASR accuracy figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline CCA analysis on ASR-cleaned XDF files "
            "using xdf_loader, ASR, and cca.classifier"
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

    df_results = run_cca_offline_asr(xdf_paths, duration=args.duration, display="mobile")
    if df_results.empty:
        print("No results to save.")
        return

    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = results_dir / "cca_offline_asr_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved ASR CCA results CSV to {csv_path}")

    # Basic statistics
    overall_acc = df_results["correct"].mean()
    print(f"Overall ASR+CCA accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    acc_by_subject = df_results.groupby("subject")["correct"].mean()
    print("\nAccuracy by subject (ASR-cleaned):")
    for subj, acc in acc_by_subject.items():
        print(f"  {subj}: {acc:.3f} ({acc*100:.1f}%)")

    acc_fig_path = results_dir / "cca_offline_asr_accuracy_by_freq.png"
    _plot_accuracy_by_freq_asr(df_results, acc_fig_path)


if __name__ == "__main__":
    main()

