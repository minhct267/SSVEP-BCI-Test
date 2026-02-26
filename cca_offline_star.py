"""
CCA Offline Analysis with STAR
==============================

Run SSVEP CCA classification offline on all XDF files in a directory,
but first clean the continuous EEG with Sparse Time-Artifact Removal
(STAR). This allows you to compare decoding accuracy before vs after
STAR preprocessing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cca import classifier
from cca_offline import _cut_trial_segment, _extract_trials_from_markers
from meegkit import star as star_mod
from xdf_loader import load_xdf_eeg


def run_cca_offline_star(
    xdf_paths: Iterable[Path],
    duration: float = 4.0,
    display: str = "mobile",
    rank: int = 2,
) -> pd.DataFrame:
    """
    Run offline CCA on STAR-cleaned EEG for all provided XDF files.

    Parameters
    ----------
    xdf_paths : iterable of Path
        Paths to .xdf files.
    duration : float
        Segment length in seconds used for CCA (default: 4.0).
    display : str
        Passed to `classifier` to select frequency ordering
        (default: 'mobile').
    rank : int
        STAR rank parameter controlling the clean subspace dimension
        (default: 2).

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
        print(
            f"  Loaded EEG: {n_samples} samples, {n_channels} channels, "
            f"srate={srate:.2f} Hz | STAR rank={rank}"
        )

        # ------------------------------------------------------------------
        # Apply STAR to the continuous recording
        # ------------------------------------------------------------------
        try:
            y, w, info = star_mod.star(eeg_data, rank)
        except Exception as e:
            print(f"  [WARNING] STAR failed on {xdf_path} ({e}), skipping subject.")
            continue

        if y.shape != eeg_data.shape:
            print(
                f"  [WARNING] STAR output shape {y.shape} != input shape {eeg_data.shape}, "
                "skipping subject."
            )
            continue

        eeg_clean = y

        # ------------------------------------------------------------------
        # CCA on STAR-cleaned data
        # ------------------------------------------------------------------
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
                print(f"  [INFO] Trial {trial_idx}: invalid or empty segment after STAR, skipping.")
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
                    "pipeline": "star",
                }
            )

    if not records:
        print("No valid trials found across all XDF files (STAR).")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    return df


def _plot_accuracy_by_freq_star(df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart of accuracy vs frequency for STAR-cleaned data."""
    if df.empty:
        print("No data to plot accuracy by frequency (STAR).")
        return

    acc_by_freq = df.groupby("true_freq")["correct"].mean().reset_index()

    # Order of frequencies and discrete x positions
    freq_order = [7.0, 8.0, 9.0, 11.0, 7.5, 8.5]
    x_pos = np.arange(len(freq_order))

    acc_map = {
        float(f): float(a)
        for f, a in zip(acc_by_freq["true_freq"], acc_by_freq["correct"])
    }
    acc = [acc_map.get(f, np.nan) for f in freq_order]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_pos, acc, width=0.6)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Accuracy")
    plt.title("CCA Offline Accuracy by Frequency (STAR-cleaned)")
    plt.xticks(x_pos, [str(f) for f in freq_order])
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, acc):
        if np.isnan(val):
            label = "NA"
            y_text = 0.02
        else:
            label = f"{val:.2f}"
            y_text = bar.get_height() + 0.02
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved STAR accuracy figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline CCA analysis on STAR-cleaned XDF files "
            "using xdf_loader, STAR, and cca.classifier"
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
        "--rank",
        type=int,
        default=2,
        help="STAR rank parameter (default: 2).",
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

    df_results = run_cca_offline_star(
        xdf_paths,
        duration=args.duration,
        display="mobile",
        rank=args.rank,
    )
    if df_results.empty:
        print("No results to save (STAR).")
        return

    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = results_dir / "cca_offline_star_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved STAR CCA results CSV to {csv_path}")

    # Basic statistics
    overall_acc = df_results["correct"].mean()
    print(f"Overall STAR+CCA accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    acc_by_subject = df_results.groupby("subject")["correct"].mean()
    print("\nAccuracy by subject (STAR-cleaned):")
    for subj, acc in acc_by_subject.items():
        print(f"  {subj}: {acc:.3f} ({acc*100:.1f}%)")

    acc_fig_path = results_dir / "cca_offline_star_accuracy_by_freq.png"
    _plot_accuracy_by_freq_star(df_results, acc_fig_path)


if __name__ == "__main__":
    main()

