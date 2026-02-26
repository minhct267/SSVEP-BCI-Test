"""
Visualize Preprocessing Pipelines - Compare Raw vs RD, SNS, RD+SNS
==================================================================

Loads one XDF file, applies the same preprocessing steps as cca_offline_rd,
cca_offline_sns, and cca_offline_rd_sns, then plots multi-channel EEG for each
pipeline in one figure for visual comparison.

Pipeline summary:
- Raw: load with apply_filter=False; first plot = DC removal + bandpass 2-45 Hz (order 3).
- RD: raw -> robust_detrend -> bandpass 2-45 Hz (order 3).
- SNS: load with bandpass+DC -> sns (meegkit).
- RD+SNS: raw -> robust_detrend -> sns -> bandpass 2-45 Hz.

Usage:
    python visualize_preprocessing_pipelines.py
    python visualize_preprocessing_pipelines.py raw/SS09.xdf --window 10
    python visualize_preprocessing_pipelines.py path/to/file.xdf --stream 0 --save
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from meegkit.detrend import detrend as robust_detrend
from meegkit.sns import sns as sns_denoise
from xdf_loader import bandpass_filter_eeg, load_xdf_eeg, remove_dc_offset


def _remove_dc_for_display(data: np.ndarray) -> np.ndarray:
    """Remove DC offset per channel for display (same as xdf_visualizer)."""
    if data.ndim != 2:
        return data
    return data - np.mean(data, axis=0, keepdims=True)


def _channel_spacing(data_display: np.ndarray, n_channels: int) -> float:
    """Compute vertical spacing between channels (robust, from xdf_visualizer)."""
    channel_amp = np.zeros(n_channels)
    for ci in range(n_channels):
        p2, p98 = np.percentile(data_display[:, ci], [2, 98])
        channel_amp[ci] = p98 - p2
    median_amp = np.median(channel_amp)
    if median_amp > 0:
        return float(median_amp * 1.2)
    channel_std = np.std(data_display, axis=0)
    median_std = np.median(channel_std)
    if median_std > 0:
        return float(median_std * 5)
    return 1.0


def _plot_panel(
    ax: plt.Axes,
    data: np.ndarray,
    srate: float,
    ch_labels: List[str],
    title: str,
    window_start: float,
    window_sec: float,
    markers: Optional[Any] = None,
    marker_ts: Optional[np.ndarray] = None,
    eeg_ts: Optional[np.ndarray] = None,
) -> None:
    """
    Draw one multi-channel EEG panel (same style as xdf_visualizer.plot_multichannel).
    data: (n_samples, n_channels). Plots the segment [window_start, window_start+window_sec].
    """
    n_samples, n_channels = data.shape
    start_idx = int(window_start * srate)
    win_samples = int(window_sec * srate)
    end_idx = min(start_idx + win_samples, n_samples)
    if start_idx >= end_idx:
        ax.set_title(title)
        return

    data_seg = data[start_idx:end_idx, :]
    data_display = _remove_dc_for_display(data_seg)
    spacing = _channel_spacing(data_display, n_channels)
    time_axis = np.arange(data_seg.shape[0]) / srate

    for ci in range(n_channels):
        offset = (n_channels - 1 - ci) * spacing
        ax.plot(
            time_axis,
            data_display[:, ci] + offset,
            linewidth=0.5,
            color="#2c3e50",
        )

    if markers is not None and marker_ts is not None and eeg_ts is not None and len(eeg_ts) > 0:
        eeg_start = float(eeg_ts[0])
        for mi in range(len(markers)):
            rec_time = float(marker_ts[mi]) - eeg_start
            if window_start <= rec_time <= window_start + window_sec:
                x_in_plot = rec_time - window_start
                ax.axvline(x_in_plot, color="#e74c3c", alpha=0.6, linewidth=0.8, linestyle="--")
                label = markers[mi]
                if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                    label = str(label[0])
                else:
                    label = str(label)
                ax.text(
                    x_in_plot,
                    (n_channels - 0.3) * spacing,
                    label,
                    fontsize=6,
                    color="#e74c3c",
                    rotation=90,
                    va="bottom",
                    ha="right",
                )

    ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else window_sec)
    ax.set_ylim(-spacing, n_channels * spacing)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    yticks = [(n_channels - 1 - ci) * spacing for ci in range(n_channels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ch_labels, fontsize=7)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")


def run_pipelines(
    xdf_path: Path,
    stream_idx: Optional[int] = None,
    window_sec: float = 10.0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Load XDF, run the three preprocessing pipelines, and plot 4 panels for comparison.
    """
    print(f"Loading XDF: {xdf_path}")
    # Load raw (for RD and RD+SNS)
    try:
        eeg_raw, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
            xdf_path, stream_idx=stream_idx, apply_filter=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to load raw: {e}")
        return

    # Load filtered (for SNS pipeline only)
    try:
        eeg_filtered, _, _, _, _, _, _ = load_xdf_eeg(
            xdf_path,
            stream_idx=stream_idx,
            apply_filter=True,
            low=2.0,
            high=45.0,
            order=3,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load filtered: {e}")
        return

    n_samples, n_channels = eeg_raw.shape
    print(f"  Channels: {n_channels}, srate: {srate:.1f} Hz, samples: {n_samples}")

    # Pipeline RD: raw -> robust_detrend -> bandpass 2-45 Hz (order 3)
    try:
        eeg_rd, _weights, _basis = robust_detrend(
            eeg_raw,
            order=3,
            basis="polynomials",
            threshold=3,
            n_iter=4,
            show=False,
        )
        eeg_rd = bandpass_filter_eeg(eeg_rd, srate, low=2.0, high=45.0, order=3)
        print("  Pipeline RD (robust detrend+bandpass 2-45 Hz): OK")
    except Exception as e:
        print(f"  [WARNING] Pipeline RD failed: {e}")
        eeg_rd = eeg_raw.copy()

    # Pipeline SNS: bandpass+DC -> sns (same as cca_offline_sns)
    try:
        X = eeg_filtered[:, :, None]
        eeg_sns, _sns_matrix = sns_denoise(X, n_neighbors=7, skip=0)
        eeg_sns = eeg_sns[:, :, 0]
        print("  Pipeline SNS (bandpass+DC+SNS): OK")
    except Exception as e:
        print(f"  [WARNING] Pipeline SNS failed: {e}")
        eeg_sns = eeg_filtered.copy()

    # Pipeline RD+SNS: raw -> robust_detrend -> sns -> bandpass (same as cca_offline_rd_sns)
    try:
        eeg_rd_temp, _w2, _b2 = robust_detrend(
            eeg_raw,
            order=3,
            basis="polynomials",
            threshold=3,
            n_iter=4,
            show=False,
        )
        X_rd = eeg_rd_temp[:, :, None]
        eeg_rd_sns, _sm = sns_denoise(X_rd, n_neighbors=7, skip=0)
        eeg_rd_sns = eeg_rd_sns[:, :, 0]
        eeg_rd_sns = bandpass_filter_eeg(eeg_rd_sns, srate, low=2.0, high=45.0, order=3)
        print("  Pipeline RD+SNS (robust_detrend+SNS+bandpass): OK")
    except Exception as e:
        print(f"  [WARNING] Pipeline RD+SNS failed: {e}")
        eeg_rd_sns = eeg_raw.copy()

    # First plot: raw -> DC removal -> bandpass 2-45 Hz (order 3)
    eeg_raw_display = _remove_dc_for_display(eeg_raw)
    eeg_raw_display = bandpass_filter_eeg(eeg_raw_display, srate, low=2.0, high=45.0, order=3)

    window_start = 0.0
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, max(10, n_channels * 0.8 + 4)),
        sharex=True,
    )
    fig.patch.set_facecolor("#fafafa")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    _plot_panel(
        axes[0],
        eeg_raw_display,
        srate,
        ch_labels,
        "Raw (DC removal + bandpass 2-45 Hz, order 3)",
        window_start,
        window_sec,
        markers,
        marker_ts,
        eeg_ts,
    )
    _plot_panel(
        axes[1],
        eeg_rd,
        srate,
        ch_labels,
        "Pipeline RD (robust detrend+bandpass 2-45 Hz)",
        window_start,
        window_sec,
        markers,
        marker_ts,
        eeg_ts,
    )
    _plot_panel(
        axes[2],
        eeg_sns,
        srate,
        ch_labels,
        "Pipeline SNS (bandpass+DC+SNS)",
        window_start,
        window_sec,
        markers,
        marker_ts,
        eeg_ts,
    )
    _plot_panel(
        axes[3],
        eeg_rd_sns,
        srate,
        ch_labels,
        "Pipeline RD+SNS (robust detrend+SNS+bandpass)",
        window_start,
        window_sec,
        markers,
        marker_ts,
        eeg_ts,
    )

    fig.suptitle(
        f"Preprocessing pipelines comparison — {xdf_path.name} (first {window_sec}s)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and compare EEG after Raw, RD, SNS, and RD+SNS preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="raw/SS09.xdf",
        help="Path to .xdf file (default: raw/SS09.xdf)",
    )
    parser.add_argument(
        "--stream",
        "-s",
        type=int,
        default=None,
        help="EEG stream index (default: auto-detect)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=float,
        default=10.0,
        help="Time window in seconds to plot (default: 10)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figure to results/ as PNG",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory for saved figure (default: results)",
    )

    args = parser.parse_args()
    xdf_path = Path(args.filepath)
    if not xdf_path.exists():
        print(f"[ERROR] File not found: {xdf_path}")
        return

    save_path = None
    if args.save:
        stem = xdf_path.stem
        save_path = Path(args.out_dir) / f"preprocessing_pipelines_{stem}.png"

    run_pipelines(
        xdf_path,
        stream_idx=args.stream,
        window_sec=args.window,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
