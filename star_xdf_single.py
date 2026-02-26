"""
STAR on single XDF file
=======================

Apply Sparse Time-Artifact Removal (STAR) from meegkit to EEG data loaded
from a single XDF file, and generate before/after plots to illustrate how
STAR attenuates sparse, short-lived artifacts while preserving the ongoing
SSVEP response.

Pipeline:
- Load EEG via ``load_xdf_eeg`` from ``xdf_loader.py`` (default: band-pass
  2–45 Hz + DC removal, as in the CCA pipeline).
- Apply STAR on the continuous recording.
- Plot before/after traces and a zoomed window around the strongest STAR
  correction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from meegkit import star
from xdf_loader import load_xdf_eeg


def run_star_on_xdf(
    xdf_path: Path,
    apply_filter: bool = True,
    rank: int = 2,
    n_chans_plot: int = 8,
) -> Dict[str, Any]:
    """
    Run STAR on a single XDF file and return raw/clean data and metadata.

    Parameters
    ----------
    xdf_path : Path
        Path to the XDF file.
    apply_filter : bool
        Whether to apply the default 2–45 Hz band-pass + DC removal in
        ``load_xdf_eeg`` (default: True).
    rank : int
        STAR rank parameter controlling the dimensionality of the clean
        subspace (default: 2).
    n_chans_plot : int
        Number of channels to focus on when plotting (default: 8).

    Returns
    -------
    result : dict
        Dictionary containing:
        - eeg_raw: np.ndarray, shape (n_chans_plot, n_samples)
        - eeg_clean: np.ndarray, same shape as eeg_raw
        - times: np.ndarray, time vector for displayed samples
        - srate: float, sampling rate
        - ch_labels: list of channel labels (for displayed channels)
        - diff_power: np.ndarray, per-sample RMS difference (all channels)
        - meta: dict with extra information (file path, durations, etc.)
    """
    xdf_path = Path(xdf_path)
    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    print(f"Loading XDF file: {xdf_path}")

    eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
        xdf_path,
        apply_filter=apply_filter,
    )

    n_samples, n_channels = eeg_data.shape
    duration_sec = n_samples / srate if srate > 0 else float("nan")
    print(f"  Samples: {n_samples}, Channels: {n_channels}, Duration: {duration_sec:.2f} s")
    print(f"  STAR rank: {rank}, apply_filter={apply_filter}")

    # STAR expects data as (n_samples, n_chans)
    x = np.asarray(eeg_data, dtype=float)

    # Apply STAR
    y, w, info = star.star(x, rank)

    if y.shape != x.shape:
        raise RuntimeError(
            f"STAR output has unexpected shape {y.shape} (expected {x.shape})."
        )

    # Compute per-sample RMS difference across channels to locate strongest correction
    diff = x - y
    diff_power = np.sqrt(np.mean(diff**2, axis=1))

    times = np.arange(n_samples) / srate

    # Limit channels for plotting (transpose to (n_chans, n_samples) for plotting)
    n_plot = min(n_chans_plot, n_channels)
    eeg_raw = x[:, :].T[:n_plot, :]
    eeg_clean = y[:, :].T[:n_plot, :]
    ch_labels_plot = ch_labels[:n_plot]

    meta: Dict[str, Any] = {
        "file": str(xdf_path),
        "srate": srate,
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_channels_plot": n_plot,
        "rank": rank,
        "apply_filter": apply_filter,
    }

    return {
        "eeg_raw": eeg_raw,
        "eeg_clean": eeg_clean,
        "times": times,
        "srate": srate,
        "ch_labels": ch_labels_plot,
        "diff_power": diff_power,
        "meta": meta,
    }


def _plot_star_result(
    eeg_raw: np.ndarray,
    eeg_clean: np.ndarray,
    times: np.ndarray,
    ch_labels: Tuple[str, ...] | Any,
    diff_power: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path | None = None,
    zoom_duration: float = 10.0,
    plot_scale: float = 0.5,
) -> None:
    """
    Plot before/after STAR for a subset of channels and a zoomed window.
    """
    n_chans, n_samples = eeg_raw.shape
    assert eeg_clean.shape == eeg_raw.shape
    assert times.shape[0] == n_samples

    if n_chans == 0 or n_samples == 0:
        print("Nothing to plot (empty data).")
        return

    # Robust y-limits based on (scaled) raw data
    amp = np.abs(eeg_raw * plot_scale)
    ylim = float(np.percentile(amp, 99.0))
    if not np.isfinite(ylim) or ylim <= 0:
        ylim = float(np.max(amp)) if np.isfinite(np.max(amp)) else 1.0

    fig, axes = plt.subplots(n_chans, 1, sharex=True, figsize=(10, 6))
    if n_chans == 1:
        axes = [axes]

    title = f"STAR before/after - {os.path.basename(meta.get('file', 'unknown'))}"
    fig.suptitle(title)

    montage_labels = ["O2", "PO4", "P2", "Oz", "POz", "P1", "PO3", "O1"]

    for i in range(n_chans):
        ax = axes[i]
        ax.plot(times, eeg_raw[i] * plot_scale, lw=0.5, label="before STAR" if i == 0 else None)
        ax.plot(times, eeg_clean[i] * plot_scale, lw=0.5, label="after STAR" if i == 0 else None)
        ax.set_ylim([-ylim, ylim])
        if n_chans <= len(montage_labels):
            label = montage_labels[i]
        else:
            label = ch_labels[i] if i < len(ch_labels) else f"Ch{i}"
        ax.set_ylabel(label)
        ax.set_yticks([])

    axes[-1].set_xlabel("Time (s)")

    # Put legend on the first axis, outside to the right
    axes[0].legend(fontsize="small", bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.subplots_adjust(hspace=0, right=0.75)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved STAR before/after figure to {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # ------------------------------------------------------------------
    # Zoomed view around the region with strongest STAR correction
    # ------------------------------------------------------------------
    if n_samples == 0 or diff_power.size != n_samples:
        return

    if not np.any(np.isfinite(diff_power)):
        return

    center_idx = int(np.nanargmax(diff_power))
    if center_idx <= 0 or center_idx >= n_samples:
        return

    half = zoom_duration / 2.0
    t_center = times[center_idx]
    t0 = max(times[0], t_center - half)
    t1 = min(times[-1], t_center + half)
    if t1 <= t0:
        return

    zoom_mask = (times >= t0) & (times <= t1)
    if not np.any(zoom_mask):
        return

    times_z = times[zoom_mask]
    eeg_raw_z = eeg_raw[:, zoom_mask]
    eeg_clean_z = eeg_clean[:, zoom_mask]

    # Robust y-limits for zoomed window
    amp_z = np.abs(eeg_raw_z * plot_scale)
    ylim_z = float(np.percentile(amp_z, 99.0))
    if not np.isfinite(ylim_z) or ylim_z <= 0:
        ylim_z = float(np.max(amp_z)) if np.isfinite(np.max(amp_z)) else 1.0

    fig2, axes2 = plt.subplots(n_chans, 1, sharex=True, figsize=(10, 4))
    if n_chans == 1:
        axes2 = [axes2]

    title2 = f"STAR cleaned EEG - {os.path.basename(meta.get('file', 'unknown'))}"
    fig2.suptitle(title2)

    for i in range(n_chans):
        ax = axes2[i]
        ax.plot(times_z, eeg_raw_z[i] * plot_scale, lw=0.6, label="before STAR" if i == 0 else None)
        ax.plot(times_z, eeg_clean_z[i] * plot_scale, lw=0.6, label="after STAR" if i == 0 else None)
        ax.set_ylim([-ylim_z, ylim_z])
        if n_chans <= len(montage_labels):
            label = montage_labels[i]
        else:
            label = ch_labels[i] if i < len(ch_labels) else f"Ch{i}"
        ax.set_ylabel(label)
        ax.set_yticks([])

    axes2[-1].set_xlabel("Time (s)")

    # Legend for zoomed view (before/after STAR), placed outside top-right
    axes2[0].legend(
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    if out_path is not None:
        zoom_path = out_path.with_name(out_path.stem + "_star_zoom" + out_path.suffix)
        zoom_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(zoom_path, dpi=150)
        print(f"Saved STAR zoom figure to {zoom_path}")
        plt.close(fig2)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run meegkit STAR on a single XDF file and plot before/after EEG.",
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the .xdf file.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable the default 2–45 Hz bandpass filter in xdf_loader.load_xdf_eeg.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="STAR rank parameter (default: 2).",
    )
    parser.add_argument(
        "--n-chans-plot",
        type=int,
        default=8,
        help="Number of channels to plot (default: 8).",
    )
    parser.add_argument(
        "--plot-scale",
        type=float,
        default=0.5,
        help="Multiplicative factor applied to EEG amplitudes in plots (default: 0.5).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save figures (default: results).",
    )

    args = parser.parse_args()

    xdf_path = Path(args.filepath)
    if not xdf_path.exists():
        print(f"[ERROR] XDF file not found: {xdf_path}")
        return

    result = run_star_on_xdf(
        xdf_path=xdf_path,
        apply_filter=not args.no_filter,
        rank=args.rank,
        n_chans_plot=args.n_chans_plot,
    )

    eeg_raw = result["eeg_raw"]
    eeg_clean = result["eeg_clean"]
    times = result["times"]
    ch_labels = result["ch_labels"]
    diff_power = result["diff_power"]
    meta = result["meta"]

    print(
        f"Finished STAR for {meta['file']} | "
        f"duration={meta['duration_sec']:.2f}s | "
        f"n_channels={meta['n_channels_plot']} | "
        f"rank={meta['rank']} | apply_filter={meta['apply_filter']}"
    )

    results_dir = Path(args.results_dir)
    stem = xdf_path.stem
    fig_name = f"{stem}_star.png"
    out_path = results_dir / fig_name

    _plot_star_result(
        eeg_raw=eeg_raw,
        eeg_clean=eeg_clean,
        times=times,
        ch_labels=tuple(ch_labels),
        diff_power=diff_power,
        meta=meta,
        out_path=out_path,
        zoom_duration=10.0,
        plot_scale=args.plot_scale,
    )


if __name__ == "__main__":
    main()

