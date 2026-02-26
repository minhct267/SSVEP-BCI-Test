"""
ASR on single XDF file
======================

Apply Artifact Subspace Reconstruction (ASR) from meegkit to EEG data
loaded from a single XDF file, and generate before/after plots to
illustrate detection and reconstruction of bad segments.

Default pipeline:
- Load EEG (and markers if present) via `load_xdf_eeg` from `xdf_loader.py`
  with bandpass 2–45 Hz and DC removal.
- Train ASR on a calibration segment.
- Apply ASR in sliding windows over the full recording.
- Plot before/after traces and highlight the calibration region and
  samples selected by ASR during calibration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from xdf_loader import load_xdf_eeg, remove_dc_offset


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

    import re  # local import to avoid polluting global namespace

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

    This matches the CCA offline analysis, where we discard the
    first 1 s of stimulation and use the last 4 s for decoding.
    """
    if eeg_ts is None or len(eeg_ts) == 0:
        return None, 0.0

    # Index of Stop marker
    stop_idx = int(np.searchsorted(eeg_ts, stop_time, side="right"))
    stop_idx = max(1, min(stop_idx, len(eeg_ts)))

    target_len = int(duration * srate)
    start_idx = max(0, stop_idx - target_len)

    used_len = stop_idx - start_idx
    if used_len <= 0:
        return None, 0.0

    segment = eeg_data[start_idx:stop_idx, :]
    return segment, used_len / srate


def run_asr_on_xdf(
    xdf_path: Path,
    mode: str = "bp_asr",
    calib_start: float = 0.0,
    calib_duration: float = 30.0,
    window_len: float = 1.0,
    step_len: float = 1.0,
    n_chans_plot: int = 8,
) -> Dict[str, Any]:
    """
    Run ASR on a single XDF file and return raw/clean data and metadata.

    Parameters
    ----------
    xdf_path : Path
        Path to the XDF file.
    mode : {"bp_asr", "raw_asr", "compare"}
        Processing mode. For now, "bp_asr" is the main supported mode.
        - "bp_asr": bandpass + DC removal in `load_xdf_eeg`, then ASR.
        - "raw_asr": disable bandpass, keep only DC removal, then ASR.
        - "compare": currently behaves like "bp_asr" but kept for future
          extension.
    calib_start : float
        Calibration start time (seconds from beginning of recording).
    calib_duration : float
        Duration of calibration segment in seconds.
    window_len : float
        ASR window length in seconds.
    step_len : float
        ASR window step in seconds.
    n_chans_plot : int
        Number of channels to focus on when plotting.

    Returns
    -------
    result : dict
        Dictionary containing:
        - eeg_raw: np.ndarray, shape (n_samples_display, n_chans_plot)
        - eeg_clean: np.ndarray, same shape as eeg_raw
        - times: np.ndarray, time vector for displayed samples
        - srate: float, sampling rate
        - ch_labels: list of channel labels (for displayed channels)
        - calib_times: np.ndarray, time vector for calibration samples
        - calib_mask: np.ndarray[bool], mask of selected samples in calib
        - meta: dict with extra information (file path, durations, etc.)
    """
    xdf_path = Path(xdf_path)
    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    print(f"Loading XDF file: {xdf_path}")

    if mode == "bp_asr" or mode == "compare":
        eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
            xdf_path,
            apply_filter=True,
        )
    elif mode == "raw_asr":
        eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
            xdf_path,
            apply_filter=False,
        )
        eeg_data = remove_dc_offset(eeg_data)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'bp_asr', 'raw_asr' or 'compare'.")

    n_samples, n_channels = eeg_data.shape
    duration_sec = n_samples / srate if srate > 0 else float("nan")
    print(f"  Samples: {n_samples}, Channels: {n_channels}, Duration: {duration_sec:.2f} s")

    # ASR expects data as (n_channels, n_times)
    data_T = eeg_data.T  # (n_channels, n_times)

    # ------------------------------------------------------------------
    # Choose calibration data
    # 1) Preferred: concatenate all 4 s SSVEP segments across trials
    #    (same windows that are used for CCA).
    # 2) Fallback: continuous window [calib_start, calib_start+calib_duration].
    # ------------------------------------------------------------------
    calib_mode = "continuous"
    calib_data: np.ndarray
    calib_start_idx = 0
    calib_stop_idx = 0
    calib_total_dur = 0.0

    if markers is not None and marker_ts is not None:
        trials = _extract_trials_from_markers(markers, marker_ts)
        segments: List[np.ndarray] = []
        if trials:
            for trial in trials:
                start_time = float(trial["start_time"])
                stop_time = float(trial["stop_time"])
                seg, used_dur = _cut_trial_segment(
                    eeg_data=eeg_data,
                    eeg_ts=eeg_ts,
                    start_time=start_time,
                    stop_time=stop_time,
                    srate=srate,
                    duration=4.0,
                )
                if seg is None or used_dur <= 0:
                    continue
                segments.append(seg)
                calib_total_dur += used_dur

        if segments:
            concat = np.concatenate(segments, axis=0)  # (n_samples_calib, n_channels)
            calib_data = concat.T
            calib_mode = "trial_concat"
            print(
                f"  Calibration mode: trial_concat "
                f"({len(segments)} segments, total {calib_total_dur:.2f} s)"
            )
        else:
            print("  [INFO] No valid trial segments for calibration, falling back to continuous.")

    if calib_mode == "continuous":
        calib_start_idx = int(max(0.0, calib_start) * srate)
        calib_stop_idx = int((calib_start + calib_duration) * srate)
        calib_start_idx = min(calib_start_idx, n_samples - 1)
        calib_stop_idx = min(max(calib_stop_idx, calib_start_idx + 1), n_samples)

        calib_data = data_T[:, calib_start_idx:calib_stop_idx]
        calib_total_dur = (calib_stop_idx - calib_start_idx) / srate
        print(
            f"  Calibration mode: continuous "
            f"(start={calib_start_idx} [{calib_start_idx / srate:.2f} s], "
            f"stop={calib_stop_idx} [{calib_stop_idx / srate:.2f} s], "
            f"len={calib_data.shape[1]} samples)"
        )

    calib_len = calib_data.shape[1]
    if calib_len <= 0:
        raise RuntimeError("Invalid calibration data (zero length).")

    asr = ASR(method="euclid")
    _, sample_mask = asr.fit(calib_data)

    if sample_mask is None or sample_mask.size == 0:
        # Fallback: all calibration samples treated as selected
        sample_mask = np.ones(calib_len, dtype=bool)
    else:
        sample_mask = np.asarray(sample_mask).astype(bool).ravel()
        if sample_mask.size != calib_len:
            # If meegkit returns mask with a different length, truncate/pad
            if sample_mask.size > calib_len:
                sample_mask = sample_mask[:calib_len]
            else:
                pad = np.ones(calib_len - sample_mask.size, dtype=bool)
                sample_mask = np.concatenate([sample_mask, pad])

    # Sliding-window ASR over the full recording
    win_samp = int(round(window_len * srate))
    step_samp = int(round(step_len * srate))
    win_samp = max(1, win_samp)
    step_samp = max(1, step_samp)

    if n_samples < win_samp:
        raise RuntimeError(
            f"Recording too short for window_len={window_len}s (need at least {win_samp} samples)."
        )

    X = sliding_window(data_T, window=win_samp, step=step_samp)
    # X shape: (n_channels, n_windows, win_samp)
    Y = np.zeros_like(X)
    n_windows = X.shape[1]
    print(f"  Running ASR on {n_windows} windows (win={win_samp} samples, step={step_samp})")

    for i in range(n_windows):
        Y[:, i, :] = asr.transform(X[:, i, :])

    # Reconstruct continuous signal (may be slightly shorter than original)
    clean_T = Y.reshape(n_channels, -1)
    n_samples_clean = clean_T.shape[1]

    # Truncate raw to match cleaned length
    data_T_trunc = data_T[:, :n_samples_clean]

    times = np.arange(n_samples_clean) / srate

    # Limit channels for plotting
    n_plot = min(n_chans_plot, n_channels)
    eeg_raw = data_T_trunc[:n_plot, :].copy()
    eeg_clean = clean_T[:n_plot, :].copy()
    ch_labels_plot = ch_labels[:n_plot]

    # Only continuous calibration can be mapped cleanly onto time axis
    if calib_mode == "continuous":
        calib_idx = np.arange(calib_start_idx, calib_stop_idx)
        calib_idx = calib_idx[calib_idx < n_samples_clean]
        calib_len_plot = calib_idx.size
        if calib_len_plot > 0:
            calib_times = calib_idx / srate
            calib_mask = sample_mask[:calib_len_plot]
        else:
            calib_times = np.array([], dtype=float)
            calib_mask = np.array([], dtype=bool)

        sel_count = int(np.sum(calib_mask))
        total_count = int(calib_len_plot)
    else:
        # For trial-based calibration we cannot map samples to a single
        # contiguous time axis, but we can still report selection stats.
        calib_times = np.array([], dtype=float)
        calib_mask = np.array([], dtype=bool)
        sel_count = int(np.sum(sample_mask))
        total_count = int(sample_mask.size)

    if total_count > 0:
        keep_ratio = sel_count / float(total_count)
        artefact_ratio = 1.0 - keep_ratio
    else:
        keep_ratio = 0.0
        artefact_ratio = float("nan")

    print(
        f"  Calibration: selected={sel_count}/{total_count} samples "
        f"({keep_ratio * 100.0:.1f}% kept)"
    )

    meta: Dict[str, Any] = {
        "file": str(xdf_path),
        "mode": mode,
        "srate": srate,
        "duration_sec": duration_sec,
        "n_samples_raw": n_samples,
        "n_samples_clean": n_samples_clean,
        "n_channels": n_channels,
        "n_channels_plot": n_plot,
        "calib_mode": calib_mode,
        "calib_total_dur_sec": calib_total_dur,
        "calib_start_sec": calib_start_idx / srate if calib_mode == "continuous" else float("nan"),
        "calib_stop_sec": calib_stop_idx / srate if calib_mode == "continuous" else float("nan"),
        "calib_artefact_ratio": artefact_ratio,
    }

    return {
        "eeg_raw": eeg_raw,
        "eeg_clean": eeg_clean,
        "times": times,
        "srate": srate,
        "ch_labels": ch_labels_plot,
        "calib_times": calib_times,
        "calib_mask": calib_mask,
        "meta": meta,
    }


def _plot_asr_result(
    eeg_raw: np.ndarray,
    eeg_clean: np.ndarray,
    times: np.ndarray,
    ch_labels: Tuple[str, ...] | Any,
    calib_times: np.ndarray,
    calib_mask: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path | None = None,
    zoom_duration: float = 10.0,
    make_zoom: bool = True,
    plot_scale: float = 0.5,
) -> None:
    """
    Plot before/after ASR for a subset of channels and highlight calibration.
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

    title = f"ASR before/after - {os.path.basename(meta.get('file', 'unknown'))}"
    fig.suptitle(title)

    for i in range(n_chans):
        ax = axes[i]

        # Calibration window shading (full window)
        if calib_times.size > 0:
            ax.fill_between(
                calib_times,
                0,
                1,
                color="grey",
                alpha=0.3,
                transform=ax.get_xaxis_transform(),
                label="calibration window" if i == 0 else None,
            )

            # Selected samples within calibration (hatched)
            ax.fill_between(
                calib_times,
                0,
                1,
                where=calib_mask,
                transform=ax.get_xaxis_transform(),
                facecolor="none",
                hatch="...",
                edgecolor="k",
                label="selected samples" if i == 0 else None,
            )

        ax.plot(times, eeg_raw[i] * plot_scale, lw=0.5, label="before ASR" if i == 0 else None)
        ax.plot(times, eeg_clean[i] * plot_scale, lw=0.5, label="after ASR" if i == 0 else None)
        ax.set_ylim([-ylim, ylim])
        # Channel label mapping for SSVEP montage
        montage_labels = ["O2", "PO4", "P2", "Oz", "POz", "P1", "PO3", "O1"]
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
        print(f"Saved ASR before/after figure to {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # ------------------------------------------------------------------
    # Optional zoomed view around the region with strongest ASR correction
    # ------------------------------------------------------------------
    if not make_zoom or n_samples == 0:
        return

    diff = eeg_raw - eeg_clean
    diff_power = np.sqrt(np.mean(diff**2, axis=0))
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
    diff_z = diff_power[zoom_mask]

    # Robust y-limits for zoomed window
    amp_z = np.abs(eeg_raw_z * plot_scale)
    ylim_z = float(np.percentile(amp_z, 99.0))
    if not np.isfinite(ylim_z) or ylim_z <= 0:
        ylim_z = float(np.max(amp_z)) if np.isfinite(np.max(amp_z)) else 1.0

    fig2, axes2 = plt.subplots(n_chans, 1, sharex=True, figsize=(10, 4))
    if n_chans == 1:
        axes2 = [axes2]

    title2 = f"ASR cleaned EEG - {os.path.basename(meta.get('file', 'unknown'))}"
    fig2.suptitle(title2)

    for i in range(n_chans):
        ax = axes2[i]
        ax.plot(times_z, eeg_raw_z[i] * plot_scale, lw=0.6, label="before ASR" if i == 0 else None)
        ax.plot(times_z, eeg_clean_z[i] * plot_scale, lw=0.6, label="after ASR" if i == 0 else None)
        ax.set_ylim([-ylim_z, ylim_z])
        montage_labels = ["O2", "PO4", "P2", "Oz", "POz", "P1", "PO3", "O1"]
        if n_chans <= len(montage_labels):
            label = montage_labels[i]
        else:
            label = ch_labels[i] if i < len(ch_labels) else f"Ch{i}"
        ax.set_ylabel(label)
        ax.set_yticks([])

    axes2[-1].set_xlabel("Time (s)")

    # Legend for zoomed view (before/after ASR), placed outside top-right
    axes2[0].legend(
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(0.99, 2.4),
        borderaxespad=0.0,
    )

    if out_path is not None:
        zoom_path = out_path.with_name(out_path.stem + "_zoom" + out_path.suffix)
        zoom_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(zoom_path, dpi=150)
        print(f"Saved ASR zoom figure to {zoom_path}")
        plt.close(fig2)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run meegkit ASR on a single XDF file and plot before/after EEG.",
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the .xdf file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bp_asr",
        choices=["bp_asr", "raw_asr", "compare"],
        help="Processing mode (default: bp_asr).",
    )
    parser.add_argument(
        "--calib-start",
        type=float,
        default=0.0,
        help="Calibration start time in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--calib-duration",
        type=float,
        default=30.0,
        help="Calibration duration in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--window-len",
        type=float,
        default=1.0,
        help="ASR window length in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--step-len",
        type=float,
        default=1.0,
        help="ASR window step in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--n-chans-plot",
        type=int,
        default=4,
        help="Number of channels to plot (default: 4).",
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

    result = run_asr_on_xdf(
        xdf_path=xdf_path,
        mode=args.mode,
        calib_start=args.calib_start,
        calib_duration=args.calib_duration,
        window_len=args.window_len,
        step_len=args.step_len,
        n_chans_plot=args.n_chans_plot,
    )

    eeg_raw = result["eeg_raw"]
    eeg_clean = result["eeg_clean"]
    times = result["times"]
    ch_labels = result["ch_labels"]
    calib_times = result["calib_times"]
    calib_mask = result["calib_mask"]
    meta = result["meta"]

    print(
        f"Finished ASR for {meta['file']} | mode={meta['mode']} | "
        f"duration={meta['duration_sec']:.2f}s | "
        f"n_channels={meta['n_channels_plot']} | "
        f"calib={meta['calib_start_sec']:.2f}-{meta['calib_stop_sec']:.2f}s"
    )

    results_dir = Path(args.results_dir)
    stem = xdf_path.stem
    fig_name = f"{stem}_asr_{args.mode}.png"
    out_path = results_dir / fig_name

    _plot_asr_result(
        eeg_raw=eeg_raw,
        eeg_clean=eeg_clean,
        times=times,
        ch_labels=tuple(ch_labels),
        calib_times=calib_times,
        calib_mask=calib_mask,
        meta=meta,
        out_path=out_path,
        zoom_duration=10.0,
        make_zoom=True,
        plot_scale=args.plot_scale,
    )


if __name__ == "__main__":
    main()

