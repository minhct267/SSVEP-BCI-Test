"""
XDF Loader - Load EEG data from .xdf files and apply a simple bandpass filter
=============================================================================

This module provides:
- A programmatic API: `load_xdf_eeg(...)`
- A simple CLI: `python xdf_loader.py raw/SS09.xdf --save-npz ss09.npz`

It reuses the same conventions as `xdf_inspector.py` and `xdf_visualizer.py`:
- Uses `pyxdf.load_xdf` to read streams
- Auto-selects the EEG stream if `--stream` is not specified
- Optionally finds a marker stream (markers/events)
- Applies a Butterworth bandpass filter (default: 2–45 Hz, order 3)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyxdf
from scipy import signal


# ---------------------------------------------------------------------------
# Utility functions (adapted from xdf_visualizer.py)
# ---------------------------------------------------------------------------

def _get_info_field(info: dict, field: str, default: str = "") -> str:
    val = info.get(field)
    if val is None:
        return default
    if isinstance(val, list):
        return str(val[0]) if len(val) > 0 else default
    return str(val)


def _get_channel_labels(info: dict, n_channels: int) -> List[str]:
    """Extract channel labels, falling back to Ch0, Ch1, ... if missing."""
    labels: List[str] = []
    desc = info.get("desc")
    if desc and isinstance(desc, list) and len(desc) > 0:
        desc_dict = desc[0]
        if isinstance(desc_dict, dict):
            channels = desc_dict.get("channels")
            if channels and isinstance(channels, list) and len(channels) > 0:
                ch_list = channels[0]
                if isinstance(ch_list, dict):
                    channel_entries = ch_list.get("channel", [])
                    if isinstance(channel_entries, list):
                        for ch in channel_entries:
                            if isinstance(ch, dict):
                                label = ch.get("label", [""])[0] if isinstance(
                                    ch.get("label"), list
                                ) else ch.get("label", "")
                                labels.append(label if label else f"Ch{len(labels)}")
    if len(labels) < n_channels:
        labels.extend([f"Ch{i}" for i in range(len(labels), n_channels)])
    return labels[:n_channels]


def _find_eeg_and_marker_streams(
    streams: List[dict], forced_eeg_idx: Optional[int] = None
) -> Tuple[int, Optional[int]]:
    """
    Find EEG stream (continuous, srate > 0) and marker stream.

    If `forced_eeg_idx` is not None, that index is used for EEG and we just try
    to find a separate marker stream.
    """
    eeg_idx: Optional[int] = None
    marker_idx: Optional[int] = None

    if forced_eeg_idx is not None:
        if forced_eeg_idx < 0 or forced_eeg_idx >= len(streams):
            raise IndexError(
                f"Invalid EEG stream index {forced_eeg_idx} "
                f"(available: 0..{len(streams) - 1})"
            )
        eeg_idx = forced_eeg_idx
        for i, stream in enumerate(streams):
            if i == forced_eeg_idx:
                continue
            info = stream["info"]
            stype = _get_info_field(info, "type").lower()
            srate = float(_get_info_field(info, "nominal_srate", "0"))
            if stype in ("markers", "marker", "events") or srate == 0:
                marker_idx = i
                break
        return eeg_idx, marker_idx

    # Auto-detect EEG + marker streams
    for i, stream in enumerate(streams):
        info = stream["info"]
        stype = _get_info_field(info, "type").lower()
        srate = float(_get_info_field(info, "nominal_srate", "0"))

        if stype in ("eeg", "exg") or (srate > 0 and stype not in ("markers", "marker", "events")):
            if eeg_idx is None:
                eeg_idx = i
        elif stype in ("markers", "marker", "events") or srate == 0:
            if marker_idx is None:
                marker_idx = i

    if eeg_idx is None:
        raise RuntimeError("No EEG/data stream found in XDF file.")

    return eeg_idx, marker_idx


def bandpass_filter_eeg(
    data: np.ndarray,
    srate: float,
    low: float = 2.0,
    high: float = 45.0,
    order: int = 3,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the time axis (axis=0)."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_channels), got shape {data.shape}")

    nyq = srate / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Invalid bandpass frequencies: low={low}, high={high}, nyquist={nyq}"
        )

    low_n = low / nyq
    high_n = high / nyq
    b, a = signal.butter(order, [low_n, high_n], btype="band")
    return signal.filtfilt(b, a, data, axis=0)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """Remove DC offset (mean) from each channel."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_channels), got shape {data.shape}")
    return data - np.mean(data, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_xdf_eeg(
    filepath: str | Path,
    stream_idx: Optional[int] = None,
    low: float = 2.0,
    high: float = 45.0,
    order: int = 3,
    apply_filter: bool = True,
) -> Tuple[
    np.ndarray,
    float,
    List[str],
    np.ndarray,
    Optional[Any],
    Optional[np.ndarray],
    Dict[str, Any],
]:
    """
    Load EEG data from an XDF file and optionally apply a bandpass filter.

    Parameters
    ----------
    filepath : str | Path
        Path to the .xdf/.xdfz file.
    stream_idx : int, optional
        Index of the EEG stream to use. If None, auto-detect EEG/marker streams.
    low, high : float
        Bandpass filter cutoffs in Hz (default: 2–45 Hz).
    order : int
        Butterworth filter order (default: 3).
    apply_filter : bool
        Whether to apply the bandpass filter (default: True).

    Returns
    -------
    eeg_data : np.ndarray
        EEG data array of shape (n_samples, n_channels), optionally filtered and
        DC-offset removed.
    srate : float
        Sampling rate in Hz.
    ch_labels : list of str
        Channel labels.
    eeg_ts : np.ndarray
        EEG time stamps.
    markers : optional
        Marker time_series for the marker stream (if found), otherwise None.
    marker_ts : np.ndarray or None
        Marker time_stamps for the marker stream (if found), otherwise None.
    info : dict
        Metadata dict for the EEG stream (stream['info']).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"XDF file not found: {path}")

    streams, header = pyxdf.load_xdf(str(path))

    eeg_idx, marker_idx = _find_eeg_and_marker_streams(streams, forced_eeg_idx=stream_idx)
    eeg_stream = streams[eeg_idx]
    eeg_data = eeg_stream["time_series"]
    eeg_ts = eeg_stream["time_stamps"]
    eeg_info = eeg_stream["info"]

    # Convert data to numeric 2D array
    if isinstance(eeg_data, list):
        eeg_data = np.array(eeg_data, dtype=np.float64)
    else:
        eeg_data = np.asarray(eeg_data, dtype=np.float64)

    if eeg_data.ndim != 2:
        raise ValueError(
            f"Unexpected EEG data shape {eeg_data.shape} (expected 2D array). "
            "This stream may not contain numeric EEG samples."
        )

    # Determine sampling rate
    nominal_srate_str = _get_info_field(eeg_info, "nominal_srate", "0")
    try:
        srate = float(nominal_srate_str)
    except ValueError:
        srate = 0.0

    if srate == 0.0:
        if eeg_ts is not None and len(eeg_ts) > 1:
            srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        else:
            raise RuntimeError("Cannot determine sampling rate from XDF stream.")

    n_samples, n_channels = eeg_data.shape
    ch_labels = _get_channel_labels(eeg_info, n_channels)

    # Apply filter + DC removal if requested
    if apply_filter:
        eeg_data = bandpass_filter_eeg(eeg_data, srate, low=low, high=high, order=order)
        eeg_data = remove_dc_offset(eeg_data)

    # Marker stream (if any)
    markers: Optional[Any] = None
    marker_ts: Optional[np.ndarray] = None
    if marker_idx is not None:
        marker_stream = streams[marker_idx]
        markers = marker_stream["time_series"]
        marker_ts = marker_stream["time_stamps"]

    return eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(
    filepath: Path,
    eeg_data: np.ndarray,
    srate: float,
    ch_labels: List[str],
    eeg_ts: np.ndarray,
    markers: Optional[Any],
    marker_ts: Optional[np.ndarray],
    eeg_info: Dict[str, Any],
    low: float,
    high: float,
    order: int,
    apply_filter: bool,
) -> None:
    n_samples, n_channels = eeg_data.shape
    duration = n_samples / srate if srate > 0 else float("nan")
    name = _get_info_field(eeg_info, "name", "Unknown")

    print("=" * 60)
    print("  XDF LOADER - SUMMARY")
    print("=" * 60)
    print(f"  File           : {filepath}")
    print(f"  Stream name    : {name}")
    print(f"  Channels       : {n_channels}")
    print(f"  Sampling rate  : {srate:.2f} Hz")
    print(f"  Samples        : {n_samples:,}")
    print(f"  Duration       : {duration:.2f} s")
    print(
        f"  First channels : {', '.join(ch_labels[:8])}"
        f"{'...' if n_channels > 8 else ''}"
    )

    if apply_filter:
        print(
            f"  Filter         : Bandpass {low:.1f}–{high:.1f} Hz "
            f"(Butterworth order {order}) + DC removal"
        )
    else:
        print("  Filter         : DISABLED (--no-filter)")

    if markers is not None and marker_ts is not None:
        print(f"  Marker stream  : {len(markers)} markers")
        print(
            f"  Marker time    : {marker_ts[0]:.6f} .. {marker_ts[-1]:.6f} "
            f"(Δ={marker_ts[-1] - marker_ts[0]:.3f} s)"
        )
    else:
        print("  Marker stream  : None found")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XDF Loader - Load EEG data from .xdf and apply a bandpass filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python xdf_loader.py                         # Default: raw/SS09.xdf\n"
            "  python xdf_loader.py raw/SS09.xdf          # Specify a file\n"
            "  python xdf_loader.py raw/SS09.xdf --stream 1\n"
            "  python xdf_loader.py raw/SS09.xdf --no-filter\n"
            "  python xdf_loader.py raw/SS09.xdf --save-npz ss09_eeg.npz\n"
        ),
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="raw/SS09.xdf",
        help="Path to the .xdf file (default: raw/SS09.xdf)",
    )
    parser.add_argument(
        "--stream",
        "-s",
        type=int,
        default=None,
        help="Index of the EEG stream to load (default: auto-detect)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable the default 2–45 Hz bandpass filter",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=2.0,
        help="Low cutoff frequency for bandpass (Hz, default: 2.0)",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=45.0,
        help="High cutoff frequency for bandpass (Hz, default: 45.0)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Butterworth filter order (default: 3)",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional path to save data as .npz (eeg, srate, ch_labels, eeg_ts, markers, marker_ts)",
    )

    args = parser.parse_args()

    apply_filter = not args.no_filter

    eeg_data, srate, ch_labels, eeg_ts, markers, marker_ts, eeg_info = load_xdf_eeg(
        args.filepath,
        stream_idx=args.stream,
        low=args.low,
        high=args.high,
        order=args.order,
        apply_filter=apply_filter,
    )

    _print_summary(
        Path(args.filepath).resolve(),
        eeg_data,
        srate,
        ch_labels,
        eeg_ts,
        markers,
        marker_ts,
        eeg_info,
        low=args.low,
        high=args.high,
        order=args.order,
        apply_filter=apply_filter,
    )

    if args.save_npz is not None:
        np.savez(
            args.save_npz,
            eeg=eeg_data,
            srate=srate,
            ch_labels=np.array(ch_labels, dtype=object),
            eeg_ts=eeg_ts,
            markers=markers,
            marker_ts=marker_ts,
        )
        print(f"Saved data to {args.save_npz}")


if __name__ == "__main__":
    main()

