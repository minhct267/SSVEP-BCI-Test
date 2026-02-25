"""
XDF Inspector - Analyze and summarize the contents of .xdf files
================================================================
Reads .xdf files (recorded via Lab Streaming Layer) and displays
detailed information about the streams inside: name, type, channel
count, sampling rate, duration, etc.

Usage:
    python xdf_inspector.py                          # Default: raw/S1.xdf
    python xdf_inspector.py path/to/file.xdf         # Specify a file
    python xdf_inspector.py path/to/file.xdf --verbose  # Show more details
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pyxdf


def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable hh:mm:ss.ms format."""
    if seconds < 0 or np.isnan(seconds):
        return "N/A"
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
    return f"{int(minutes):02d}:{secs:06.3f}"


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to a human-readable format (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def get_stream_info_field(info: dict, field: str, default: str = "N/A") -> str:
    """Extract a value from the stream info dict (pyxdf returns lists per field)."""
    val = info.get(field)
    if val is None:
        return default
    if isinstance(val, list):
        return str(val[0]) if len(val) > 0 else default
    return str(val)


def get_channel_labels(info: dict) -> list:
    """Extract channel names from stream info (if available)."""
    labels = []
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
                                label = ch.get("label", [""])[0] if isinstance(ch.get("label"), list) else ch.get("label", "")
                                labels.append(label)
    return labels


def print_separator(char="=", length=72):
    print(char * length)


def print_header(title: str):
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def print_section(title: str):
    print(f"\n{'-' * 72}")
    print(f"  {title}")
    print(f"{'-' * 72}")


def inspect_xdf(filepath: str, verbose: bool = False):
    """Read and analyze the contents of an .xdf file."""

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    if not filepath.suffix.lower() in (".xdf", ".xdfz"):
        print(f"[WARNING] File does not have .xdf/.xdfz extension: {filepath}")

    file_size = filepath.stat().st_size
    file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

    print_header("XDF INSPECTOR - XDF File Analysis")
    print(f"  File     : {filepath.resolve()}")
    print(f"  Size     : {format_bytes(file_size)}")
    print(f"  Modified : {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # --- Load XDF ---
    print("  Loading XDF file...")
    try:
        streams, header = pyxdf.load_xdf(str(filepath))
    except Exception as e:
        print(f"  [ERROR] Failed to read XDF file: {e}")
        sys.exit(1)

    print(f"  Loaded successfully!\n")

    # --- File Header ---
    print_section("FILE HEADER")
    if header and isinstance(header, dict):
        info = header.get("info", {})
        if isinstance(info, dict):
            version = info.get("version", ["N/A"])
            print(f"  XDF Version : {version[0] if isinstance(version, list) else version}")
    else:
        print("  (No header information available)")

    # --- Summary ---
    print_section(f"OVERVIEW: {len(streams)} stream(s) found")

    # Classify streams
    eeg_streams = []
    marker_streams = []
    other_streams = []

    for i, stream in enumerate(streams):
        info = stream["info"]
        stream_type = get_stream_info_field(info, "type").lower()
        srate = float(get_stream_info_field(info, "nominal_srate", "0"))

        if stream_type == "eeg" or (srate > 0 and stream_type not in ("markers", "marker", "events")):
            eeg_streams.append((i, stream))
        elif stream_type in ("markers", "marker", "events") or srate == 0:
            marker_streams.append((i, stream))
        else:
            other_streams.append((i, stream))

    print(f"\n  {'Type':<20} {'Count':>10}")
    print(f"  {'-' * 20} {'-' * 10}")
    print(f"  {'EEG / Data':<20} {len(eeg_streams):>10}")
    print(f"  {'Markers / Events':<20} {len(marker_streams):>10}")
    if other_streams:
        print(f"  {'Other':<20} {len(other_streams):>10}")

    # --- Stream details ---
    for idx, stream in enumerate(streams):
        info = stream["info"]
        ts = stream["time_stamps"]
        data = stream["time_series"]

        name = get_stream_info_field(info, "name")
        stream_type = get_stream_info_field(info, "type")
        ch_count = get_stream_info_field(info, "channel_count")
        nominal_srate = get_stream_info_field(info, "nominal_srate", "0")
        ch_format = get_stream_info_field(info, "channel_format")
        source_id = get_stream_info_field(info, "source_id")
        stream_id = get_stream_info_field(info, "stream_id")
        hostname = get_stream_info_field(info, "hostname")
        created_at = get_stream_info_field(info, "created_at")

        n_samples = len(ts) if ts is not None else 0
        srate_float = float(nominal_srate) if nominal_srate != "N/A" else 0

        if n_samples > 0 and ts is not None:
            duration = ts[-1] - ts[0]
            actual_srate = (n_samples - 1) / duration if duration > 0 else 0
        else:
            duration = 0
            actual_srate = 0

        print_section(f"STREAM #{idx + 1}: {name}")
        print(f"  {'Stream ID':<24}: {stream_id}")
        print(f"  {'Name':<24}: {name}")
        print(f"  {'Type':<24}: {stream_type}")
        print(f"  {'Channels':<24}: {ch_count}")
        print(f"  {'Format':<24}: {ch_format}")
        print(f"  {'Nominal Srate (Hz)':<24}: {nominal_srate}")
        print(f"  {'Effective Srate (Hz)':<24}: {actual_srate:.4f}" if actual_srate > 0 else f"  {'Effective Srate (Hz)':<24}: Irregular")
        print(f"  {'Samples':<24}: {n_samples:,}")
        print(f"  {'Duration':<24}: {format_duration(duration)}")
        print(f"  {'Source ID':<24}: {source_id}")
        print(f"  {'Hostname':<24}: {hostname}")
        print(f"  {'Created at':<24}: {created_at}")

        if n_samples > 0 and ts is not None:
            print(f"  {'First Timestamp':<24}: {ts[0]:.6f}")
            print(f"  {'Last Timestamp':<24}: {ts[-1]:.6f}")

        # Channel labels
        ch_labels = get_channel_labels(info)
        if ch_labels:
            print(f"\n  Channel List ({len(ch_labels)}):")
            for ci, label in enumerate(ch_labels):
                print(f"    [{ci:>3}] {label}")
        elif verbose:
            print(f"\n  (No channel labels found in metadata)")

        # Data statistics (for EEG / data streams)
        if isinstance(data, np.ndarray) and data.ndim == 2 and srate_float > 0:
            print(f"\n  Data Statistics:")
            print(f"    {'Shape':<20}: {data.shape}")
            print(f"    {'Dtype':<20}: {data.dtype}")
            print(f"    {'Min':<20}: {np.nanmin(data):.6f}")
            print(f"    {'Max':<20}: {np.nanmax(data):.6f}")
            print(f"    {'Mean':<20}: {np.nanmean(data):.6f}")
            print(f"    {'Std':<20}: {np.nanstd(data):.6f}")

            if verbose:
                print(f"\n  Per-Channel Statistics:")
                n_ch = int(ch_count) if ch_count != "N/A" else data.shape[1]
                print(f"    {'Channel':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
                print(f"    {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
                for ci in range(min(n_ch, data.shape[1])):
                    ch_data = data[:, ci]
                    ch_label = ch_labels[ci] if ci < len(ch_labels) else f"Ch{ci}"
                    print(f"    {ch_label:<12} {np.nanmin(ch_data):>12.4f} {np.nanmax(ch_data):>12.4f} "
                          f"{np.nanmean(ch_data):>12.4f} {np.nanstd(ch_data):>12.4f}")

        # Marker content (for marker streams)
        elif isinstance(data, list) or (isinstance(data, np.ndarray) and srate_float == 0):
            if n_samples > 0:
                print(f"\n  Marker Contents:")
                if isinstance(data, list):
                    marker_values = [str(d[0]) if isinstance(d, list) and len(d) > 0 else str(d) for d in data]
                elif isinstance(data, np.ndarray):
                    if data.ndim == 2:
                        marker_values = [str(d[0]) for d in data]
                    else:
                        marker_values = [str(d) for d in data]
                else:
                    marker_values = [str(data)]

                unique_markers = sorted(set(marker_values))
                print(f"    Total markers   : {len(marker_values)}")
                print(f"    Unique markers  : {len(unique_markers)}")
                print(f"\n    {'Marker':<30} {'Count':>10}")
                print(f"    {'-' * 30} {'-' * 10}")
                for m in unique_markers:
                    count = marker_values.count(m)
                    print(f"    {m:<30} {count:>10}")

                if verbose and len(marker_values) <= 100:
                    print(f"\n    Detailed list (timestamp | marker):")
                    for mi in range(len(marker_values)):
                        t = ts[mi] if ts is not None and mi < len(ts) else 0
                        print(f"    [{mi:>4}] {t:>14.6f} | {marker_values[mi]}")
                elif verbose:
                    print(f"\n    (Too many markers ({len(marker_values)}), showing first 20 only)")
                    for mi in range(min(20, len(marker_values))):
                        t = ts[mi] if ts is not None and mi < len(ts) else 0
                        print(f"    [{mi:>4}] {t:>14.6f} | {marker_values[mi]}")

    # --- Timing overview ---
    print_section("TIMING OVERVIEW")
    all_starts = []
    all_ends = []
    for stream in streams:
        ts = stream["time_stamps"]
        if ts is not None and len(ts) > 0:
            all_starts.append(ts[0])
            all_ends.append(ts[-1])

    if all_starts:
        global_start = min(all_starts)
        global_end = max(all_ends)
        total_duration = global_end - global_start
        print(f"  Total recording duration : {format_duration(total_duration)}")
        print(f"  Earliest timestamp       : {global_start:.6f}")
        print(f"  Latest timestamp         : {global_end:.6f}")

        print(f"\n  Stream Timeline:")
        for idx, stream in enumerate(streams):
            info = stream["info"]
            ts = stream["time_stamps"]
            name = get_stream_info_field(info, "name")
            if ts is not None and len(ts) > 0:
                offset_start = ts[0] - global_start
                offset_end = ts[-1] - global_start
                bar_len = 40
                bar_start = int((offset_start / total_duration) * bar_len) if total_duration > 0 else 0
                bar_end = int((offset_end / total_duration) * bar_len) if total_duration > 0 else bar_len
                bar = "." * bar_start + "#" * max(1, bar_end - bar_start) + "." * (bar_len - bar_end)
                print(f"  #{idx + 1:>2} {name:<20} |{bar}| {format_duration(offset_end - offset_start)}")

    print_separator("=")
    print("  Inspection complete!")
    print_separator("=")


def main():
    parser = argparse.ArgumentParser(
        description="XDF Inspector - Analyze and summarize the contents of .xdf files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xdf_inspector.py                          # Default: raw/S1.xdf
  python xdf_inspector.py raw/S1.xdf             # Specify a file
  python xdf_inspector.py raw/S1.xdf --verbose   # Show more details
        """
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="raw/S1.xdf",
        help="Path to the .xdf file (default: raw/S1.xdf)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show more details (per-channel stats, marker list, ...)"
    )

    args = parser.parse_args()
    inspect_xdf(args.filepath, args.verbose)


if __name__ == "__main__":
    main()
