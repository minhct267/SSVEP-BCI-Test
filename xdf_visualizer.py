"""
XDF Visualizer - Visualize EEG data from .xdf files (EEGLAB-style)
===================================================================
Visualization tool for EEG data recorded via Lab Streaming Layer,
providing plots similar to EEGLAB in MATLAB:

  1. Multi-channel time series (scrollable)
  2. Power Spectral Density (PSD)
  3. Spectrogram (Time-Frequency)
  4. Marker/Event overlay on EEG signals
  5. Interactive scroll & zoom

Usage:
    python xdf_visualizer.py                            # Default: raw/S1.xdf
    python xdf_visualizer.py path/to/file.xdf           # Specify a file
    python xdf_visualizer.py path/to/file.xdf --stream 0 --window 10
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pyxdf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch
from scipy import signal


# --- Utility functions --------------------------------------------------------

def get_info_field(info: dict, field: str, default: str = "") -> str:
    val = info.get(field)
    if val is None:
        return default
    if isinstance(val, list):
        return str(val[0]) if len(val) > 0 else default
    return str(val)


def get_channel_labels(info: dict, n_channels: int) -> list:
    """Extract channel names from metadata, falling back to Ch0, Ch1, ..."""
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
                                labels.append(label if label else f"Ch{len(labels)}")
    if len(labels) < n_channels:
        labels.extend([f"Ch{i}" for i in range(len(labels), n_channels)])
    return labels[:n_channels]


def find_eeg_stream(streams: list) -> tuple:
    """Find the EEG stream (continuous, srate > 0) and marker stream."""
    eeg_idx = None
    marker_idx = None

    for i, stream in enumerate(streams):
        info = stream["info"]
        stype = get_info_field(info, "type").lower()
        srate = float(get_info_field(info, "nominal_srate", "0"))

        if stype in ("eeg", "exg") or (srate > 0 and stype not in ("markers", "marker", "events")):
            if eeg_idx is None:
                eeg_idx = i
        elif stype in ("markers", "marker", "events") or srate == 0:
            if marker_idx is None:
                marker_idx = i

    return eeg_idx, marker_idx


def bandpass_filter(data: np.ndarray, srate: float, low: float = 1.0, high: float = 50.0, order: int = 4):
    """Apply a Butterworth bandpass filter."""
    nyq = srate / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = signal.butter(order, [low_n, high_n], btype="band")
    return signal.filtfilt(b, a, data, axis=0)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """Remove DC offset (mean) from each channel -- essential for EEG display."""
    return data - np.mean(data, axis=0, keepdims=True)


def print_data_diagnostics(data: np.ndarray, ch_labels: list, title: str = "Data"):
    """Print detailed data statistics for debugging."""
    n_samples, n_channels = data.shape
    print(f"\n  {title} Diagnostics:")
    print(f"    Shape          : {data.shape}")
    print(f"    Dtype          : {data.dtype}")
    print(f"    Global min     : {np.min(data):.6f}")
    print(f"    Global max     : {np.max(data):.6f}")
    print(f"    Global mean    : {np.mean(data):.6f}")
    print(f"    Global std     : {np.std(data):.6f}")
    print(f"    {'Channel':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for ci in range(min(n_channels, 16)):
        ch = data[:, ci]
        label = ch_labels[ci] if ci < len(ch_labels) else f"Ch{ci}"
        print(f"    {label:<12} {np.min(ch):>12.4f} {np.max(ch):>12.4f} "
              f"{np.mean(ch):>12.4f} {np.std(ch):>12.4f}")

    all_zero = np.allclose(data, 0, atol=1e-10)
    all_const = np.allclose(np.std(data, axis=0), 0, atol=1e-10)
    if all_zero:
        print(f"\n    [WARNING] All data values are zero! Check recording/device.")
    elif all_const:
        print(f"\n    [WARNING] All channels are constant (no variation)!")


# --- Plot 1: Multi-channel time series (EEGLAB Scroll-style) -----------------

def plot_multichannel(data: np.ndarray, srate: float, ch_labels: list,
                      markers=None, marker_ts=None, eeg_ts=None,
                      window_sec: float = 10.0, title: str = "EEG Time Series"):
    """
    Display multi-channel EEG signals in a scrollable view (EEGLAB scroll style).
    - DC offset is removed per channel (mean-subtracted)
    - Each channel is offset along the Y axis
    - A slider allows scrolling through time
    - Markers are shown as vertical dashed lines
    """
    n_samples, n_channels = data.shape
    duration = n_samples / srate

    # CRITICAL: Remove DC offset from each channel before display.
    # Raw EEG almost always has a large DC bias that pushes signals outside
    # the visible Y-axis range, making them appear flat/invisible.
    data_display = remove_dc_offset(data)

    # Compute inter-channel spacing using robust statistics.
    # Use the 2nd-98th percentile range as a robust amplitude measure,
    # which is less sensitive to outliers/artifacts than plain std.
    channel_amp = np.zeros(n_channels)
    for ci in range(n_channels):
        p2, p98 = np.percentile(data_display[:, ci], [2, 98])
        channel_amp[ci] = p98 - p2

    median_amp = np.median(channel_amp)
    if median_amp > 0:
        spacing = median_amp * 1.2
    else:
        channel_std = np.std(data_display, axis=0)
        median_std = np.median(channel_std)
        if median_std > 0:
            spacing = median_std * 5
        else:
            spacing = 1.0
            print("  [WARNING] All channels appear flat/constant in this view.")

    fig, ax = plt.subplots(figsize=(14, max(6, n_channels * 0.5 + 2)))
    plt.subplots_adjust(bottom=0.18, top=0.93, left=0.1, right=0.95)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")

    time = np.arange(n_samples) / srate

    win_samples = int(window_sec * srate)

    lines = []
    for ci in range(n_channels):
        offset = (n_channels - 1 - ci) * spacing
        seg = data_display[:win_samples, ci] + offset
        line, = ax.plot(time[:win_samples], seg, linewidth=0.5, color="#2c3e50")
        lines.append(line)

    marker_lines = []
    marker_texts = []
    if markers is not None and marker_ts is not None and eeg_ts is not None:
        eeg_start = eeg_ts[0]
        for mi in range(len(markers)):
            mt = marker_ts[mi] - eeg_start
            if 0 <= mt <= window_sec:
                vl = ax.axvline(mt, color="#e74c3c", alpha=0.6, linewidth=1, linestyle="--")
                marker_lines.append(vl)
                mv = str(markers[mi][0]) if isinstance(markers[mi], (list, np.ndarray)) else str(markers[mi])
                txt = ax.text(mt, (n_channels - 0.3) * spacing, mv,
                              fontsize=7, color="#e74c3c", rotation=90, va="bottom", ha="right")
                marker_texts.append(txt)

    ax.set_xlim(0, window_sec)
    ax.set_ylim(-spacing, n_channels * spacing)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    yticks = [(n_channels - 1 - ci) * spacing for ci in range(n_channels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ch_labels, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    ax_slider = plt.axes([0.1, 0.05, 0.75, 0.03], facecolor="#ecf0f1")
    max_start = max(0, duration - window_sec)
    slider = Slider(ax_slider, "Time (s)", 0, max_start,
                    valinit=0, valstep=0.5, color="#3498db")

    def update(val):
        start_time = slider.val
        start_sample = int(start_time * srate)
        end_sample = min(start_sample + win_samples, n_samples)

        t_seg = time[start_sample:end_sample]
        for ci in range(n_channels):
            offset = (n_channels - 1 - ci) * spacing
            seg = data_display[start_sample:end_sample, ci] + offset
            lines[ci].set_xdata(t_seg)
            lines[ci].set_ydata(seg)

        ax.set_xlim(start_time, start_time + window_sec)

        for vl in marker_lines:
            vl.remove()
        for txt in marker_texts:
            txt.remove()
        marker_lines.clear()
        marker_texts.clear()

        if markers is not None and marker_ts is not None and eeg_ts is not None:
            eeg_start = eeg_ts[0]
            for mi in range(len(markers)):
                mt = marker_ts[mi] - eeg_start
                if start_time <= mt <= start_time + window_sec:
                    vl = ax.axvline(mt, color="#e74c3c", alpha=0.6, linewidth=1, linestyle="--")
                    marker_lines.append(vl)
                    mv = str(markers[mi][0]) if isinstance(markers[mi], (list, np.ndarray)) else str(markers[mi])
                    txt = ax.text(mt, (n_channels - 0.3) * spacing, mv,
                                  fontsize=7, color="#e74c3c", rotation=90, va="bottom", ha="right")
                    marker_texts.append(txt)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        step = window_sec * 0.5
        if event.key == "right":
            slider.set_val(min(slider.val + step, max_start))
        elif event.key == "left":
            slider.set_val(max(slider.val - step, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


# --- Plot 2: Power Spectral Density (PSD) ------------------------------------

def plot_psd(data: np.ndarray, srate: float, ch_labels: list,
             fmin: float = 0.5, fmax: float = 60.0, title: str = "Power Spectral Density"):
    """
    Plot PSD for all channels (similar to EEGLAB spectopo).
    Uses Welch's method.
    """
    n_samples, n_channels = data.shape

    nperseg = min(int(srate * 2), n_samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, wspace=0.3)

    colors = plt.cm.tab20(np.linspace(0, 1, n_channels))

    all_psd = []
    for ci in range(n_channels):
        freqs, psd = signal.welch(data[:, ci], fs=srate, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= fmin) & (freqs <= fmax)
        psd_db = 10 * np.log10(psd[mask] + 1e-20)
        all_psd.append(psd_db)
        ax1.plot(freqs[mask], psd_db, linewidth=0.8, color=colors[ci],
                 alpha=0.7, label=ch_labels[ci])

    ax1.set_xlabel("Frequency (Hz)", fontsize=11)
    ax1.set_ylabel("Power (dB)", fontsize=11)
    ax1.set_title("PSD per Channel", fontsize=12)
    ax1.grid(True, alpha=0.3)
    if n_channels <= 16:
        ax1.legend(fontsize=7, loc="upper right", ncol=2)

    # Mark common SSVEP frequency bands
    ssvep_freqs = [7, 7.5, 8, 8.5, 9, 11]
    for f in ssvep_freqs:
        if fmin <= f <= fmax:
            ax1.axvline(f, color="#e74c3c", alpha=0.15, linewidth=1, linestyle=":")

    # Average PSD
    all_psd_arr = np.array(all_psd)
    mean_psd = np.mean(all_psd_arr, axis=0)
    std_psd = np.std(all_psd_arr, axis=0)
    freqs_masked = freqs[mask]

    ax2.plot(freqs_masked, mean_psd, color="#2c3e50", linewidth=2, label="Mean")
    ax2.fill_between(freqs_masked, mean_psd - std_psd, mean_psd + std_psd,
                     alpha=0.2, color="#3498db")
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Power (dB)", fontsize=11)
    ax2.set_title("Mean PSD +/- Std", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    for f in ssvep_freqs:
        if fmin <= f <= fmax:
            ax2.axvline(f, color="#e74c3c", alpha=0.15, linewidth=1, linestyle=":")
            ax2.text(f, ax2.get_ylim()[1], f"{f}Hz", fontsize=7, ha="center",
                     va="bottom", color="#e74c3c", alpha=0.6)

    plt.show()


# --- Plot 3: Spectrogram (Time-Frequency) ------------------------------------

def plot_spectrogram(data: np.ndarray, srate: float, ch_labels: list,
                     channel_idx: int = 0, fmin: float = 0.5, fmax: float = 60.0,
                     title: str = "Spectrogram (Time-Frequency)"):
    """
    Plot spectrogram for a specific channel (similar to EEGLAB time-frequency plot).
    DC offset is removed before computing spectrogram.
    """
    ch_data = data[:, channel_idx] - np.mean(data[:, channel_idx])
    n_samples = len(ch_data)

    nperseg = min(int(srate * 2), n_samples)
    noverlap = int(nperseg * 0.9)

    freqs, times, Sxx = signal.spectrogram(ch_data, fs=srate,
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            window="hann")

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    Sxx_db = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [1, 3]},
                                    sharex=True)
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(f"{title} - Channel: {ch_labels[channel_idx]}", fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.1)

    time_axis = np.arange(n_samples) / srate
    ax1.plot(time_axis, ch_data, linewidth=0.3, color="#2c3e50")
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title(f"Signal (DC removed) - {ch_labels[channel_idx]}", fontsize=11)
    ax1.grid(True, alpha=0.3)

    im = ax2.pcolormesh(times, freqs[freq_mask], Sxx_db,
                         shading="gouraud", cmap="jet")
    ax2.set_ylabel("Frequency (Hz)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax2, pad=0.02, aspect=30)
    cbar.set_label("Power (dB)", fontsize=10)

    ssvep_freqs = [7, 7.5, 8, 8.5, 9, 11]
    for f in ssvep_freqs:
        if fmin <= f <= fmax:
            ax2.axhline(f, color="white", alpha=0.3, linewidth=0.5, linestyle="--")

    plt.show()


# --- Plot 4: All channels overview -------------------------------------------

def plot_channel_overview(data: np.ndarray, srate: float, ch_labels: list,
                          title: str = "Channel Overview"):
    """
    Show a quick overview of all channels in a subplot grid.
    Each channel gets its own small waveform panel (DC offset removed).
    """
    n_samples, n_channels = data.shape
    n_cols = min(4, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))

    # Remove DC offset for each channel
    data_centered = remove_dc_offset(data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(4, n_rows * 2)),
                              sharex=True, sharey=False)
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.06)

    if n_channels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    time = np.arange(n_samples) / srate
    if n_samples > 5000:
        step = n_samples // 5000
        time_ds = time[::step]
        data_ds = data_centered[::step, :]
    else:
        time_ds = time
        data_ds = data_centered

    for ci in range(n_channels):
        row = ci // n_cols
        col = ci % n_cols
        ax = axes[row, col]
        ax.plot(time_ds, data_ds[:, ci], linewidth=0.3, color="#2c3e50")
        ax.set_title(ch_labels[ci], fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for ci in range(n_channels, n_rows * n_cols):
        row = ci // n_cols
        col = ci % n_cols
        axes[row, col].set_visible(False)

    plt.show()


# --- Main Dashboard -----------------------------------------------------------

def visualize_xdf(filepath: str, stream_idx: int = None, window_sec: float = 10.0,
                  filter_low: float = None, filter_high: float = None,
                  no_filter: bool = False, fmax: float = 60.0):
    """Main function: load XDF and display plots."""

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    print(f"Loading: {filepath}")
    streams, header = pyxdf.load_xdf(str(filepath))
    print(f"Loaded {len(streams)} stream(s)")

    # Print all streams for visibility
    print(f"\n  Available streams:")
    for i, s in enumerate(streams):
        s_name = get_info_field(s["info"], "name")
        s_type = get_info_field(s["info"], "type")
        s_srate = get_info_field(s["info"], "nominal_srate", "0")
        s_ch = get_info_field(s["info"], "channel_count", "?")
        s_fmt = get_info_field(s["info"], "channel_format", "?")
        n_samp = len(s["time_stamps"]) if s["time_stamps"] is not None else 0
        print(f"    [{i}] {s_name} (type={s_type}, srate={s_srate}, "
              f"ch={s_ch}, fmt={s_fmt}, samples={n_samp:,})")

    # Select EEG stream
    if stream_idx is not None:
        if stream_idx >= len(streams):
            print(f"[ERROR] Stream index {stream_idx} is invalid (only {len(streams)} streams available)")
            sys.exit(1)
        eeg_idx = stream_idx
        marker_idx = None
        for i, s in enumerate(streams):
            stype = get_info_field(s["info"], "type").lower()
            if stype in ("markers", "marker", "events") and i != stream_idx:
                marker_idx = i
                break
    else:
        eeg_idx, marker_idx = find_eeg_stream(streams)

    if eeg_idx is None:
        print("[ERROR] No EEG/data stream found!")
        print("\nPlease specify stream index: python xdf_visualizer.py file.xdf --stream <index>")
        sys.exit(1)

    eeg_stream = streams[eeg_idx]
    eeg_data = eeg_stream["time_series"]
    eeg_ts = eeg_stream["time_stamps"]
    eeg_info = eeg_stream["info"]

    # Validate data format -- pyxdf returns lists for string streams
    if isinstance(eeg_data, list):
        print("[WARNING] EEG data is in string/list format. Attempting conversion...")
        try:
            eeg_data = np.array(eeg_data, dtype=np.float64)
        except (ValueError, TypeError):
            print("[ERROR] Cannot convert EEG data to numeric format!")
            print("  This stream may contain string data (markers, not EEG).")
            print("  Try specifying a different stream: --stream <index>")
            sys.exit(1)

    if eeg_data.ndim != 2:
        print(f"[ERROR] Unexpected data shape: {eeg_data.shape} (expected 2D array)")
        sys.exit(1)

    srate = float(get_info_field(eeg_info, "nominal_srate", "0"))
    if srate == 0:
        if len(eeg_ts) > 1:
            srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        else:
            print("[ERROR] Cannot determine sampling rate!")
            sys.exit(1)

    n_samples, n_channels = eeg_data.shape
    ch_labels = get_channel_labels(eeg_info, n_channels)

    name = get_info_field(eeg_info, "name", "Unknown")
    duration = n_samples / srate

    print(f"\n{'=' * 55}")
    print(f"  SELECTED EEG STREAM")
    print(f"{'=' * 55}")
    print(f"  Name          : {name} (index {eeg_idx})")
    print(f"  Channels      : {n_channels}")
    print(f"  Srate (Hz)    : {srate:.1f}")
    print(f"  Samples       : {n_samples:,}")
    print(f"  Duration      : {duration:.2f}s")
    print(f"  Ch Labels     : {', '.join(ch_labels[:8])}{'...' if n_channels > 8 else ''}")

    # Convert and validate data
    plot_data = eeg_data.copy().astype(np.float64)

    # Print raw data diagnostics BEFORE any processing
    print_data_diagnostics(plot_data, ch_labels, title="Raw Data")

    # Apply bandpass filter:
    # By default, apply 1-50 Hz bandpass to remove DC offset and high-freq noise.
    # Use --no-filter to disable. Use --filter LOW HIGH to customize.
    if no_filter:
        print("\n  Filtering: DISABLED (--no-filter)")
    elif filter_low is not None or filter_high is not None:
        low = filter_low if filter_low is not None else 1.0
        high = filter_high if filter_high is not None else min(srate / 2 - 1, fmax)
        print(f"\n  Applying bandpass filter: {low}-{high} Hz")
        try:
            plot_data = bandpass_filter(plot_data, srate, low, high)
        except Exception as e:
            print(f"  [WARNING] Could not apply filter: {e}. Using raw data.")
    else:
        low, high = 1.0, min(50.0, srate / 2 - 1)
        print(f"\n  Applying default bandpass filter: {low}-{high} Hz")
        print(f"  (Use --no-filter to disable, or --filter LOW HIGH to customize)")
        try:
            plot_data = bandpass_filter(plot_data, srate, low, high)
        except Exception as e:
            print(f"  [WARNING] Could not apply filter: {e}. Using raw data.")

    # Print filtered data diagnostics
    if not no_filter:
        print_data_diagnostics(plot_data, ch_labels, title="Filtered Data")

    print(f"{'=' * 55}")

    # Markers
    markers = None
    marker_ts_data = None
    if marker_idx is not None:
        marker_stream = streams[marker_idx]
        markers = marker_stream["time_series"]
        marker_ts_data = marker_stream["time_stamps"]
        m_name = get_info_field(marker_stream["info"], "name", "Markers")
        print(f"\n  Marker stream : {m_name} ({len(markers)} markers)")
    else:
        print(f"\n  Marker stream : (none found)")

    # --- Plot selection menu ---
    print("\n" + "=" * 50)
    print("  SELECT A PLOT TO DISPLAY")
    print("=" * 50)
    print("  [1] Multi-channel scroll view (EEGLAB-style)")
    print("  [2] Power Spectral Density (PSD)")
    print("  [3] Spectrogram (Time-Frequency)")
    print("  [4] All channels overview")
    print("  [5] Show all (sequentially)")
    print("  [0] Exit")
    print("=" * 50)

    while True:
        try:
            choice = input("\nEnter choice (0-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == "0":
            print("Exiting.")
            break

        elif choice == "1":
            print("Plotting multi-channel scroll view...")
            plot_multichannel(
                plot_data, srate, ch_labels,
                markers=markers, marker_ts=marker_ts_data, eeg_ts=eeg_ts,
                window_sec=window_sec,
                title=f"EEG Time Series - {name} ({srate:.0f} Hz, {n_channels} ch)"
            )

        elif choice == "2":
            print("Plotting PSD...")
            plot_psd(
                plot_data, srate, ch_labels,
                fmax=fmax,
                title=f"Power Spectral Density - {name}"
            )

        elif choice == "3":
            if n_channels > 1:
                print(f"Select channel for spectrogram (0-{n_channels - 1}):")
                for ci in range(n_channels):
                    print(f"  [{ci}] {ch_labels[ci]}")
                try:
                    ch_choice = int(input("Channel: ").strip())
                    if ch_choice < 0 or ch_choice >= n_channels:
                        print("Invalid channel!")
                        continue
                except ValueError:
                    ch_choice = 0
            else:
                ch_choice = 0

            print(f"Plotting spectrogram for channel {ch_labels[ch_choice]}...")
            plot_spectrogram(
                plot_data, srate, ch_labels,
                channel_idx=ch_choice, fmax=fmax,
                title=f"Spectrogram - {name}"
            )

        elif choice == "4":
            print("Plotting channel overview...")
            plot_channel_overview(
                plot_data, srate, ch_labels,
                title=f"Channel Overview - {name} ({srate:.0f} Hz)"
            )

        elif choice == "5":
            print("Showing all plots sequentially...")
            print("  [1/4] Multi-channel scroll view...")
            plot_multichannel(
                plot_data, srate, ch_labels,
                markers=markers, marker_ts=marker_ts_data, eeg_ts=eeg_ts,
                window_sec=window_sec,
                title=f"EEG Time Series - {name} ({srate:.0f} Hz, {n_channels} ch)"
            )
            print("  [2/4] PSD...")
            plot_psd(
                plot_data, srate, ch_labels,
                fmax=fmax,
                title=f"Power Spectral Density - {name}"
            )
            print("  [3/4] Spectrogram (channel 0)...")
            plot_spectrogram(
                plot_data, srate, ch_labels,
                channel_idx=0, fmax=fmax,
                title=f"Spectrogram - {name}"
            )
            print("  [4/4] Channel overview...")
            plot_channel_overview(
                plot_data, srate, ch_labels,
                title=f"Channel Overview - {name} ({srate:.0f} Hz)"
            )

        else:
            print("Invalid choice! Enter a number from 0-5.")


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XDF Visualizer - Visualize EEG data from .xdf files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xdf_visualizer.py                               # Default: raw/S1.xdf
  python xdf_visualizer.py raw/S1.xdf                  # Specify a file
  python xdf_visualizer.py raw/S1.xdf --stream 0       # Select specific stream
  python xdf_visualizer.py raw/S1.xdf --window 5       # 5-second window
  python xdf_visualizer.py raw/S1.xdf --filter 1 50    # Bandpass filter 1-50 Hz
  python xdf_visualizer.py raw/S1.xdf --no-filter      # Disable default filter
  python xdf_visualizer.py raw/S1.xdf --fmax 80        # PSD/spectrogram up to 80 Hz
        """
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="raw/S1.xdf",
        help="Path to the .xdf file (default: raw/S1.xdf)"
    )
    parser.add_argument(
        "--stream", "-s",
        type=int,
        default=None,
        help="Index of the EEG stream to visualize (default: auto-detect)"
    )
    parser.add_argument(
        "--window", "-w",
        type=float,
        default=10.0,
        help="Display window width in seconds (default: 10)"
    )
    parser.add_argument(
        "--filter", "-f",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Bandpass filter (Hz), e.g.: --filter 1 50"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable the default 1-50 Hz bandpass filter"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=60.0,
        help="Maximum frequency for PSD/spectrogram (Hz, default: 60)"
    )

    args = parser.parse_args()

    filter_low = args.filter[0] if args.filter else None
    filter_high = args.filter[1] if args.filter else None

    visualize_xdf(
        args.filepath,
        stream_idx=args.stream,
        window_sec=args.window,
        filter_low=filter_low,
        filter_high=filter_high,
        no_filter=args.no_filter,
        fmax=args.fmax
    )


if __name__ == "__main__":
    main()
