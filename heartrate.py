import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def remove_outliers(signal_array, threshold=3.0, debug=False, label=""):
    signal_array = np.array(signal_array, dtype=float)
    diffs = np.diff(signal_array)

    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))
    mad_std = mad * 1.4826

    is_outlier = np.zeros(len(signal_array), dtype=bool)
    deviation_scores = np.zeros(len(signal_array))

    for i in range(1, len(signal_array)):
        jump = signal_array[i] - signal_array[i - 1]
        score = abs(jump - median_diff) / mad_std if mad_std > 0 else 0
        deviation_scores[i] = score
        if score > threshold:
            is_outlier[i] = True

    if debug:
        print(f"--- {label} ---")
        print(f"mad_std: {mad_std:.5f}")
        print(f"Outliers flagged: {is_outlier.sum()} / {len(signal_array)}")

    good_idx = np.where(~is_outlier)[0]
    bad_idx = np.where(is_outlier)[0]

    cleaned = signal_array.copy()
    if len(bad_idx) > 0:
        cleaned[bad_idx] = np.interp(bad_idx, good_idx, signal_array[good_idx])

    return cleaned


def process_signal(npz_path, label, threshold=3.0):
    """Load, clean, filter, and FFT a saved rPPG signal. Returns everything needed to plot."""
    data = np.load(npz_path)
    G = data["G"]
    fps = float(data["fps"])

    print(f"[{label}] Loaded {len(G)} frames at {fps:.2f} fps")

    G_clean = remove_outliers(G, threshold=threshold, debug=True, label=label)

    G_detrended = signal.detrend(G_clean)

    low_hz = 0.7
    high_hz = 4.0
    nyquist = fps / 2
    b, a = signal.butter(N=3, Wn=[low_hz / nyquist, high_hz / nyquist], btype="band")
    G_filtered = signal.filtfilt(b, a, G_detrended)

    n = len(G_filtered)
    freqs = np.fft.rfftfreq(n, d=1 / fps)
    fft_vals = np.abs(np.fft.rfft(G_filtered))

    valid = (freqs >= low_hz) & (freqs <= high_hz)
    peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
    bpm = peak_freq * 60

    print(f"[{label}] Estimated heart rate: {bpm:.1f} bpm")

    return {
        "raw": G,
        "clean": G_clean,
        "filtered": G_filtered,
        "freqs": freqs[valid],
        "fft_vals": fft_vals[valid],
        "peak_freq": peak_freq,
        "bpm": bpm,
        "label": label,
    }


# --- Process both signals ---
landmark_result = process_signal("rppg_signal.npz", "Landmark hull")
forehead_result = process_signal("rppg_signal_forehead.npz", "Forehead bbox")

# --- Plot side by side: landmark (left column) vs forehead (right column) ---
fig, axs = plt.subplots(4, 2, figsize=(16, 10))

for col, result in enumerate([landmark_result, forehead_result]):
    axs[0, col].plot(result["raw"])
    axs[0, col].set_title(f"{result['label']} — raw signal")

    axs[1, col].plot(result["clean"])
    axs[1, col].set_title(f"{result['label']} — after outlier removal")

    axs[2, col].plot(result["filtered"])
    axs[2, col].set_title(f"{result['label']} — filtered (bandpassed)")

    axs[3, col].plot(result["freqs"], result["fft_vals"])
    axs[3, col].axvline(result["peak_freq"], color="r", linestyle="--")
    axs[3, col].set_title(f"{result['label']} — FFT, peak at {result['bpm']:.1f} bpm")
    axs[3, col].set_xlabel("Hz")

plt.tight_layout()
plt.savefig("rppg_comparison_plot.png")
plt.show()