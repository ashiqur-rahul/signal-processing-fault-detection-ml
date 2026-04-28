import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pathlib import Path

from generate_data import generate_signal

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_time_signal(label: str):
    sampling_rate = 1000
    t, signal = generate_signal(label=label, sampling_rate=sampling_rate, seed=42)

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Time-Domain Signal: {label.capitalize()}")
    plt.tight_layout()
    path = FIG_DIR / f"time_signal_{label}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

def plot_frequency_spectrum(label: str):
    sampling_rate = 1000
    _, signal = generate_signal(label=label, sampling_rate=sampling_rate, seed=42)

    fft_values = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1 / sampling_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_values)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Spectrum: {label.capitalize()}")
    plt.tight_layout()
    path = FIG_DIR / f"spectrum_{label}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

def main():
    for label in ["normal", "fault"]:
        plot_time_signal(label)
        plot_frequency_spectrum(label)

if __name__ == "__main__":
    main()
