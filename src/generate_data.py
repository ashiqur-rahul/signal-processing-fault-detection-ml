import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_signal(label: str, sampling_rate: int = 1000, duration: float = 1.0, seed: int | None = None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    signal = (
        1.0 * np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
    )
    signal = signal + rng.normal(0, 0.15, len(t))

    if label == "fault":
        fault_component = 0.8 * np.sin(2 * np.pi * 260 * t)
        burst = np.zeros_like(t)
        burst_start = int(0.35 * len(t))
        burst_end = int(0.55 * len(t))
        burst[burst_start:burst_end] = 1.5 * np.sin(2 * np.pi * 400 * t[burst_start:burst_end])
        signal = 1.2 * signal + fault_component + burst

    return t, signal

def create_dataset(n_samples_per_class: int = 300, sampling_rate: int = 1000, duration: float = 1.0):
    rows = []

    for label in ["normal", "fault"]:
        for i in range(n_samples_per_class):
            _, signal = generate_signal(
                label=label,
                sampling_rate=sampling_rate,
                duration=duration,
                seed=i if label == "normal" else i + 10000,
            )
            row = {"label": label}
            for j, value in enumerate(signal):
                row[f"x_{j}"] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = DATA_DIR / "synthetic_signals.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    create_dataset()
