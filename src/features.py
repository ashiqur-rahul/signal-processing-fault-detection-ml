import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

def band_energy(freqs, fft_values, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(fft_values[mask] ** 2))

def extract_features(signal: np.ndarray, sampling_rate: int = 1000) -> dict:
    signal = np.asarray(signal, dtype=float)

    fft_values = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1 / sampling_rate)

    fft_sum = np.sum(fft_values) + 1e-12
    dominant_frequency = freqs[np.argmax(fft_values)]
    spectral_centroid = np.sum(freqs * fft_values) / fft_sum
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_values) / fft_sum)

    features = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak_to_peak": float(np.ptp(signal)),
        "energy": float(np.sum(signal ** 2)),
        "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)),
        "dominant_frequency": float(dominant_frequency),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "low_band_energy": band_energy(freqs, fft_values, 0, 150),
        "mid_band_energy": band_energy(freqs, fft_values, 150, 300),
        "high_band_energy": band_energy(freqs, fft_values, 300, 500),
    }

    return features

def build_feature_table(df, sampling_rate: int = 1000):
    feature_rows = []
    signal_columns = [col for col in df.columns if col.startswith("x_")]

    for _, row in df.iterrows():
        signal = row[signal_columns].values.astype(float)
        features = extract_features(signal, sampling_rate=sampling_rate)
        features["label"] = row["label"]
        feature_rows.append(features)

    return feature_rows
