import numpy as np
from src.features import extract_features

def test_extract_features_contains_expected_keys():
    signal = np.sin(np.linspace(0, 2 * np.pi, 1000))
    features = extract_features(signal)

    expected_keys = {
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "energy",
        "skewness",
        "kurtosis",
        "dominant_frequency",
        "spectral_centroid",
        "spectral_bandwidth",
        "low_band_energy",
        "mid_band_energy",
        "high_band_energy",
    }

    assert expected_keys.issubset(features.keys())

def test_extract_features_returns_finite_numbers():
    signal = np.random.normal(0, 1, 1000)
    features = extract_features(signal)

    for value in features.values():
        assert np.isfinite(value)
