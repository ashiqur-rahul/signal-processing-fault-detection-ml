# Signal Processing Fault Detection with Machine Learning

A Python-based project for detecting faults in sensor signals using signal processing and machine learning techniques.

This project demonstrates how time-domain and frequency-domain signal features can be extracted and used to classify system conditions (normal vs faulty).

---

## 🚀 Project Overview

Many engineering systems (e.g., ultrasonic inspection, predictive maintenance, embedded sensing, communication systems) rely on analyzing noisy sensor signals.

This project simulates such signals and builds a full pipeline:

- Signal generation (normal vs fault)
- Signal visualization (time + frequency domain)
- Feature extraction (statistical + spectral)
- Machine learning classification
- Model evaluation and reporting

---

## 🧠 Key Concepts Demonstrated

- Signal Processing (Fourier Transform, spectral analysis)
- Feature Engineering (time + frequency domain)
- Machine Learning (classification pipeline)
- System-level thinking (data → features → model → evaluation)
- Engineering diagnostics workflow

---

## 📂 Project Structure

signal-processing-fault-detection-ml/
├── data/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── run_pipeline.py
├── tests/
├── docs/
├── notebooks/
├── requirements.txt
└── README.md

---

## ⚙️ How It Works

### 1. Signal Generation

Two types of signals are simulated:

- Normal signal: low-frequency components + noise
- Fault signal: additional high-frequency components + bursts

---

### 2. Feature Extraction

Time-domain features:
- Mean
- Standard deviation
- RMS
- Peak-to-peak amplitude
- Energy
- Skewness
- Kurtosis

Frequency-domain features:
- Dominant frequency
- Spectral centroid
- Spectral bandwidth
- Frequency band energy (low / mid / high)

---

### 3. Machine Learning

- Model: Random Forest
- Pipeline:
  - Feature scaling
  - Training & testing split
  - Classification

---

## 📊 Results

### Time-domain signals

![Normal Signal](outputs/figures/time_signal_normal.png)  
![Fault Signal](outputs/figures/time_signal_fault.png)

---

### Frequency-domain analysis

![Normal Spectrum](outputs/figures/spectrum_normal.png)  
![Fault Spectrum](outputs/figures/spectrum_fault.png)

---

### Model Evaluation

![Confusion Matrix](outputs/figures/confusion_matrix.png)

![Feature Importance](outputs/figures/feature_importance.png)

---

## 📈 Performance

The model achieves very high accuracy on the dataset.

Note: Since the dataset is synthetically generated with clearly separable patterns, near-perfect accuracy is expected. This project focuses on demonstrating the signal processing + ML workflow rather than real-world performance limits.

---

## ▶️ How to Run

### 1. Create environment

python -m venv .venv

### 2. Activate environment

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Run full pipeline

python src/run_pipeline.py

---

## 🧪 Testing

pytest

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 🎯 Relevance

This project is relevant to:

- Signal Processing Engineer roles
- AI / Machine Learning Engineer roles
- Embedded Systems
- Sensor Data Analysis
- Predictive Maintenance
- Industrial Inspection Systems

---

## 🔮 Future Improvements

- Add wavelet transform features
- Use real sensor datasets (ultrasound, vibration)
- Add deep learning models (CNN on raw signals)
- Deploy on edge devices (Jetson / Raspberry Pi)
- Build a Streamlit dashboard

---

## 👤 Author

Ashiqur Rahman Rahul  
AI, Signal Processing, and Embedded Systems Engineer  
Berlin, Germany
