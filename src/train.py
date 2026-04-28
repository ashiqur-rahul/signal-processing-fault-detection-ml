import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

from features import build_feature_table

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_signals.csv"
MODEL_PATH = ROOT / "models" / "fault_detection_model.pkl"
FEATURE_PATH = ROOT / "data" / "features.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/generate_data.py")

    df = pd.read_csv(DATA_PATH)
    features_df = pd.DataFrame(build_feature_table(df))
    features_df.to_csv(FEATURE_PATH, index=False)

    X = features_df.drop(columns=["label"])
    y = features_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            random_state=42,
            class_weight="balanced"
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report")
    print(report)

    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature table saved to: {FEATURE_PATH}")
    print(f"Report saved to: {REPORT_DIR / 'classification_report.txt'}")

if __name__ == "__main__":
    main()
