import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "fault_detection_model.pkl"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError("Feature table not found. Run: python src/train.py")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")

    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    model = load(MODEL_PATH)
    y_pred = model.predict(X)

    report = classification_report(y, y_pred)
    print("Evaluation on full dataset")
    print(report)

    (REPORT_DIR / "full_dataset_evaluation.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y, y_pred, labels=["normal", "fault"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "fault"])
    disp.plot()
    plt.title("Confusion Matrix")
    output_path = FIG_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(importance_df["feature"].head(10)[::-1], importance_df["importance"].head(10)[::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    importance_path = FIG_DIR / "feature_importance.png"
    plt.savefig(importance_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to: {importance_path}")

if __name__ == "__main__":
    main()
