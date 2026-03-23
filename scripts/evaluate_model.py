import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)


def load_sentiment140():
    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        header=None
    )

    if df.shape[1] < 6:
        raise ValueError("Fișierul Sentiment140 nu are formatul așteptat.")

    df = df[[0, 5]].copy()
    df.columns = ["target", "text"]

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: 0, 4: 1})

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""]

    return df[["text", "label"]]


def main():
    print("Loading Sentiment140 dataset...")
    df = load_sentiment140()

    X = df["text"]
    y = df["label"]

    print("Creating baseline train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Saving shared test set for later comparisons...")
    joblib.dump(X_test, EVAL_DIR / "X_test.pkl")
    joblib.dump(y_test, EVAL_DIR / "y_test.pkl")

    baseline_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=10000
        )),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Training baseline Logistic Regression model...")
    baseline_model.fit(X_train, y_train)

    print("Evaluating baseline model...")
    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn)
    }

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    model_path = MODELS_DIR / "lr_sent140_tfidf.pkl"
    metrics_path = EVAL_DIR / "metrics.json"
    cm_path = EVAL_DIR / "confusion_matrix.csv"

    print("Saving baseline model...")
    joblib.dump(baseline_model, model_path)

    print("Saving metrics...")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Saving confusion matrix...")
    cm_df.to_csv(cm_path)

    print("\nDone.")
    print(f"Baseline model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Shared X_test saved to: {EVAL_DIR / 'X_test.pkl'}")
    print(f"Shared y_test saved to: {EVAL_DIR / 'y_test.pkl'}")


if __name__ == "__main__":
    main()