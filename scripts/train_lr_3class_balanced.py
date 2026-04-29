import json
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "twitter_training.csv"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "lr_3class_balanced.joblib"
METRICS_PATH = EVAL_DIR / "lr_3class_metrics.json"
CM_PATH = EVAL_DIR / "lr_3class_confusion_matrix.csv"
REPORT_PATH = EVAL_DIR / "lr_3class_classification_report.csv"

LABEL_MAP = {
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    df = pd.read_csv(DATA_PATH, header=None)

    # Format Kaggle uzual:
    # 0 = tweet id, 1 = entity, 2 = sentiment, 3 = text
    df = df[[2, 3]].copy()
    df.columns = ["label", "text"]

    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["label"] = df["label"].map(LABEL_MAP)

    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].apply(clean_text)

    df = df[df["text_clean"] != ""].copy()

    return df[["text_clean", "label"]]


def main():
    print("Incarc datasetul nou cu 3 clase...")
    df = load_dataset()

    print("Distributie clase:")
    print(df["label"].value_counts())

    X = df["text_clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("Antrenez Logistic Regression 3 clase cu class_weight='balanced'...")
    pipeline.fit(X_train, y_train)

    print("Evaluez modelul...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {x}" for x in labels],
        columns=[f"Predicted {x}" for x in labels],
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    cm_df.to_csv(CM_PATH)
    report_df.to_csv(REPORT_PATH)

    print("GATA.")
    print(f"Model salvat: {MODEL_PATH}")
    print(f"Metrici salvate: {METRICS_PATH}")
    print(f"Confusion matrix salvata: {CM_PATH}")
    print(f"Classification report salvat: {REPORT_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()