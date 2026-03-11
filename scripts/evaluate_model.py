import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_lr_sent140.joblib"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_sent140.joblib"
DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
OUT_DIR = BASE_DIR / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

# Sentiment140 original: fara header
df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=["target", "ids", "date", "flag", "user", "text"]
)

# pastram doar clasele negative si pozitive
df = df[df["target"].isin([0, 4])].copy()

# convertim la 0/1
df["label"] = df["target"].map({0: 0, 4: 1})

# eliminam valori lipsa
df = df.dropna(subset=["text", "label"])

X = df["text"].astype(str)
y = df["label"].astype(int)

print("Rows used:", len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading model and TF-IDF...")
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "true_positive": int(tp),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame(
    cm,
    index=["actual_neg", "actual_pos"],
    columns=["pred_neg", "pred_pos"]
).to_csv(OUT_DIR / "confusion_matrix.csv")

print("Evaluarea a fost salvată.")

# CV simplu pe reprezentarea curenta
X_all_tfidf = tfidf.transform(X)
cv_scores = cross_val_score(model, X_all_tfidf, y, cv=5, scoring="accuracy")

print("CV accuracy scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))