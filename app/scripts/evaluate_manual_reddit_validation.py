import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

INPUT_PATH = EVAL_DIR / "reddit_manual_validation_sample.csv"
OUTPUT_METRICS_PATH = EVAL_DIR / "reddit_manual_validation_metrics.json"
OUTPUT_REPORT_PATH = EVAL_DIR / "reddit_manual_validation_report.csv"
OUTPUT_CM_PATH = EVAL_DIR / "reddit_manual_validation_confusion_matrix.csv"

LABELS = ["negative", "neutral", "positive"]

METHOD_COLUMNS = {
    "lr_3class_balanced": "lr_label",
    "vader": "vader_label",
    "deep_learning_transformer": "dl_label",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if "manual_label" not in df.columns:
        raise ValueError("Lipseste coloana manual_label.")

    df["manual_label"] = df["manual_label"].astype(str).str.lower().str.strip()
    df = df[df["manual_label"].isin(LABELS)].copy()

    if df.empty:
        raise ValueError("Nu exista randuri etichetate manual corect.")

    metrics = {}
    reports = []

    for method, pred_col in METHOD_COLUMNS.items():
        valid = df[df[pred_col].notna()].copy()
        valid[pred_col] = valid[pred_col].astype(str).str.lower().str.strip()
        valid = valid[valid[pred_col].isin(LABELS)]

        y_true = valid["manual_label"]
        y_pred = valid[pred_col]

        method_metrics = {
            "sample_size": int(len(valid)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        metrics[method] = method_metrics

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        report_df.insert(0, "method", method)
        report_df.insert(1, "class", report_df.index)
        reports.append(report_df)

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in LABELS],
            columns=[f"Predicted {label}" for label in LABELS],
        )
        cm_df.insert(0, "method", method)

        if OUTPUT_CM_PATH.exists():
            old = pd.read_csv(OUTPUT_CM_PATH, index_col=0)
            old = old[old["method"] != method]
            cm_df = pd.concat([old, cm_df])

        cm_df.to_csv(OUTPUT_CM_PATH)

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    final_report = pd.concat(reports)
    final_report.to_csv(OUTPUT_REPORT_PATH, index=False)

    print("GATA.")
    print(f"Randuri validate manual: {len(df)}")
    print(f"Metrici salvate: {OUTPUT_METRICS_PATH}")
    print(f"Raport salvat: {OUTPUT_REPORT_PATH}")
    print(f"Confusion matrix salvata: {OUTPUT_CM_PATH}")


if __name__ == "__main__":
    main()