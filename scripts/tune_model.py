import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)


def load_sentiment140():
    """
    Încarcă fișierul Sentiment140 și extrage coloana target + text.
    """

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        header=None
    )

    if df.shape[1] >= 6:
        df = df[[0, 5]].copy()
        df.columns = ["target", "text"]
    else:
        raise ValueError(
            "Fișierul Sentiment140 nu are formatul așteptat. "
            "Mă aștept la cel puțin 6 coloane."
        )

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: 0, 4: 1})

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""]

    return df[["text", "label"]]


def main():
    print("Loading dataset...")
    df = load_sentiment140()

    print("Sampling 100000 rows for faster tuning...")
    df = df.sample(n=100000, random_state=42)

    X = df["text"]
    y = df["label"]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(random_state=42))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [5000],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.1, 1, 5],
        "clf__solver": ["liblinear"],
        "clf__max_iter": [1000]
    }

    print("Running GridSearchCV...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\nBest params:")
    print(grid.best_params_)

    print("\nBest CV score:")
    print(grid.best_score_)

    best_model = grid.best_estimator_

    print("\nEvaluating best model on test set...")
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "best_cv_score": float(grid.best_score_)
    }

    best_params = grid.best_params_

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    model_path = MODELS_DIR / "lr_sent140_tfidf_tuned.pkl"
    metrics_path = EVAL_DIR / "metrics_tuned.json"
    best_params_path = EVAL_DIR / "best_params.json"
    cm_path = EVAL_DIR / "confusion_matrix_tuned.csv"

    print("\nSaving tuned model...")
    joblib.dump(best_model, model_path)

    print("Saving metrics...")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Saving best params...")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print("Saving confusion matrix...")
    cm_df.to_csv(cm_path)

    print("\nDone.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Best params saved to: {best_params_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()