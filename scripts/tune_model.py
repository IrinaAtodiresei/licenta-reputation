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
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

X_TRAIN_PATH = EVAL_DIR / "X_train.pkl"
X_TEST_PATH = EVAL_DIR / "X_test.pkl"
Y_TRAIN_PATH = EVAL_DIR / "y_train.pkl"
Y_TEST_PATH = EVAL_DIR / "y_test.pkl"

BEST_PARAMS_PATH = EVAL_DIR / "best_params.json"
CV_RESULTS_PATH = EVAL_DIR / "cv_results.csv"
METRICS_TUNED_PATH = EVAL_DIR / "metrics_tuned.json"
CM_TUNED_PATH = EVAL_DIR / "confusion_matrix_tuned.csv"
TUNED_MODEL_PATH = MODELS_DIR / "lr_sent140_tuned.joblib"

MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)


def load_train_test_data():
    missing = [
        str(path.name)
        for path in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Lipsesc fisierele necesare pentru tuning: {', '.join(missing)}. "
            f"Ruleaza mai intai scriptul care salveaza split-ul train/test."
        )

    X_train = pd.read_pickle(X_TRAIN_PATH)
    X_test = pd.read_pickle(X_TEST_PATH)
    y_train = pd.read_pickle(Y_TRAIN_PATH)
    y_test = pd.read_pickle(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_param_grid():
    return {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
    }


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    print("Incărcare train/test split...")
    X_train, X_test, y_train, y_test = load_train_test_data()

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    pipeline = build_pipeline()
    param_grid = build_param_grid()

    print("Pornesc GridSearchCV...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    print("Tuning finalizat."
          "")
    print("Best params:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)

    best_model = grid.best_estimator_

    print("Evaluez modelul tuned pe test set...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tuned_metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "best_cv_f1": float(grid.best_score_),
    }

    save_json(BEST_PARAMS_PATH, grid.best_params_)
    save_json(METRICS_TUNED_PATH, tuned_metrics)

    cv_results = pd.DataFrame(grid.cv_results_)

    cols_to_keep = [
        "params",
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    cv_results = cv_results[cols_to_keep].sort_values("rank_test_score", ascending=True)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    cm_df.to_csv(CM_TUNED_PATH)

    joblib.dump(best_model, TUNED_MODEL_PATH)

    print(f"Salvat: {BEST_PARAMS_PATH}")
    print(f"Salvat: {CV_RESULTS_PATH}")
    print(f"Salvat: {METRICS_TUNED_PATH}")
    print(f"Salvat: {CM_TUNED_PATH}")
    print(f"Salvat: {TUNED_MODEL_PATH}")

    print("Metrici finale pe test set:")
    for key, value in tuned_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()