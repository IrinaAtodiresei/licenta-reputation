import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "training.1600000.processed.noemoticon.csv"

# Sentiment140 columns: target, id, date, flag, user, text
COLS = ["target", "id", "date", "flag", "user", "text"]


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)      # remove URLs
    s = re.sub(r"@\w+", " ", s)                 # remove @mentions
    s = re.sub(r"#", " ", s)                    # keep hashtag text, remove #
    s = re.sub(r"[^a-z\s']", " ", s)            # keep letters + spaces + '
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    print("Loading CSV... (this can take a bit)")
    df = pd.read_csv(
        DATA_PATH,
        encoding="ISO-8859-1",   # Sentiment140 needs latin/ISO encoding often
        header=None,
        names=COLS
    )

    # Keep only negative (0) and positive (4)
    df = df[df["target"].isin([0, 4])].copy()

    # Map: 0 -> 0 (neg), 4 -> 1 (pos)
    df["label"] = (df["target"] == 4).astype(int)

    # Clean text
    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    X = df["text_clean"].values
    y = df["label"].values

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Vectorizing TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, pred)
    print("Accuracy:", round(acc, 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred, digits=4))

    print("Saving model + vectorizer...")
    joblib.dump(model, "../models/sentiment_lr_sent140.joblib")
    joblib.dump(vectorizer, "../models/tfidf_sent140.joblib")

    print("DONE ✅ Files saved:")
    print("- sentiment_lr_sent140.joblib")
    print("- tfidf_sent140.joblib")


if __name__ == "__main__":
    main()