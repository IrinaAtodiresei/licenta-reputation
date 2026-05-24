from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
EVAL_DIR = BASE_DIR / "evaluation"

EVAL_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"]
    )

    df = df[["target", "text"]].copy()
    df = df[df["target"].isin([0, 4])]
    df["target"] = df["target"].map({0: 0, 4: 1})

    return df


def main():
    df = load_data()

    X = df["text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # SALVARE COMPLETĂ
    X_train.to_pickle(EVAL_DIR / "X_train.pkl")
    X_test.to_pickle(EVAL_DIR / "X_test.pkl")
    y_train.to_pickle(EVAL_DIR / "y_train.pkl")
    y_test.to_pickle(EVAL_DIR / "y_test.pkl")

    print("Train/Test split salvat cu succes.")


if __name__ == "__main__":
    main()