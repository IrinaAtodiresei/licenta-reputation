import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
OUT_DIR = BASE_DIR / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=["target", "ids", "date", "flag", "user", "text"]
)

initial_rows = len(df)

# păstrăm doar clasele negative și pozitive
df = df[df["target"].isin([0, 4])].copy()

# transformăm în 0 / 1
df["label"] = df["target"].map({0: 0, 4: 1})

null_texts = df["text"].isna().sum()
null_labels = df["label"].isna().sum()

df = df.dropna(subset=["text", "label"])
after_dropna = len(df)

duplicates = df.duplicated(subset=["text", "label"]).sum()
df = df.drop_duplicates(subset=["text", "label"])
after_dedup = len(df)

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

df["text_length"] = df["text"].str.len()

report = {
    "initial_rows": int(initial_rows),
    "null_texts": int(null_texts),
    "null_labels": int(null_labels),
    "after_dropna": int(after_dropna),
    "duplicates_removed": int(duplicates),
    "final_rows": int(len(df)),
    "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
    "avg_text_length": float(df["text_length"].mean()),
    "median_text_length": float(df["text_length"].median()),
    "min_text_length": int(df["text_length"].min()),
    "max_text_length": int(df["text_length"].max())
}

with open(OUT_DIR / "data_cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Raportul de curățare a fost salvat.")