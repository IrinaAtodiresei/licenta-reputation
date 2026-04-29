import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "app" / ".env")

PG_DSN = os.getenv("PG_DSN")
OUTPUT_PATH = BASE_DIR / "evaluation" / "reddit_manual_validation_sample.csv"

if not PG_DSN:
    raise RuntimeError("PG_DSN lipseste din .env")


def main():
    query = """
        SELECT
            m.mention_id,
            c.name AS company_name,
            COALESCE(m.title, '') AS title,
            COALESCE(m.content, '') AS content,
            m.author,
            m.published_at,
            lr.label AS lr_label,
            lr.score AS lr_score,
            vader.label AS vader_label,
            vader.score AS vader_score,
            dl.label AS dl_label,
            dl.score AS dl_score
        FROM reputation.mention m
        JOIN reputation.companies c ON m.company_id = c.company_id
        LEFT JOIN reputation.sentiment_result lr
            ON m.mention_id = lr.mention_id
           AND lr.method = 'lr_3class_balanced'
        LEFT JOIN reputation.sentiment_result vader
            ON m.mention_id = vader.mention_id
           AND vader.method = 'vader'
        LEFT JOIN reputation.sentiment_result dl
            ON m.mention_id = dl.mention_id
           AND dl.method = 'deep_learning_transformer'
        WHERE lr.label IS NOT NULL
          AND vader.label IS NOT NULL
          AND dl.label IS NOT NULL
        ORDER BY random()
        LIMIT 500;
    """

    with psycopg.connect(PG_DSN) as conn:
        df = pd.read_sql_query(query, conn)

    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()

    df = df[
        [
            "mention_id",
            "company_name",
            "title",
            "content",
            "text",
            "author",
            "published_at",
            "lr_label",
            "lr_score",
            "vader_label",
            "vader_score",
            "dl_label",
            "dl_score",
        ]
    ]

    df["manual_label"] = ""

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Fisier salvat aici: {OUTPUT_PATH}")
    print("Completeaza coloana manual_label cu: positive / neutral / negative")


if __name__ == "__main__":
    main()