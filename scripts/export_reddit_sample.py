import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / "app" / ".env")

OUTPUT_PATH = BASE_DIR / "reddit_sample.csv"

PG_DSN = os.getenv("PG_DSN")

if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env / secrets")

def main():
    conn = psycopg.connect(PG_DSN)

    query = """
        SELECT
            m.mention_id,
            m.title,
            m.content,
            m.author,
            m.published_at
        FROM reputation.mention m
        WHERE m.title IS NOT NULL
        ORDER BY m.published_at DESC
        LIMIT 150;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df["text"] = (df["title"] + " " + df["content"]).str.strip()
    df = df[df["text"] != ""]

    df = df[["mention_id", "author", "published_at", "text"]]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()