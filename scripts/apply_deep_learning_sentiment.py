import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "app" / ".env")

PG_DSN = os.getenv("PG_DSN")

if not PG_DSN:
    raise RuntimeError("PG_DSN lipseste din .env")


METHOD_DL = "deep_learning_transformer"


def map_label(label: str):
    label = label.lower()

    if "positive" in label:
        return "positive"

    if "negative" in label:
        return "negative"

    if "neutral" in label:
        return "neutral"

    return "neutral"


def main():
    print("Incarc modelul Deep Learning...")

    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512
    )

    print("Ma conectez la baza de date...")

    with psycopg.connect(PG_DSN) as conn:
        query = """
            SELECT
                m.mention_id,
                COALESCE(m.title, '') AS title,
                COALESCE(m.content, '') AS content
            FROM reputation.mention m
            WHERE NOT EXISTS (
                SELECT 1
                FROM reputation.sentiment_result s
                WHERE s.mention_id = m.mention_id
                  AND s.method = %s
            )
            ORDER BY m.mention_id;
        """

        df = pd.read_sql_query(query, conn, params=(METHOD_DL,))

        print(f"Mentiuni fara Deep Learning: {len(df)}")

        if df.empty:
            print("Nu exista mentiuni noi de procesat.")
            return

        df["text"] = (df["title"] + " " + df["content"]).str.strip()
        df = df[df["text"] != ""]

        inserted = 0

        with conn.cursor() as cur:
            for index, row in df.iterrows():
                text = row["text"]

                try:
                    result = classifier(text[:3000])[0]

                    label = map_label(result["label"])
                    score = float(result["score"])

                    cur.execute("""
                        INSERT INTO reputation.sentiment_result(
                            mention_id,
                            method,
                            label,
                            score,
                            created_at
                        )
                        VALUES (%s, %s, %s, %s, now())
                        ON CONFLICT (mention_id, method) DO NOTHING;
                    """, (
                        int(row["mention_id"]),
                        METHOD_DL,
                        label,
                        score
                    ))

                    inserted += cur.rowcount

                    if inserted % 100 == 0:
                        conn.commit()
                        print(f"Procesate/inserate pana acum: {inserted}")

                except Exception as e:
                    print(f"Eroare la mention_id={row['mention_id']}: {e}")

            conn.commit()

        print(f"GATA. Rezultate Deep Learning inserate: {inserted}")


if __name__ == "__main__":
    main()