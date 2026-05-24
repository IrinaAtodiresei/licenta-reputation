import os
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "app" / ".env")

PG_DSN = os.getenv("PG_DSN")
METHOD_LR = "lr_3class_balanced"

MODEL_PATH = BASE_DIR / "models" / "lr_3class_balanced.joblib"

if not PG_DSN:
    raise RuntimeError("PG_DSN lipseste din .env")

model = joblib.load(MODEL_PATH)


def main():
    with psycopg.connect(PG_DSN) as conn:
        df = pd.read_sql_query("""
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
        """, conn, params=(METHOD_LR,))

        print(f"Mentiuni fara LR nou: {len(df)}")

        if df.empty:
            return

        df["text"] = (df["title"] + " " + df["content"]).str.strip()
        df = df[df["text"] != ""]

        inserted = 0

        with conn.cursor() as cur:
            for _, row in df.iterrows():
                text = row["text"]

                label = model.predict([text])[0]
                proba = model.predict_proba([text])[0]
                score = float(proba.max())

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
                    METHOD_LR,
                    label,
                    score
                ))

                inserted += cur.rowcount

                if inserted % 500 == 0:
                    conn.commit()
                    print(f"Inserate pana acum: {inserted}")

            conn.commit()

        print(f"GATA. LR nou inserat: {inserted}")


if __name__ == "__main__":
    main()