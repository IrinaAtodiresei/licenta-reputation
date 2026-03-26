import os
import json
import time
import requests
import joblib
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import psycopg


# ------------------------------------------------------------
# CONFIG din .env
# ------------------------------------------------------------
load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env / secrets")

KEYWORDS_RAW = os.getenv(
    "KEYWORDS",
    'Apple OR iPhone OR iOS OR MacBook OR Samsung OR Galaxy OR OneUI OR Google OR Pixel OR Android'
)

KEYWORDS = [
    k.strip().strip('"').strip("'").lower()
    for k in KEYWORDS_RAW.split("OR")
    if k.strip()
]

SUBREDDITS = [
    s.strip()
    for s in os.getenv("SUBREDDITS", "technology,apple,samsung,google").split(",")
    if s.strip()
]

HEADERS = {
    "User-Agent": os.getenv(
        "REDDIT_USER_AGENT",
        "script:reputation_licenta:v1.0 (by u/vibin_n_thrivin_)"
    )
}

# ------------------------------------------------------------
# COMPANIES (din DB)
# ------------------------------------------------------------
COMPANY_ID = {"apple": 1, "samsung": 2, "google": 3}

COMPANY_MAP = {
    "apple": [
        "apple", "iphone", "ios", "ipad", "mac", "macbook", "airpods", "imessage", "facetime"
    ],
    "samsung": [
        "samsung", "galaxy", "one ui", "z fold", "z flip", "s23", "s24", "note", "galaxy watch"
    ],
    "google": [
        "google", "pixel", "chrome", "gmail", "youtube", "android", "google assistant"
    ],
}


def detect_company_id(text: str):
    t = text.lower()
    hits = []
    for comp, keys in COMPANY_MAP.items():
        if any(k in t for k in keys):
            hits.append(comp)
    if not hits:
        return None
    return COMPANY_ID[hits[0]]


# ------------------------------------------------------------
# LOAD ML MODEL (LogReg + TF-IDF)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # app -> LICENTA
MODEL_PATH = BASE_DIR / "models" / "sentiment_lr_sent140.joblib"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_sent140.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc modelul: {MODEL_PATH}")
if not TFIDF_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc TF-IDF-ul: {TFIDF_PATH}")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

analyzer = SentimentIntensityAnalyzer()

METHOD_LR = "lr_sent140_tfidf"
METHOD_VADER = "vader"


def ensure_tables_exist(cur):

    # minim sanity check: schema reputation
    cur.execute("SELECT to_regnamespace('reputation') IS NOT NULL;")
    ok = cur.fetchone()[0]
    if not ok:
        raise RuntimeError("Schema 'reputation' nu exista în DB. Trebuie restore/migrate intai.")


def main():
    # ------------------------------------------------------------
    # DB CONNECT (psycopg v3)
    # ------------------------------------------------------------
    conn = psycopg.connect(PG_DSN, autocommit=True)
    cur = conn.cursor()

    ensure_tables_exist(cur)

    # ------------------------------------------------------------
    # SOURCE (reputation.source)
    # ------------------------------------------------------------
    cur.execute("""
        INSERT INTO reputation.source(code, name)
        VALUES ('reddit_public', 'Reddit JSON API')
        ON CONFLICT (code) DO NOTHING;
    """)
    cur.execute("SELECT source_id FROM reputation.source WHERE code='reddit_public';")
    source_id = cur.fetchone()[0]

    # ------------------------------------------------------------
    # JOB RUN (reputation.job_run)
    # ------------------------------------------------------------
    cur.execute("""
        INSERT INTO reputation.job_run(source_id, query)
        VALUES (%s, %s)
        RETURNING job_run_id;
    """, (
        source_id,
        f"SUBREDDITS={','.join(SUBREDDITS)}; KEYWORDS={KEYWORDS_RAW}; METHODS={METHOD_LR}+{METHOD_VADER}"
    ))
    job_run_id = cur.fetchone()[0]

    mentions_new = 0
    sent_lr_rows_inserted = 0
    sent_vader_rows_inserted = 0
    mentions_labeled_company = 0

    print("🔍 Colectez postări Reddit (API public /new.json)...")

    # ------------------------------------------------------------
    # FETCH + INSERT mention + sentiment_result
    # ------------------------------------------------------------
    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=50"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()

            children = data.get("data", {}).get("children", [])
            for child in children:
                post = child.get("data", {})
                title = post.get("title", "") or ""
                content = post.get("selftext", "") or ""
                combined_raw = (title + " " + content).strip()

                if not combined_raw:
                    continue

                combined = combined_raw.lower()

                # filtrare keywords
                if not any(k in combined for k in KEYWORDS):
                    continue

                author = post.get("author")
                external_id = post.get("id")
                created_utc = post.get("created_utc", 0)
                published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

                company_id = detect_company_id(combined)


                cur.execute("""
                    INSERT INTO reputation.mention(
                        company_id,
                        source_id,
                        external_id,
                        title,
                        content,
                        author,
                        published_at,
                        collected_at,
                        job_run_id
                    )
                    VALUES (
                        %s,
                        %s, %s, %s, %s, %s, %s, now(), %s
                    )
                    ON CONFLICT (source_id, external_id) DO NOTHING
                    RETURNING mention_id;
                """, (
                    company_id,
                    source_id,
                    external_id,
                    title,
                    content,
                    author,
                    published_at,
                    job_run_id
                ))

                row = cur.fetchone()
                if row:
                    mention_id = row[0]
                    mentions_new += 1
                else:
                    cur.execute("""
                        SELECT mention_id, company_id
                        FROM reputation.mention
                        WHERE source_id=%s AND external_id=%s;
                    """, (source_id, external_id))
                    mention_id, existing_company_id = cur.fetchone()

                    # completează company_id dacă era NULL
                    if existing_company_id is None and company_id is not None:
                        cur.execute("""
                            UPDATE reputation.mention
                            SET company_id=%s
                            WHERE mention_id=%s;
                        """, (company_id, mention_id))

                if company_id is not None:
                    mentions_labeled_company += 1

                # ---- LR Sent140
                X = tfidf.transform([combined])
                pred = int(model.predict(X)[0])                  # 1=pos, 0=neg
                proba_pos = float(model.predict_proba(X)[0][1])  # P(pos)
                lr_label = "positive" if pred == 1 else "negative"

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
                """, (mention_id, METHOD_LR, lr_label, proba_pos))
                if cur.rowcount == 1:
                    sent_lr_rows_inserted += 1

                # ---- VADER
                v = analyzer.polarity_scores(combined)
                vader_compound = float(v["compound"])

                if vader_compound >= 0.05:
                    vader_label = "positive"
                elif vader_compound <= -0.05:
                    vader_label = "negative"
                else:
                    vader_label = "neutral"

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
                """, (mention_id, METHOD_VADER, vader_label, vader_compound))
                if cur.rowcount == 1:
                    sent_vader_rows_inserted += 1

            time.sleep(1)

        except Exception as e:
            print(f" Eroare la {sub}: {e}")

    print(f" Mentiuni NOI salvate in DB: {mentions_new}")
    print(f" Sentiment LR rows inserate: {sent_lr_rows_inserted}")
    print(f" Sentiment VADER rows inserate: {sent_vader_rows_inserted}")
    print(f" Mentiuni cu company_id detectat (in acest run): {mentions_labeled_company}")

    # ------------------------------------------------------------
    # UPDATE job_run
    # ------------------------------------------------------------
    cur.execute("""
        UPDATE reputation.job_run
        SET status='done', finished_at=now(), stats=%s
        WHERE job_run_id=%s;
    """, (json.dumps({
        "mentions_new": mentions_new,
        "sentiment_lr_rows_inserted": sent_lr_rows_inserted,
        "sentiment_vader_rows_inserted": sent_vader_rows_inserted,
        "mentions_labeled_company": mentions_labeled_company,
        "subreddits": SUBREDDITS,
        "keywords": KEYWORDS_RAW,
        "methods": [METHOD_LR, METHOD_VADER]
    }), job_run_id))

    conn.close()
    print(" Conexiunea la baza de date a fost inchisa.")


if __name__ == "__main__":
    main()