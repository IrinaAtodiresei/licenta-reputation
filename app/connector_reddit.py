import os
import json
import time
import psycopg2
import requests
import joblib
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------
# CONFIG din .env
# ------------------------------------------------------------
load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env")

# KEYWORDS din .env:
# KEYWORDS=Apple OR iPhone OR iOS OR MacBook OR Samsung OR Galaxy OR OneUI OR Google OR Pixel OR Android
KEYWORDS_RAW = os.getenv(
    "KEYWORDS",
    'Apple OR iPhone OR iOS OR MacBook OR Samsung OR Galaxy OR OneUI OR Google OR Pixel OR Android'
)

KEYWORDS = [
    k.strip().strip('"').strip("'").lower()
    for k in KEYWORDS_RAW.split("OR")
    if k.strip()
]

# SUBREDDITS din .env:
# SUBREDDITS=technology,apple,samsung,google,gadgets,technews,android
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
# COMPANIES (din DB-ul tău)
# Apple=1, Samsung=2, Google=3
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
    """
    Returnează company_id pentru prima companie detectată în text.
    Dacă nu găsește, returnează None.
    """
    t = text.lower()

    hits = []
    for comp, keys in COMPANY_MAP.items():
        if any(k in t for k in keys):
            hits.append(comp)

    if not hits:
        return None

    # simplu: prima potrivire
    return COMPANY_ID[hits[0]]


# ------------------------------------------------------------
# LOAD ML MODEL (LogReg + TF-IDF)
# proiect:
# LICENTA/
#   app/connector_reddit.py
#   models/sentiment_lr_sent140.joblib
#   models/tfidf_sent140.joblib
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

# VADER analyzer
analyzer = SentimentIntensityAnalyzer()

METHOD_LR = "lr_sent140_tfidf"
METHOD_VADER = "vader"

# ------------------------------------------------------------
# DB CONNECT
# ------------------------------------------------------------
conn = psycopg2.connect(PG_DSN)
conn.autocommit = True
cur = conn.cursor()

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
            combined = (title + " " + content).strip().lower()

            if not combined:
                continue

            # filtrare după keywords
            if not any(k in combined for k in KEYWORDS):
                continue

            author = post.get("author")
            external_id = post.get("id")
            created_utc = post.get("created_utc", 0)
            published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

            # detect company_id
            company_id = detect_company_id(combined)

            # 1) INSERT mention + RETURNING mention_id (sau SELECT dacă exista)
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

                # dacă deja exista și company_id era NULL, îl completăm acum (doar dacă avem)
                if existing_company_id is None and company_id is not None:
                    cur.execute("""
                        UPDATE reputation.mention
                        SET company_id=%s
                        WHERE mention_id=%s;
                    """, (company_id, mention_id))

            if company_id is not None:
                mentions_labeled_company += 1

            # 2) PREDICT sentiment (LR)
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
            """, (
                mention_id,
                METHOD_LR,
                lr_label,
                proba_pos
            ))
            if cur.rowcount == 1:
                sent_lr_rows_inserted += 1

            # 3) PREDICT sentiment (VADER)
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
            """, (
                mention_id,
                METHOD_VADER,
                vader_label,
                vader_compound
            ))
            if cur.rowcount == 1:
                sent_vader_rows_inserted += 1

        # mic sleep să eviți rate limiting
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Eroare la {sub}: {e}")

print(f"✅ Mențiuni NOI salvate în DB: {mentions_new}")
print(f"✅ Sentiment LR rows inserate: {sent_lr_rows_inserted}")
print(f"✅ Sentiment VADER rows inserate: {sent_vader_rows_inserted}")
print(f"🏷️ Mențiuni cu company_id detectat (în acest run): {mentions_labeled_company}")

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
print("🔒 Conexiunea la baza de date a fost închisă.")