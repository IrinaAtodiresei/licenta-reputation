import os
import json
import time
import requests
import joblib
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import psycopg


print("SCRIPT PORNIT - se incarca librariile...", flush=True)

load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env / secrets")


HEADERS = {
    "User-Agent": os.getenv(
        "REDDIT_USER_AGENT",
        "script:reputation_licenta:v3.1 (by u/vibin_n_thrivin_)"
    )
}


SUBREDDITS = [
    "samsung",
    "galaxy_samsung",
    "SamsungSupport"
]


KEYWORDS = [
    "apple", "iphone", "ios", "ipad", "macbook", "airpods", "mac",
    "samsung", "galaxy", "one ui", "z fold", "z flip", "s23", "s24", "s25",
    "google", "pixel", "android", "chrome", "gmail", "youtube", "gemini"
]


COMPANY_ID = {
    "apple": 1,
    "samsung": 2,
    "google": 3
}


COMPANY_MAP = {
    "apple": [
        "apple", "iphone", "ios", "ipad", "mac", "macbook",
        "airpods", "imessage", "facetime"
    ],
    "samsung": [
        "samsung", "galaxy", "one ui", "z fold", "z flip",
        "s23", "s24", "s25", "note", "galaxy watch"
    ],
    "google": [
        "google", "pixel", "chrome", "gmail", "youtube",
        "android", "google assistant", "gemini"
    ],
}


BASE_ENDPOINTS = [
    "new",
    "hot",
    "top",
    "rising",
]


TOP_TIME_FILTERS = [
    "week",
    "month",
    "year",
    "all"
]


# Pentru test rapid. După ce merge, poți crește la 5.
MAX_PAGES_PER_ENDPOINT = 2
LIMIT_PER_PAGE = 100

COLLECT_COMMENTS = True
MAX_POSTS_FOR_COMMENTS_PER_SUBREDDIT = 5
MAX_COMMENTS_PER_POST = 15

SLEEP_BETWEEN_REQUESTS = 1.5


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_lr_sent140.joblib"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_sent140.joblib"

print("Incarc modelul Logistic Regression si TF-IDF...", flush=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc modelul: {MODEL_PATH}")

if not TFIDF_PATH.exists():
    raise FileNotFoundError(f"Nu găsesc TF-IDF-ul: {TFIDF_PATH}")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

print("Modelul si TF-IDF-ul au fost incarcate.", flush=True)

analyzer = SentimentIntensityAnalyzer()

METHOD_LR = "lr_sent140_tfidf"
METHOD_VADER = "vader"


def detect_company_id(text: str):
    text = text.lower()
    hits = []

    for company, keywords in COMPANY_MAP.items():
        if any(keyword in text for keyword in keywords):
            hits.append(company)

    if not hits:
        return None

    return COMPANY_ID[hits[0]]


def text_is_relevant(text: str):
    text = text.lower()
    return any(keyword in text for keyword in KEYWORDS)


def ensure_tables_exist(cur):
    cur.execute("SELECT to_regnamespace('reputation') IS NOT NULL;")
    ok = cur.fetchone()[0]

    if not ok:
        raise RuntimeError("Schema reputation nu există în DB.")


def insert_source(cur):
    cur.execute("""
        INSERT INTO reputation.source(code, name)
        VALUES ('reddit_public', 'Reddit JSON API')
        ON CONFLICT (code) DO NOTHING;
    """)

    cur.execute("""
        SELECT source_id
        FROM reputation.source
        WHERE code = 'reddit_public';
    """)

    return cur.fetchone()[0]


def create_job_run(cur, source_id):
    query_text = {
        "subreddits": SUBREDDITS,
        "keywords": KEYWORDS,
        "base_endpoints": BASE_ENDPOINTS,
        "max_pages_per_endpoint": MAX_PAGES_PER_ENDPOINT,
        "collect_comments": COLLECT_COMMENTS,
        "max_comments_per_post": MAX_COMMENTS_PER_POST,
        "methods": [METHOD_LR, METHOD_VADER]
    }

    cur.execute("""
        INSERT INTO reputation.job_run(source_id, query)
        VALUES (%s, %s)
        RETURNING job_run_id;
    """, (source_id, json.dumps(query_text)))

    return cur.fetchone()[0]


def classify_and_save_sentiment(cur, mention_id, text):
    X = tfidf.transform([text])

    pred = int(model.predict(X)[0])
    proba_pos = float(model.predict_proba(X)[0][1])
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

    lr_inserted = 1 if cur.rowcount == 1 else 0

    vader_scores = analyzer.polarity_scores(text)
    vader_compound = float(vader_scores["compound"])

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

    vader_inserted = 1 if cur.rowcount == 1 else 0

    return lr_inserted, vader_inserted


def save_mention_and_sentiment(
    cur,
    source_id,
    job_run_id,
    external_id,
    title,
    content,
    author,
    created_utc
):
    title = title or ""
    content = content or ""
    combined_raw = (title + " " + content).strip()

    if not combined_raw:
        return {
            "new_mention": 0,
            "lr_inserted": 0,
            "vader_inserted": 0,
            "company_detected": 0
        }

    combined = combined_raw.lower()

    if not text_is_relevant(combined):
        return {
            "new_mention": 0,
            "lr_inserted": 0,
            "vader_inserted": 0,
            "company_detected": 0
        }

    company_id = detect_company_id(combined)

    if company_id is None:
        return {
            "new_mention": 0,
            "lr_inserted": 0,
            "vader_inserted": 0,
            "company_detected": 0
        }

    try:
        published_at = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
    except Exception:
        published_at = datetime.now(timezone.utc)

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
        VALUES (%s, %s, %s, %s, %s, %s, %s, now(), %s)
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
    new_mention = 0

    if row:
        mention_id = row[0]
        new_mention = 1
    else:
        cur.execute("""
            SELECT mention_id, company_id
            FROM reputation.mention
            WHERE source_id = %s AND external_id = %s;
        """, (source_id, external_id))

        existing = cur.fetchone()

        if not existing:
            return {
                "new_mention": 0,
                "lr_inserted": 0,
                "vader_inserted": 0,
                "company_detected": 0
            }

        mention_id, existing_company_id = existing

        if existing_company_id is None and company_id is not None:
            cur.execute("""
                UPDATE reputation.mention
                SET company_id = %s
                WHERE mention_id = %s;
            """, (company_id, mention_id))

    lr_inserted, vader_inserted = classify_and_save_sentiment(
        cur=cur,
        mention_id=mention_id,
        text=combined
    )

    return {
        "new_mention": new_mention,
        "lr_inserted": lr_inserted,
        "vader_inserted": vader_inserted,
        "company_detected": 1
    }


def make_request(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
    except requests.exceptions.Timeout:
        print("    Timeout la request. Sar peste.", flush=True)
        return None
    except requests.exceptions.RequestException as e:
        print(f"    Eroare request: {e}", flush=True)
        return None

    if response.status_code == 429:
        print("    Rate limit atins. Aștept 60 secunde...", flush=True)
        time.sleep(60)
        return None

    if response.status_code in [403, 404]:
        print(f"    Endpoint indisponibil: {url}", flush=True)
        return None

    response.raise_for_status()
    return response.json()


def fetch_listing_posts(subreddit, endpoint_type, time_filter=None):
    posts = []
    after = None

    for page in range(1, MAX_PAGES_PER_ENDPOINT + 1):
        if endpoint_type == "top":
            url = (
                f"https://www.reddit.com/r/{subreddit}/top.json"
                f"?limit={LIMIT_PER_PAGE}&t={time_filter}"
            )
        else:
            url = (
                f"https://www.reddit.com/r/{subreddit}/{endpoint_type}.json"
                f"?limit={LIMIT_PER_PAGE}"
            )

        if after:
            url += f"&after={after}"

        print(f"    Page {page}: {url}", flush=True)

        data = make_request(url)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        after = data.get("data", {}).get("after")

        for child in children:
            post = child.get("data", {})
            if post:
                posts.append(post)

        if not after:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return posts


def fetch_search_posts(subreddit, keyword):
    posts = []
    after = None
    keyword_encoded = quote(keyword)

    for page in range(1, MAX_PAGES_PER_ENDPOINT + 1):
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={keyword_encoded}&restrict_sr=1&sort=new&limit={LIMIT_PER_PAGE}"
        )

        if after:
            url += f"&after={after}"

        print(f"    Search '{keyword}', page {page}", flush=True)

        data = make_request(url)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        after = data.get("data", {}).get("after")

        for child in children:
            post = child.get("data", {})
            if post:
                posts.append(post)

        if not after:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return posts


def flatten_comments(comment_listing, max_comments):
    comments = []

    def walk(items):
        nonlocal comments

        if len(comments) >= max_comments:
            return

        for item in items:
            if len(comments) >= max_comments:
                return

            if item.get("kind") != "t1":
                continue

            data = item.get("data", {})
            body = data.get("body", "")

            if body and body not in ["[deleted]", "[removed]"]:
                comments.append(data)

            replies = data.get("replies")
            if isinstance(replies, dict):
                reply_items = replies.get("data", {}).get("children", [])
                walk(reply_items)

    walk(comment_listing)
    return comments


def fetch_comments_for_post(subreddit, post_id):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=100"

    print(f"    Iau comentarii pentru postarea {post_id}", flush=True)

    data = make_request(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    comment_listing = data[1].get("data", {}).get("children", [])
    return flatten_comments(comment_listing, MAX_COMMENTS_PER_POST)


def process_result(result, totals):
    totals["total_new_mentions"] += result["new_mention"]
    totals["total_lr_inserted"] += result["lr_inserted"]
    totals["total_vader_inserted"] += result["vader_inserted"]
    totals["total_company_detected"] += result["company_detected"]


def main():
    print("START SCRIPT - am intrat in main()", flush=True)

    print("Ma conectez la baza de date...", flush=True)
    conn = psycopg.connect(PG_DSN, autocommit=True, connect_timeout=10)
    cur = conn.cursor()
    print("Conexiune DB reusita.", flush=True)

    ensure_tables_exist(cur)
    print("Schema reputation exista.", flush=True)

    source_id = insert_source(cur)
    print(f"source_id = {source_id}", flush=True)

    job_run_id = create_job_run(cur, source_id)
    print(f"job_run_id = {job_run_id}", flush=True)

    totals = {
        "total_new_mentions": 0,
        "total_lr_inserted": 0,
        "total_vader_inserted": 0,
        "total_company_detected": 0,
        "total_posts_seen": 0,
        "total_comments_seen": 0
    }

    print("Incep colectarea extinsa Reddit cu pagination + search + comments...", flush=True)

    for subreddit in SUBREDDITS:
        print("\n==============================", flush=True)
        print(f"Subreddit: r/{subreddit}", flush=True)
        print("==============================", flush=True)

        all_posts = {}

        for endpoint_type in BASE_ENDPOINTS:
            if endpoint_type == "top":
                for time_filter in TOP_TIME_FILTERS:
                    print(f"\n  Endpoint: top, t={time_filter}", flush=True)
                    posts = fetch_listing_posts(subreddit, endpoint_type, time_filter)

                    for post in posts:
                        post_id = post.get("id")
                        if post_id:
                            all_posts[post_id] = post
            else:
                print(f"\n  Endpoint: {endpoint_type}", flush=True)
                posts = fetch_listing_posts(subreddit, endpoint_type)

                for post in posts:
                    post_id = post.get("id")
                    if post_id:
                        all_posts[post_id] = post

        for keyword in KEYWORDS:
            print(f"\n  Search keyword: {keyword}", flush=True)
            posts = fetch_search_posts(subreddit, keyword)

            for post in posts:
                post_id = post.get("id")
                if post_id:
                    all_posts[post_id] = post

        print(f"\n  Postari unice gasite in r/{subreddit}: {len(all_posts)}", flush=True)

        comment_posts_processed = 0

        for post_id, post in all_posts.items():
            totals["total_posts_seen"] += 1

            title = post.get("title", "") or ""
            content = post.get("selftext", "") or ""
            author = post.get("author")
            created_utc = post.get("created_utc", 0)

            result = save_mention_and_sentiment(
                cur=cur,
                source_id=source_id,
                job_run_id=job_run_id,
                external_id=f"post_{post_id}",
                title=title,
                content=content,
                author=author,
                created_utc=created_utc
            )
            process_result(result, totals)

            if COLLECT_COMMENTS and comment_posts_processed < MAX_POSTS_FOR_COMMENTS_PER_SUBREDDIT:
                combined_post_text = f"{title} {content}".lower()

                if text_is_relevant(combined_post_text):
                    comments = fetch_comments_for_post(subreddit, post_id)
                    comment_posts_processed += 1
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

                    for comment in comments:
                        totals["total_comments_seen"] += 1

                        comment_id = comment.get("id")
                        comment_body = comment.get("body", "")
                        comment_author = comment.get("author")
                        comment_created_utc = comment.get("created_utc", 0)

                        if not comment_id:
                            continue

                        comment_result = save_mention_and_sentiment(
                            cur=cur,
                            source_id=source_id,
                            job_run_id=job_run_id,
                            external_id=f"comment_{comment_id}",
                            title=title,
                            content=comment_body,
                            author=comment_author,
                            created_utc=comment_created_utc
                        )
                        process_result(comment_result, totals)

        print(f"\n  Total nou pana acum: {totals['total_new_mentions']}", flush=True)

    stats = {
        **totals,
        "subreddits": SUBREDDITS,
        "keywords": KEYWORDS,
        "base_endpoints": BASE_ENDPOINTS,
        "top_time_filters": TOP_TIME_FILTERS,
        "max_pages_per_endpoint": MAX_PAGES_PER_ENDPOINT,
        "collect_comments": COLLECT_COMMENTS,
        "max_posts_for_comments_per_subreddit": MAX_POSTS_FOR_COMMENTS_PER_SUBREDDIT,
        "max_comments_per_post": MAX_COMMENTS_PER_POST,
        "methods": [METHOD_LR, METHOD_VADER]
    }

    cur.execute("""
        UPDATE reputation.job_run
        SET status = 'done',
            finished_at = now(),
            stats = %s
        WHERE job_run_id = %s;
    """, (json.dumps(stats), job_run_id))

    conn.close()

    print("\nGATA.", flush=True)
    print(f"Mentiuni noi salvate: {totals['total_new_mentions']}", flush=True)
    print(f"Sentiment LR inserat: {totals['total_lr_inserted']}", flush=True)
    print(f"Sentiment VADER inserat: {totals['total_vader_inserted']}", flush=True)
    print(f"Mentiuni cu companie detectata: {totals['total_company_detected']}", flush=True)
    print(f"Postari analizate: {totals['total_posts_seen']}", flush=True)
    print(f"Comentarii analizate: {totals['total_comments_seen']}", flush=True)


if __name__ == "__main__":
    main()