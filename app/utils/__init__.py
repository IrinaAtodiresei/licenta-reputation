import os
import pandas as pd
import streamlit as st
from .db import safe_read_df


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)


def load_html(path: str):
    """Încarcă un fișier HTML din assets/ și îl afișează în Streamlit."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        st.markdown(html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Fișierul HTML nu a fost găsit: {path}")


def get_mock_reddit_data_path() -> str:
    return os.path.join(DATA_DIR, "mock_reddit_comments.csv")


def ensure_mock_data() -> str:
    """Generează un CSV cu date mock dacă nu există."""
    path = get_mock_reddit_data_path()
    if os.path.exists(path):
        return path

    data = [
        {
            "comment_id": "c1",
            "text": "I love my new iPhone, Apple did a great job!",
            "subreddit": "apple",
            "author": "user_apple_fan",
            "created_utc": "2026-04-29 10:00:00",
        },
        {
            "comment_id": "c2",
            "text": "Google services are reliable but sometimes confusing.",
            "subreddit": "google",
            "author": "user_google",
            "created_utc": "2026-04-29 10:05:00",
        },
        {
            "comment_id": "c3",
            "text": "Samsung phones are good but overpriced.",
            "subreddit": "samsung",
            "author": "user_samsung",
            "created_utc": "2026-04-29 10:10:00",
        },
        {
            "comment_id": "c4",
            "text": "I hate the latest update from Google, too many bugs.",
            "subreddit": "google",
            "author": "user_bug",
            "created_utc": "2026-04-29 10:15:00",
        },
        {
            "comment_id": "c5",
            "text": "Apple ecosystem is amazing but the prices are insane.",
            "subreddit": "apple",
            "author": "user_mixed",
            "created_utc": "2026-04-29 10:20:00",
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path


def load_reddit_dataframe() -> pd.DataFrame:
    sql = """
        SELECT
            m.mention_id,
            c.name AS company,
            m.external_id,
            CASE
                WHEN m.external_id LIKE 'post_%%' THEN 'post'
                WHEN m.external_id LIKE 'comment_%%' THEN 'comment'
                ELSE 'unknown'
            END AS reddit_object_type,
            REPLACE(REPLACE(m.external_id, 'post_', ''), 'comment_', '') AS reddit_object_id,
            m.author,
            m.published_at AS created_utc,
            m.collected_at,
            m.title,
            m.content AS text,
            m.subreddit,
            m.post_id,
            m.comment_id,
            m.reddit_url
        FROM reputation.mention m
        LEFT JOIN reputation.companies c ON m.company_id = c.company_id
        ORDER BY m.published_at DESC
    """
    return safe_read_df(sql)
__all__ = ["load_reddit_dataframe"]


