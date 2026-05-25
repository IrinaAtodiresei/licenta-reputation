# utils.py
import os
import base64
import pandas as pd
import streamlit as st

DATA_DIR = "data"
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
    """
    Încarcă datele REALE din PostgreSQL pentru tab-ul Proof of Source.
    NU folosește mock, NU folosește CSV.
    """
    sql = """
        SELECT
            m.mention_id,
            c.name AS company,
            m.subreddit,
            m.author,
            m.content AS text,
            m.published_at AS created_utc,
            m.comment_id,
            m.post_id,
            m.reddit_url
        FROM reputation.mention m
        LEFT JOIN reputation.companies c
            ON m.company_id = c.company_id
        ORDER BY m.published_at DESC
    """

    return safe_read_df(sql)




def app_header():
    """Header global (în locul header-ului HTML repetat)."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    logo_b64 = base64.b64encode(open(logo_path, "rb").read()).decode()
    logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:52px; height:52px; border-radius:12px; object-fit:contain;" />'

    st.markdown(
        f"""
    <div style="display:flex; align-items:center; gap:16px; background:white;
                padding:24px 32px; box-shadow:0 2px 6px rgba(0,0,0,0.06); 
                border-radius:12px; margin-bottom:24px;">
        <div>{logo_img}</div>
        <div>
            <h1 style="margin:0; color:#FF8A00;">Reputation & Sentiment Dashboard</h1>
            <p style="margin:4px 0 0; color:#FFC300;">Analiză reputațională pentru companii tehnologice (Apple · Google · Samsung)</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def app_footer():
    """Footer global (în locul footer-ului HTML repetat)."""
    st.markdown(
        """
    <div style='text-align:center; margin-top:40px; padding:20px; 
                background:#fff; border-top:2px solid #FF8A00; color:#555;'>
        Reputation Dashboard — Licență 2026 — Irina Atodiresei<br>
        Built with Streamlit · Machine Learning · LLM · Groq · LangChain
    </div>
    """,
        unsafe_allow_html=True,
    )
