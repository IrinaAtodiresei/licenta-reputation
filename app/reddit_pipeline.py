import requests
import pandas as pd
from typing import List, Dict
import streamlit as st

from utils import load_reddit_dataframe, ensure_mock_data

REDDIT_BASE_URL = "https://oauth.reddit.com"  # OAuth base
HEADERS_FALLBACK = {"User-Agent": "ReputationDashboard/0.1 by student"}

# ============================================================
# Detectăm internetul
# ============================================================
def internet_is_available() -> bool:
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except:
        return False

# ============================================================
# DEMO MODE automat dacă pică netul
# ============================================================
DEMO_MODE = not internet_is_available()

# ============================================================
# DEMO MOCK — EXACT 60 comentarii
# ============================================================
def generate_demo_mock(n=60):
    companies = ["Apple", "Google", "Samsung"]
    rows = []
    for i in range(n):
        comp = companies[i % 3]
        rows.append({
            "comment_id": f"demo_{i}",
            "text": f"Demo comment {i} about {comp}",
            "subreddit": comp.lower(),
            "author": f"user_{i}",
            "created_utc": 1710000000 + i,
            "company": comp,
        })
    return pd.DataFrame(rows)

# ============================================================
# LIMITARE — DOAR ÎN DEMO MODE (60 rânduri)
# ============================================================
MAX_ROWS_DEMO = 60

def limit_df(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    """Limitează dataframe la max_rows (sau MAX_ROWS_DEMO în DEMO_MODE)."""
    if df is None:
        return pd.DataFrame()
    
    if DEMO_MODE:
        limit = MAX_ROWS_DEMO
    else:
        limit = max_rows if max_rows is not None else None
    
    if limit and len(df) > limit:
        return df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)

# ============================================================
# OAUTH TOKEN
# ============================================================
def get_reddit_token() -> str:
    """
    Ia token OAuth folosind client_id / client_secret din st.secrets.
    Funcționează atât pe Streamlit Cloud, cât și local (dacă ai secrets).
    """
    client_id = st.secrets["REDDIT_CLIENT_ID"]
    client_secret = st.secrets["REDDIT_CLIENT_SECRET"]
    user_agent = st.secrets["REDDIT_USER_AGENT"]

    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": user_agent}

    resp = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth,
        data=data,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Nu am putut obține token OAuth de la Reddit.")
    return token

# ============================================================
# FETCH REDDIT COMMENTS via OAuth
# ============================================================
def fetch_reddit_comments_oauth(subreddit: str, limit: int = 50) -> List[Dict]:
    """
    Ia comentarii dintr-un subreddit folosind OAuth.
    Endpoint: https://oauth.reddit.com/r/{subreddit}/comments?limit=...
    """
    token = get_reddit_token()

    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": st.secrets["REDDIT_USER_AGENT"],
    }

    url = f"{REDDIT_BASE_URL}/r/{subreddit}/comments?limit={limit}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows: List[Dict] = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        rows.append(
            {
                "comment_id": d.get("id"),
                "text": d.get("body") or d.get("selftext") or "",
                "subreddit": d.get("subreddit", subreddit),
                "author": d.get("author"),
                "created_utc": d.get("created_utc"),
            }
        )
    return rows

# ============================================================
# FALLBACK DATA — LIMITARE DOAR ÎN DEMO MODE
# ============================================================
def fallback_data(max_rows: int = None) -> pd.DataFrame:
    """
    Fallback automat în 3 niveluri:
    1) DB
    2) CSV local
    3) mock local
    În DEMO_MODE, limitare la 60. Altfel, respectă max_rows.
    """

    if DEMO_MODE:
        return generate_demo_mock(MAX_ROWS_DEMO)

    # 1) DB fallback
    try:
        df = load_reddit_dataframe()
        if df is not None and len(df) > 0:
            df = df.rename(columns={"content": "text"})
            df = df[["comment_id", "text", "subreddit", "author", "created_utc"]]
            return limit_df(df, max_rows)
    except Exception as e:
        print("DB fallback error:", e)

    # 2) CSV fallback
    try:
        csv_path = ensure_mock_data()
        csv_limit = max_rows if max_rows else 1000
        df = pd.read_csv(csv_path, nrows=csv_limit)
        return limit_df(df, max_rows)
    except Exception as e:
        print("CSV fallback error:", e)

    # 3) mock local final fallback
    return generate_demo_mock(MAX_ROWS_DEMO)

# ============================================================
# FETCH REDDIT COMMENTS — CU OAUTH + FALLBACK
# ============================================================
def fetch_reddit_comments(subreddits: List[str], limit: int = 50, max_rows: int = None) -> pd.DataFrame:
    """
    În mod normal: folosește OAuth.
    Dacă ceva crapă (token, request, etc.) → fallback_data.
    În DEMO_MODE → mock direct.
    """

    if DEMO_MODE:
        return generate_demo_mock(MAX_ROWS_DEMO)

    rows: List[Dict] = []

    try:
        for sub in subreddits:
            rows.extend(fetch_reddit_comments_oauth(sub, limit))
    except Exception as e:
        print("OAuth fetch error:", e)
        return fallback_data(max_rows)

    if not rows:
        return fallback_data(max_rows)

    return limit_df(pd.DataFrame(rows), max_rows)

# ============================================================
# EXTRACT FIELDS
# ============================================================
def extract_fields(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    cols = ["comment_id", "text", "subreddit", "author", "created_utc"]
    existing = [c for c in cols if c in df.columns]
    return limit_df(df[existing].copy(), max_rows)


#MAP TO COMPANY

def map_to_company(df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
    def _map(row):
        sub = str(row.get("subreddit", "")).lower()
        text = str(row.get("text", "")).lower()

        if "apple" in sub or "iphone" in text or "mac" in text:
            return "Apple"
        if "google" in sub or "android" in text or "pixel" in text:
            return "Google"
        if "samsung" in sub or "galaxy" in text:
            return "Samsung"
        return "Other"

    df = df.copy()
    df["company"] = df.apply(_map, axis=1)
    df = df[df["company"].isin(["Apple", "Google", "Samsung"])]
    return limit_df(df, max_rows)

#PIPELINE COMPLET

def run_reddit_pipeline_live(limit_per_sub: int = 20) -> pd.DataFrame:
    subreddits = ["apple", "google", "samsung"]

    max_rows_limit = MAX_ROWS_DEMO if DEMO_MODE else (limit_per_sub * 3)

    try:
        raw = fetch_reddit_comments(subreddits, limit_per_sub, max_rows_limit)
    except Exception as e:
        print("Pipeline fetch error:", e)
        raw = fallback_data(max_rows_limit)

    extracted = extract_fields(raw, max_rows_limit)
    mapped = map_to_company(extracted, max_rows_limit)

    if DEMO_MODE:
        mapped["dl_label"] = "disabled"

    return limit_df(mapped, max_rows_limit)
