import os
import re
import streamlit as st
from dotenv import load_dotenv, dotenv_values
import pandas as pd
import psycopg
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Get the absolute path to the .env file (should be in project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

print(f"[DB] Project root: {PROJECT_ROOT}")
print(f"[DB] ENV file path: {ENV_FILE_PATH}")
print(f"[DB] ENV file exists: {os.path.exists(ENV_FILE_PATH)}")

# Încărcăm .env din calea absolută
load_dotenv(ENV_FILE_PATH)

# Încercăm să luăm PG_DSN din ENV
PG_DSN = os.environ.get("PG_DSN")

# Dacă nu există, îl citim DIRECT din fișierul .env
if not PG_DSN:
    env_dict = dotenv_values(ENV_FILE_PATH)
    PG_DSN = env_dict.get("PG_DSN") or env_dict.get("DATABASE_URL")

print(f"[DB] PG_DSN loaded: {PG_DSN[:50] if PG_DSN else 'NOT FOUND'}...")



def sanitize_neon_dsn(dsn: str) -> str:
    """Remove search_path and related options from DSN that Neon pooler doesn't support."""
    if not dsn:
        return dsn

    # Parse the DSN
    parsed = urlparse(dsn)
    query_dict = dict(parse_qsl(parsed.query, keep_blank_values=True))
    
    # Remove 'options' entirely if it contains search_path
    options = query_dict.get("options", "")
    if options and "search_path" in options.lower():
        query_dict.pop("options", None)
    
    # Rebuild the DSN without the problematic options
    sanitized = urlunparse(parsed._replace(query=urlencode(query_dict)))
    
    # Additional check: if the DSN still contains search_path anywhere, try to remove it
    if "search_path" in sanitized.lower():
        # For DSNs with embedded search_path parameter
        sanitized = re.sub(r'[?&]search_path=[^&]*', '', sanitized)
        sanitized = re.sub(r'[?&]options=[^&]*search_path[^&]*', '', sanitized)
    
    return sanitized


def get_conn():
    if not PG_DSN:
        raise RuntimeError("PG_DSN lipsește din .env")

    # Debug: show what DSN we're trying to use
    print(f"[DB] Original DSN: {PG_DSN[:80]}..." if len(PG_DSN) > 80 else f"[DB] Original DSN: {PG_DSN}")
    
    sanitized_dsn = sanitize_neon_dsn(PG_DSN)
    print(f"[DB] Sanitized DSN: {sanitized_dsn[:80]}..." if len(sanitized_dsn) > 80 else f"[DB] Sanitized DSN: {sanitized_dsn}")
    
    conn = psycopg.connect(sanitized_dsn)
    return conn



def read_df(sql: str, params=None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame."""
    print("Executing SQL:", sql)

    try:
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.warning(f"Database query failed: {e}")
        return pd.DataFrame()


def safe_read_df(sql: str, params=None) -> pd.DataFrame:
    """Safely read DataFrame with error handling."""
    try:
        return read_df(sql, params=params)
    except Exception as e:
        st.warning(f"Data could not be loaded: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1)
def load_summary(method: str) -> pd.DataFrame:
    df = safe_read_df(
        """
            SELECT *
            FROM reputation.v_company_sentiment_summary
            WHERE method = %s
        """,
        params=(method,),
    )

    if df is None or df.empty:
        return df

    # 🔥 FIX: curățăm HTML + Markdown AICI
    from chains.reputation_marketing_chain import strip_html, strip_markdown

    for col in ["title", "content"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(strip_html)
                .apply(strip_markdown)
            )

    return df




@st.cache_data(ttl=1)
def load_disagree_company() -> pd.DataFrame:
    """Load company-level disagreement statistics between methods."""
    return safe_read_df(
        """
            SELECT *
            FROM reputation.v_company_method_disagreement
        """
    )


@st.cache_data(ttl=1)
def load_disagreements() -> pd.DataFrame:
    """Load individual disagreement records between Logistic Regression and VADER."""
    return safe_read_df(
        """
            SELECT *
            FROM reputation.v_sentiment_disagreements
        """
    )


def load_disagree_company_lr_dl() -> pd.DataFrame:
    """Load LR vs Deep Learning Transformer disagreement at company level."""
    return safe_read_df(
        """
            SELECT *
            FROM reputation.v_company_lr_dl_disagreement
        """
    )


def load_disagreements_lr_dl() -> pd.DataFrame:
    """Load individual LR vs Deep Learning Transformer disagreement records."""
    return safe_read_df(
        """
            SELECT *
            FROM reputation.v_lr_dl_sentiment_disagreements
        """
    )



@st.cache_data(ttl=1)
def load_negative_examples(
    method: str,
    company: str = "All",
    limit: int = 30,
    order_direction: str = "DESC"
) -> pd.DataFrame:
    """Load negative sentiment examples for visualization (cleaned of HTML/Markdown)."""

    if company == "All":
        df = safe_read_df(f"""
            SELECT
                m.mention_id,
                c.name AS company_name,
                m.title,
                m.content,
                m.author,
                m.published_at,
                s.label,
                s.score
            FROM reputation.mention m
            JOIN reputation.companies c ON m.company_id = c.company_id
            JOIN reputation.sentiment_result s ON m.mention_id = s.mention_id
            WHERE s.method = %s
              AND s.label = 'negative'
            ORDER BY s.score {order_direction}
            LIMIT {limit};
        """, params=(method,))
    else:
        df = safe_read_df(f"""
            SELECT
                m.mention_id,
                c.name AS company_name,
                m.title,
                m.content,
                m.author,
                m.published_at,
                s.label,
                s.score
            FROM reputation.mention m
            JOIN reputation.companies c ON m.company_id = c.company_id
            JOIN reputation.sentiment_result s ON m.mention_id = s.mention_id
            WHERE s.method = %s
              AND s.label = 'negative'
              AND c.name = %s
            ORDER BY s.score {order_direction}
            LIMIT {limit};
        """, params=(method, company))

    # Dacă nu avem date, returnăm direct
    if df is None or df.empty:
        return df

    # 🔥 Anti‑HTML total — curățăm TOT înainte să ajungă la LLM
    from chains.reputation_marketing_chain import strip_html, strip_markdown

    for col in ["title", "content"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(strip_html)
                .apply(strip_markdown)
            )

    return df



def load_all_examples(method: str, company: str = "All", limit: int = 40, order_direction: str = "DESC") -> pd.DataFrame:
    """Load all sentiment examples for analysis."""
    if company == "All":
        return safe_read_df(f"""
            SELECT
                m.mention_id,
                c.name AS company_name,
                m.title,
                m.content,
                m.author,
                m.published_at,
                s.label,
                s.score
            FROM reputation.mention m
            JOIN reputation.companies c ON m.company_id = c.company_id
            JOIN reputation.sentiment_result s ON m.mention_id = s.mention_id
            WHERE s.method = %s
            ORDER BY m.published_at {order_direction}
            LIMIT {limit};
        """, params=(method,))
    else:
        return safe_read_df(f"""
            SELECT
                m.mention_id,
                c.name AS company_name,
                m.title,
                m.content,
                m.author,
                m.published_at,
                s.label,
                s.score
            FROM reputation.mention m
            JOIN reputation.companies c ON m.company_id = c.company_id
            JOIN reputation.sentiment_result s ON m.mention_id = s.mention_id
            WHERE s.method = %s
              AND c.name = %s
            ORDER BY m.published_at {order_direction}
            LIMIT {limit};
        """, params=(method, company))
def get_manual_cm_for_method(path, method):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # eliminăm coloana 'method' dacă există
    if "method" in df.columns:
        df = df.drop(columns=["method"])

    # eliminăm prima coloană textuală (numele rândurilor)
    df = df.drop(df.columns[0], axis=1)

    # convertim numeric (forțat)
    df = df.apply(pd.to_numeric, errors="coerce")

    # blocurile sunt pe rânduri, câte 3 rânduri per metodă
    block_map = {
        "lr_3class_balanced": (0, 3),
        "vader": (3, 6),
        "deep_learning_transformer": (6, 9),
    }

    start, end = block_map.get(method, (0, 3))

    df_block = df.iloc[start:end, :3]

    df_block.index = ["negative", "neutral", "positive"]
    df_block.columns = ["negative", "neutral", "positive"]

    return df_block



