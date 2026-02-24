import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import psycopg


load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env")

st.set_page_config(page_title="Reputation Dashboard", layout="wide")


def get_conn():
    # psycopg3 connection
    return psycopg.connect(PG_DSN)


def read_df(sql: str, params=None) -> pd.DataFrame:
    # pandas poate folosi conexiunea DB-API; psycopg3 e ok
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


METHODS = {
    "Logistic Regression (Sent140)": "lr_sent140_tfidf",
    "VADER": "vader",
}

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

method_name = st.sidebar.radio("Method", list(METHODS.keys()), index=0)
method = METHODS[method_name]

company = st.sidebar.selectbox("Company", ["All", "Apple", "Samsung", "Google"], index=0)
only_disagreements = st.sidebar.checkbox("Show only disagreements", value=True)
limit_rows = st.sidebar.slider("Rows in disagreements table", 20, 500, 100, 20)
refresh = st.sidebar.button("🔄 Refresh data")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: dacă nu vezi date, verifică view-urile în schema reputation.")

# -----------------------------
# Load data (cache in session)
# -----------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if refresh or not st.session_state.loaded:
    try:
        summary = read_df("""
            SELECT *
            FROM reputation.v_company_sentiment_summary
            WHERE method = %s
        """, params=(method,))
    except Exception:
        st.error("Nu găsesc view-ul reputation.v_company_sentiment_summary. Creează-l în DB (schema reputation).")
        st.stop()

    disagree_company = read_df("""
        SELECT *
        FROM reputation.v_company_method_disagreement
    """)

    disagreements = read_df("""
        SELECT *
        FROM reputation.v_sentiment_disagreements
    """)

    st.session_state.summary = summary
    st.session_state.disagree_company = disagree_company
    st.session_state.disagreements = disagreements
    st.session_state.loaded = True

summary = st.session_state.summary.copy()
disagree_company = st.session_state.disagree_company.copy()
disagreements = st.session_state.disagreements.copy()

# filter company
if company != "All":
    summary = summary[summary["company_name"] == company]
    disagree_company = disagree_company[disagree_company["company_name"] == company]
    disagreements = disagreements[disagreements["company_name"] == company]

# -----------------------------
# Header
# -----------------------------
st.title("📊 Reputation & Sentiment Dashboard")
st.subheader(f"Method: {method_name}")

# -----------------------------
# KPI boxes
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

total_mentions = int(summary["mentions"].sum()) if not summary.empty else 0
avg_score = float(summary["avg_score"].mean()) if not summary.empty else 0.0

# in view: "pct_negative" (din pozele tale)
pct_neg = float(summary["pct_negative"].mean()) if not summary.empty else 0.0
pct_diff = float(disagree_company["pct_different"].mean()) if not disagree_company.empty else 0.0

col1.metric("Total mentions (filtered)", f"{total_mentions}")
col2.metric("Avg score", f"{avg_score:.4f}")
col3.metric("% Negative (avg)", f"{pct_neg:.2f}%")
col4.metric("% Disagreement LR vs VADER", f"{pct_diff:.2f}%")

st.markdown("---")

# -----------------------------
# Summary table + download button
# -----------------------------
st.subheader("Company summary (selected method)")

cols_order = ["company_id", "company_name", "method", "mentions", "avg_score",
             "positives", "negatives", "neutrals", "pct_negative"]

# dacă view-ul tău are exact coloanele astea, perfect
summary = summary[cols_order].sort_values("mentions", ascending=False)
st.dataframe(summary, use_container_width=True)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download summary CSV",
    data=csv_bytes,
    file_name=f"summary_{method}.csv",
    mime="text/csv",
)

# -----------------------------
# Charts
# -----------------------------
st.subheader("Charts")
c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Mentions per company")
    if not summary.empty:
        st.bar_chart(summary.set_index("company_name")[["mentions"]])

with c2:
    st.caption("% Negative per company")
    if not summary.empty:
        st.bar_chart(summary.set_index("company_name")[["pct_negative"]])

with c3:
    st.caption("Avg score per company")
    if not summary.empty:
        st.bar_chart(summary.set_index("company_name")[["avg_score"]])

st.markdown("---")

# -----------------------------
# Disagreement summary
# -----------------------------
st.subheader("LR vs VADER disagreement (by company)")
disagree_company = disagree_company.sort_values("pct_different", ascending=False)
st.dataframe(disagree_company, use_container_width=True)

# -----------------------------
# Disagreements list
# -----------------------------
st.subheader("Posts where LR and VADER disagree")

keep_cols = [
    "mention_id", "company_name", "title", "author", "published_at",
    "lr_label", "lr_score", "vader_label", "vader_score"
]

if not disagreements.empty:
    disagreements = disagreements[keep_cols].sort_values("published_at", ascending=False).head(limit_rows)

st.dataframe(disagreements, use_container_width=True)
st.caption("Tip: folosește filtrele din stânga + butonul Refresh.")