import os
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env")

st.set_page_config(page_title="Reputation Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def get_conn():
    return psycopg2.connect(PG_DSN)

def read_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def method_label(m: str) -> str:
    return "Logistic Regression (Sent140)" if m == "lr_sent140_tfidf" else "VADER"

METHODS = {
    "Logistic Regression (Sent140)": "lr_sent140_tfidf",
    "VADER": "vader",
}

# -----------------------------
# Sidebar controls (butoane + filtre)
# -----------------------------
st.sidebar.title("Controls")

method_name = st.sidebar.radio("Method", list(METHODS.keys()), index=0)
method = METHODS[method_name]

company = st.sidebar.selectbox("Company", ["All", "Apple", "Samsung", "Google"], index=0)

only_disagreements = st.sidebar.checkbox("Show only disagreements", value=True)

limit_rows = st.sidebar.slider("Rows in disagreements table", min_value=20, max_value=500, value=100, step=20)

refresh = st.sidebar.button("🔄 Refresh data")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: dacă nu vezi date, verifică view-urile în schema reputation.")

# -----------------------------
# Load data (cache in session)
# -----------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if refresh or not st.session_state.loaded:
    # summary per company/method
    try:
        summary = read_df("""
            SELECT *
            FROM reputation.v_company_sentiment_summary
            WHERE method = %s
        """, params=(method,))
    except Exception as e:
        st.error("Nu găsesc view-ul reputation.v_company_sentiment_summary. Creează-l în pgAdmin (ți l-am dat mai sus).")
        st.stop()

    # disagreements per company
    disagree_company = read_df("""
        SELECT *
        FROM reputation.v_company_method_disagreement
    """)

    # disagreements list
    base_sql = """
        SELECT *
        FROM reputation.v_sentiment_disagreements
    """
    if only_disagreements:
        # view-ul deja e disagreements-only, deci e ok oricum
        pass

    disagreements = read_df(base_sql)

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

# order columns nicely
cols_order = ["company_id", "company_name", "method", "mentions", "avg_score", "positives", "negatives", "neutrals", "pct_negative"]
summary = summary[cols_order].sort_values("mentions", ascending=False)

st.dataframe(summary, use_container_width=True)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download summary CSV",
    data=csv_bytes,
    file_name=f"summary_{method}.csv",
    mime="text/csv"
)

# -----------------------------
# Charts (simple, no extra libs)
# -----------------------------
st.subheader("Charts")

c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Mentions per company")
    if not summary.empty:
        chart_df = summary[["company_name", "mentions"]].set_index("company_name")
        st.bar_chart(chart_df)

with c2:
    st.caption("% Negative per company")
    if not summary.empty:
        chart_df = summary[["company_name", "pct_negative"]].set_index("company_name")
        st.bar_chart(chart_df)

with c3:
    st.caption("Avg score per company")
    if not summary.empty:
        chart_df = summary[["company_name", "avg_score"]].set_index("company_name")
        st.bar_chart(chart_df)

st.markdown("---")

# -----------------------------
# Disagreement summary (LR vs VADER)
# -----------------------------
st.subheader("LR vs VADER disagreement (by company)")

disagree_company = disagree_company.sort_values("pct_different", ascending=False)
st.dataframe(disagree_company, use_container_width=True)

# -----------------------------
# Disagreements list (table)
# -----------------------------
st.subheader("Posts where LR and VADER disagree")

# keep useful columns
keep_cols = [
    "mention_id", "company_name", "title", "author", "published_at",
    "lr_label", "lr_score", "vader_label", "vader_score"
]
if not disagreements.empty:
    disagreements = disagreements[keep_cols].sort_values("published_at", ascending=False).head(limit_rows)

st.dataframe(disagreements, use_container_width=True)

st.caption("Tip: folosește filtrele din stânga + butonul Refresh.")