import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import psycopg  # ✅ psycopg v3


load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env / secrets")

st.set_page_config(page_title="Reputation Dashboard", layout="wide")


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    # autocommit False e ok pt SELECT-uri
    return psycopg.connect(PG_DSN)

def read_df(sql: str, params=None) -> pd.DataFrame:
    """
    Citește query -> pandas DataFrame (compatibil psycopg v3).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)


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

limit_rows = st.sidebar.slider(
    "Rows in disagreements table",
    min_value=20,
    max_value=500,
    value=100,
    step=20
)

refresh = st.sidebar.button("🔄 Refresh data")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: dacă nu vezi date, verifică view-urile în schema reputation.")


# -----------------------------
# Load data (session cache)
# -----------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

def load_data(selected_method: str):
    # 1) summary per company/method
    summary = read_df("""
        SELECT *
        FROM reputation.v_company_sentiment_summary
        WHERE method = %s
    """, (selected_method,))

    # 2) disagreement per company (LR vs VADER)
    disagree_company = read_df("""
        SELECT *
        FROM reputation.v_company_method_disagreement
    """)

    # 3) disagreements list
    disagreements = read_df("""
        SELECT *
        FROM reputation.v_sentiment_disagreements
    """)

    return summary, disagree_company, disagreements


if refresh or not st.session_state.loaded or st.session_state.get("method_loaded") != method:
    try:
        summary, disagree_company, disagreements = load_data(method)
    except Exception:
        st.error(
            "Nu găsesc unul dintre view-uri în DB.\n\n"
            "Trebuie să existe:\n"
            "- reputation.v_company_sentiment_summary\n"
            "- reputation.v_company_method_disagreement\n"
            "- reputation.v_sentiment_disagreements\n"
        )
        st.stop()

    st.session_state.summary = summary
    st.session_state.disagree_company = disagree_company
    st.session_state.disagreements = disagreements
    st.session_state.loaded = True
    st.session_state.method_loaded = method

summary = st.session_state.summary.copy()
disagree_company = st.session_state.disagree_company.copy()
disagreements = st.session_state.disagreements.copy()


# -----------------------------
# Filters
# -----------------------------
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
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

total_mentions = int(summary["mentions"].sum()) if not summary.empty else 0
avg_score = float(summary["avg_score"].mean()) if not summary.empty else 0.0

# în view-ul tău coloana e pct_negative (din ce ai arătat în tabel)
pct_neg = float(summary["pct_negative"].mean()) if (not summary.empty and "pct_negative" in summary.columns) else 0.0

pct_diff = float(disagree_company["pct_different"].mean()) if (not disagree_company.empty and "pct_different" in disagree_company.columns) else 0.0

col1.metric("Total mentions (filtered)", f"{total_mentions}")
col2.metric("Avg score", f"{avg_score:.4f}")
col3.metric("% Negative (avg)", f"{pct_neg:.2f}%")
col4.metric("% Disagreement LR vs VADER", f"{pct_diff:.2f}%")

st.markdown("---")


# -----------------------------
# Summary table + download
# -----------------------------
st.subheader("Company summary (selected method)")

cols_order = ["company_id", "company_name", "method", "mentions", "avg_score", "positives", "negatives", "neutrals", "pct_negative"]
cols_order = [c for c in cols_order if c in summary.columns]  # safety

summary_view = summary[cols_order].sort_values("mentions", ascending=False) if not summary.empty else summary

st.dataframe(summary_view, use_container_width=True)

csv_bytes = summary_view.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download summary CSV",
    data=csv_bytes,
    file_name=f"summary_{method}.csv",
    mime="text/csv"
)


# -----------------------------
# Charts
# -----------------------------
st.subheader("Charts")

c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Mentions per company")
    if not summary_view.empty:
        chart_df = summary_view[["company_name", "mentions"]].set_index("company_name")
        st.bar_chart(chart_df)

with c2:
    st.caption("% Negative per company")
    if not summary_view.empty and "pct_negative" in summary_view.columns:
        chart_df = summary_view[["company_name", "pct_negative"]].set_index("company_name")
        st.bar_chart(chart_df)

with c3:
    st.caption("Avg score per company")
    if not summary_view.empty:
        chart_df = summary_view[["company_name", "avg_score"]].set_index("company_name")
        st.bar_chart(chart_df)

st.markdown("---")


# -----------------------------
# Disagreement summary
# -----------------------------
st.subheader("LR vs VADER disagreement (by company)")
if not disagree_company.empty:
    disagree_company_view = disagree_company.sort_values("pct_different", ascending=False) if "pct_different" in disagree_company.columns else disagree_company
    st.dataframe(disagree_company_view, use_container_width=True)
else:
    st.info("Nu există încă date pentru disagreement.")


# -----------------------------
# Disagreements list
# -----------------------------
st.subheader("Posts where LR and VADER disagree")

keep_cols = [
    "mention_id", "company_name", "title", "author", "published_at",
    "lr_label", "lr_score", "vader_label", "vader_score"
]
keep_cols = [c for c in keep_cols if c in disagreements.columns]

if not disagreements.empty:
    disagreements_view = disagreements[keep_cols].sort_values("published_at", ascending=False).head(limit_rows)
    if only_disagreements:
        # view-ul tău e deja disagreements-only, deci nu filtrăm suplimentar
        pass
    st.dataframe(disagreements_view, use_container_width=True)
else:
    st.info("Nu există încă rânduri în v_sentiment_disagreements.")

st.caption("Tip: schimbă filtrele din stânga și apasă Refresh.")