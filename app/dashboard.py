import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg


load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    st.error("PG_DSN lipsește. Setează-l în Streamlit Cloud → Settings → Secrets.")
    st.stop()

st.set_page_config(page_title="Reputation Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"
CM_PATH = EVAL_DIR / "confusion_matrix.csv"
METRICS_PATH = EVAL_DIR / "metrics.json"


def get_conn():
    return psycopg.connect(PG_DSN)


def read_df(sql: str, params=None) -> pd.DataFrame:
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

# -----------------------------
# Filter company
# -----------------------------
if company != "All":
    summary = summary[summary["company_name"] == company]
    disagree_company = disagree_company[disagree_company["company_name"] == company]
    disagreements = disagreements[disagreements["company_name"] == company]

# dacă user-ul debifează disagreements
if not only_disagreements:
    disagreements = disagreements.copy()

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

cols_order = [
    "company_id", "company_name", "method", "mentions", "avg_score",
    "positives", "negatives", "neutrals", "pct_negative"
]

if not summary.empty:
    summary = summary[cols_order].sort_values("mentions", ascending=False)
    st.dataframe(summary, use_container_width=True)

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download summary CSV",
        data=csv_bytes,
        file_name=f"summary_{method}.csv",
        mime="text/csv",
    )
else:
    st.info("Nu există date pentru filtrul selectat.")

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
if not disagree_company.empty:
    disagree_company = disagree_company.sort_values("pct_different", ascending=False)
    st.dataframe(disagree_company, use_container_width=True)
else:
    st.info("Nu există date despre disagreement pentru filtrul selectat.")

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
else:
    st.info("Nu există postări în care LR și VADER să difere pentru filtrul selectat.")

st.caption("Tip: folosește filtrele din stânga + butonul Refresh.")

# -----------------------------
# Model evaluation
# -----------------------------
st.markdown("---")
st.subheader("Model evaluation")

if method == "lr_sent140_tfidf":

    if CM_PATH.exists() and METRICS_PATH.exists():

        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        cm_df = pd.read_csv(CM_PATH, index_col=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1-score", f"{metrics['f1']:.4f}")

        st.markdown("### Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("### Confusion matrix chart")
        st.bar_chart(cm_df)

        st.markdown("### Detailed counts")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("True Positive", f"{metrics['true_positive']}")
        d2.metric("True Negative", f"{metrics['true_negative']}")
        d3.metric("False Positive", f"{metrics['false_positive']}")
        d4.metric("False Negative", f"{metrics['false_negative']}")

    else:
        st.info("Nu există încă fișierele de evaluare. Rulează scripts/evaluate_model.py.")

else:

    st.info(
        "VADER is a rule-based sentiment analyzer, therefore confusion matrix "
        "evaluation is not applicable. Confusion matrix is shown only for the "
        "Logistic Regression model trained on the Sent140 dataset."
    )