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

METRICS_PATH = EVAL_DIR / "metrics.json"
CM_PATH = EVAL_DIR / "confusion_matrix.csv"


def get_conn():
    return psycopg.connect(PG_DSN)


def read_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def generate_interpretation(company, method_name, total_mentions, avg_score, pct_negative, pct_disagreement):
    if company == "All":
        company_text = "for the selected companies"
    else:
        company_text = f"for {company}"

    if method_name == "Logistic Regression (Sent140)":
        method_text = "using the Logistic Regression model trained on Sent140"
    else:
        method_text = "using the VADER rule-based sentiment analyzer"

    if avg_score >= 0.60:
        sentiment_text = (
            "The overall sentiment appears predominantly positive, which suggests a generally favorable online perception."
        )
    elif avg_score >= 0.45:
        sentiment_text = (
            "The overall sentiment is mixed to moderately positive, which suggests a balanced perception with both positive and negative signals."
        )
    else:
        sentiment_text = (
            "The overall sentiment leans negative, which may suggest weaker public perception or more critical discussions."
        )

    if pct_negative >= 40:
        negative_text = (
            "The share of negative mentions is high, which may indicate visible reputation risks."
        )
    elif pct_negative >= 25:
        negative_text = (
            "There is a noticeable share of negative mentions, but negativity is not dominant."
        )
    else:
        negative_text = (
            "The share of negative mentions is relatively low, which supports a more stable reputation profile."
        )

    if pct_disagreement >= 30:
        disagreement_text = (
            "The disagreement between Logistic Regression and VADER is high, so the sentiment results should be interpreted with caution."
        )
    elif pct_disagreement >= 15:
        disagreement_text = (
            "There is a moderate disagreement between Logistic Regression and VADER, which suggests that some posts are more difficult to classify consistently."
        )
    else:
        disagreement_text = (
            "The disagreement between Logistic Regression and VADER is relatively low, which suggests a fairly consistent sentiment pattern."
        )

    return (
        f"Based on the selected filters, {total_mentions} mentions were analyzed {company_text} {method_text}. "
        f"{sentiment_text} {negative_text} {disagreement_text}"
    )


def generate_method_note(method_name):
    if method_name == "Logistic Regression (Sent140)":
        return (
            "This view uses a machine learning model trained on the Sent140 dataset. "
            "It is useful for a more data-driven sentiment classification approach."
        )
    return (
        "This view uses VADER, a lexicon and rule-based sentiment analyzer. "
        "It is useful for quick sentiment scoring, but it may interpret context differently than the machine learning model."
    )


def generate_confusion_matrix_interpretation(metrics):
    tp = metrics.get("true_positive", 0)
    tn = metrics.get("true_negative", 0)
    fp = metrics.get("false_positive", 0)
    fn = metrics.get("false_negative", 0)
    accuracy = metrics.get("accuracy", 0.0)

    total = tp + tn + fp + fn
    correct = tp + tn
    wrong = fp + fn

    return (
        f"The Logistic Regression model was evaluated using the confusion matrix. "
        f"It correctly classified {tp:,} positive messages (True Positive) and {tn:,} negative messages (True Negative). "
        f"At the same time, it made classification errors: {fp:,} negative messages were incorrectly classified as positive "
        f"(False Positive), and {fn:,} positive messages were incorrectly classified as negative (False Negative). "
        f"Overall, the model correctly classified {correct:,} messages and misclassified {wrong:,} messages, "
        f"out of a total of {total:,} evaluated messages. This corresponds to an accuracy of {accuracy:.3f}, "
        f"which is approximately {accuracy * 100:.0f}% correct predictions."
    )


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
# Interpretation
# -----------------------------
st.subheader("Interpretation")

if total_mentions > 0:
    interpretation = generate_interpretation(
        company=company,
        method_name=method_name,
        total_mentions=total_mentions,
        avg_score=avg_score,
        pct_negative=pct_neg,
        pct_disagreement=pct_diff
    )
    st.info(interpretation)
else:
    st.info("No data is available for the current selection, so no interpretation can be generated.")

st.markdown("### Method note")
st.write(generate_method_note(method_name))

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

if method == "vader":
    st.info(
        "VADER is a rule-based sentiment analyzer, so confusion matrix validation and offline model-training "
        "metrics are not applicable here. The VADER view is used only for the Reddit sentiment analysis above."
    )

elif method == "lr_sent140_tfidf":
    base_metrics = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            base_metrics = json.load(f)

    st.markdown("### Logistic Regression validation")

    if base_metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{base_metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{base_metrics['precision']:.4f}")
        c3.metric("Recall", f"{base_metrics['recall']:.4f}")
        c4.metric("F1-score", f"{base_metrics['f1']:.4f}")

        if CM_PATH.exists():
            cm_df = pd.read_csv(CM_PATH, index_col=0)
            st.markdown("### Confusion matrix")
            st.dataframe(cm_df, use_container_width=True)

        st.markdown("### Interpretation (confusion matrix)")
        st.info(generate_confusion_matrix_interpretation(base_metrics))

    else:
        st.warning("Logistic Regression evaluation files were not found.")

    st.markdown("---")
    st.subheader("Offline experiment: Logistic Regression vs DistilBERT on Sentiment140")

    st.info(
        "The section below presents a separate offline comparison between TF-IDF + Logistic Regression "
        "and DistilBERT on the Sentiment140 dataset. It is not based on the Reddit data displayed above. "
        "Both methods were evaluated on exactly the same sampled subset of 10,000 instances "
        "(sampled from the original 1.6M dataset), using the same 80% / 20% train-test split."
    )

    experiment_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
        "TF-IDF + Logistic Regression": [0.7653, 0.7631, 0.7734, 0.7682, 0.8463],
        "DistilBERT": [0.8075, 0.8178, 0.7908, 0.8041, 0.8892],
        "Improvement (%)": ["+5.52%", "+7.17%", "+2.25%", "+4.67%", "+5.06%"]
    })

    st.markdown("### Experimental setup")
    st.write(
        "- Same sampled subset: 10,000 instances from Sentiment140\n"
        "- Same train/test split: 80% training / 20% testing\n"
        "- Same evaluation setting for both models"
    )

    st.markdown("### Comparison results")
    st.dataframe(experiment_df, use_container_width=True)

    st.markdown("### Interpretation")
    st.success(
        "Both methods were evaluated on exactly the same subset of data: the same 10,000-instance sample "
        "drawn from the original 1.6M Sentiment140 dataset, using the same 80% / 20% train-test split. "
        "DistilBERT achieved better results on all metrics: Accuracy (+5.52%), Precision (+7.17%), "
        "Recall (+2.25%), F1-score (+4.67%), and ROC AUC (+5.06%)."
    )

    st.markdown("### Overall conclusion")
    st.write(
        "Overall, DistilBERT is approximately 4.7% more performant than the TF-IDF + Logistic Regression model, "
        "based on the F1-score, which indicates a better balance between precision and recall."
    )

    st.caption(
        "Important: this comparison refers only to the Sentiment140 offline experiment on a fixed 10,000-instance sample, "
        "not to the Reddit posts analyzed in the dashboard above."
    )