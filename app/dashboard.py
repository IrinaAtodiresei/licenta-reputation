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

CM_TUNED_PATH = EVAL_DIR / "confusion_matrix_tuned.csv"
METRICS_TUNED_PATH = EVAL_DIR / "metrics_tuned.json"
BEST_PARAMS_PATH = EVAL_DIR / "best_params.json"


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


def format_best_params_text(best_params):
    if not best_params:
        return "No tuned hyperparameters are available."

    mapping = {
        "clf__C": "Regularization strength (C)",
        "clf__max_iter": "Maximum iterations",
        "clf__solver": "Solver",
        "tfidf__max_features": "Maximum TF-IDF features",
        "tfidf__min_df": "Minimum document frequency",
        "tfidf__ngram_range": "N-gram range",
        "tfidf__max_df": "Maximum document frequency",
    }

    lines = []
    for key, value in best_params.items():
        label = mapping.get(key, key)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def generate_tuned_model_interpretation(metrics, best_params):
    f1 = metrics.get("f1", 0.0)
    acc = metrics.get("accuracy", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)

    if f1 >= 0.80:
        performance_text = (
            "The tuned Logistic Regression model shows strong classification performance."
        )
    elif f1 >= 0.70:
        performance_text = (
            "The tuned Logistic Regression model shows good classification performance."
        )
    else:
        performance_text = (
            "The tuned Logistic Regression model shows acceptable performance, but there is still room for improvement."
        )

    ngram = best_params.get("tfidf__ngram_range")
    min_df = best_params.get("tfidf__min_df")
    max_features = best_params.get("tfidf__max_features")
    c_value = best_params.get("clf__C")
    solver = best_params.get("clf__solver")

    config_text = (
        f"The best configuration uses n-grams {ngram}, max_features={max_features}, "
        f"min_df={min_df}, C={c_value}, and solver={solver}. "
    )

    explanation_text = (
        "This suggests that the model benefits from capturing short word combinations, "
        "filtering rare terms, and using a balanced regularization setting."
    )

    metrics_text = (
        f"On evaluation, it obtained Accuracy = {acc:.4f}, Precision = {precision:.4f}, "
        f"Recall = {recall:.4f}, and F1-score = {f1:.4f}. "
    )

    return performance_text + " " + metrics_text + config_text + explanation_text


def generate_comparison_text(base_metrics, tuned_metrics):
    base_f1 = base_metrics.get("f1", 0.0)
    tuned_f1 = tuned_metrics.get("f1", 0.0)
    base_acc = base_metrics.get("accuracy", 0.0)
    tuned_acc = tuned_metrics.get("accuracy", 0.0)

    delta_f1 = tuned_f1 - base_f1
    delta_acc = tuned_acc - base_acc

    if delta_f1 > 0:
        improvement_text = (
            f"The tuned model improves the F1-score by {delta_f1:.4f} and the accuracy by {delta_acc:.4f} compared to the initial model."
        )
    elif delta_f1 < 0:
        improvement_text = (
            f"The tuned model decreases the F1-score by {abs(delta_f1):.4f} and changes the accuracy by {delta_acc:.4f} compared to the initial model."
        )
    else:
        improvement_text = (
            f"The tuned model keeps the F1-score at a similar level, while the accuracy changes by {delta_acc:.4f}."
        )

    if tuned_f1 >= base_f1:
        conclusion_text = (
            "Overall, hyperparameter tuning had a positive or neutral effect on model performance."
        )
    else:
        conclusion_text = (
            "Overall, the initial configuration remains more effective than the tuned variant."
        )

    return improvement_text + " " + conclusion_text


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

if method == "lr_sent140_tfidf":

    base_metrics = None
    tuned_metrics = None
    best_params = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            base_metrics = json.load(f)

    if METRICS_TUNED_PATH.exists():
        with open(METRICS_TUNED_PATH, "r", encoding="utf-8") as f:
            tuned_metrics = json.load(f)

    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            best_params = json.load(f)

    # -------------------------
    # Initial model
    # -------------------------
    if base_metrics is not None:
        st.markdown("### Initial Logistic Regression model")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{base_metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{base_metrics['precision']:.4f}")
        c3.metric("Recall", f"{base_metrics['recall']:.4f}")
        c4.metric("F1-score", f"{base_metrics['f1']:.4f}")

        if CM_PATH.exists():
            cm_df = pd.read_csv(CM_PATH, index_col=0)
            st.markdown("#### Initial confusion matrix")
            st.dataframe(cm_df, use_container_width=True)

    else:
        st.info("Initial evaluation files were not found.")

    st.markdown("---")

    # -------------------------
    # Tuned model
    # -------------------------
    if tuned_metrics is not None:
        st.markdown("### Tuned Logistic Regression model")

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Accuracy", f"{tuned_metrics['accuracy']:.4f}")
        t2.metric("Precision", f"{tuned_metrics['precision']:.4f}")
        t3.metric("Recall", f"{tuned_metrics['recall']:.4f}")
        t4.metric("F1-score", f"{tuned_metrics['f1']:.4f}")

        if CM_TUNED_PATH.exists():
            cm_tuned_df = pd.read_csv(CM_TUNED_PATH, index_col=0)
            st.markdown("#### Tuned confusion matrix")
            st.dataframe(cm_tuned_df, use_container_width=True)

        st.markdown("#### Tuned model interpretation")
        if best_params is not None:
            st.info(generate_tuned_model_interpretation(tuned_metrics, best_params))
        else:
            st.info(
                f"The tuned Logistic Regression model obtained Accuracy = {tuned_metrics['accuracy']:.4f}, "
                f"Precision = {tuned_metrics['precision']:.4f}, Recall = {tuned_metrics['recall']:.4f}, "
                f"and F1-score = {tuned_metrics['f1']:.4f}."
            )

        if best_params is not None:
            st.markdown("#### Best hyperparameters")
            st.markdown(format_best_params_text(best_params))

    else:
        st.info("Tuned evaluation files were not found. Run scripts/tune_model.py first.")

    st.markdown("---")

    # -------------------------
    # Comparison
    # -------------------------
    st.markdown("### Initial vs tuned model comparison")

    if base_metrics is not None and tuned_metrics is not None:
        comp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Initial model": [
                base_metrics["accuracy"],
                base_metrics["precision"],
                base_metrics["recall"],
                base_metrics["f1"]
            ],
            "Tuned model": [
                tuned_metrics["accuracy"],
                tuned_metrics["precision"],
                tuned_metrics["recall"],
                tuned_metrics["f1"]
            ]
        })

        comp_df["Difference"] = comp_df["Tuned model"] - comp_df["Initial model"]
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("#### Comparison interpretation")
        st.success(generate_comparison_text(base_metrics, tuned_metrics))

    else:
        st.info("Both initial and tuned metrics are required for comparison.")

else:
    st.info(
        "VADER is a rule-based sentiment analyzer, therefore confusion matrix "
        "evaluation is not applicable. Confusion matrix and hyperparameter tuning "
        "are shown only for the Logistic Regression model trained on the Sent140 dataset."
    )