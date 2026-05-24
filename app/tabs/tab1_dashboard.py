import streamlit as st
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

from utils.db import (
    load_summary,
    load_disagree_company,
    load_disagreements,
    safe_read_df,
)
from utils.filters import filter_by_company, reorder_summary_columns
from utils.kpis import compute_kpis
from utils.charts import (
    render_metrics_grid,
    render_mentions_chart,
    render_pct_negative_chart,
    render_avg_score_chart,
)
from sentiment_models import predict_logreg, predict_vader

# Paths for evaluation files
from utils.paths import (
    LR_METRICS_PATH,
    LR_CM_PATH,
    LR_REPORT_PATH,
    MANUAL_SAMPLE_PATH,
    MANUAL_METRICS_PATH,
    MANUAL_CM_PATH,
    MANUAL_REPORT_PATH,
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def plot_confusion_matrix_heatmap(cm_df, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_normalized_confusion_matrix(cm_df, title="Normalized Confusion Matrix"):
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", ax=ax, vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def format_thousands_dot(n):
    return f"{int(n):,}".replace(",", ".")

def load_lr_evaluation():
    try:
        with open(LR_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except:
        metrics = {}

    try:
        cm_raw = pd.read_csv(LR_CM_PATH, index_col=0)
    except:
        cm_raw = pd.DataFrame()

    try:
        report = pd.read_csv(LR_REPORT_PATH)
    except:
        report = pd.DataFrame()

    return metrics, cm_raw, report

def load_manual_validation():
    try:
        manual_sample = pd.read_csv(MANUAL_SAMPLE_PATH)
    except:
        manual_sample = pd.DataFrame()

    try:
        with open(MANUAL_METRICS_PATH, "r", encoding="utf-8") as f:
            manual_metrics = json.load(f)
    except:
        manual_metrics = {}

    try:
        manual_cm = pd.read_csv(MANUAL_CM_PATH, index_col=0)
    except:
        manual_cm = pd.DataFrame()

    try:
        manual_report = pd.read_csv(MANUAL_REPORT_PATH)
    except:
        manual_report = pd.DataFrame()

    return manual_sample, manual_metrics, manual_cm, manual_report


# ------------------------------------------------------------
# MAIN DASHBOARD TAB
# ------------------------------------------------------------

def render_tab1(base_df, rows_slider: int):

    st.markdown("""
    <style>
        .dashboard-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
        }
        .dashboard-title {
            margin: 0 0 8px 0;
            color: #0B72E5;
            font-size: 22px;
            font-weight: 700;
        }
        .dashboard-text {
            margin: 0;
            color: #475569;
            line-height: 1.75;
        }
        .kpi-box {
            background: #F9FAFB;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        }
        .kpi-label {
            font-size: 13px;
            color: #6B7280;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 24px;
            font-weight: 700;
            color: #111827;
        }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------
    METHODS = {
        "Logistic Regression (Sent140)": "lr_sent140_tfidf",
        "VADER": "vader",
    }

    st.sidebar.title("Dashboard Controls")
    method_name = st.sidebar.radio("Method", list(METHODS.keys()))
    method = METHODS[method_name]
    company = st.sidebar.selectbox("Company", ["All", "Apple", "Samsung", "Google"])
    limit_rows = st.sidebar.slider("Rows in tables", 20, 500, rows_slider, 20)
    refresh = st.sidebar.button("🔄 Refresh data")

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    if "dashboard_loaded" not in st.session_state or refresh:
        try:
            summary = load_summary(method)
            disagree_company = load_disagree_company()
            disagreements = load_disagreements()
        except:
            summary, disagree_company, disagreements = create_fallback_data(base_df, method)

        st.session_state.summary = summary
        st.session_state.disagree_company = disagree_company
        st.session_state.disagreements = disagreements
        st.session_state.dashboard_loaded = True

    summary = st.session_state.summary.copy()
    disagree_company = st.session_state.disagree_company.copy()
    disagreements = st.session_state.disagreements.copy()

    summary, disagree_company, disagreements = filter_by_company(
        summary, disagree_company, disagreements, company
    )

    kpis = compute_kpis(summary, disagree_company)

    # ------------------------------------------------------------
    # WHAT DOES THIS APP DO?
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">What does this app do?</h2>
        <p class="dashboard-text">
            This dashboard analyzes Reddit discussions about Apple, Samsung and Google using
            Logistic Regression, VADER and Deep Learning models. It provides sentiment metrics,
            disagreement analysis, model evaluation and manual validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # OVERVIEW OF COLLECTED DATA
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Overview of collected data</h2>
        <p class="dashboard-text">High-level metrics for the selected method and company.</p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi-box"><div class="kpi-label">Total mentions</div>
                    <div class="kpi-value">{kpis['total_mentions']}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-box"><div class="kpi-label">Avg score</div>
                    <div class="kpi-value">{kpis['avg_score']:.4f}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-box"><div class="kpi-label">% Negative</div>
                    <div class="kpi-value">{kpis['pct_negative']:.2f}%</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-box"><div class="kpi-label">% Disagreement</div>
                    <div class="kpi-value">{kpis['pct_disagreement']:.2f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SENTIMENT DISTRIBUTION BY METHOD
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Sentiment distribution by method</h2>
    """, unsafe_allow_html=True)

    method_distribution = safe_read_df("""
        SELECT method,
               SUM(CASE WHEN label='positive' THEN 1 ELSE 0 END) AS positive,
               SUM(CASE WHEN label='neutral' THEN 1 ELSE 0 END) AS neutral,
               SUM(CASE WHEN label='negative' THEN 1 ELSE 0 END) AS negative
        FROM reputation.sentiment_result
        GROUP BY method
        ORDER BY method;
    """)

    if not method_distribution.empty:
        st.dataframe(method_distribution, use_container_width=True)
        st.bar_chart(method_distribution.set_index("method")[["positive", "neutral", "negative"]])
    else:
        st.info("No data available.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SENTIMENT BREAKDOWN
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Sentiment breakdown</h2>
    """, unsafe_allow_html=True)

    pos = summary["positives"].sum()
    neg = summary["negatives"].sum()
    neu = summary["neutrals"].sum()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="kpi-box"><div class="kpi-label">Positive</div>
                    <div class="kpi-value">{pos}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-box"><div class="kpi-label">Negative</div>
                    <div class="kpi-value">{neg}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-box"><div class="kpi-label">Neutral</div>
                    <div class="kpi-value">{neu}</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # COMPANY SUMMARY
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Company summary</h2>
    """, unsafe_allow_html=True)

    ordered_summary = reorder_summary_columns(summary)
    st.dataframe(ordered_summary, use_container_width=True)

    st.download_button(
        "⬇ Download summary CSV",
        ordered_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{method}.csv",
        mime="text/csv",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # CHARTS
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Charts</h2>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_mentions_chart(summary)
    with c2:
        render_pct_negative_chart(summary)
    with c3:
        render_avg_score_chart(summary)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # MOST NEGATIVE MENTIONS
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Most negative mentions</h2>
    """, unsafe_allow_html=True)

    negative_examples = safe_read_df(f"""
        SELECT company_name, title, content, score, published_at
        FROM reputation.v_most_negative_mentions
        WHERE method = %s
        ORDER BY score ASC
        LIMIT 30
    """, params=(method,))

    if company != "All":
        negative_examples = negative_examples[negative_examples["company_name"] == company]

    if not negative_examples.empty:
        st.dataframe(negative_examples, use_container_width=True)
    else:
        st.info("No negative examples available.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Model evaluation</h2>
    """, unsafe_allow_html=True)

    lr_metrics, lr_cm_raw, lr_report = load_lr_evaluation()

    if lr_metrics:
        metric_cards = [
            {"label": "Accuracy", "value": f"{lr_metrics.get('accuracy', 0):.4f}"},
            {"label": "Precision macro", "value": f"{lr_metrics.get('precision_macro', 0):.4f}"},
            {"label": "Recall macro", "value": f"{lr_metrics.get('recall_macro', 0):.4f}"},
            {"label": "F1 macro", "value": f"{lr_metrics.get('f1_macro', 0):.4f}"},
        ]
        render_metrics_grid(metric_cards, columns=4)

    if not lr_cm_raw.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Raw confusion matrix")
            st.pyplot(plot_confusion_matrix_heatmap(lr_cm_raw))
        with c2:
            st.markdown("### Normalized confusion matrix")
            st.pyplot(plot_normalized_confusion_matrix(lr_cm_raw))

    if not lr_report.empty:
        st.markdown("### Classification report")
        st.dataframe(lr_report, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # MANUAL VALIDATION
    # ------------------------------------------------------------
    st.markdown("""
    <div class="dashboard-card">
        <h2 class="dashboard-title">Manual Reddit validation</h2>
    """, unsafe_allow_html=True)

    manual_sample, manual_metrics, manual_cm, manual_report = load_manual_validation()

    if method in manual_metrics:
        mm = manual_metrics[method]

        cards = [
            {"label": "Sample size", "value": mm.get("sample_size", 0)},
            {"label": "Accuracy", "value": f"{mm.get('accuracy', 0):.4f}"},
            {"label": "F1 macro", "value": f"{mm.get('f1_macro', 0):.4f}"},
        ]
        render_metrics_grid(cards, columns=3)

        if not manual_cm.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Raw confusion matrix")
                st.pyplot(plot_confusion_matrix_heatmap(manual_cm))
            with c2:
                st.markdown("### Normalized confusion matrix")
                st.pyplot(plot_normalized_confusion_matrix(manual_cm))

        if not manual_report.empty:
            st.markdown("### Manual classification report")
            st.dataframe(manual_report, use_container_width=True)

        st.markdown("### Manual sample")
        st.dataframe(manual_sample.head(50), use_container_width=True)

    else:
        st.info("No manual validation available for this method.")

    st.markdown("</div>", unsafe_allow_html=True)
