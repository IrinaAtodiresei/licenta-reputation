import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    st.error("PG_DSN lipsește. Setează-l în Streamlit Cloud → Settings → Secrets")
    st.stop()

st.set_page_config(page_title="Reputation Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

LR_METRICS_PATH = EVAL_DIR / "lr_3class_metrics.json"
LR_CM_PATH = EVAL_DIR / "lr_3class_confusion_matrix.csv"
LR_REPORT_PATH = EVAL_DIR / "lr_3class_classification_report.csv"

MANUAL_SAMPLE_PATH = EVAL_DIR / "reddit_manual_validation_sample.csv"
MANUAL_METRICS_PATH = EVAL_DIR / "reddit_manual_validation_metrics.json"
MANUAL_REPORT_PATH = EVAL_DIR / "reddit_manual_validation_report.csv"
MANUAL_CM_PATH = EVAL_DIR / "reddit_manual_validation_confusion_matrix.csv"


# ------------------------------------------------------------
# DB HELPERS
# ------------------------------------------------------------
def get_conn():
    return psycopg.connect(PG_DSN)


def read_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def safe_read_df(sql: str, params=None) -> pd.DataFrame:
    try:
        return read_df(sql, params=params)
    except Exception as e:
        st.warning(f"Nu s-au putut încărca datele pentru această secțiune: {e}")
        return pd.DataFrame()


def format_thousands_dot(value):
    return f"{int(value):,}".replace(",", ".")


# ------------------------------------------------------------
# TEXT GENERATION HELPERS
# ------------------------------------------------------------
def generate_interpretation(
    company,
    method_name,
    total_mentions,
    avg_score,
    pct_negative,
    pct_disagreement,
):
    if company == "All":
        company_text = "for the selected companies"
    else:
        company_text = f"for {company}"

    if method_name == "Logistic Regression 3-class balanced":
        method_text = (
            "using the balanced 3-class Logistic Regression model trained on a Twitter sentiment dataset"
        )

        if avg_score >= 0.60:
            sentiment_text = (
                "The model shows relatively high average confidence in its sentiment classifications."
            )
        elif avg_score >= 0.45:
            sentiment_text = (
                "The model shows moderate average confidence, which is acceptable for short and informal Reddit texts."
            )
        else:
            sentiment_text = (
                "The model shows lower average confidence, which suggests that many texts are ambiguous or difficult to classify."
            )

        comparison_text = (
            "This view presents Logistic Regression as the main baseline method. "
            "Disagreement analysis is shown separately in the VADER and Deep Learning Transformer views."
        )

    elif method_name == "VADER":
        method_text = "using the VADER rule-based sentiment analyzer"

        if avg_score >= 0.20:
            sentiment_text = (
                "The overall sentiment appears positive based on the VADER compound score."
            )
        elif avg_score >= -0.05:
            sentiment_text = (
                "The overall sentiment is relatively neutral or mixed based on the VADER compound score."
            )
        else:
            sentiment_text = (
                "The overall sentiment leans negative based on the VADER compound score."
            )

        comparison_text = (
            f"The disagreement between Logistic Regression and VADER is {pct_disagreement:.2f}%, "
            "which shows how often the statistical model and the rule-based method classify Reddit mentions differently."
        )

    else:
        method_text = "using the Deep Learning Transformer model"

        if avg_score >= 0.60:
            sentiment_text = (
                "The overall sentiment appears more confidently classified by the transformer model."
            )
        elif avg_score >= 0.45:
            sentiment_text = (
                "The overall sentiment is mixed, with moderate confidence from the transformer model."
            )
        else:
            sentiment_text = (
                "The transformer model shows lower confidence, which may indicate ambiguous or context-dependent mentions."
            )

        comparison_text = (
            f"The disagreement between Logistic Regression and the Deep Learning Transformer is {pct_disagreement:.2f}%, "
            "which highlights the differences between a classic machine learning model and a contextual deep learning model."
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

    return (
        f"Based on the selected filters, {format_thousands_dot(total_mentions)} mentions were analyzed "
        f"{company_text} {method_text}. {sentiment_text} {negative_text} {comparison_text}"
    )


def generate_method_note(method_name):
    if method_name == "Logistic Regression 3-class balanced":
        return (
            "This view uses a balanced 3-class Logistic Regression model trained on a Twitter sentiment dataset "
            "with positive, neutral, and negative labels. The text is represented with TF-IDF features, "
            "and the final classification is made using Logistic Regression."
        )

    if method_name == "VADER":
        return (
            "This view uses VADER, a lexicon and rule-based sentiment analyzer. "
            "It does not require model training and is useful for quick sentiment scoring, "
            "but it may interpret context differently than machine learning models."
        )

    return (
        "This view uses a Deep Learning Transformer model pre-trained for sentiment analysis. "
        "In this application, the model is used for inference on the collected Reddit mentions, without local fine-tuning."
    )


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    im = ax.imshow(cm_df.values, cmap="Blues")

    labels = ["Neg", "Neu", "Pos"]

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.set_title(title, fontsize=8, pad=6)

    threshold = cm_df.values.max() / 2 if cm_df.values.size > 0 else 0

    for i in range(3):
        for j in range(3):
            value = int(cm_df.iloc[i, j])
            ax.text(
                j,
                i,
                format_thousands_dot(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    return fig


def plot_normalized_confusion_matrix(cm_df: pd.DataFrame, title="Normalized Confusion Matrix"):
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=100)

    labels = ["Neg", "Neu", "Pos"]

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.set_title(title, fontsize=8, pad=6)

    for i in range(3):
        for j in range(3):
            value = cm_norm.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 50 else "black",
                fontsize=7,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    return fig

def get_manual_cm_for_method(cm_path: Path, method: str):
    if not cm_path.exists():
        return pd.DataFrame()

    cm_all = pd.read_csv(cm_path, index_col=0)

    if "method" not in cm_all.columns:
        return pd.DataFrame()

    cm_method = cm_all[cm_all["method"] == method].drop(columns=["method"])

    return cm_method

# ------------------------------------------------------------
# METHODS
# ------------------------------------------------------------
METHODS = {
    "Logistic Regression 3-class balanced": "lr_3class_balanced",
    "VADER": "vader",
    "Deep Learning Transformer": "deep_learning_transformer",
}


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "loaded_method" not in st.session_state:
    st.session_state.loaded_method = None

if "summary" not in st.session_state:
    st.session_state.summary = pd.DataFrame()

if "comparison_company" not in st.session_state:
    st.session_state.comparison_company = pd.DataFrame()

if "comparison_rows" not in st.session_state:
    st.session_state.comparison_rows = pd.DataFrame()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.caption(
    "Choose a sentiment analysis method and optionally filter the results by company."
)

method_name = st.sidebar.radio("Method", list(METHODS.keys()), index=0)
method = METHODS[method_name]

limit_rows = st.sidebar.slider("Rows in disagreement table", 20, 500, 100, 20)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
need_reload = (
    not st.session_state.loaded
    or st.session_state.loaded_method != method
)

if need_reload:
    try:
        summary = read_df("""
            SELECT *
            FROM reputation.v_company_sentiment_summary
            WHERE method = %s
        """, params=(method,))
    except Exception:
        st.error("Nu gasesc view-ul reputation.v_company_sentiment_summary")
        st.stop()

    if method == "deep_learning_transformer":
        comparison_title = "Logistic Regression vs Deep Learning Transformer"

        try:
            comparison_company = read_df("""
                SELECT *
                FROM reputation.v_company_lr_dl_disagreement
            """)
        except Exception:
            st.error("Nu gasesc view-ul reputation.v_company_lr_dl_disagreement.")
            st.stop()

        try:
            comparison_rows = read_df("""
                SELECT *
                FROM reputation.v_lr_dl_sentiment_disagreements
            """)
        except Exception:
            st.error("Nu gasesc view-ul reputation.v_lr_dl_sentiment_disagreements.")
            st.stop()

    elif method == "vader":
        comparison_title = "Logistic Regression vs VADER"

        try:
            comparison_company = read_df("""
                SELECT *
                FROM reputation.v_company_method_disagreement
            """)
        except Exception:
            st.error("Nu gasesc view-ul reputation.v_company_method_disagreement.")
            st.stop()

        try:
            comparison_rows = read_df("""
                SELECT *
                FROM reputation.v_sentiment_disagreements
            """)
        except Exception:
            st.error("Nu gasesc view-ul reputation.v_sentiment_disagreements.")
            st.stop()

    else:
        comparison_title = ""
        comparison_company = pd.DataFrame()
        comparison_rows = pd.DataFrame()

    st.session_state.summary = summary
    st.session_state.comparison_company = comparison_company
    st.session_state.comparison_rows = comparison_rows
    st.session_state.comparison_title = comparison_title
    st.session_state.loaded = True
    st.session_state.loaded_method = method

summary = st.session_state.summary.copy()
comparison_company = st.session_state.comparison_company.copy()
comparison_rows = st.session_state.comparison_rows.copy()
comparison_title = st.session_state.get("comparison_title", "")


# ------------------------------------------------------------
# COMPANY FILTER
# ------------------------------------------------------------
if not summary.empty and "company_name" in summary.columns:
    company_options = ["All"] + sorted(summary["company_name"].dropna().unique().tolist())
else:
    company_options = ["All"]

company = st.sidebar.selectbox("Company", company_options, index=0)

if company != "All":
    summary = summary[summary["company_name"] == company]

    if not comparison_company.empty and "company_name" in comparison_company.columns:
        comparison_company = comparison_company[comparison_company["company_name"] == company]

    if not comparison_rows.empty and "company_name" in comparison_rows.columns:
        comparison_rows = comparison_rows[comparison_rows["company_name"] == company]


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("Reputation & Sentiment Dashboard")
st.markdown("""
### What does this app do?

This application analyzes public Reddit discussions about major tech companies 
and automatically detects whether the sentiment is **positive, neutral, or negative**.

It compares three different approaches:
- Logistic Regression (machine learning baseline)
- VADER (rule-based)
- Deep Learning Transformer

Use the controls on the left to explore how sentiment changes across companies and methods.
""")
st.subheader(f"Method: {method_name}")

st.markdown("### Overview of collected data")
st.info(
    "The dataset contains Reddit posts and comments collected through the Reddit public JSON API. "
    "The collection process used selected technology-related subreddits, keyword filtering, pagination, "
    "search endpoints, pagination, and comment extraction. "
    "The collected mentions are stored in a PostgreSQL database and analyzed using three sentiment analysis methods."
)


# ------------------------------------------------------------
# KPI BOXES
# ------------------------------------------------------------
if method == "lr_3class_balanced":
    col1, col2, col3 = st.columns(3)
else:
    col1, col2, col3, col4 = st.columns(4)

total_mentions = int(summary["mentions"].sum()) if not summary.empty and "mentions" in summary.columns else 0

if not summary.empty and total_mentions > 0 and "avg_score" in summary.columns:
    avg_score = float((summary["avg_score"] * summary["mentions"]).sum() / total_mentions)
else:
    avg_score = 0.0

if not summary.empty and total_mentions > 0 and "negatives" in summary.columns:
    total_negatives = int(summary["negatives"].sum())
    pct_neg = float((total_negatives / total_mentions) * 100)
else:
    total_negatives = 0
    pct_neg = 0.0

if not comparison_company.empty:
    if "different_mentions" in comparison_company.columns and "total_mentions" in comparison_company.columns:
        total_different = comparison_company["different_mentions"].sum()
        total_posts = comparison_company["total_mentions"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    elif "pct_different" in comparison_company.columns:
        pct_diff = float(comparison_company["pct_different"].mean())
    else:
        pct_diff = 0.0
else:
    pct_diff = 0.0

col1.metric("Total mentions (filtered)", format_thousands_dot(total_mentions))
col2.metric("Model confidence", f"{avg_score:.4f}")
col3.metric("% Negative", f"{pct_neg:.2f}%")

if method == "vader":
    col4.metric("% Disagreement LR vs VADER", f"{pct_diff:.2f}%")
elif method == "deep_learning_transformer":
    col4.metric("% Disagreement LR vs DL", f"{pct_diff:.2f}%")

st.caption(
    "Note: Avg confidence represents the average confidence/probability of the selected model. "
    "For VADER, the score represents the compound sentiment score."
)

st.caption("The dashboard uses the data currently stored in the PostgreSQL database.")


# ------------------------------------------------------------
# SENTIMENT DISTRIBUTION
# ------------------------------------------------------------
st.markdown("### Sentiment distribution by method")

method_distribution = safe_read_df("""
    SELECT
        method,
        SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE WHEN label = 'neutral' THEN 1 ELSE 0 END) AS neutral,
        SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) AS negative,
        COUNT(*) AS total
    FROM reputation.sentiment_result
    GROUP BY method
    ORDER BY method;
""")

if not method_distribution.empty:
    st.dataframe(method_distribution, use_container_width=True)

    chart_df = method_distribution.set_index("method")[["positive", "neutral", "negative"]]
    st.bar_chart(chart_df)
else:
    st.info("Nu există date suficiente pentru distribuția sentimentului pe metode.")


# ------------------------------------------------------------
# EXTRA OVERVIEW
# ------------------------------------------------------------
st.markdown("### Sentiment breakdown")

extra1, extra2, extra3 = st.columns(3)

positive_count = int(summary["positives"].sum()) if not summary.empty and "positives" in summary.columns else 0
negative_count = int(summary["negatives"].sum()) if not summary.empty and "negatives" in summary.columns else 0
neutral_count = int(summary["neutrals"].sum()) if not summary.empty and "neutrals" in summary.columns else 0

extra1.metric("Positive mentions", format_thousands_dot(positive_count))
extra2.metric("Negative mentions", format_thousands_dot(negative_count))
extra3.metric("Neutral mentions", format_thousands_dot(neutral_count))

st.markdown("---")


# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
st.subheader("Company summary (selected method)")

cols_order = [
    "company_id",
    "company_name",
    "method",
    "mentions",
    "avg_score",
    "positives",
    "negatives",
    "neutrals",
    "pct_negative",
]

if not summary.empty:
    existing_cols = [col for col in cols_order if col in summary.columns]
    summary = summary[existing_cols].sort_values("mentions", ascending=False)
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


# ------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------
st.markdown("### Mentions by company")

st.caption(
    "Note: Differences in mention volume reflect the frequency of Reddit discussions captured from the selected "
    "subreddits and keywords. A higher number of mentions does not necessarily indicate higher overall popularity "
    "or better reputation."
)

if not summary.empty and "company_name" in summary.columns and "mentions" in summary.columns:
    mentions_chart = summary[["company_name", "mentions"]].sort_values("mentions", ascending=True)
    st.bar_chart(mentions_chart.set_index("company_name"))
else:
    st.info("Nu există date suficiente pentru graficul de volum.")

st.markdown("### Negative sentiment by company")

if not summary.empty and "company_name" in summary.columns and "pct_negative" in summary.columns:
    negative_chart = summary[["company_name", "pct_negative"]].sort_values("pct_negative", ascending=True)
    st.bar_chart(negative_chart.set_index("company_name"))
else:
    st.info("Nu există date suficiente pentru graficul de sentiment negativ.")


# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
st.subheader("Interpretation")

if total_mentions > 0:
    interpretation = generate_interpretation(
        company=company,
        method_name=method_name,
        total_mentions=total_mentions,
        avg_score=avg_score,
        pct_negative=pct_neg,
        pct_disagreement=pct_diff,
    )
    st.info(interpretation)
else:
    st.info("No data is available for the current selection, so no interpretation can be generated.")

st.markdown("### Method note")
st.write(generate_method_note(method_name))

st.markdown("---")


# ------------------------------------------------------------
# METHOD DISAGREEMENT
# ------------------------------------------------------------
if method in ["vader", "deep_learning_transformer"]:
    st.subheader(f"Method disagreement: {comparison_title}")

    if not comparison_company.empty:
        sort_col = "pct_different" if "pct_different" in comparison_company.columns else comparison_company.columns[-1]
        comparison_company = comparison_company.sort_values(sort_col, ascending=False)
        st.dataframe(comparison_company, use_container_width=True)
    else:
        st.info("Nu exista date despre disagreement pentru filtrul selectat.")

    st.subheader(f"Posts where {comparison_title} disagree")

    if method == "deep_learning_transformer":
        keep_cols = [
            "mention_id",
            "company_name",
            "title",
            "author",
            "published_at",
            "lr_label",
            "lr_score",
            "dl_label",
            "dl_score",
        ]
    else:
        keep_cols = [
            "mention_id",
            "company_name",
            "title",
            "author",
            "published_at",
            "lr_label",
            "lr_score",
            "vader_label",
            "vader_score",
        ]

    if not comparison_rows.empty:
        existing_keep_cols = [col for col in keep_cols if col in comparison_rows.columns]

        if existing_keep_cols:
            sort_col = "published_at" if "published_at" in comparison_rows.columns else existing_keep_cols[0]
            comparison_rows = comparison_rows[existing_keep_cols].sort_values(
                sort_col,
                ascending=False,
            ).head(limit_rows)
            st.dataframe(comparison_rows, use_container_width=True)
        else:
            st.info("Nu există coloanele necesare pentru afișarea comparației.")
    else:
        st.info("Nu exista postări în care metodele selectate să difere pentru filtrul selectat.")

    st.markdown("---")


# ------------------------------------------------------------
# MOST NEGATIVE MENTIONS
# ------------------------------------------------------------
st.subheader("Most negative mentions")

if method == "deep_learning_transformer":
    order_direction = "DESC"
else:
    order_direction = "ASC"

negative_examples = safe_read_df(f"""
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
    LIMIT 30;
""", params=(method,))

if company != "All" and not negative_examples.empty:
    negative_examples = negative_examples[negative_examples["company_name"] == company]

if not negative_examples.empty:
    st.dataframe(negative_examples.head(10), use_container_width=True)
else:
    st.info("Nu există exemple negative pentru filtrul selectat.")


# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Model evaluation")

if method == "lr_3class_balanced":
    st.caption(
        "Note: the Logistic Regression model evaluation metrics and confusion matrix are computed on the new "
        "3-class Twitter sentiment dataset, not on the collected Reddit mentions."
    )

    if LR_METRICS_PATH.exists():
        with open(LR_METRICS_PATH, "r", encoding="utf-8") as f:
            lr_metrics = json.load(f)

        st.markdown("### Logistic Regression 3-class balanced metrics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{lr_metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision macro", f"{lr_metrics.get('precision_macro', 0):.4f}")
        c3.metric("Recall macro", f"{lr_metrics.get('recall_macro', 0):.4f}")
        c4.metric("F1 macro", f"{lr_metrics.get('f1_macro', 0):.4f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Precision weighted", f"{lr_metrics.get('precision_weighted', 0):.4f}")
        c6.metric("Recall weighted", f"{lr_metrics.get('recall_weighted', 0):.4f}")
        c7.metric("F1 weighted", f"{lr_metrics.get('f1_weighted', 0):.4f}")

    else:
        st.warning("Nu gasesc fisierul evaluation/lr_3class_metrics.json.")

    if LR_CM_PATH.exists():
        cm_df = pd.read_csv(LR_CM_PATH, index_col=0)

        st.markdown("### Logistic Regression confusion matrices")
        st.caption(
            "The raw matrix shows the number of predictions, while the normalized matrix shows percentages per real class. "
            "Rows represent actual labels and columns represent predicted labels."
        )

        col_raw, col_norm = st.columns(2)

        with col_raw:
            st.markdown("#### Raw confusion matrix")
            fig_raw = plot_confusion_matrix_heatmap(cm_df, title="Raw")
            st.pyplot(fig_raw, use_container_width=False)

        with col_norm:
            st.markdown("#### Normalized confusion matrix")
            fig_norm = plot_normalized_confusion_matrix(cm_df, title="Normalized (%)")
            st.pyplot(fig_norm, use_container_width=False)

        st.caption(
            "Observation: The model performs best on negative sentiment (~90%), while neutral is slightly harder to classify (~84%). "
            "Most errors occur between neighboring sentiment classes."
        )
    else:
        st.warning("Nu gasesc fisierul evaluation/lr_3class_confusion_matrix.csv.")

    if LR_REPORT_PATH.exists():
        report_df = pd.read_csv(LR_REPORT_PATH, index_col=0)
        st.markdown("### Classification report")
        st.dataframe(report_df, use_container_width=True)

elif method == "vader":
    st.info(
        "VADER is a rule-based sentiment analyzer, so hyperparameter tuning, "
        "confusion matrix validation, and offline model-training metrics are not applicable here. "
        "The VADER view is used only for the Reddit sentiment analysis above."
    )

elif method == "deep_learning_transformer":
    st.info(
        "The Deep Learning Transformer model is used for sentiment inference on the collected Reddit mentions. "
        "In this application, it was not trained from scratch and was not hyperparameter-tuned locally. "
        "Therefore, local training metrics and confusion matrices are not shown for this method."
    )

# ------------------------------------------------------------
# MANUAL REDDIT VALIDATION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Manual Reddit validation")

st.info(
    "This section evaluates the selected sentiment analysis method on a manually labeled Reddit sample. "
    "The manual labels are used as ground truth, while the model predictions are compared against them."
)

if not MANUAL_METRICS_PATH.exists() or not MANUAL_SAMPLE_PATH.exists():
    st.warning(
        "Manual validation files are not available yet. "
        "Export a Reddit sample, label the manual_label column, then run the manual evaluation script."
    )
else:
    with open(MANUAL_METRICS_PATH, "r", encoding="utf-8") as f:
        manual_metrics = json.load(f)

    if method not in manual_metrics:
        st.warning("Nu exista metrici manuale pentru metoda selectata.")
    else:
        selected_manual_metrics = manual_metrics[method]

        st.markdown(f"### Manual validation results: {method_name}")

        mv1, mv2, mv3, mv4 = st.columns(4)

        mv1.metric("Manual sample size", format_thousands_dot(selected_manual_metrics.get("sample_size", 0)))
        mv2.metric("Accuracy", f"{selected_manual_metrics.get('accuracy', 0):.4f}")
        mv3.metric("Precision macro", f"{selected_manual_metrics.get('precision_macro', 0):.4f}")
        mv4.metric("F1 macro", f"{selected_manual_metrics.get('f1_macro', 0):.4f}")

        mv5, mv6, mv7 = st.columns(3)

        mv5.metric("Recall macro", f"{selected_manual_metrics.get('recall_macro', 0):.4f}")
        mv6.metric("F1 weighted", f"{selected_manual_metrics.get('f1_weighted', 0):.4f}")
        mv7.metric("Accuracy on Reddit", f"{selected_manual_metrics.get('accuracy', 0) * 100:.2f}%")

        st.caption(
            "Unlike the previous training evaluation, these metrics are computed directly on Reddit mentions "
            "that were manually labeled by the author of the project."
        )

        manual_cm_df = get_manual_cm_for_method(MANUAL_CM_PATH, method)

        if not manual_cm_df.empty:
            st.markdown("### Manual validation confusion matrices")

            cm_left, cm_right = st.columns(2)

            with cm_left:
                st.markdown("#### Raw confusion matrix")
                fig_manual_raw = plot_confusion_matrix_heatmap(
                    manual_cm_df,
                    title="Manual Reddit Validation - Raw"
                )
                st.pyplot(fig_manual_raw, use_container_width=False)

            with cm_right:
                st.markdown("#### Normalized confusion matrix")
                fig_manual_norm = plot_normalized_confusion_matrix(
                    manual_cm_df,
                    title="Manual Reddit Validation - Normalized (%)"
                )
                st.pyplot(fig_manual_norm, use_container_width=False)

        if MANUAL_REPORT_PATH.exists():
            manual_report_df = pd.read_csv(MANUAL_REPORT_PATH)

            if "method" in manual_report_df.columns:
                manual_report_df = manual_report_df[manual_report_df["method"] == method]

            st.markdown("### Manual validation classification report")
            st.dataframe(manual_report_df, use_container_width=True)

        manual_df = pd.read_csv(MANUAL_SAMPLE_PATH)

        if company != "All" and "company_name" in manual_df.columns:
            manual_df = manual_df[manual_df["company_name"] == company]

        pred_col_map = {
            "lr_3class_balanced": "lr_label",
            "vader": "vader_label",
            "deep_learning_transformer": "dl_label",
        }

        selected_pred_col = pred_col_map.get(method)

        if selected_pred_col and selected_pred_col in manual_df.columns:
            manual_df["is_correct"] = manual_df["manual_label"] == manual_df[selected_pred_col]

            display_cols = [
                "mention_id",
                "company_name",
                "text",
                "manual_label",
                selected_pred_col,
                "is_correct",
            ]

            existing_display_cols = [col for col in display_cols if col in manual_df.columns]

            st.markdown("### Manually labeled Reddit examples")
            st.dataframe(
                manual_df[existing_display_cols].head(50),
                use_container_width=True
            )

            st.download_button(
                label="⬇ Download manual validation sample",
                data=manual_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"manual_validation_{method}.csv",
                mime="text/csv",
            )

        st.markdown("### Manual validation interpretation")

        acc = selected_manual_metrics.get("accuracy", 0)
        f1 = selected_manual_metrics.get("f1_macro", 0)

        if acc >= 0.75:
            manual_text = (
                f"The selected method performs well on the manually labeled Reddit sample, "
                f"with an accuracy of {acc:.4f} and a macro F1-score of {f1:.4f}."
            )
        elif acc >= 0.60:
            manual_text = (
                f"The selected method has moderate performance on the manually labeled Reddit sample, "
                f"with an accuracy of {acc:.4f}. This suggests that Reddit language is more difficult "
                f"than the original training data."
            )
        else:
            manual_text = (
                f"The selected method has limited performance on the manually labeled Reddit sample, "
                f"with an accuracy of {acc:.4f}. This shows that model predictions should be interpreted carefully "
                f"when applied to informal Reddit discussions."
            )

        st.info(manual_text)