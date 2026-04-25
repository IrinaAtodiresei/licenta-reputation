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

METRICS_PATH = EVAL_DIR / "metrics.json"
CM_PATH = EVAL_DIR / "confusion_matrix.csv"

METRICS_TUNED_PATH = EVAL_DIR / "metrics_tuned.json"
CM_TUNED_PATH = EVAL_DIR / "confusion_matrix_tuned.csv"
BEST_PARAMS_PATH = EVAL_DIR / "best_params.json"


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

    if method_name == "Logistic Regression (Sent140)":
        method_text = "using the Logistic Regression model trained on Sentiment140"

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

        comparison_text = (
            f"The disagreement between Logistic Regression and VADER is {pct_disagreement:.2f}%, "
            "so the differences between the statistical model and the rule-based analyzer should be interpreted carefully."
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
            "which shows that the two methods do not always classify the same Reddit mentions identically."
        )

    else:
        method_text = "using the Deep Learning Transformer model"

        if avg_score >= 0.60:
            sentiment_text = (
                "The overall sentiment appears predominantly positive based on the transformer model."
            )
        elif avg_score >= 0.45:
            sentiment_text = (
                "The overall sentiment is mixed based on the transformer model."
            )
        else:
            sentiment_text = (
                "The overall sentiment leans negative based on the transformer model."
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
    if method_name == "Logistic Regression (Sent140)":
        return (
            "This view uses a machine learning model trained on the Sentiment140 dataset. "
            "The text is represented with TF-IDF features, and the final classification is made using Logistic Regression."
        )

    if method_name == "VADER":
        return (
            "This view uses VADER, a lexicon and rule-based sentiment analyzer. "
            "It does not require model training and is useful for quick sentiment scoring, but it may interpret context differently than machine learning models."
        )

    return (
        "This view uses a Deep Learning Transformer model pre-trained for sentiment analysis. "
        "In this application, the model is used for inference on the collected Reddit mentions, without local fine-tuning."
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
        f"It correctly classified {format_thousands_dot(tp)} positive messages and "
        f"{format_thousands_dot(tn)} negative messages. "
        f"It also made {format_thousands_dot(fp)} false positive errors and "
        f"{format_thousands_dot(fn)} false negative errors. "
        f"Overall, the model correctly classified {format_thousands_dot(correct)} messages and misclassified "
        f"{format_thousands_dot(wrong)} messages, out of {format_thousands_dot(total)} evaluated messages. "
        f"This corresponds to an accuracy of {accuracy:.3f}, approximately {accuracy * 100:.0f}% correct predictions."
    )


def format_best_params_text(best_params):
    if not best_params:
        return "No tuned hyperparameters are available."

    mapping = {
        "clf__C": "Regularization strength (C)",
        "clf__max_iter": "Maximum iterations",
        "clf__solver": "Solver",
        "clf__penalty": "Penalty",
        "tfidf__max_features": "Maximum TF-IDF features",
        "tfidf__min_df": "Minimum document frequency",
        "tfidf__ngram_range": "N-gram range",
        "tfidf__max_df": "Maximum document frequency",
    }

    lines = []
    for key, value in best_params.items():
        label = mapping.get(key, key)
        lines.append(f"- **{label}**: {value}")

    return "\n".join(lines)


def generate_tuned_model_interpretation(metrics, best_params):
    f1 = metrics.get("f1", 0.0)
    acc = metrics.get("accuracy", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)

    if f1 >= 0.80:
        performance_text = "The tuned Logistic Regression model shows strong classification performance."
    elif f1 >= 0.70:
        performance_text = "The tuned Logistic Regression model shows good classification performance."
    else:
        performance_text = "The tuned Logistic Regression model shows acceptable performance, but there is still room for improvement."

    if best_params:
        ngram = best_params.get("tfidf__ngram_range")
        min_df = best_params.get("tfidf__min_df")
        max_features = best_params.get("tfidf__max_features")
        c_value = best_params.get("clf__C")
        solver = best_params.get("clf__solver")

        config_text = (
            f"The best configuration uses n-grams {ngram}, "
            f"max_features={max_features}, min_df={min_df}, "
            f"C={c_value}, and solver={solver}. "
        )
    else:
        config_text = ""

    metrics_text = (
        f"On evaluation, it obtained Accuracy = {acc:.4f}, "
        f"Precision = {precision:.4f}, Recall = {recall:.4f}, "
        f"and F1-score = {f1:.4f}. "
    )

    explanation_text = (
        "The tuning process identified the best configuration within the tested parameter grid, "
        "although it did not improve performance on the test set."
    )

    return performance_text + " " + metrics_text + config_text + explanation_text


def generate_comparison_text(base_metrics, tuned_metrics):
    base_f1 = base_metrics.get("f1", 0.0)
    tuned_f1 = tuned_metrics.get("f1", 0.0)

    base_acc = base_metrics.get("accuracy", 0.0)
    tuned_acc = tuned_metrics.get("accuracy", 0.0)

    base_precision = base_metrics.get("precision", 0.0)
    tuned_precision = tuned_metrics.get("precision", 0.0)

    base_recall = base_metrics.get("recall", 0.0)
    tuned_recall = tuned_metrics.get("recall", 0.0)

    delta_f1 = tuned_f1 - base_f1
    delta_acc = tuned_acc - base_acc
    delta_precision = tuned_precision - base_precision
    delta_recall = tuned_recall - base_recall

    text = (
        f"Compared to the initial model, the tuned model changes Accuracy by {delta_acc:.4f}, "
        f"Precision by {delta_precision:.4f}, Recall by {delta_recall:.4f}, "
        f"and F1-score by {delta_f1:.4f}. "
    )

    if delta_f1 > 0:
        text += (
            "Overall, the tuning process had a positive effect, since the tuned model achieved "
            "a better balance between precision and recall."
        )
    elif delta_f1 < 0:
        text += (
            "Overall, the tuning process did not improve the final balance between precision and recall, "
            "so the initial configuration remains stronger."
        )
    else:
        text += (
            "Overall, the tuning process produced very similar results to the initial model."
        )

    return text


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 4.8))

    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    threshold = cm_df.values.max() / 2 if cm_df.values.size > 0 else 0

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = int(cm_df.iloc[i, j])
            text_color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                format_thousands_dot(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# METHODS
# ------------------------------------------------------------
METHODS = {
    "Logistic Regression (Sent140)": "lr_sent140_tfidf",
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

method_name = st.sidebar.radio("Method", list(METHODS.keys()), index=0)
method = METHODS[method_name]

limit_rows = st.sidebar.slider("Rows in disagreement table", 20, 500, 100, 20)

if st.sidebar.button("Reload data"):
    st.session_state.loaded = False
    st.rerun()


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
            st.error("Nu gasesc view-ul reputation.v_company_lr_dl_disagreement. Ruleaza SQL-ul pentru view-ul LR vs Deep Learning.")
            st.stop()

        try:
            comparison_rows = read_df("""
                SELECT *
                FROM reputation.v_lr_dl_sentiment_disagreements
            """)
        except Exception:
            st.error("Nu gasesc view-ul reputation.v_lr_dl_sentiment_disagreements. Ruleaza SQL-ul pentru view-ul LR vs Deep Learning.")
            st.stop()

    else:
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
st.subheader(f"Method: {method_name}")


# ------------------------------------------------------------
# KPI BOXES
# ------------------------------------------------------------
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
    if "different_label_count" in comparison_company.columns and "total" in comparison_company.columns:
        total_different = comparison_company["different_label_count"].sum()
        total_posts = comparison_company["total"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    elif "different_count" in comparison_company.columns and "total_posts" in comparison_company.columns:
        total_different = comparison_company["different_count"].sum()
        total_posts = comparison_company["total_posts"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    elif "different_mentions" in comparison_company.columns and "total_mentions" in comparison_company.columns:
        total_different = comparison_company["different_mentions"].sum()
        total_posts = comparison_company["total_mentions"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    else:
        pct_diff = float(comparison_company["pct_different"].mean()) if "pct_different" in comparison_company.columns else 0.0
else:
    pct_diff = 0.0

col1.metric("Total mentions (filtered)", format_thousands_dot(total_mentions))
col2.metric("Avg score", f"{avg_score:.4f}")
col3.metric("% Negative", f"{pct_neg:.2f}%")

if method == "deep_learning_transformer":
    col4.metric("% Disagreement LR vs DL", f"{pct_diff:.2f}%")
else:
    col4.metric("% Disagreement LR vs VADER", f"{pct_diff:.2f}%")

st.caption("The dashboard uses the data currently stored in the PostgreSQL database.")

st.markdown("### Extra overview")

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


# ------------------------------------------------------------
# MOST NEGATIVE MENTIONS
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Most negative mentions")

negative_examples = safe_read_df("""
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
    ORDER BY s.score ASC
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
st.caption(
    "Note: the Logistic Regression model evaluation metrics and confusion matrices are computed on the Sentiment140 test set, "
    "not on the collected Reddit mentions. The Reddit data is used for reputation monitoring and dashboard analysis."
)

st.subheader("Model evaluation")

if method == "vader":
    st.info(
        "VADER is a rule-based sentiment analyzer, so hyperparameter tuning, "
        "confusion matrix validation, and offline model-training metrics are not applicable here. "
        "The VADER view is used only for the Reddit sentiment analysis above."
    )

elif method == "deep_learning_transformer":
    st.info(
        "The Deep Learning Transformer model is used for sentiment inference on the collected Reddit mentions. "
        "In this application, it was not trained from scratch and was not hyperparameter-tuned locally. "
        "Therefore, hyperparameter tuning results, confusion matrices, and Sentiment140 training metrics are not shown for this method."
    )

elif method == "lr_sent140_tfidf":
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

    st.markdown("### Hyperparameter tuning")
    st.write(
        "This section evaluates the Logistic Regression model on the Sentiment140 test set. "
        "These metrics measure the model's classification performance on labeled training data and are separate "
        "from the Reddit reputation analysis displayed above. "
        "To improve the Logistic Regression model, a hyperparameter tuning stage was introduced, using different "
        "combinations of TF-IDF and Logistic Regression parameters. The best configuration was selected based on validation performance."
    )

    # -------------------------
    # Initial model
    # -------------------------
    st.markdown("### Initial Logistic Regression model")

    if base_metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{base_metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{base_metrics['precision']:.4f}")
        c3.metric("Recall", f"{base_metrics['recall']:.4f}")
        c4.metric("F1-score", f"{base_metrics['f1']:.4f}")

        if CM_PATH.exists():
            cm_df = pd.read_csv(CM_PATH, index_col=0)
            cm_df.index = ["Actual Negative", "Actual Positive"]
            cm_df.columns = ["Predicted Negative", "Predicted Positive"]

            st.markdown("#### Initial confusion matrix")
            fig = plot_confusion_matrix_heatmap(cm_df, title="Initial Confusion Matrix")
            st.pyplot(fig)

        st.markdown("#### Interpretation")
        st.info(
            "This is the baseline Logistic Regression model before hyperparameter tuning. "
            "Its results serve as the reference point for evaluating whether tuning improved performance."
        )
        st.info(generate_confusion_matrix_interpretation(base_metrics))
    else:
        st.warning("Initial evaluation files were not found.")

    st.markdown("---")

    # -------------------------
    # Tuned model
    # -------------------------
    st.markdown("### Tuned Logistic Regression model")

    if tuned_metrics is not None:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Accuracy", f"{tuned_metrics['accuracy']:.4f}")
        t2.metric("Precision", f"{tuned_metrics['precision']:.4f}")
        t3.metric("Recall", f"{tuned_metrics['recall']:.4f}")
        t4.metric("F1-score", f"{tuned_metrics['f1']:.4f}")

        if CM_TUNED_PATH.exists():
            cm_tuned_df = pd.read_csv(CM_TUNED_PATH, index_col=0)
            cm_tuned_df.index = ["Actual Negative", "Actual Positive"]
            cm_tuned_df.columns = ["Predicted Negative", "Predicted Positive"]

            st.markdown("#### Tuned confusion matrix")
            fig_tuned = plot_confusion_matrix_heatmap(cm_tuned_df, title="Tuned Confusion Matrix")
            st.pyplot(fig_tuned)

        st.markdown("#### Best hyperparameters")
        if best_params is not None:
            st.markdown(format_best_params_text(best_params))
        else:
            st.info("Best hyperparameters file was not found.")

        st.markdown("#### Interpretation")
        st.success(generate_tuned_model_interpretation(tuned_metrics, best_params))
    else:
        st.warning("Tuned evaluation files were not found. Run scripts/tune_model.py first.")

    st.markdown("---")

    # -------------------------
    # Comparison
    # -------------------------
    st.markdown("### Initial vs tuned model comparison")

    if base_metrics is not None and tuned_metrics is not None:
        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Initial model": [
                base_metrics["accuracy"],
                base_metrics["precision"],
                base_metrics["recall"],
                base_metrics["f1"],
            ],
            "Tuned model": [
                tuned_metrics["accuracy"],
                tuned_metrics["precision"],
                tuned_metrics["recall"],
                tuned_metrics["f1"],
            ],
        })

        comparison_df["Difference"] = comparison_df["Tuned model"] - comparison_df["Initial model"]
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("#### Comparison interpretation")
        st.info(generate_comparison_text(base_metrics, tuned_metrics))
    else:
        st.info("Both initial and tuned metrics are required for comparison.")

    st.markdown("---")