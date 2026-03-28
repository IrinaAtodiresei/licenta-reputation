import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg
import matplotlib.pyplot as plt

load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    st.error("PG_DSN lipsește. Seteazs-l în Streamlit Cloud → Settings → Secrets")
    st.stop()

st.set_page_config(page_title="Reputation Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

METRICS_PATH = EVAL_DIR / "metrics.json"
CM_PATH = EVAL_DIR / "confusion_matrix.csv"

METRICS_TUNED_PATH = EVAL_DIR / "metrics_tuned.json"
CM_TUNED_PATH = EVAL_DIR / "confusion_matrix_tuned.csv"
BEST_PARAMS_PATH = EVAL_DIR / "best_params.json"

LR_BASELINE_REDDIT_METRICS_PATH = EVAL_DIR / "lr_baseline_reddit_metrics.json"
DISTILBERT_REDDIT_METRICS_PATH = EVAL_DIR / "distilbert_reddit_metrics.json"


def get_conn():
    return psycopg.connect(PG_DSN)


def read_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def format_thousands_dot(value):
    return f"{int(value):,}".replace(",", ".")


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
        f"Based on the selected filters, {format_thousands_dot(total_mentions)} mentions were analyzed {company_text} {method_text}. "
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
        f"It correctly classified {format_thousands_dot(tp)} positive messages (True Positive) and "
        f"{format_thousands_dot(tn)} negative messages (True Negative). "
        f"At the same time, it made classification errors: {format_thousands_dot(fp)} negative messages were "
        f"incorrectly classified as positive (False Positive), and {format_thousands_dot(fn)} positive messages "
        f"were incorrectly classified as negative (False Negative). "
        f"Overall, the model correctly classified {format_thousands_dot(correct)} messages and misclassified "
        f"{format_thousands_dot(wrong)} messages, out of a total of {format_thousands_dot(total)} evaluated messages. "
        f"This corresponds to an accuracy of {accuracy:.3f}, which is approximately {accuracy * 100:.0f}% correct predictions."
    )


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


def pct_improvement(new, old):
    if old == 0:
        return "N/A"
    return f"{((new - old) / old) * 100:+.2f}%"


METHODS = {
    "Logistic Regression (Sent140)": "lr_sent140_tfidf",
    "VADER": "vader",
}

# -----------------------------
# Session state init
# -----------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "loaded_method" not in st.session_state:
    st.session_state.loaded_method = None

if "summary" not in st.session_state:
    st.session_state.summary = pd.DataFrame()

if "disagree_company" not in st.session_state:
    st.session_state.disagree_company = pd.DataFrame()

if "disagreements" not in st.session_state:
    st.session_state.disagreements = pd.DataFrame()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

method_name = st.sidebar.radio("Method", list(METHODS.keys()), index=0)
method = METHODS[method_name]

limit_rows = st.sidebar.slider("Rows in disagreements table", 20, 500, 100, 20)


# -----------------------------
# Reload logic
# -----------------------------
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

    try:
        disagree_company = read_df("""
            SELECT *
            FROM reputation.v_company_method_disagreement
        """)
    except Exception:
        st.error("Nu gasesc view-ul reputation.v_company_method_disagreement.")
        st.stop()

    try:
        disagreements = read_df("""
            SELECT *
            FROM reputation.v_sentiment_disagreements
        """)
    except Exception:
        st.error("Nu gasesc view-ul reputation.v_sentiment_disagreements.")
        st.stop()

    st.session_state.summary = summary
    st.session_state.disagree_company = disagree_company
    st.session_state.disagreements = disagreements
    st.session_state.loaded = True
    st.session_state.loaded_method = method

summary = st.session_state.summary.copy()
disagree_company = st.session_state.disagree_company.copy()
disagreements = st.session_state.disagreements.copy()

# -----------------------------
# Dynamic company options
# -----------------------------
if not summary.empty and "company_name" in summary.columns:
    company_options = ["All"] + sorted(summary["company_name"].dropna().unique().tolist())
else:
    company_options = ["All"]

company = st.sidebar.selectbox("Company", company_options, index=0)

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
st.title("Reputation & Sentiment Dashboard")
st.subheader(f"Method: {method_name}")

# -----------------------------
# KPI boxes
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

total_mentions = int(summary["mentions"].sum()) if not summary.empty else 0

if not summary.empty and total_mentions > 0:
    avg_score = float((summary["avg_score"] * summary["mentions"]).sum() / total_mentions)
else:
    avg_score = 0.0

if not summary.empty and total_mentions > 0 and "negatives" in summary.columns:
    total_negatives = int(summary["negatives"].sum())
    pct_neg = float((total_negatives / total_mentions) * 100)
else:
    pct_neg = 0.0

if not disagree_company.empty:
    if "different_count" in disagree_company.columns and "total_posts" in disagree_company.columns:
        total_different = disagree_company["different_count"].sum()
        total_posts = disagree_company["total_posts"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    elif "different_mentions" in disagree_company.columns and "total_mentions" in disagree_company.columns:
        total_different = disagree_company["different_mentions"].sum()
        total_posts = disagree_company["total_mentions"].sum()
        pct_diff = float((total_different / total_posts) * 100) if total_posts > 0 else 0.0
    else:
        pct_diff = float(disagree_company["pct_different"].mean()) if "pct_different" in disagree_company.columns else 0.0
else:
    pct_diff = 0.0

col1.metric("Total mentions (filtered)", format_thousands_dot(total_mentions))
col2.metric("Avg score", f"{avg_score:.4f}")
col3.metric("% Negative", f"{pct_neg:.2f}%")
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
    sort_col = "pct_different" if "pct_different" in disagree_company.columns else disagree_company.columns[-1]
    disagree_company = disagree_company.sort_values(sort_col, ascending=False)
    st.dataframe(disagree_company, use_container_width=True)
else:
    st.info("Nu exista date despre disagreement pentru filtrul selectat.")

# -----------------------------
# Disagreements list
# -----------------------------
st.subheader("Posts where LR and VADER disagree")

keep_cols = [
    "mention_id", "company_name", "title", "author", "published_at",
    "lr_label", "lr_score", "vader_label", "vader_score"
]

if not disagreements.empty:
    existing_keep_cols = [col for col in keep_cols if col in disagreements.columns]
    sort_col = "published_at" if "published_at" in disagreements.columns else existing_keep_cols[0]
    disagreements = disagreements[existing_keep_cols].sort_values(
        sort_col, ascending=False
    ).head(limit_rows)
    st.dataframe(disagreements, use_container_width=True)
else:
    st.info("Nu exista postări în care LR și VADER să difere pentru filtrul selectat.")

# -----------------------------
# Model evaluation
# -----------------------------
st.markdown("---")
st.subheader("Model evaluation")

if method == "vader":
    st.info(
        "VADER is a rule-based sentiment analyzer, so hyperparameter tuning, "
        "confusion matrix validation, and offline model-training metrics are not applicable here. "
        "The VADER view is used only for the Reddit sentiment analysis above."
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
        "To improve the Logistic Regression model, a hyperparameter tuning stage was introduced. "
        "The model was run multiple times using different combinations of TF-IDF and Logistic Regression parameters. "
        "The best configuration was selected based on validation performance."
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
            ]
        })

        comparison_df["Difference"] = comparison_df["Tuned model"] - comparison_df["Initial model"]
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("#### Comparison interpretation")
        st.info(generate_comparison_text(base_metrics, tuned_metrics))
    else:
        st.info("Both initial and tuned metrics are required for comparison.")

    st.markdown("---")

# -------------------------
# Baseline LR vs DistilBERT
# -------------------------
st.subheader("Baseline Logistic Regression vs DistilBERT")

st.info(
    "This section compares the baseline (untuned) Logistic Regression model with DistilBERT "
    "on the same labeled Reddit comments sample. The results are based on real offline runs."
)

lr_baseline_reddit_metrics = None
distilbert_metrics = None

if LR_BASELINE_REDDIT_METRICS_PATH.exists():
    with open(LR_BASELINE_REDDIT_METRICS_PATH, "r", encoding="utf-8") as f:
        lr_baseline_reddit_metrics = json.load(f)

if DISTILBERT_REDDIT_METRICS_PATH.exists():
    with open(DISTILBERT_REDDIT_METRICS_PATH, "r", encoding="utf-8") as f:
        distilbert_metrics = json.load(f)

if lr_baseline_reddit_metrics is not None and distilbert_metrics is not None:
    baseline_acc = lr_baseline_reddit_metrics.get("accuracy", 0.0)
    baseline_precision = lr_baseline_reddit_metrics.get("precision", 0.0)
    baseline_recall = lr_baseline_reddit_metrics.get("recall", 0.0)
    baseline_f1 = lr_baseline_reddit_metrics.get("f1", 0.0)

    distil_acc = distilbert_metrics.get("accuracy", 0.0)
    distil_precision = distilbert_metrics.get("precision", 0.0)
    distil_recall = distilbert_metrics.get("recall", 0.0)
    distil_f1 = distilbert_metrics.get("f1", 0.0)

    experiment_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Baseline Logistic Regression": [
            baseline_acc,
            baseline_precision,
            baseline_recall,
            baseline_f1,
        ],
        "DistilBERT": [
            distil_acc,
            distil_precision,
            distil_recall,
            distil_f1,
        ],
        "Improvement (%)": [
            pct_improvement(distil_acc, baseline_acc),
            pct_improvement(distil_precision, baseline_precision),
            pct_improvement(distil_recall, baseline_recall),
            pct_improvement(distil_f1, baseline_f1),
        ]
    })

    st.markdown("### Comparison results")
    st.dataframe(experiment_df, use_container_width=True)

    st.markdown("### Interpretation")
    delta_f1 = distil_f1 - baseline_f1

    if delta_f1 > 0:
        st.success(
            f"DistilBERT outperformed the baseline Logistic Regression model on the labeled Reddit comments sample. "
            f"Compared with the untuned TF-IDF + Logistic Regression baseline, DistilBERT improved "
            f"Accuracy by {pct_improvement(distil_acc, baseline_acc)}, "
            f"Precision by {pct_improvement(distil_precision, baseline_precision)}, "
            f"Recall by {pct_improvement(distil_recall, baseline_recall)}, and "
            f"F1-score by {pct_improvement(distil_f1, baseline_f1)}. "
            f"Overall, this suggests that DistilBERT captured contextual information better on the real labeled Reddit sample."
        )
    elif delta_f1 < 0:
        st.warning(
            f"DistilBERT did not outperform the baseline Logistic Regression model on this Reddit sample. "
            f"The F1-score changed by {delta_f1:.4f}, which suggests that the baseline remained stronger."
        )
    else:
        st.info(
            "DistilBERT and the baseline Logistic Regression model achieved very similar results "
            "on the labeled Reddit comments sample."
        )

    st.caption(
        "Important: this comparison is based on the same labeled Reddit comments sample "
        "and is separate from the Logistic Regression hyperparameter tuning section above."
    )
else:
    st.info(
        "This section cannot be displayed yet. Make sure lr_baseline_reddit_metrics.json "
        "and distilbert_reddit_metrics.json are available in the evaluation folder."
    )