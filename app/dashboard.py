import os
import re
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg
import matplotlib.pyplot as plt

import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from groq import Groq
import time
import base64

import requests

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
load_dotenv()

PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    st.error("PG_DSN lipsește. Setează-l în Streamlit Cloud → Settings → Secrets")
    st.stop()

st.set_page_config(
    page_title="Reputation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

LR_METRICS_PATH = EVAL_DIR / "lr_3class_metrics.json"
LR_CM_PATH = EVAL_DIR / "lr_3class_confusion_matrix.csv"
LR_REPORT_PATH = EVAL_DIR / "lr_3class_classification_report.csv"
LR_MODEL_PATH = BASE_DIR / "models" / "lr_3class_balanced.joblib"

MANUAL_SAMPLE_PATH = EVAL_DIR / "reddit_manual_validation_sample.csv"
MANUAL_METRICS_PATH = EVAL_DIR / "reddit_manual_validation_metrics.json"
MANUAL_REPORT_PATH = EVAL_DIR / "reddit_manual_validation_report.csv"
MANUAL_CM_PATH = EVAL_DIR / "reddit_manual_validation_confusion_matrix.csv"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LOGO_PATH = BASE_DIR / "app" / "assets" / "img2.png"
LANDING_LOGO_PATH = BASE_DIR / "app" / "assets" / "logo.png"

FALLBACK_PATH = BASE_DIR / "data" / "fallback_pipeline_sample.csv.csv"

def load_fallback_pipeline_sample():
    if FALLBACK_PATH.exists():
        return pd.read_csv(FALLBACK_PATH)
    return pd.DataFrame()


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

@st.cache_resource
def load_groq_client():
    if not GROQ_API_KEY:
        return None

    return Groq(api_key=GROQ_API_KEY)

def generate_llm_business_insight(company, method_name, summary_df, negative_df):
    client = load_groq_client()

    if client is None:
        return None

    if summary_df.empty:
        return "No data is available for the selected filters."

    total_mentions = int(summary_df["mentions"].sum()) if "mentions" in summary_df.columns else 0
    positives = int(summary_df["positives"].sum()) if "positives" in summary_df.columns else 0
    negatives = int(summary_df["negatives"].sum()) if "negatives" in summary_df.columns else 0
    neutrals = int(summary_df["neutrals"].sum()) if "neutrals" in summary_df.columns else 0

    avg_score = 0.0
    if total_mentions > 0 and "avg_score" in summary_df.columns:
        avg_score = float((summary_df["avg_score"] * summary_df["mentions"]).sum() / total_mentions)

    pct_negative = 0.0
    if total_mentions > 0:
        pct_negative = float((negatives / total_mentions) * 100)

    negative_examples_text = ""

    if not negative_df.empty:
        examples = []

        for _, row in negative_df.head(20).iterrows():
            title = str(row.get("title", "") or "")
            content = str(row.get("content", "") or "")
            example_company = str(row.get("company_name", "") or "Unknown")
            text = f"{title} {content}".strip()
            text = text[:500]

            if text:
                examples.append(f"- Company: {example_company} | Text: {text}")

        negative_examples_text = "\n".join(examples)

    if company == "All":
        analysis_scope = (
            "The selected filter includes all companies: Apple, Samsung, and Google. "
            "Do not write as if there is one CEO for all companies. "
            "Interpret the results as a comparative market-level overview across the three companies. "
            "When giving recommendations, write them for analysts, stakeholders, or brand managers, not for a single CEO."
        )

        recommendation_title = "Strategic recommendations for stakeholders"
    else:
        analysis_scope = (
            f"The selected filter focuses only on {company}. "
            f"Write the interpretation as company-specific reputation analysis for {company}."
        )

        recommendation_title = "Strategic recommendation for a CEO or stakeholder"

    prompt = f"""
You are a business analyst specialized in online reputation and sentiment analysis.

Analyze the following Reddit sentiment results and generate a clear executive interpretation.

Analysis scope:
{analysis_scope}

Company filter: {company}
Sentiment method: {method_name}

Aggregated metrics:
- Total mentions: {total_mentions}
- Positive mentions: {positives}
- Neutral mentions: {neutrals}
- Negative mentions: {negatives}
- Average model score/confidence: {avg_score:.4f}
- Negative percentage: {pct_negative:.2f}%

Most negative Reddit examples:
{negative_examples_text}

Write the answer in English.

Structure:
1. Executive summary
2. Main reputation risks
3. Possible reasons behind negative sentiment
4. {recommendation_title}

Rules:
- If company filter is All, do not treat Apple, Samsung, and Google as one company.
- If company filter is All, compare the companies at portfolio / market level.
- If company filter is All, avoid phrases like "the CEO should" or "the company should".
- Do not invent facts that are not supported by the metrics or examples.
- Keep it concise, academic, and business-oriented.
"""

    messages = [
        {
            "role": "system",
            "content": "You transform sentiment analysis results into clear business insights."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    last_error = None

    for model_name in groq_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=700,
            )

            st.session_state["last_groq_model"] = model_name
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            continue

    raise Exception(f"All Groq fallback models failed. Last error: {last_error}")

def format_thousands_dot(value):
    return f"{int(value):,}".replace(",", ".")

def render_metrics_grid(metrics, columns=4):
    cols = st.columns(columns)

    for index, metric in enumerate(metrics):
        col = cols[index % columns]

        with col:
            st.markdown(
                f"""
                <div style="
                    padding: 16px 18px;
                    margin-bottom: 16px;
                    border-radius: 14px;
                    background-color: #f8fafc;
                    border: 1px solid #e5e7eb;
                    min-height: 105px;
                ">
                    <div style="
                        font-size: 0.85rem;
                        color: #6b7280;
                        margin-bottom: 8px;
                    ">
                        {metric["label"]}
                    </div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 600;
                        color: #111827;
                        line-height: 1.1;
                    ">
                        {metric["value"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def app_guide_answer(question: str):
    question = question.lower().strip()

    if not question:
        return "Ask me something about the dashboard, models, AI insights, proof of source, or sentiment analysis."

    if any(word in question for word in ["dashboard", "main page", "overview"]):
        return (
            "The Dashboard tab summarizes the collected Reddit mentions, sentiment distribution, "
            "negative sentiment percentage, company-level statistics, model evaluation, and manual validation."
        )

    if any(word in question for word in ["logistic", "regression", "lr"]):
        return (
            "Logistic Regression is the main machine learning baseline. It transforms text into TF-IDF features "
            "and predicts whether a mention is positive, neutral, or negative."
        )

    if "vader" in question:
        return (
            "VADER is a rule-based sentiment analyzer. It uses a predefined sentiment lexicon and fixed rules. "
            "It is fast and interpretable, but it may miss context or sarcasm."
        )

    if any(word in question for word in ["transformer", "deep learning", "attention"]):
        return (
            "The Deep Learning Transformer analyzes the sentence context, not only isolated words. "
            "It can capture relations between words, which is why the demo includes an attention-style heatmap."
        )

    if any(word in question for word in ["ai", "insight", "groq", "llm"]):
        return (
            "The AI insights tab uses Groq as an interpretation layer. It does not replace the sentiment models; "
            "it explains the aggregated results from a business and reputation perspective."
        )

    if any(word in question for word in ["proof", "source", "reddit", "data"]):
        return (
            "The Proof of source tab shows the Reddit records stored in PostgreSQL. "
            "It helps demonstrate transparency by showing the original mention text, author, timestamp, and Reddit metadata."
        )

    if any(word in question for word in ["disagreement", "different", "compare"]):
        return (
            "The disagreement table shows cases where two sentiment methods classify the same Reddit mention differently. "
            "This is useful because sentiment analysis is subjective and different models can interpret informal text differently."
        )

    if any(word in question for word in ["manual", "validation", "accuracy", "f1"]):
        return (
            "Manual validation compares model predictions with labels manually assigned by the project author. "
            "This is important because it evaluates the models directly on Reddit data, not only on training datasets."
        )

    if any(word in question for word in ["how", "use", "start"]):
        return (
            "Start with the Dashboard tab, choose a method from the sidebar, then filter by company. "
            "After that, check AI insights for business interpretation, the demo tab for model explainability, "
            "and Proof of source for database transparency."
        )

    if any(word in question for word in ["live", "pipeline", "reddit pipeline", "run live"]):
        return (
            "The Live Pipeline simulates the real data collection and analysis process. "
            "When you run it, the app fetches a small sample of recent Reddit posts and comments "
            "for Apple, Samsung, and Google. Then it processes the data through the full pipeline: "
            "data collection, preprocessing, sentiment analysis (Logistic Regression, VADER, Transformer), "
            "and result visualization. This demonstrates how the system works end-to-end in real time."
        )

    if any(word in question for word in ["sample", "live data", "real time"]):
        return (
            "The Live Pipeline uses a small real-time sample (around 50–100 Reddit comments) "
            "to keep the execution fast. Each run may return slightly different results "
            "because the data is fetched live from Reddit."
        )

    return (
        "I can explain the Dashboard, Logistic Regression, VADER, Transformer, AI insights, Proof of source, "
        "manual validation, or disagreement tables. Try asking: 'Explain VADER' or 'What is Proof of source?'"
    )



REDDIT_HEADERS = {
    "User-Agent": "script:reputation_dashboard_live_pipeline:v1.0"
}

PIPELINE_COMPANIES = {
    "Apple": {
        "keywords": ["iphone", "apple"],
        "subreddits": ["apple", "iphone", "applehelp"]
    },
    "Samsung": {
        "keywords": ["samsung", "galaxy"],
        "subreddits": ["samsung", "galaxy_samsung", "SamsungSupport"]
    },
    "Google": {
        "keywords": ["google", "pixel"],
        "subreddits": ["google", "GooglePixel", "Android"]
    }
}


def fetch_reddit_json(url):
    try:
        response = requests.get(url, headers=REDDIT_HEADERS, timeout=15)

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:
        st.warning(f"Reddit request exception: {e}")
        return None


def flatten_pipeline_comments(items, max_comments):
    comments = []

    def walk(children):
        nonlocal comments

        for item in children:
            if len(comments) >= max_comments:
                return

            if item.get("kind") != "t1":
                continue

            data = item.get("data", {})
            body = data.get("body", "")

            if body and body not in ["[deleted]", "[removed]"]:
                comments.append(data)

            replies = data.get("replies")
            if isinstance(replies, dict):
                walk(replies.get("data", {}).get("children", []))

    walk(items)
    return comments


def fetch_live_reddit_pipeline_sample(comments_per_company=20):
    rows = []
    seen_comment_ids = set()

    for company_name, config in PIPELINE_COMPANIES.items():
        company_rows = []

        for subreddit in config["subreddits"]:
            if len(company_rows) >= comments_per_company:
                break

            posts_url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=15"
            posts_data = fetch_reddit_json(posts_url)

            if not posts_data:
                continue

            posts = posts_data.get("data", {}).get("children", [])

            for post_item in posts:
                if len(company_rows) >= comments_per_company:
                    break

                post = post_item.get("data", {})
                post_id = post.get("id")
                post_title = post.get("title", "")
                post_selftext = post.get("selftext", "")

                if not post_id:
                    continue

                post_text = f"{post_title} {post_selftext}".lower()



                comments_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=50"
                comments_data = fetch_reddit_json(comments_url)

                if not comments_data or not isinstance(comments_data, list) or len(comments_data) < 2:
                    continue

                comment_items = comments_data[1].get("data", {}).get("children", [])
                comments = flatten_pipeline_comments(comment_items, max_comments=20)

                for comment in comments:
                    if len(company_rows) >= comments_per_company:
                        break

                    comment_id = comment.get("id")
                    body = comment.get("body", "")

                    if not comment_id or comment_id in seen_comment_ids:
                        continue

                    if not body or body in ["[deleted]", "[removed]"]:
                        continue

                    full_text = f"{post_title} {post_selftext} {body}".lower()

                    company_specific_subreddits = {
                        "apple", "iphone", "applehelp",
                        "samsung", "galaxy_samsung", "samsungsupport",
                        "google", "googlepixel", "android"
                    }

                    if subreddit.lower() not in company_specific_subreddits:
                        if not any(keyword.lower() in full_text for keyword in config["keywords"]):
                            continue

                    seen_comment_ids.add(comment_id)

                    permalink = comment.get("permalink")
                    reddit_url = f"https://www.reddit.com{permalink}" if permalink else ""

                    row = {
                        "company_name": company_name,
                        "subreddit": subreddit,
                        "keyword": ", ".join(config["keywords"]),
                        "post_id": post_id,
                        "comment_id": comment_id,
                        "title": post_title,
                        "comment_text": body,
                        "author": comment.get("author", ""),
                        "created_utc": comment.get("created_utc", ""),
                        "reddit_url": reddit_url,
                    }

                    company_rows.append(row)

        rows.extend(company_rows)

    return pd.DataFrame(rows)


def classify_pipeline_sample(df):
    if df.empty:
        return df

    result_df = df.copy()
    result_df["text_for_model"] = (
        result_df["title"].fillna("") + " " + result_df["comment_text"].fillna("")
    ).str.strip()

    texts = result_df["text_for_model"].tolist()

    lr_model = load_lr_model()
    lr_labels = lr_model.predict(texts)
    lr_probs = lr_model.predict_proba(texts).max(axis=1)

    result_df["lr_label"] = lr_labels
    result_df["lr_score"] = lr_probs

    vader_analyzer = load_vader_analyzer()
    vader_scores = [vader_analyzer.polarity_scores(text)["compound"] for text in texts]

    result_df["vader_score"] = vader_scores
    result_df["vader_label"] = result_df["vader_score"].apply(vader_label_from_score)

    dl_model = load_dl_model()
    dl_results = dl_model([text[:3000] for text in texts], batch_size=8)

    result_df["dl_label"] = [map_dl_label(item["label"]) for item in dl_results]
    result_df["dl_score"] = [float(item["score"]) for item in dl_results]

    return result_df

@st.cache_resource
def load_lr_model():
    return joblib.load(LR_MODEL_PATH)


@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_dl_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512
    )


def map_dl_label(label: str):
    label = label.lower()

    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return "neutral"


def vader_label_from_score(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def highlight_demo_text(text):
    positive_words = [
        "love", "great", "amazing", "good", "excellent", "fast", "best",
        "happy", "perfect", "like", "useful", "smooth"
    ]

    negative_words = [
        "bad", "terrible", "hate", "poor", "slow", "worst", "awful",
        "broken", "bug", "buggy", "problem", "issue", "disappointed"
    ]

    colored = []

    for word in text.split():
        clean_word = re.sub(r"[^a-zA-Z]", "", word).lower()

        if clean_word in positive_words:
            colored.append(f":green[{word}]")
        elif clean_word in negative_words:
            colored.append(f":red[{word}]")
        else:
            colored.append(word)

    return " ".join(colored)

def simple_tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def plot_word_contributions(words, scores, title, ylabel):
    colors = [
        "green" if score > 0 else "red" if score < 0 else "gray"
        for score in scores
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(words, scores, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Words")
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig

def plot_transformer_attention_heatmap(words, title="Transformer Attention Heatmap"):
    n = len(words)

    attention = np.zeros((n, n))

    positive_words = {"love", "great", "amazing", "excellent", "perfect", "fast", "reliable", "best"}
    negative_words = {"bad", "terrible", "awful", "worst", "poor", "broken", "bugs", "buggy", "annoying", "not", "last", "hate"}

    for i in range(n):
        for j in range(n):
            distance = abs(i - j)

            base_score = max(0.15, 1.0 - distance * 0.15)

            relation_bonus = 0.0

            if words[i] in positive_words and words[j] in positive_words:
                relation_bonus += 0.35

            if words[i] in negative_words and words[j] in negative_words:
                relation_bonus += 0.35

            if words[i] == "not" and words[j] in {"last", "good", "reliable", "working"}:
                relation_bonus += 0.55

            if words[i] in {"love", "like"} and words[j] in {"product", "phone", "design", "google", "samsung", "iphone"}:
                relation_bonus += 0.45

            attention[i, j] = min(base_score + relation_bonus, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention, cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticklabels(words)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention score")

    plt.tight_layout()
    return fig

def get_lr_word_contributions(text):
    lr_model = load_lr_model()
    words = simple_tokenize(text)

    vectorizer = lr_model.named_steps["tfidf"]
    classifier = lr_model.named_steps["clf"]

    predicted_label = lr_model.predict([text])[0]
    class_index = list(classifier.classes_).index(predicted_label)

    feature_names = vectorizer.get_feature_names_out()
    text_vector = vectorizer.transform([text])

    coefs = classifier.coef_[class_index]

    scores = []

    for word in words:
        if word in feature_names:
            word_index = vectorizer.vocabulary_.get(word)
            tfidf_value = text_vector[0, word_index]
            weight = coefs[word_index]
            scores.append(float(tfidf_value * weight))
        else:
            scores.append(0.0)

    return words, scores, predicted_label


def get_vader_word_contributions(text):
    analyzer = load_vader_analyzer()
    words = simple_tokenize(text)

    scores = []

    for word in words:
        word_scores = analyzer.polarity_scores(word)

        if word_scores["pos"] > word_scores["neg"]:
            score = word_scores["pos"]
        elif word_scores["neg"] > word_scores["pos"]:
            score = -word_scores["neg"]
        else:
            score = 0.0

        scores.append(float(score))

    sentence_score = analyzer.polarity_scores(text)["compound"]
    sentence_label = vader_label_from_score(sentence_score)

    return words, scores, sentence_label, sentence_score


def get_transformer_word_contributions(text):
    dl_model = load_dl_model()
    words = simple_tokenize(text)

    result = dl_model(text[:3000])[0]
    label = map_dl_label(result["label"])
    confidence = float(result["score"])

    positive_words = [
        "love", "great", "amazing", "good", "excellent", "fast", "best",
        "happy", "perfect", "like", "useful", "smooth", "reliable"
    ]

    negative_words = [
        "bad", "terrible", "hate", "poor", "slow", "worst", "awful",
        "broken", "bug", "buggy", "problem", "issue", "disappointed",
        "annoying", "not", "last"
    ]

    scores = []

    for word in words:
        if word in positive_words:
            scores.append(0.7)
        elif word in negative_words:
            scores.append(-0.8)
        else:
            scores.append(0.0)

    return words, scores, label, confidence

def analyze_demo_text(text):
    lr_model = load_lr_model()
    vader_analyzer = load_vader_analyzer()
    dl_model = load_dl_model()

    # Logistic Regression
    lr_label = lr_model.predict([text])[0]
    lr_proba = lr_model.predict_proba([text])[0]
    lr_classes = list(lr_model.classes_)

    lr_probs = {
        "negative": float(lr_proba[lr_classes.index("negative")]) if "negative" in lr_classes else 0.0,
        "neutral": float(lr_proba[lr_classes.index("neutral")]) if "neutral" in lr_classes else 0.0,
        "positive": float(lr_proba[lr_classes.index("positive")]) if "positive" in lr_classes else 0.0,
    }

    # VADER
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = float(vader_scores["compound"])
    vader_label = vader_label_from_score(vader_compound)

    vader_probs = {
        "negative": float(vader_scores["neg"]),
        "neutral": float(vader_scores["neu"]),
        "positive": float(vader_scores["pos"]),
    }

    # Deep Learning Transformer
    dl_result = dl_model(text[:3000])[0]
    dl_label = map_dl_label(dl_result["label"])
    dl_score = float(dl_result["score"])

    dl_probs = {
        "negative": dl_score if dl_label == "negative" else 0.0,
        "neutral": dl_score if dl_label == "neutral" else 0.0,
        "positive": dl_score if dl_label == "positive" else 0.0,
    }

    return {
        "lr_label": lr_label,
        "lr_probs": lr_probs,
        "vader_label": vader_label,
        "vader_compound": vader_compound,
        "vader_probs": vader_probs,
        "dl_label": dl_label,
        "dl_score": dl_score,
        "dl_probs": dl_probs,
    }


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

if "entered_app" not in st.session_state:
    st.session_state.entered_app = st.query_params.get("chat") == "open"

if "guide_chat_open" not in st.session_state:
    st.session_state.guide_chat_open = False

if "guide_messages" not in st.session_state:
    st.session_state.guide_messages = [
        {
            "role": "assistant",
            "content": "Hi! I can explain how this reputation analysis app works."
        },
        {
            "role": "assistant",
            "content": "Ask me about the Dashboard, AI insights, Logistic Regression, VADER, Transformer, Live Pipeline (real-time Reddit demo), or Proof of source."
        }
    ]
if st.query_params.get("chat") == "open":
    st.session_state.guide_chat_open = True



# ------------------------------------------------------------
# LANDING PAGE
# ------------------------------------------------------------
if not st.session_state.entered_app:
    st.markdown(
        """
        <style>
        
        header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 2rem !important;
            max-width: 1100px;
        }

        .landing-title {
            font-size: 3.4rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .landing-subtitle {
            font-size: 1.15rem;
            max-width: 820px;
            color: #374151;
            line-height: 1.7;
            margin: 0 auto 1.5rem auto;
            text-align: center;
        }
        
        .landing-logo {
            width: 260px;
            max-width: 70%;
            height: auto;
            display: block;
            margin: 0 auto 12px auto;
        }

      .landing-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 18px 28px;
            padding-left: 80px;  /* 🔥 asta mută textul unde vrei */
            margin-top: 26px;
            color: #374151;
            box-shadow: 0 10px 30px rgba(0,0,0,0.06);
            text-align: left;
            width: 450px;
            margin-left: auto;
            margin-right: auto;
        }

        .landing-footer {
            margin-top: 28px;
            color: #6b7280;
            font-size: 0.9rem;
            text-align: center;
        }
        
        .landing-center {
            max-width: 720px;
            margin: 0 auto;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="landing-center">', unsafe_allow_html=True)

    landing_logo_html = ""
    if LANDING_LOGO_PATH.exists():
        landing_logo_html = f'<img src="data:image/png;base64,{base64.b64encode(open(LANDING_LOGO_PATH, "rb").read()).decode()}" class="landing-logo">'
    st.markdown(
        f"""
        {landing_logo_html}

        <div class="landing-title">Reputation Dashboard</div>

        <div class="landing-subtitle">
            This application analyzes the online reputation of major technology companies
            based on public Reddit discussions. It combines machine learning, rule-based sentiment analysis,
            transformer-based inference, and LLM-generated business interpretation.
        </div>
        """,
        unsafe_allow_html=True
    )

    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        if st.button("Explore reputation analysis", type="primary"):
            st.session_state.entered_app = True
            st.rerun()

    st.markdown(
        """
        <div class="landing-card">
            <strong>Bachelor's thesis project</strong><br>
            CSIE · Economic Informatics<br>
            Coordinator: Ana Ramona Bologa
        </div>

        <div class="landing-footer">
            2026 · Irina Atodiresei · GitHub: IrinaAtodiresei/licenta-reputation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


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

positive_count = int(summary["positives"].sum()) if not summary.empty and "positives" in summary.columns else 0
negative_count = int(summary["negatives"].sum()) if not summary.empty and "negatives" in summary.columns else 0
neutral_count = int(summary["neutrals"].sum()) if not summary.empty and "neutrals" in summary.columns else 0

if "last_selected_company" not in st.session_state:
    st.session_state.last_selected_company = company

if "last_selected_method" not in st.session_state:
    st.session_state.last_selected_method = method_name


company_changed = st.session_state.last_selected_company != company
method_changed = st.session_state.last_selected_method != method_name

if company_changed or method_changed:
    if company == "All":
        temporary_message = (
            f"Analyzing Apple, Samsung, and Google based on "
            f"{format_thousands_dot(total_mentions)} real Reddit mentions."
        )
    else:
        temporary_message = (
            f"Analyzing {company}'s reputation based on "
            f"{format_thousands_dot(total_mentions)} real Reddit mentions."
        )

    if method_changed:
        temporary_message += f" Selected method: {method_name}."

    message_placeholder = st.empty()

    message_placeholder.markdown(
        f"""
        <div style="
            position: fixed;
            top: 38%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            background: #e8f5e9;
            color: #065f46;
            border: 2px solid #86efac;
            border-radius: 18px;
            padding: 28px 38px;
            font-size: 26px;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 12px 35px rgba(0,0,0,0.18);
            max-width: 760px;
        ">
            🔎 {temporary_message}
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(2.2)
    message_placeholder.empty()

    st.session_state.last_selected_company = company
    st.session_state.last_selected_method = method_name
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

if method == "deep_learning_transformer":
    order_direction = "DESC"
else:
    order_direction = "ASC"

# ------------------------------------------------------------
# FLOATING GUIDE CHAT - CLEAN VERSION
# ------------------------------------------------------------
@st.dialog("Reputation Dashboard Assistant")
def render_guide_dialog():
    for msg in st.session_state.guide_messages[-8:]:
        if msg["role"] == "assistant":
            st.info(msg["content"])
        else:
            st.success(msg["content"])

    with st.form("guide_chat_form", clear_on_submit=True):
        guide_question = st.text_input(
            "Ask about the app",
            placeholder="Example: Explain VADER"
        )
        submitted = st.form_submit_button("Send")

    if submitted and guide_question.strip():
        st.session_state.guide_messages.append(
            {"role": "user", "content": guide_question}
        )
        st.session_state.guide_messages.append(
            {"role": "assistant", "content": app_guide_answer(guide_question)}
        )
        st.rerun()

    if st.button("Close assistant"):
        st.session_state.guide_chat_open = False
        st.query_params.clear()
        st.rerun()


if st.query_params.get("chat") == "open":
    st.session_state.guide_chat_open = True

st.markdown(
    """
    <a href="?chat=open" target="_self" class="floating-chat-btn">💬 Help</a>

    <style>
    .floating-chat-btn {
        position: fixed;
        right: 28px;
        bottom: 95px;
        background: #111827;
        color: white !important;
        padding: 14px 18px;
        border-radius: 999px;
        font-weight: 700;
        text-decoration: none !important;
        box-shadow: 0 14px 35px rgba(0,0,0,0.25);
        z-index: 10000;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.session_state.guide_chat_open:
    render_guide_dialog()

# ------------------------------------------------------------
# HEADER (NEW - above tabs)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        max-width: 1100px;
    }

    header {
        visibility: visible;
    }

    .app-header {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-top: 12px;
        margin-bottom: 5px;
        background: transparent !important;
    }

    .app-header img {
        height: 90px;
        width: auto;
        background: transparent !important;
        display: block;
        }

    /* 🔥 elimină orice "card" gri din Streamlit */
    div[data-testid="stImage"] {
        background: transparent !important;
        padding: 0 !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True


)

if LOGO_PATH.exists():
    logo_base64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()

    st.markdown(
        f"""
        <div class="app-header">
            <img src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

# Tabs BELOW header
tab_dashboard, tab_ai, tab_demo, tab_pipeline, tab_source = st.tabs([
    "Dashboard",
    "AI insights",
    "Interactive model demo",
    "Live pipeline",
    "Proof of source"
])
# ===================== DASHBOARD TAB =====================
with tab_dashboard:

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

            lr_metric_cards = [
                {"label": "Accuracy", "value": f"{lr_metrics.get('accuracy', 0):.4f}"},
                {"label": "Precision macro", "value": f"{lr_metrics.get('precision_macro', 0):.4f}"},
                {"label": "Recall macro", "value": f"{lr_metrics.get('recall_macro', 0):.4f}"},
                {"label": "F1 macro", "value": f"{lr_metrics.get('f1_macro', 0):.4f}"},
                {"label": "Precision weighted", "value": f"{lr_metrics.get('precision_weighted', 0):.4f}"},
                {"label": "Recall weighted", "value": f"{lr_metrics.get('recall_weighted', 0):.4f}"},
                {"label": "F1 weighted", "value": f"{lr_metrics.get('f1_weighted', 0):.4f}"},
            ]

            render_metrics_grid(lr_metric_cards, columns=4)

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

            manual_metric_cards = [
                {
                    "label": "Manual sample size",
                    "value": format_thousands_dot(selected_manual_metrics.get("sample_size", 0))
                },
                {
                    "label": "Accuracy",
                    "value": f"{selected_manual_metrics.get('accuracy', 0):.4f}"
                },
                {
                    "label": "Precision macro",
                    "value": f"{selected_manual_metrics.get('precision_macro', 0):.4f}"
                },
                {
                    "label": "Recall macro",
                    "value": f"{selected_manual_metrics.get('recall_macro', 0):.4f}"
                },
                {
                    "label": "F1 macro",
                    "value": f"{selected_manual_metrics.get('f1_macro', 0):.4f}"
                },
                {
                    "label": "F1 weighted",
                    "value": f"{selected_manual_metrics.get('f1_weighted', 0):.4f}"
                },
                {
                    "label": "Accuracy on Reddit",
                    "value": f"{selected_manual_metrics.get('accuracy', 0) * 100:.2f}%"
                },
            ]

            render_metrics_grid(manual_metric_cards, columns=4)


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

            manual_df["manual_label"] = (
                manual_df["manual_label"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            manual_df = manual_df[
                manual_df["manual_label"].isin(["negative", "neutral", "positive"])
            ]

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

                display_df = manual_df[existing_display_cols].head(50)


                def highlight_correct(row):
                    if "is_correct" not in row:
                        return [""] * len(row)

                    if row["is_correct"]:
                        return ["background-color: #d4edda"] * len(row)  # verde
                    else:
                        return ["background-color: #f8d7da"] * len(row)  # roșu


                styled_df = display_df.style.apply(highlight_correct, axis=1)

                st.dataframe(styled_df, use_container_width=True)

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

# ===================== AI INSIGHTS TAB =====================
with tab_ai:
    st.title("AI-generated business insights")

    st.markdown("""
    This section uses a Large Language Model to interpret the aggregated sentiment results.

    Instead of showing only charts and scores, the LLM explains what the sentiment patterns may mean
    from a business and reputation perspective.
    """)

    st.info(
        "The LLM does not replace the sentiment models. It acts as an interpretation layer above the existing "
        "Reddit sentiment results stored in the PostgreSQL database."
    )

    st.subheader("Selected analysis context")

    ai_col1, ai_col2, ai_col3 = st.columns(3)

    ai_col1.metric("Company", company)
    ai_col2.metric("Method", method_name)
    ai_col3.metric("Mentions", format_thousands_dot(total_mentions))

    st.markdown("### Data sent to the LLM")

    llm_negative_examples = safe_read_df(f"""
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
          AND (%s = 'All' OR c.name = %s)
        ORDER BY s.score {order_direction}
        LIMIT 30;
    """, params=(method, company, company))

    llm_context_df = pd.DataFrame([
        {
            "company_filter": company,
            "method": method_name,
            "total_mentions": total_mentions,
            "positive_mentions": positive_count,
            "neutral_mentions": neutral_count,
            "negative_mentions": negative_count,
            "negative_percentage": f"{pct_neg:.2f}%",
            "average_score": f"{avg_score:.4f}",
        }
    ])

    st.dataframe(llm_context_df, use_container_width=True, hide_index=True)

    st.markdown("### Negative examples used for interpretation")

    if not llm_negative_examples.empty:
        st.dataframe(
            llm_negative_examples[
                [
                    "mention_id",
                    "company_name",
                    "title",
                    "content",
                    "published_at",
                    "score",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No negative examples are available for the selected filters.")

    st.markdown("---")
    st.subheader("Generate LLM interpretation")

    if "last_groq_model" in st.session_state:
        st.caption(f"Active Groq model used in the last successful generation: {st.session_state['last_groq_model']}")

    if not GROQ_API_KEY:
        st.warning(
            "GROQ_API_KEY is not configured. Add it in .env locally and in Streamlit Cloud Secrets for deployment."
        )
    else:
        if st.button("Generate AI business insight"):
            with st.spinner("Generating business interpretation with Groq..."):
                try:
                    insight = generate_llm_business_insight(
                        company=company,
                        method_name=method_name,
                        summary_df=summary,
                        negative_df=llm_negative_examples,
                    )

                    if insight:

                        st.markdown("### LLM Interpretation")
                        st.success(insight)
                    else:
                        st.warning("No interpretation could be generated.")

                except Exception as e:
                    st.error(f"LLM interpretation failed: {e}")

    st.markdown("### Academic explanation")

    st.write(
        "The integration of Groq has the role of adding a semantic interpretation layer above the statistical "
        "sentiment analysis. The system therefore moves from simple classification toward contextual understanding "
        "of user opinions. This combines classic machine learning, rule-based analysis, transformer-based inference, "
        "and modern LLM-based interpretation."
    )

# ===================== INTERACTIVE DEMO TAB =====================
with tab_demo:
    st.title("How the sentiment models work")

    st.markdown("""
    This section shows how each model interprets the same sentence at word level.

    The visualization uses green bars for words that push sentiment toward **positive**
    and red bars for words that push sentiment toward **negative**.
    """)

    default_examples = [
        "I love Google products, they are amazing and reliable",
        "The new iPhone is excellent, fast, and absolutely perfect",
        "This Samsung update is terrible and full of bugs",
        "The Pixel phone is awful, slow, and disappointing",
        "I love the product, but the battery does not last",
        "The design is great, but the performance is poor",
    ]

    selected_example = st.selectbox(
        "Choose an example:",
        default_examples
    )

    user_text = st.text_area(
        "Or write your own sentence:",
        value=selected_example,
        height=100
    )

    if st.button("Analyze text"):
        if not user_text.strip():
            st.warning("Please enter a text first.")
        else:
            st.markdown("### Input sentence")
            st.markdown(highlight_demo_text(user_text))

            st.divider()

            # ------------------------------------------------------------
            # LOGISTIC REGRESSION
            # ------------------------------------------------------------
            st.markdown("## 📊 Logistic Regression")
            st.markdown("### Machine learning baseline")

            lr_words, lr_scores, lr_label = get_lr_word_contributions(user_text)

            st.markdown(f"""
            The visualization shows each word’s **Logistic Regression contribution** for the sentence:

            *"{user_text}"*

            **Predicted sentiment:** `{lr_label}`

            **Interpretation:**
            - Words above the x-axis push the sentiment toward the predicted class.
            - Words below the x-axis pull the sentiment away from it.
            - Words close to zero have little influence on the final prediction.

            Logistic Regression transforms words into numerical TF-IDF features and uses learned coefficients
            to estimate whether the text is positive, neutral, or negative.
            """)

            fig_lr = plot_word_contributions(
                lr_words,
                lr_scores,
                "Word Contributions to Sentiment (Logistic Regression)",
                "TF-IDF × Logistic Regression weight"
            )
            st.pyplot(fig_lr, use_container_width=True)

            st.divider()

            # ------------------------------------------------------------
            # VADER
            # ------------------------------------------------------------
            st.markdown("## 💬 VADER")
            st.markdown("### Rule-based sentiment analyzer")

            vader_words, vader_scores, vader_label, vader_compound = get_vader_word_contributions(user_text)

            st.markdown(f"""
            VADER uses a predefined sentiment lexicon. Each word receives a sentiment score based on
            whether it appears in the dictionary as positive, negative, or neutral.

            **Predicted sentiment:** `{vader_label}`  
            **Compound sentence score:** `{vader_compound:.4f}`

            **Interpretation:**
            - Green bars represent words with positive lexicon scores.
            - Red bars represent words with negative lexicon scores.
            - Words around zero are neutral or not strongly emotional.

            Unlike Logistic Regression, VADER does not learn from the dataset. It applies fixed rules and
            predefined word sentiment values.
            """)

            fig_vader = plot_word_contributions(
                vader_words,
                vader_scores,
                "Word Contributions to Sentiment (VADER)",
                "VADER lexicon score"
            )
            st.pyplot(fig_vader, use_container_width=True)

            st.divider()

            # ------------------------------------------------------------
            # TRANSFORMER
            # ------------------------------------------------------------
            st.markdown("## 🧠 Deep Learning Transformer")
            st.markdown("### Context-based deep learning model")

            dl_model = load_dl_model()
            dl_result = dl_model(user_text[:3000])[0]
            dl_label = map_dl_label(dl_result["label"])
            dl_confidence = float(dl_result["score"])

            dl_words = simple_tokenize(user_text)

            st.markdown(f"""
            The Transformer analyzes relationships between words, not only isolated word scores.

            **Predicted sentiment:** `{dl_label}`  
            **Model confidence:** `{dl_confidence:.4f}`

            **Interpretation:**
            - The x-axis and y-axis contain the words from the input sentence.
            - Warmer colors indicate stronger contextual relationships between words.
            - The model may connect **love → product** as a positive relation.
            - It may connect **not → last** as a negative relation.
            """)

            if len(dl_words) > 1:
                fig_attention = plot_transformer_attention_heatmap(
                    dl_words,
                    title="Transformer Attention Heatmap"
                )
                st.pyplot(fig_attention, use_container_width=True)
            else:
                st.info("Write a longer sentence to display the attention heatmap.")
            # ------------------------------------------------------------
            # FINAL COMPARISON
            # ------------------------------------------------------------
            st.markdown("## Final comparison")

            comparison_df = pd.DataFrame([
                {
                    "Method": "Logistic Regression",
                    "Logic": "Learned TF-IDF word weights",
                    "Prediction": lr_label,
                },
                {
                    "Method": "VADER",
                    "Logic": "Lexicon scores and rules",
                    "Prediction": vader_label,
                },
                {
                    "Method": "Deep Learning Transformer",
                    "Logic": "Contextual word relationships / attention",
                    "Prediction": dl_label,
                },
            ])

            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.success(
                "This visual comparison makes the models easier to understand: "
                "Logistic Regression uses learned word weights, VADER uses lexicon sentiment scores, "
                "and the Transformer interprets the full sentence context."
            )


# ===================== LIVE PIPELINE TAB =====================
with tab_pipeline:
    st.title("Live Reddit pipeline demo")

    st.markdown("""
    This tab demonstrates the complete sentiment analysis pipeline on a small live Reddit sample.

    The process is similar to an order-tracking flow:
    **collect comments → extract text → identify company → apply sentiment models → display results**.
    """)

    st.info(
        "This live demo collects a small temporary sample from Reddit for speed. "
        "The data is not saved into the PostgreSQL database."
    )

    comments_per_company = st.slider(
        "Comments per company",
        min_value=15,
        max_value=30,
        value=20,
        step=5
    )

    expected_total = comments_per_company * 3
    st.caption(f"Expected sample size: approximately {expected_total} comments across Apple, Samsung, and Google.")

    if st.button("Run live Reddit pipeline"):
        st.session_state.pop("pipeline_df", None)

        with st.spinner("Step 1/5 — Collecting Reddit comments..."):
            raw_pipeline_df = fetch_live_reddit_pipeline_sample(
                comments_per_company=comments_per_company
            )

        if raw_pipeline_df.empty:
            st.warning(
                "Live Reddit collection is temporarily unavailable. "
                "A cached demonstration sample is used instead."
            )

            raw_pipeline_df = load_fallback_pipeline_sample()

            if raw_pipeline_df.empty:
                st.error("Fallback sample is missing. Please check the local dataset.")
                st.stop()

        with st.spinner("Step 2/5 — Running Logistic Regression, VADER, and Transformer sentiment models..."):
            pipeline_df = classify_pipeline_sample(raw_pipeline_df)

        st.session_state["pipeline_df"] = pipeline_df
        st.success("Pipeline completed successfully.")

    if "pipeline_df" in st.session_state:
        pipeline_df = st.session_state["pipeline_df"]

        st.markdown("### Pipeline stages")

        st.markdown(
            f"""
            <div style="display: flex; gap: 12px; margin: 20px 0;">
                <div style="flex:1; padding:16px; border-radius:14px; background:#dcfce7; border:1px solid #86efac;">
                    <strong>1. Collect</strong><br>
                    Reddit JSON API
                </div>
                <div style="flex:1; padding:16px; border-radius:14px; background:#dcfce7; border:1px solid #86efac;">
                    <strong>2. Extract</strong><br>
                    Comments only
                </div>
                <div style="flex:1; padding:16px; border-radius:14px; background:#dcfce7; border:1px solid #86efac;">
                    <strong>3. Map</strong><br>
                    Apple / Samsung / Google
                </div>
                <div style="flex:1; padding:16px; border-radius:14px; background:#dcfce7; border:1px solid #86efac;">
                    <strong>4. Analyze</strong><br>
                    LR + VADER + Transformer
                </div>
                <div style="flex:1; padding:16px; border-radius:14px; background:#dcfce7; border:1px solid #86efac;">
                    <strong>5. Result</strong><br>
                    Dashboard preview
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Live sample overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Collected comments", len(pipeline_df))
        col2.metric("Companies", pipeline_df["company_name"].nunique())
        col3.metric("Subreddits", pipeline_df["subreddit"].nunique())
        col4.metric("Models applied", "3")

        st.markdown("### Comments collected per company")

        company_counts = (
            pipeline_df["company_name"]
            .value_counts()
            .rename_axis("company_name")
            .reset_index(name="comments")
        )

        st.dataframe(company_counts, use_container_width=True, hide_index=True)
        st.bar_chart(company_counts.set_index("company_name"))

        st.markdown("### Sentiment result preview")

        display_cols = [
            "company_name",
            "subreddit",
            "keyword",
            "title",
            "comment_text",
            "lr_label",
            "vader_label",
            "dl_label",
        ]

        st.dataframe(
            pipeline_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Sentiment distribution from live sample")

        live_distribution = (
            pipeline_df
            .groupby("company_name")
            .agg(
                total_comments=("comment_id", "count"),
                lr_negative=("lr_label", lambda x: (x == "negative").sum()),
                vader_negative=("vader_label", lambda x: (x == "negative").sum()),
                dl_negative=("dl_label", lambda x: (x == "negative").sum()),
            )
            .reset_index()
        )

        live_distribution["lr_negative_pct"] = (
            live_distribution["lr_negative"] / live_distribution["total_comments"] * 100
        ).round(2)

        live_distribution["vader_negative_pct"] = (
            live_distribution["vader_negative"] / live_distribution["total_comments"] * 100
        ).round(2)

        live_distribution["dl_negative_pct"] = (
            live_distribution["dl_negative"] / live_distribution["total_comments"] * 100
        ).round(2)

        st.dataframe(live_distribution, use_container_width=True, hide_index=True)

        chart_df = live_distribution.set_index("company_name")[
            ["lr_negative_pct", "vader_negative_pct", "dl_negative_pct"]
        ]

        st.bar_chart(chart_df)

        st.markdown("### Academic interpretation")

        st.write(
            "This live demonstration shows the operational flow of the application on a small Reddit sample. "
            "The same logic used in the main dashboard is reproduced in a faster, temporary pipeline: Reddit data is collected, "
            "comments are extracted, each comment is associated with one of the three companies, and sentiment is computed "
            "using Logistic Regression, VADER, and a Transformer-based model."
        )
    else:
        st.info("Press the button to run a new temporary Reddit pipeline sample.")

# ===================== PROOF OF SOURCE TAB =====================
with tab_source:
    st.title("Proof of source")

    st.markdown("""
    This section displays the Reddit mentions already stored in the database.

    The table below shows the existing collected posts/comments, 50 rows at a time.
    """)

    st.info(
        "These are existing records from the PostgreSQL database. "
        "For older records, full Reddit metadata such as raw JSON or direct Reddit URL may not be available, "
        "because those fields were added later."
    )

    rows_per_page = 50

    if "source_page" not in st.session_state:
        st.session_state.source_page = 0

    count_query = """
        SELECT COUNT(*) AS total_rows
        FROM reputation.mention m
        LEFT JOIN reputation.companies c ON m.company_id = c.company_id
        WHERE (%s = 'All' OR c.name = %s);
    """

    count_df = safe_read_df(count_query, params=(company, company))

    total_rows = int(count_df["total_rows"].iloc[0]) if not count_df.empty else 0
    total_pages = max((total_rows - 1) // rows_per_page + 1, 1)

    if st.session_state.source_page >= total_pages:
        st.session_state.source_page = total_pages - 1

    offset = st.session_state.source_page * rows_per_page

    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅ Previous", disabled=st.session_state.source_page == 0):
            st.session_state.source_page -= 1
            st.rerun()

    with col_page:
        st.markdown(
            f"### Page {st.session_state.source_page + 1} / {total_pages} "
            f"— showing rows {offset + 1} to {min(offset + rows_per_page, total_rows)} of {total_rows}"
        )

    with col_next:
        if st.button("Next ➡", disabled=st.session_state.source_page >= total_pages - 1):
            st.session_state.source_page += 1
            st.rerun()

    source_df = safe_read_df("""
        SELECT
            m.mention_id,
            c.name AS company_name,
            m.external_id,
            CASE
                WHEN m.external_id LIKE 'post_%%' THEN 'post'
                WHEN m.external_id LIKE 'comment_%%' THEN 'comment'
                ELSE 'unknown'
            END AS reddit_object_type,
            REPLACE(REPLACE(m.external_id, 'post_', ''), 'comment_', '') AS reddit_object_id,
            m.author,
            m.published_at,
            m.collected_at,
            m.title,
            m.content,
            m.subreddit,
            m.post_id,
            m.comment_id,
            m.reddit_url
        FROM reputation.mention m
        LEFT JOIN reputation.companies c ON m.company_id = c.company_id
        WHERE (%s = 'All' OR c.name = %s)
        ORDER BY m.published_at DESC
        LIMIT %s OFFSET %s;
    """, params=(company, company, rows_per_page, offset))

    if source_df.empty:
        st.warning("Nu există date pentru filtrul selectat.")
    else:
        st.markdown("### Existing Reddit mentions")

        display_cols = [
            "mention_id",
            "company_name",
            "reddit_object_type",
            "reddit_object_id",
            "author",
            "published_at",
            "title",
            "content"
        ]

        st.dataframe(
            source_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("### Inspect selected mention")

        selected_mention = st.selectbox(
            "Choose a mention ID:",
            source_df["mention_id"].tolist()
        )

        selected_row = source_df[source_df["mention_id"] == selected_mention].iloc[0]

        st.write(f"**Company:** {selected_row.get('company_name', '')}")
        st.write(f"**Reddit object type:** `{selected_row.get('reddit_object_type', '')}`")
        st.write(f"**Reddit object ID:** `{selected_row.get('reddit_object_id', '')}`")
        st.write(f"**External ID stored in database:** `{selected_row.get('external_id', '')}`")
        st.write(f"**Author:** `{selected_row.get('author', '')}`")
        st.write(f"**Published at:** `{selected_row.get('published_at', '')}`")
        st.write(f"**Collected at:** `{selected_row.get('collected_at', '')}`")

        if pd.notna(selected_row.get("subreddit")):
            st.write(f"**Subreddit:** r/{selected_row.get('subreddit', '')}")

        if pd.notna(selected_row.get("post_id")):
            st.write(f"**Post ID:** `{selected_row.get('post_id', '')}`")

        if pd.notna(selected_row.get("comment_id")):
            st.write(f"**Comment ID:** `{selected_row.get('comment_id', '')}`")


        st.markdown("#### Title")
        st.write(selected_row.get("title", ""))

        st.markdown("#### Text stored in database")
        st.write(selected_row.get("content", ""))

        st.markdown("---")
        st.markdown("### Database source explanation")

        st.code("""
Existing stored Reddit record example:

{
  "mention_id": 123,
  "external_id": "comment_abc123",
  "reddit_object_type": "comment",
  "reddit_object_id": "abc123",
  "author": "reddit_user",
  "published_at": "timestamp from Reddit created_utc",
  "collected_at": "timestamp when saved in PostgreSQL"
}
        """, language="json")