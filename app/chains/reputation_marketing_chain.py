import os
import time
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ============================================================
# LLM INITIALIZER
# ============================================================
def get_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=900,
        groq_api_key=api_key,
    )


# ============================================================
# SANITIZERS — HTML + Markdown + whitespace
# ============================================================
def _safe_text(x):
    if x is None:
        return ""
    x = str(x).strip()
    return "" if x.lower() == "nan" else x


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(x) -> str:
    x = _safe_text(x)
    x = strip_html(x)
    x = strip_markdown(x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


# ============================================================
# PROMPTS — cu regulă anti‑HTML + control de stil
# ============================================================
ANTI_HTML_RULE = """
IMPORTANT:
- Do NOT output HTML, CSS, XML, JSX, or code of any kind.
- Output plain text only.
"""

DETAIL_STYLE_RULE = """
Detail level: {detail_level}
Bullet mode: {bullet_mode}

Rules:
- If bullet_mode is true, respond ONLY with bullet points (one idea per bullet), no paragraphs.
- If bullet_mode is false, respond with short sentences/mini-paragraphs.
- If detail_level is "Low", keep it very concise (2–3 bullets / 2–3 fraze).
- If detail_level is "Medium", oferă un nivel moderat de detaliu.
- If detail_level is "High", poți adăuga câteva detalii suplimentare, dar rămâi clar și structurat.
"""


EXECUTIVE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=[
        "company", "analysis_scope", "method_name",
        "total_mentions", "positive_mentions", "neutral_mentions",
        "negative_mentions", "negative_percentage", "average_score",
        "negative_examples",
        "detail_level", "bullet_mode",
    ],
    template=f"""
You are assisting in writing the Executive Summary for a marketing reputation dashboard.

Analysis scope:
{{analysis_scope}}

Task:
Write an executive summary using this structure:
1. General trend.
2. Dominant themes.
3. Main opportunity.

Context:
Company: {{company}}
Method: {{method_name}}
Total mentions: {{total_mentions}}
Positive: {{positive_mentions}}
Neutral: {{neutral_mentions}}
Negative: {{negative_mentions}}
Negative percentage: {{negative_percentage}}
Average score: {{average_score}}

Negative examples:
{{negative_examples}}

{DETAIL_STYLE_RULE}

Strict rules:
- Write in English.
- No invented facts.
{ANTI_HTML_RULE}
"""
)


GENERAL_SENTIMENT_PROMPT = PromptTemplate(
    input_variables=[
        "company", "analysis_scope", "method_name",
        "total_mentions", "positive_mentions", "neutral_mentions",
        "negative_mentions", "negative_percentage", "average_score",
        "examples",
        "detail_level", "bullet_mode",
    ],
    template=f"""
You are writing the General Sentiment card.

Analysis scope:
{{analysis_scope}}

Task:
Explain the overall public sentiment.

Context:
Company: {{company}}
Method: {{method_name}}
Total mentions: {{total_mentions}}
Positive: {{positive_mentions}}
Neutral: {{neutral_mentions}}
Negative: {{negative_mentions}}
Negative percentage: {{negative_percentage}}
Average score: {{average_score}}

Examples:
{{examples}}

{DETAIL_STYLE_RULE}

Strict rules:
- Write in English.
- Aim for a concise explanation (max ~80 words if bullet_mode is false).
- No invented facts.
{ANTI_HTML_RULE}
"""
)


RECURRING_THEMES_PROMPT = PromptTemplate(
    input_variables=["company", "analysis_scope", "examples", "detail_level", "bullet_mode"],
    template=f"""
You are writing the Top 3 Recurring Themes card.

Analysis scope:
{{analysis_scope}}

Task:
Identify exactly 3 recurring themes based ONLY on the examples.

Examples:
{{examples}}

{DETAIL_STYLE_RULE}

Strict rules:
- Write in English.
- Return exactly 3 themes.
- If bullet_mode is true, one theme per bullet.
- If bullet_mode is false, one short sentence per theme.
- No invented themes.
{ANTI_HTML_RULE}
"""
)


REPUTATION_RISKS_PROMPT = PromptTemplate(
    input_variables=[
        "company", "analysis_scope",
        "negative_percentage", "negative_examples",
        "detail_level", "bullet_mode",
    ],
    template=f"""
You are writing the Reputation Risks card.

Analysis scope:
{{analysis_scope}}

Task:
Identify the main reputation risks based on the negative percentage and examples.

Negative percentage: {{negative_percentage}}

Negative examples:
{{negative_examples}}

{DETAIL_STYLE_RULE}

Strict rules:
- Write in English.
- No invented facts.
{ANTI_HTML_RULE}
"""
)


MARKETING_RECOMMENDATIONS_PROMPT = PromptTemplate(
    input_variables=[
        "company", "analysis_scope",
        "sentiment_context", "negative_examples",
        "detail_level", "bullet_mode",
    ],
    template=f"""
You are writing the Marketing Recommendations card.

Analysis scope:
{{analysis_scope}}

Sentiment context:
{{sentiment_context}}

Negative examples:
{{negative_examples}}

Task:
Generate practical marketing recommendations.

{DETAIL_STYLE_RULE}

Strict rules:
- Write in English.
- Focus on actionable, concrete suggestions.
- No invented facts.
{ANTI_HTML_RULE}
"""
)


# ============================================================
# HELPER
# ============================================================
def _build_analysis_scope(company: str) -> str:
    if company == "All":
        return (
            "This is a portfolio-level analysis across Apple, Samsung, and Google. "
            "Do not describe the results as one brand."
        )
    return (
        f"This analysis focuses ONLY on {company}. "
        "Do not mention other companies."
    )


# ============================================================
# MAIN FUNCTION
# ============================================================
def generate_marketing_ai_insights(
    company: str,
    period: str,
    method_name: str,
    summary_df,
    negative_df,
    model_name: str = "llama-3.3-70b-versatile",
    detail_level: str = "Medium",
    bullet_mode: bool = True,
):

    llm = get_llm(model_name=model_name)

    # ---------- METRICS ----------
    total_mentions = int(summary_df["mentions"].sum()) if "mentions" in summary_df else 0
    positive_mentions = int(summary_df["positives"].sum()) if "positives" in summary_df else 0
    neutral_mentions = int(summary_df["neutrals"].sum()) if "neutrals" in summary_df else 0
    negative_mentions = int(summary_df["negatives"].sum()) if "negatives" in summary_df else 0

    negative_percentage = (negative_mentions / total_mentions * 100) if total_mentions > 0 else 0.0
    average_score = (
        float((summary_df["avg_score"] * summary_df["mentions"]).sum() / total_mentions)
        if total_mentions > 0 and "avg_score" in summary_df
        else 0.0
    )

    # ---------- EXAMPLES ----------
    def build_examples(df):
        if df is None or df.empty:
            return "- No examples available"
        rows = []
        for _, row in df.head(12).iterrows():
            title = clean_text(row.get("title"))
            content = clean_text(row.get("content"))
            text = f"{title} {content}".strip()
            if len(text) > 5:
                rows.append(f"- {text[:400]}")
        return "\n".join(rows) if rows else "- No valid examples available"

    negative_examples = build_examples(negative_df)
    examples = build_examples(summary_df)

    sentiment_context = clean_text(f"""
Company: {company}
Method: {method_name}
Total mentions: {total_mentions}
Positive: {positive_mentions}
Neutral: {neutral_mentions}
Negative: {negative_mentions}
Negative percentage: {negative_percentage:.2f}%
Average score: {average_score:.4f}
""")

    # ---------- CONTEXT ----------
    context = {
        "company": company,
        "analysis_scope": _build_analysis_scope(company),
        "method_name": method_name,
        "total_mentions": total_mentions,
        "positive_mentions": positive_mentions,
        "neutral_mentions": neutral_mentions,
        "negative_mentions": negative_mentions,
        "negative_percentage": negative_percentage,
        "average_score": average_score,
        "negative_examples": negative_examples,
        "examples": examples,
        "sentiment_context": sentiment_context,
        "detail_level": detail_level,
        "bullet_mode": str(bullet_mode),  # ca text pentru prompt
    }

    context = {k: clean_text(v) if isinstance(v, str) else v for k, v in context.items()}

    # ---------- TRACE ----------
    chains_run = []
    timings = {}
    raw_outputs = {}

    def run_chain(name, prompt_obj, ctx):
        start = time.time()
        chain = prompt_obj | llm
        result = chain.invoke(ctx)
        end = time.time()

        raw = result.content
        clean = clean_text(raw)

        chains_run.append(name)
        timings[name] = round(end - start, 3)
        raw_outputs[name] = raw

        return clean

    executive_summary = run_chain("executive_summary", EXECUTIVE_SUMMARY_PROMPT, context)
    general_sentiment = run_chain("general_sentiment", GENERAL_SENTIMENT_PROMPT, context)
    recurring_themes = run_chain("recurring_themes", RECURRING_THEMES_PROMPT, context)
    reputation_risks = run_chain("reputation_risks", REPUTATION_RISKS_PROMPT, context)
    marketing_recommendations = run_chain("marketing_recommendations", MARKETING_RECOMMENDATIONS_PROMPT, context)

    return {
        "executive_summary": executive_summary,
        "general_sentiment": general_sentiment,
        "recurring_themes": recurring_themes,
        "reputation_risks": reputation_risks,
        "marketing_recommendations": marketing_recommendations,
        "sentiment_context": sentiment_context,
        "detail_level": detail_level,
        "bullet_mode": bullet_mode,

        "context": context,
        "raw_output": raw_outputs,
        "timings": timings,
        "chains": chains_run,
        "full_prompt": "Multiple prompts used (one per card). See 'raw_output' for each chain.",
    }
