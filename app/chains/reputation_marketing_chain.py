import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return None

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=900,
        groq_api_key=groq_api_key,
    )


EXECUTIVE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "method_name",
        "total_mentions",
        "positive_mentions",
        "neutral_mentions",
        "negative_mentions",
        "negative_percentage",
        "average_score",
        "negative_examples",
    ],
    template="""
You are assisting in writing the "Executive Summary" section for the AI Insights tab of a Streamlit reputation dashboard.

The dashboard is used by a marketing department to understand online reputation based on Reddit sentiment analysis.

Analysis scope:
{analysis_scope}

Task:
Rewrite and expand the executive summary using the following structure:
1. General trend – 1–2 sentences about the overall sentiment direction.
2. Dominant themes – 3 main themes, each explained in 1 sentence.
3. Main opportunity – 1 clear sentence focused on marketing or strategic opportunity.

Context:
Company: {company}
Sentiment method: {method_name}
Total mentions: {total_mentions}
Positive mentions: {positive_mentions}
Neutral mentions: {neutral_mentions}
Negative mentions: {negative_mentions}
Negative percentage: {negative_percentage}
Average score/confidence: {average_score}

Negative Reddit examples:
{negative_examples}

Strict rules:
- Write in English.
- If Company is "All", do NOT describe the result as one brand, one company, or one CEO.
- If Company is not "All", write ONLY about the selected company.
- Do not mention Apple, Samsung, or Google unless they are the selected company or Company is "All".
- Do not invent market facts, product facts, financial facts, or events that are not present in the input.
- Base the interpretation only on the metrics and Reddit examples.
- If evidence is limited, say that the conclusion is indicative, not definitive.
- Keep the structure identical.
- Produce only the rewritten text.
"""
)


GENERAL_SENTIMENT_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "method_name",
        "total_mentions",
        "positive_mentions",
        "neutral_mentions",
        "negative_mentions",
        "negative_percentage",
        "average_score",
        "examples",
    ],
    template="""
You are writing the "General Sentiment" card for a marketing reputation dashboard.

Analysis scope:
{analysis_scope}

Task:
Explain the overall public sentiment for the selected analysis scope.

Context:
Company: {company}
Method: {method_name}
Total mentions: {total_mentions}
Positive mentions: {positive_mentions}
Neutral mentions: {neutral_mentions}
Negative mentions: {negative_mentions}
Negative percentage: {negative_percentage}
Average score/confidence: {average_score}

Reddit examples:
{examples}

Strict rules:
- Write in English.
- Maximum 80 words.
- If Company is "All", write about the selected companies as a portfolio, not as one brand.
- If Company is not "All", write ONLY about the selected company.
- Do not mention other companies unless Company is "All".
- Business-oriented tone.
- Explain what the sentiment suggests for reputation perception.
- Do not invent facts.
- Produce only the final card text.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
"""
)


RECURRING_THEMES_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "examples",
    ],
    template="""
You are writing the "Top 3 Recurring Themes" card for a marketing reputation dashboard.

Analysis scope:
{analysis_scope}

Task:
Identify the top 3 recurring discussion themes based only on the Reddit examples below.

Context:
Company: {company}

Reddit examples:
{examples}

Strict rules:
- Write in English.
- Return exactly 3 numbered themes.
- Each theme must have a short explanation.
- If Company is "All", themes may refer to the selected companies as a portfolio.
- If Company is not "All", themes must refer ONLY to the selected company.
- Do not mention other companies unless Company is "All".
- Do not invent themes that are not supported by the examples.
- If the examples are insufficient, mention that the themes are indicative.
- Produce only the final card text.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
"""
)


REPUTATION_RISKS_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "negative_percentage",
        "negative_examples",
    ],
    template="""
You are writing the "Reputation Risks" card for a marketing reputation dashboard.

Analysis scope:
{analysis_scope}

Task:
Identify the main reputation risks based on the negative sentiment percentage and negative Reddit examples.

Context:
Company: {company}
Negative percentage: {negative_percentage}

Negative Reddit examples:
{negative_examples}

Strict rules:
- Write in English.
- Maximum 90 words.
- If Company is "All", discuss risks at portfolio or market-sample level.
- If Company is not "All", discuss risks ONLY for the selected company.
- Do not mention other companies unless Company is "All".
- Focus on concrete reputation risks, not generic statements.
- Do not invent unsupported facts.
- Explain why these risks matter for marketing and brand trust.
- Produce only the final card text.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
"""
)


MARKETING_RECOMMENDATIONS_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "sentiment_context",
        "negative_examples",
    ],
    template="""
You are writing the "Marketing Recommendations" card for a reputation intelligence dashboard.

Analysis scope:
{analysis_scope}

Task:
Generate practical marketing recommendations based only on the sentiment context and Reddit examples.

Sentiment context:
{sentiment_context}

Negative Reddit examples:
{negative_examples}

Strict rules:
- Write in English.
- Maximum 100 words.
- If Company is "All", recommendations must target analysts, stakeholders, or brand managers across the selected companies.
- If Company is not "All", recommendations must target the selected company only.
- Do not mention other companies unless Company is "All".
- Do not write as if there is one CEO when Company is "All".
- Recommendations must be linked to the observed sentiment and risks.
- Do not invent campaign results, financial data, product launches, or market facts.
- Produce only the final card text.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
"""
)


NARRATIVE_INTERPRETATION_PROMPT = PromptTemplate(
    input_variables=[
        "company",
        "analysis_scope",
        "method_name",
        "sentiment_context",
        "examples",
    ],
    template="""
You are writing the "AI-generated Narrative Interpretation" section for a Streamlit reputation dashboard.

Analysis scope:
{analysis_scope}

Task:
Write a narrative business interpretation of the online reputation reflected in the selected analysis scope.

Context:
Company: {company}
Method: {method_name}

Sentiment context:
{sentiment_context}

Reddit examples:
{examples}

Strict rules:
- Write in English.
- 1 coherent paragraph.
- 120–160 words.
- If Company is "All", interpret the results as a market-level or portfolio-level overview across selected companies.
- If Company is not "All", write ONLY about the selected company.
- Do not mention other companies unless Company is "All".
- Business-oriented and suitable for a marketing department.
- Explain what the public perception suggests and how marketing could react.
- Do not invent facts beyond the provided data.
- If the evidence is limited, present conclusions as indicative.
- Produce only the paragraph.
- If Company is NOT "All", you MUST write ONLY about the selected company.
- If Company is NOT "All", mentioning other companies is forbidden.
- Never mention Apple, Samsung, or Google unless they are the selected company.
- If Company is "All", write about the companies collectively as a portfolio.
- Never mix portfolio-level wording with single-company wording.
- Do not invent facts.
"""
)


def run_chain(prompt_template, variables):
    llm = get_llm()

    if llm is None:
        return "GROQ_API_KEY is not configured."

    chain = prompt_template | llm
    response = chain.invoke(variables)

    return response.content


def generate_marketing_ai_insights(context):
    if "analysis_scope" not in context:
        company = context.get("company", "All")

        if company == "All":
            context["analysis_scope"] = (
                "This is a portfolio-level analysis across Apple, Samsung, and Google. "
                "Do not describe the results as one brand, one company, or one CEO."
            )
        else:
            context["analysis_scope"] = (
                f"This analysis focuses ONLY on {company}. "
                "Do not mention other companies. Do not compare brands. "
                "Do not use portfolio-level wording."
            )

    executive_summary = run_chain(EXECUTIVE_SUMMARY_PROMPT, context)
    general_sentiment = run_chain(GENERAL_SENTIMENT_PROMPT, context)
    recurring_themes = run_chain(RECURRING_THEMES_PROMPT, context)
    reputation_risks = run_chain(REPUTATION_RISKS_PROMPT, context)
    marketing_recommendations = run_chain(MARKETING_RECOMMENDATIONS_PROMPT, context)
    narrative_interpretation = run_chain(NARRATIVE_INTERPRETATION_PROMPT, context)

    return {
        "executive_summary": executive_summary,
        "general_sentiment": general_sentiment,
        "recurring_themes": recurring_themes,
        "reputation_risks": reputation_risks,
        "marketing_recommendations": marketing_recommendations,
        "narrative_interpretation": narrative_interpretation,
    }