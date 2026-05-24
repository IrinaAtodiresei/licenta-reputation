# ------------------------------------------------------------
# INTERPRETATION HELPERS (EN + RO)
# ------------------------------------------------------------

def generate_interpretation(
    company,
    method_name,
    total_mentions,
    avg_score,
    pct_negative,
    pct_disagreement,
):
    # Company text
    if company == "All":
        company_text = "for the selected companies"
    else:
        company_text = f"for {company}"

    # Method-specific logic
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

    # Negative share
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
        f"Based on the selected filters, {total_mentions:,} mentions were analyzed "
        f"{company_text} {method_text}. {sentiment_text} {negative_text} {comparison_text}"
    )


# ------------------------------------------------------------
# ROMANIAN VERSION
# ------------------------------------------------------------

def generate_interpretation_ro(
    company,
    method_name,
    total_mentions,
    avg_score,
    pct_negative,
    pct_disagreement,
):
    # Company text
    if company == "All":
        company_text = "pentru companiile selectate"
    else:
        company_text = f"pentru {company}"

    # Method-specific logic
    if method_name == "Logistic Regression 3-class balanced":
        method_text = (
            "folosind modelul Logistic Regression echilibrat pe 3 clase, antrenat pe un set de date de sentiment din Twitter"
        )

        if avg_score >= 0.60:
            sentiment_text = (
                "Modelul prezintă un nivel ridicat al încrederii medii în clasificările sale."
            )
        elif avg_score >= 0.45:
            sentiment_text = (
                "Modelul prezintă un nivel moderat al încrederii medii, adecvat pentru texte scurte și informale de pe Reddit."
            )
        else:
            sentiment_text = (
                "Modelul prezintă o încredere medie scăzută, ceea ce sugerează că multe texte sunt ambigue sau dificil de clasificat."
            )

        comparison_text = (
            "Această vizualizare prezintă Logistic Regression ca metodă de bază. "
            "Analiza dezacordului este afișată separat în vizualizările VADER și Deep Learning Transformer."
        )

    elif method_name == "VADER":
        method_text = "folosind analizorul de sentiment bazat pe reguli VADER"

        if avg_score >= 0.20:
            sentiment_text = (
                "Sentimentul general pare pozitiv pe baza scorului compound VADER."
            )
        elif avg_score >= -0.05:
            sentiment_text = (
                "Sentimentul general este relativ neutru sau mixt pe baza scorului compound VADER."
            )
        else:
            sentiment_text = (
                "Sentimentul general înclină spre negativ pe baza scorului compound VADER."
            )

        comparison_text = (
            f"Dezacordul dintre Logistic Regression și VADER este de {pct_disagreement:.2f}%, "
            "ceea ce arată cât de des modelul statistic și metoda bazată pe reguli clasifică diferit mențiunile de pe Reddit."
        )

    else:
        method_text = "folosind modelul Deep Learning Transformer"

        if avg_score >= 0.60:
            sentiment_text = (
                "Sentimentul general pare clasificat cu un nivel ridicat de încredere de către modelul transformer."
            )
        elif avg_score >= 0.45:
            sentiment_text = (
                "Sentimentul general este mixt, cu un nivel moderat de încredere din partea modelului transformer."
            )
        else:
            sentiment_text = (
                "Modelul transformer prezintă o încredere scăzută, ceea ce poate indica mențiuni ambigue sau dependente de context."
            )

        comparison_text = (
            f"Dezacordul dintre Logistic Regression și Deep Learning Transformer este de {pct_disagreement:.2f}%, "
            "ceea ce evidențiază diferențele dintre un model clasic de machine learning și un model contextual de deep learning."
        )

    # Negative share
    if pct_negative >= 40:
        negative_text = (
            "Ponderea mențiunilor negative este ridicată, ceea ce poate indica riscuri reputaționale vizibile."
        )
    elif pct_negative >= 25:
        negative_text = (
            "Există o pondere vizibilă a mențiunilor negative, dar negativitatea nu este dominantă."
        )
    else:
        negative_text = (
            "Ponderea mențiunilor negative este redusă, ceea ce sugerează un profil reputațional stabil."
        )

    return (
        f"Pe baza filtrelor selectate, au fost analizate {total_mentions:,} mențiuni "
        f"{company_text} {method_text}. {sentiment_text} {negative_text} {comparison_text}"
    )
