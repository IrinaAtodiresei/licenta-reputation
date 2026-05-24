import streamlit as st
from groq import Groq
# ============================================================
#  CACHE INTELIGENT PENTRU HELP (RO/EN)
# ============================================================
HELP_CACHE = {}

# ============================================================
#  CLIENT GROQ (CACHE)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ============================================================
#  EXPLICAȚII TEHNICE FIXE (fără Groq)
# ============================================================
TECHNICAL_TOPICS = {
    "Logistic Regression": {
        "RO": (
            "Logistic Regression este un model liniar utilizat pentru clasificarea "
            "probabilistică a textelor în funcție de polaritatea lor. În analiza "
            "sentimentului, modelul operează pe baza vectorizării TF‑IDF, care "
            "transformă textul într-un set de caracteristici numerice. Algoritmul "
            "învață limite de decizie între clasele Pozitiv, Neutru și Negativ prin "
            "optimizarea funcției log‑loss. Modelul se remarcă prin interpretabilitate, "
            "stabilitate și eficiență computațională, fiind adecvat pentru seturi de "
            "date de dimensiune medie."
        ),
        "EN": (
            "Logistic Regression is a linear model used for probabilistic text "
            "classification. In sentiment analysis, it operates on TF‑IDF vectorized "
            "features, learning decision boundaries between Positive, Neutral, and "
            "Negative classes through log‑loss optimization. The model is valued for "
            "its interpretability, stability, and computational efficiency, making it "
            "suitable for medium‑sized datasets."
        ),
    },

    "VADER": {
        "RO": (
            "VADER (Valence Aware Dictionary and sEntiment Reasoner) este un algoritm "
            "lexicon‑based specializat în analiza sentimentului pentru texte scurte și "
            "limbaj informal. Acesta utilizează un lexicon pre‑annotat de termeni cu "
            "polaritate asociată și un set de reguli lingvistice pentru ajustarea "
            "scorurilor, incluzând tratamentul negărilor, intensificatorilor, "
            "majusculelor și semnelor de punctuație. Rezultatul final este compound "
            "score, un scor normalizat în intervalul [-1, 1], care reflectă polaritatea "
            "globală a textului."
        ),
        "EN": (
            "VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon‑based "
            "algorithm designed for sentiment analysis of short and informal text. It "
            "uses a pre‑annotated lexicon and linguistic rules to adjust polarity, "
            "including handling of negations, intensifiers, capitalization, and "
            "punctuation. The final output is the compound score, normalized in the "
            "range [-1, 1], representing the overall sentiment polarity."
        ),
    },

    "Transformer": {
        "RO": (
            "Arhitectura Transformer se bazează pe mecanisme de self‑attention care "
            "permit captarea relațiilor semantice dintre termeni indiferent de poziția "
            "lor în secvență. Această abordare elimină limitările modelelor secvențiale "
            "tradiționale și permite procesarea paralelă a textului. În analiza "
            "sentimentului, modelele Transformer oferă performanțe superioare datorită "
            "capacității lor de a surprinde contextul global și nuanțele lingvistice "
            "complexe."
        ),
        "EN": (
            "The Transformer architecture relies on self‑attention mechanisms that "
            "capture semantic relationships between tokens regardless of their position "
            "in the sequence. This approach removes the constraints of traditional "
            "sequential models and enables parallel text processing. In sentiment "
            "analysis, Transformer models achieve superior performance due to their "
            "ability to model global context and complex linguistic nuances."
        ),
    },
}

# ============================================================
#  FUNCTIA PRINCIPALĂ
# ============================================================
def ask_groq(topic: str, lang: str) -> str:
    lang = lang.upper()
    normalized = topic.strip().lower()

    # 1. CACHE INTELIGENT
    if normalized in HELP_CACHE and lang in HELP_CACHE[normalized]:
        return HELP_CACHE[normalized][lang]

    # 2. EXPLICAȚII TEHNICE FIXE
    if normalized in TECHNICAL_TOPICS:
        answer = TECHNICAL_TOPICS[normalized][lang]
        HELP_CACHE.setdefault(normalized, {})[lang] = answer
        return answer

    # 3. APEL GROQ (doar dacă nu există în cache)
    client = get_groq_client()

    if lang == "RO":
        user_prompt = f"Explică foarte pe scurt secțiunea '{topic}' în 2–3 fraze."
    else:
        user_prompt = f"Briefly explain the '{topic}' section in 2–3 sentences."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a micro‑assistant for a sentiment analysis dashboard. "
                        "Your job is ONLY to explain the selected section in 2–3 sentences. "
                        "No greetings, no questions, no chit‑chat, no marketing tone. "
                        "Be factual, concise, and neutral. "
                        "If the user writes in Romanian, answer in Romanian. "
                        "If the user writes in English, answer in English."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        answer = response.choices[0].message.content.strip()

        # Salvăm în cache
        HELP_CACHE.setdefault(normalized, {})[lang] = answer

        return answer

    except Exception as e:
        return f"⚠️ Eroare Groq: {e}"

