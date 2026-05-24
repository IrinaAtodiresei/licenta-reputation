# ------------------------------------------------------------
# METHOD NOTE HELPERS (EN + RO)
# ------------------------------------------------------------

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


def generate_method_note_ro(method_name):
    if method_name == "Logistic Regression 3-class balanced":
        return (
            "Această vizualizare folosește un model Logistic Regression echilibrat pe 3 clase, antrenat pe un set de date "
            "de sentiment din Twitter, cu etichete pozitive, neutre și negative. Textele sunt reprezentate prin caracteristici TF‑IDF, "
            "iar clasificarea finală este realizată cu Logistic Regression."
        )

    if method_name == "VADER":
        return (
            "Această vizualizare folosește VADER, un analizor de sentiment bazat pe lexicon și reguli. "
            "Nu necesită antrenarea unui model și este util pentru scorare rapidă a sentimentului, "
            "dar poate interpreta contextul diferit față de modelele de machine learning."
        )

    return (
        "Această vizualizare folosește un model Deep Learning Transformer pre-antrenat pentru analiză de sentiment. "
        "În această aplicație, modelul este utilizat doar pentru inferență pe mențiunile colectate de pe Reddit, fără fine‑tuning local."
    )
