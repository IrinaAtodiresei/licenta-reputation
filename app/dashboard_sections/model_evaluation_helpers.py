# ============================================================
#  MODEL EVALUATION TEXT HELPERS (EN + RO)
# ============================================================

def lr_note_en():
    return (
        "The Logistic Regression model evaluation metrics and confusion matrix are computed on the "
        "3‑class Twitter sentiment dataset, not on the collected Reddit mentions."
    )

def lr_note_ro():
    return (
        "Metricile de evaluare și matricea de confuzie pentru modelul Logistic Regression sunt calculate "
        "pe setul de date Twitter cu 3 clase, nu pe mențiunile colectate de pe Reddit."
    )


def vader_note_en():
    return (
        "VADER is a rule‑based sentiment analyzer, so hyperparameter tuning, "
        "confusion matrix validation, and offline model‑training metrics are not applicable here. "
        "The VADER view is used only for the Reddit sentiment analysis above."
    )

def vader_note_ro():
    return (
        "VADER este un analizor de sentiment bazat pe reguli, astfel că ajustarea hiperparametrilor, "
        "validarea prin matrice de confuzie și metricile de antrenament offline nu sunt aplicabile aici. "
        "Vizualizarea VADER este folosită doar pentru analiza sentimentului pe mențiunile Reddit de mai sus."
    )


def transformer_note_en():
    return (
        "The Deep Learning Transformer model is used for sentiment inference on the collected Reddit mentions. "
        "In this application, it was not trained from scratch and was not hyperparameter‑tuned locally. "
        "Therefore, local training metrics and confusion matrices are not shown for this method."
    )

def transformer_note_ro():
    return (
        "Modelul Deep Learning Transformer este folosit pentru inferența sentimentului pe mențiunile colectate de pe Reddit. "
        "În această aplicație, modelul nu a fost antrenat de la zero și nu a fost ajustat local prin hyperparameter‑tuning. "
        "Prin urmare, pentru această metodă nu sunt afișate metrici de antrenament local sau matrici de confuzie."
    )
