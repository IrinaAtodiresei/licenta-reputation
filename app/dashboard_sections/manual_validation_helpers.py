# ============================================================
#  MANUAL VALIDATION TEXT HELPERS (EN + RO)
# ============================================================

def manual_intro_en():
    return (
        "This section evaluates the selected sentiment analysis method on a manually labeled Reddit sample. "
        "The manual labels are used as ground truth, while the model predictions are compared against them."
    )

def manual_intro_ro():
    return (
        "Această secțiune evaluează metoda de analiză a sentimentului selectată pe un eșantion Reddit etichetat manual. "
        "Etichetele manuale sunt folosite ca adevăr de referință, iar predicțiile modelului sunt comparate cu acestea."
    )


def manual_caption_en():
    return (
        "Unlike the previous training evaluation, these metrics are computed directly on Reddit mentions "
        "that were manually labeled by the author of the project."
    )

def manual_caption_ro():
    return (
        "Spre deosebire de evaluarea anterioară pe setul de antrenament, aceste metrici sunt calculate direct "
        "pe mențiuni Reddit etichetate manual de autorul proiectului."
    )


def manual_interpretation_good_en(acc, f1):
    return (
        f"The selected method performs well on the manually labeled Reddit sample, "
        f"with an accuracy of {acc:.4f} and a macro F1-score of {f1:.4f}."
    )

def manual_interpretation_good_ro(acc, f1):
    return (
        f"Metoda selectată are performanțe bune pe eșantionul Reddit etichetat manual, "
        f"cu o acuratețe de {acc:.4f} și un scor F1 macro de {f1:.4f}."
    )


def manual_interpretation_medium_en(acc):
    return (
        f"The selected method has moderate performance on the manually labeled Reddit sample, "
        f"with an accuracy of {acc:.4f}. This suggests that Reddit language is more difficult "
        f"than the original training data."
    )

def manual_interpretation_medium_ro(acc):
    return (
        f"Metoda selectată are performanțe moderate pe eșantionul Reddit etichetat manual, "
        f"cu o acuratețe de {acc:.4f}. Acest lucru sugerează că limbajul de pe Reddit este mai dificil "
        f"decât cel din datele de antrenament."
    )


def manual_interpretation_bad_en(acc):
    return (
        f"The selected method has limited performance on the manually labeled Reddit sample, "
        f"with an accuracy of {acc:.4f}. This shows that model predictions should be interpreted carefully "
        f"when applied to informal Reddit discussions."
    )

def manual_interpretation_bad_ro(acc):
    return (
        f"Metoda selectată are performanțe reduse pe eșantionul Reddit etichetat manual, "
        f"cu o acuratețe de {acc:.4f}. Acest lucru arată că predicțiile modelului trebuie interpretate cu atenție "
        f"când sunt aplicate discuțiilor informale de pe Reddit."
    )
