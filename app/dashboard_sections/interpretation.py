import streamlit as st
from translations import t
from dashboard_sections.interpretation_helpers import (
    generate_interpretation,
    generate_interpretation_ro,
)

def render_interpretation(kpis, method_name, lang, company, pct_disagreement):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"<h2 class='section-title'>{t('dashboard_interpretation_title', lang)}</h2>",
        unsafe_allow_html=True,
    )

    total = kpis["total_mentions"]
    avg_score = kpis["avg_score"]
    pct_negative = kpis["pct_negative"]
    #pct_disagreement = kpis["pct_disagreement"]

    if total == 0:
        st.info(t("dashboard_interpretation_no_data", lang))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if lang == "ro":
        interpretation = generate_interpretation_ro(
            company=company,
            method_name=method_name,
            total_mentions=total,
            avg_score=avg_score,
            pct_negative=pct_negative,
            pct_disagreement=pct_disagreement,
        )
    else:
        interpretation = generate_interpretation(
            company=company,
            method_name=method_name,
            total_mentions=total,
            avg_score=avg_score,
            pct_negative=pct_negative,
            pct_disagreement=pct_disagreement,
        )

    st.info(interpretation)
    st.markdown("</div>", unsafe_allow_html=True)
