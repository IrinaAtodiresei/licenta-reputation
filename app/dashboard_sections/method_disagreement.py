import streamlit as st
from translations import t

def render_method_disagreement(disagree_company, method_name, lang):
    if disagree_company.empty:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)

    title = (
        t("dashboard_disagreement_lr_vader", lang)
        if method_name == "VADER"
        else t("dashboard_disagreement_lr_dl", lang)
    )

    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)

    sort_col = (
        "pct_different"
        if "pct_different" in disagree_company.columns
        else disagree_company.columns[-1]
    )

    disagree_company = disagree_company.sort_values(sort_col, ascending=False)

    st.dataframe(disagree_company, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
