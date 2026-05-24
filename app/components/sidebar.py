import streamlit as st
from translations import t

def render_sidebar() -> dict:
    state = st.session_state
    lang = state.get("lang", "RO")

    st.sidebar.title(t("sidebar_title", lang))

    # ============================================================
    # FORM – filtrele NU declanșează rerun
    # ============================================================
    with st.sidebar.form("filters_form"):

        # LIMBA
        new_lang = st.radio(
            t("sidebar_language_selector", lang),
            ["RO", "EN"],
            index=0 if lang == "RO" else 1,
            key="lang"
        )

        # METODĂ
        method = st.radio(
            t("sidebar_method", new_lang),
            [
                "Logistic Regression 3-class balanced",
                "VADER",
                "Deep Learning Transformer",
            ],
            key="dashboard_method"
        )

        # COMPANIE
        company = st.selectbox(
            t("sidebar_company", new_lang),
            ["All", "Apple", "Google", "Samsung"],
            key="company_filter"
        )

        # ROWS
        rows_slider = st.slider(
            t("sidebar_rows_slider", new_lang),
            min_value=10,
            max_value=200,
            value=state.get("rows_slider", 50),
            key="rows_slider"
        )

        # LIMIT ROWS
        dashboard_limit_rows = st.slider(
            t("sidebar_limit_rows", new_lang),
            min_value=20,
            max_value=500,
            value=state.get("dashboard_limit_rows", 50),
            key="dashboard_limit_rows"
        )

        st.markdown("---")

        # REFRESH – singurul care declanșează rerun
        submitted = st.form_submit_button(
            t("sidebar_refresh", new_lang),
            type="primary"
        )

    # ============================================================
    # HELP – în afara form-ului (NU declanșează submit)
    # ============================================================
    if st.sidebar.button("💬 Help", key="help_btn_sidebar"):
        state.help_open = True

    st.sidebar.caption(t("sidebar_caption", new_lang))

    return {
        "dashboard_method": method,
        "company_filter": company,
        "rows_slider": rows_slider,
        "dashboard_limit_rows": dashboard_limit_rows,
        "dashboard_refresh": submitted,
        "lang": new_lang,
    }
