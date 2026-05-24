import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Reputation & Sentiment Dashboard",
    layout="wide",
    page_icon="🧠",
)

# ============================================================
#  IMPORTS
# ============================================================
from utils import load_reddit_dataframe
from tab0_home import render_home
from tabs.tab_dashboard import render_dashboard
from tabs.tab2_ai_insights import render_tab2
from tabs.tab3_interactive_demo import render_tab3
from tabs.tab4_live_pipeline import render_tab4
from tabs.tab5_proof_of_source import render_tab5

from help_llm import ask_groq   # ← NOUL HELP



from components.layout import render_base_styles
from components.header import render_header
from components.footer import render_footer
from components.sidebar import render_sidebar
from translations import t


# ============================================================
#  SESSION STATE
# ============================================================
if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_controls" not in st.session_state:
    st.session_state.show_controls = True

if "lang" not in st.session_state:
    st.session_state.lang = "RO"

if "help_open" not in st.session_state:
    st.session_state.help_open = False


# ============================================================
#  LANDING PAGE
# ============================================================
if not st.session_state.entered:
    render_home()
    st.stop()


# ============================================================
#  GLOBAL STYLES
# ============================================================
st.session_state["_components_styles_loaded"] = False
render_base_styles()

st.markdown(
    "<script>document.body.classList.remove('landing-page');</script>",
    unsafe_allow_html=True,
)


# ============================================================
#  GLOBAL HEADER
# ============================================================
render_header()
# DEBUG — afișăm cheia din secrets
#st.error(f"CHEIA DIN SECRETS: {st.secrets.get('GROQ_API_KEY')}")


# ============================================================
#  HELP DIALOG (Groq direct)
# ============================================================
# ============================================================
#  HELP DIALOG (BILINGV)
# ============================================================
@st.dialog(t("help_title", st.session_state.lang))
def help_dialog():
    lang = st.session_state.lang

    st.markdown(f"### {t('help_select_topic', lang)}")

    topic = st.selectbox(
        t("help_topic_label", lang),
        [
            "Dashboard",
            "Logistic Regression",
            "VADER",
            "Transformer",
            "Live Pipeline",
            "Proof of Source",
            "AI Insights",
        ],
        key="help_topic"
    )

    if st.button(t("help_send_button", lang), type="primary", key="help_send"):
        st.session_state.help_answer = ask_groq(topic, lang)

    if "help_answer" in st.session_state:
        st.markdown(
            f"### {t('help_explanation_label', lang)}\n"
            f"{st.session_state.help_answer}"
        )
    else:
        st.info(t("help_no_answer", lang))

# ============================================================
#  SIDEBAR CONTROLS
# ============================================================
if st.session_state.show_controls:
    sidebar_values = render_sidebar()
else:
    sidebar_values = {
        "dashboard_method": "Logistic Regression 3-class balanced",
        "company_filter": "All",
        "rows_slider": 50,
        "dashboard_limit_rows": 50,
        "dashboard_refresh": False,
        "lang": st.session_state.lang,
    }

# Trigger dialog
if st.session_state.help_open:
    st.session_state.help_answer = ""   # ← GOLIM EXPLICAȚIA
    help_dialog()
    st.session_state.help_open = False



dashboard_method = sidebar_values["dashboard_method"]
company_filter = sidebar_values["company_filter"]
rows_slider = sidebar_values["rows_slider"]
lang = sidebar_values["lang"]
dashboard_limit_rows = sidebar_values["dashboard_limit_rows"]
dashboard_refresh = sidebar_values["dashboard_refresh"]


# ============================================================
#  LOAD BASE DATA
# ============================================================
@st.cache_data(show_spinner=False)
def get_base_df_full():
    df = load_reddit_dataframe()

    if df is None or df.empty:
        st.warning(
            "⚠️ Unable to load data from database. Using mock data for demonstration."
        )
        return pd.DataFrame()

    return df


base_df_full = get_base_df_full()
base_df = base_df_full.copy() if base_df_full is not None else None

if base_df is not None and company_filter != "All":
    base_df = base_df[base_df["company"] == company_filter]


# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        t("tab_dashboard", lang),
        t("tab_ai_insights", lang),
        t("tab_interactive_demo", lang),
        t("tab_live_pipeline", lang),
        t("tab_proof_source", lang),
    ]
)

with tab1:
    render_dashboard(
        base_df_full,
        method_name=dashboard_method,
        company=company_filter,
        limit_rows=dashboard_limit_rows,
        refresh=dashboard_refresh,
    )
with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4(rows_slider, lang)

with tab5:
    render_tab5(base_df, rows_slider)


# ============================================================
#  GLOBAL FOOTER
# ============================================================
render_footer()
