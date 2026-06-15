import streamlit as st
import datetime
import pandas as pd

from translations import t
from chains.reputation_marketing_chain import (
    generate_marketing_ai_insights,
    strip_html,
    strip_markdown,
)
from utils.db import load_summary, load_negative_examples

METHODS = {
    "Logistic Regression 3-class balanced": "lr_3class_balanced",
    "VADER": "vader",
    "Deep Learning Transformer": "deep_learning_transformer",
}


def clean_output(x: str) -> str:
    if not isinstance(x, str):
        return x
    return strip_markdown(strip_html(x))


def render_insight_card(title: str, content: str, accent_class: str) -> str:
    content = clean_output(content or "")
    return f"""
    <div class="insight-card {accent_class}">
        <div class="insight-card__title">{title}</div>
        <div class="insight-card__body">{content}</div>
    </div>
    """


def render_tab2(summary_df=None, negative_df=None):
    lang = st.session_state["lang"]

    # ---------- CSS GLOBAL + BUTON MARE ----------
    st.markdown(
        """
        <style>

        /* BUTON MARE, FULL-WIDTH, ALBASTRU */
        div.stForm button[kind="primary"] {
            background-color: #0B72E5 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 1rem 1.2rem !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(11, 114, 229, 0.35) !important;
            transition: 0.2s ease-in-out !important;
        }

        div.stForm > div > div {
            width: 100% !important;
        }

        div.stForm button[kind="primary"]:hover {
            background-color: #095bb8 !important;
            box-shadow: 0 6px 18px rgba(9, 91, 184, 0.45) !important;
        }

        /* CARDURI */
        .insight-card {
            padding: 18px;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .insight-card__title {
            font-weight: 700;
            margin-bottom: 6px;
        }
        .insight-card__body {
            max-height: none !important;
            overflow: visible !important;
            text-overflow: unset !important;
            white-space: normal !important;
            font-size: 14px;
            color: #374151;
        }
        .insight-card.blue { border-left: 4px solid #3B82F6; }
        .insight-card.green { border-left: 4px solid #10B981; }
        .insight-card.orange { border-left: 4px solid #F97316; }
        .insight-card.purple { border-left: 4px solid #8B5CF6; }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- HEADER ----------
    st.markdown(
        f"""
        <div style="
            padding: 22px 24px;
            margin-bottom: 18px;
            border-radius: 18px;
            background: linear-gradient(90deg, #0B72E5 0%, #1D4ED8 40%, #0F172A 100%);
            color: white;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
        ">
            <div style="font-size: 24px; font-weight: 700;">
                {t("ai_insights_header_title", lang)}
            </div>
            <div style="font-size: 14px; opacity: 0.95;">
                {t("ai_insights_header_subtitle", lang)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    #  FREEZE UI TOTAL — TOTUL ÎN FORM
    # ============================================================
    with st.form("insights_form"):

        st.markdown(f"### {t('ai_insights_context_selector', lang)}")

        c1, c2, c3, c4 = st.columns(4)

        sentiment_method = c1.selectbox(t("ai_insights_sentiment_method", lang), list(METHODS.keys()))
        company = c2.selectbox(t("ai_insights_company", lang), ["Apple", "Google", "Samsung", "All"])
        period_options = (
            ["Last 7 days", "Last 30 days", "Last 90 days"]
            if lang == "EN"
            else ["Ultimele 7 zile", "Ultimele 30 de zile", "Ultimele 90 de zile"]
        )

        period = c3.selectbox(
            t("ai_insights_period", lang),
            period_options,
            index=1,
        )
        model = c4.selectbox(
            t("ai_insights_llm_engine", lang),
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        )

        st.markdown(f"### {t('ai_insights_which_insights', lang)}")

        c5, c6 = st.columns(2)
        with c5:
            show_summary = st.checkbox(t("ai_insights_executive_summary_checkbox", lang), True)
            show_trend = st.checkbox(t("ai_insights_general_trend_checkbox", lang), True)
            show_themes = st.checkbox(t("ai_insights_dominant_themes_checkbox", lang), True)
        with c6:
            show_risks = st.checkbox(t("ai_insights_reputation_risks_checkbox", lang), True)
            show_recs = st.checkbox(t("ai_insights_marketing_recs_checkbox", lang), True)
            bullet_mode = st.checkbox(t("ai_insights_bullet_mode_checkbox", lang), True)

        detail_level = st.select_slider(
            t("ai_insights_detail_level", lang),
            ["Low", "Medium", "High"],
            value="Medium",
        )

        presentation_mode = st.toggle(t("ai_insights_presentation_mode", lang))

        submitted = st.form_submit_button(t("ai_insights_generate_button", lang), type="primary")

    st.session_state["presentation_mode"] = presentation_mode

    summary_placeholder = st.empty()
    cards_placeholder = st.empty()

    # ---------- CACHE ----------
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}

    cache = st.session_state["ai_cache"]
    cache_key = f"{company}__{period}__{METHODS[sentiment_method]}__{model}__{detail_level}__{bullet_mode}"

    insights = st.session_state.get("ai_insights")

    # ============================================================
    #  GENERARE DOAR LA SUBMIT
    # ============================================================
    if submitted:

        with st.spinner(t("ai_insights_generating_spinner", lang)):

            # ---------- VARIANTA 2 — LOGICA AUTOMATĂ PENTRU MODEL ----------
            if bullet_mode:
                model = "llama-3.1-8b-instant"

            # ---------- ÎNCĂRCARE DATE ----------
            summary_df = load_summary(METHODS[sentiment_method])
            negative_df = load_negative_examples(METHODS[sentiment_method], company)

            # ------------------------------------------------------------
            # CALCUL PERIOADĂ RAPORTATĂ LA ULTIMA DATĂ DIN negative_df
            # ------------------------------------------------------------

            # Convertim la datetime
            negative_df["published_at"] = pd.to_datetime(negative_df["published_at"], errors="coerce")

            # 1. Ultima dată disponibilă în negative_df (date brute)
            last_date = negative_df["published_at"].max()

            days_map = {
                "Ultimele 7 zile": 7,
                "Ultimele 30 de zile": 30,
                "Ultimele 90 de zile": 90,
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
            }

            if period in days_map:
                start_date = last_date - datetime.timedelta(days=days_map[period])
            else:
                start_date = negative_df["published_at"].min()

            # Filtrăm negative_df (date brute)
            filtered_negative_df = negative_df[
                negative_df["published_at"].between(start_date, last_date)
            ]

            # summary_df NU are dată → îl lăsăm nefiltrat
            filtered_df = summary_df

            # Numărăm rândurile
            total_rows = len(negative_df)
            filtered_rows = len(filtered_negative_df)

            # Fallback dacă nu există date în perioada selectată
            if filtered_rows == 0:
                filtered_negative_df = negative_df
                filtered_rows = total_rows
                period_label = "All period (automatic fallback)" if lang == "EN" else "Toată perioada (fallback automat)"
            else:
                period_label = (
                    f"{period} (based on {last_date.strftime('%d.%m.%Y')})"
                    if lang == "EN"
                    else f"{period} (raportat la {last_date.strftime('%d.%m.%Y')})"
                )

            # ---------- PRESENTATION MODE ----------
            if presentation_mode and cache_key in cache:
                insights = cache[cache_key]
                st.info(t("ai_insights_cache_loaded_info", lang))

            else:

                def _call_llm(selected_model: str):
                    return generate_marketing_ai_insights(
                        company=company,
                        period=period_label,
                        method_name=METHODS[sentiment_method],
                        summary_df=filtered_df,
                        negative_df=filtered_negative_df,
                        model_name=selected_model,
                        detail_level=detail_level,
                        bullet_mode=bullet_mode,
                    )

                try:
                    insights = _call_llm(model)
                except Exception as e:
                    msg = str(e).lower()
                    if "rate limit" in msg or "429" in msg:
                        st.info(t("ai_insights_rate_limit_warning", lang))
                        model = "llama-3.1-8b-instant"
                        insights = _call_llm(model)
                    else:
                        raise e

                for k, v in insights.items():
                    if isinstance(v, str):
                        insights[k] = clean_output(v)

                insights["model_name"] = model
                insights["generated_at"] = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
                insights["company"] = company
                insights["period"] = period_label
                insights["sentiment_method"] = sentiment_method
                insights["detail_level"] = detail_level
                insights["bullet_mode"] = bullet_mode
                insights["total_rows"] = total_rows
                insights["filtered_rows"] = filtered_rows
                insights["last_date"] = last_date.strftime("%d.%m.%Y")

                cache[cache_key] = insights

            st.session_state["ai_insights"] = insights

    # ---------- NO INSIGHTS ----------
    insights = st.session_state.get("ai_insights")
    if not insights:
        summary_placeholder.info(t("ai_insights_no_insights_message", lang))
        return

    # ---------- METADATA ----------
    summary_placeholder.markdown(
        f"""
        <div style="font-size: 12px; color: #64748B; margin-bottom: 12px;">
            📅 <b>Ultima dată disponibilă:</b> {insights['last_date']} |
            📊 <b>Total mențiuni brute:</b> {insights['total_rows']} |
            🔍 <b>Mențiuni în perioada selectată:</b> {insights['filtered_rows']} |
            🗂️ <b>Perioadă:</b> {insights['period']} |
            ⚙️ <b>Model:</b> {insights['model_name']} |
            {insights['generated_at']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ---------- ATENȚIONARE DESPRE LIMITA DE 30 ----------
    limit_message = (
        "ℹ️ *Only the latest 30 negative mentions are displayed "
        "(performance limit). This limitation prevents excessive token consumption "
        "and keeps insight generation fast. The actual number of mentions in the "
        "selected period may be higher in the database.*"
        if lang == "EN"
        else
        "ℹ️ *Sunt afișate doar ultimele 30 de mențiuni negative "
        "(limită configurată pentru performanță). Limitarea previne consumul excesiv "
        "de tokeni și menține viteza de generare a insight-urilor. Numărul real de "
        "mențiuni din perioada selectată poate fi mai mare în baza de date.*"
    )

    st.info(limit_message)

    # ---------- CARDURI ----------
    with cards_placeholder.container():

        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")

        if show_summary:
            with row1[0]:
                st.markdown(
                    render_insight_card(
                        t("ai_insights_card_executive_summary", lang),
                        insights.get("executive_summary", ""),
                        "blue",
                    ),
                    unsafe_allow_html=True,
                )

        if show_trend:
            with row1[1]:
                st.markdown(
                    render_insight_card(
                        t("ai_insights_card_general_sentiment", lang),
                        insights.get("general_sentiment", ""),
                        "green",
                    ),
                    unsafe_allow_html=True,
                )

        if show_themes:
            with row2[0]:
                st.markdown(
                    render_insight_card(
                        t("ai_insights_card_recurring_themes", lang),
                        insights.get("recurring_themes", ""),
                        "orange",
                    ),
                    unsafe_allow_html=True,
                )

        if show_risks:
            with row2[1]:
                st.markdown(
                    render_insight_card(
                        t("ai_insights_card_reputation_risks", lang),
                        insights.get("reputation_risks", ""),
                        "purple",
                    ),
                    unsafe_allow_html=True,
                )

        if show_recs:
            st.markdown(
                render_insight_card(
                    t("ai_insights_card_marketing_recommendations", lang),
                    insights.get("marketing_recommendations", ""),
                    "blue",
                ),
                unsafe_allow_html=True,
            )

        with st.expander(t("ai_insights_llm_trace_viewer", lang)):
            st.subheader(t("ai_insights_llm_context_header", lang))
            st.json(insights.get("context", {}))

            st.subheader(t("ai_insights_llm_raw_output_header", lang))
            st.json(insights.get("raw_output", {}))

            st.subheader(t("ai_insights_llm_execution_times_header", lang))
            st.json(insights.get("timings", {}))

            st.subheader(t("ai_insights_llm_chains_header", lang))
            st.write(insights.get("chains", []))

            st.subheader(t("ai_insights_llm_prompts_note_header", lang))
            st.write(insights.get("full_prompt", t("ai_insights_llm_prompts_default_note", lang)))
