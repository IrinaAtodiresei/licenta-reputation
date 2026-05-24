import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import base64

from translations import t
from utils.db import (
    safe_read_df,
    get_manual_cm_for_method,
)
from utils.filters import reorder_summary_columns
from utils.kpis import compute_kpis
from utils.charts import (
    render_metrics_grid,
    render_mentions_chart,
    render_pct_negative_chart,
    render_avg_score_chart,
)
from utils.paths import (
    LR_METRICS_PATH,
    LR_CM_PATH,
    LR_REPORT_PATH,
    MANUAL_SAMPLE_PATH,
    MANUAL_METRICS_PATH,
    MANUAL_CM_PATH,
    MANUAL_REPORT_PATH,
)

from dashboard_sections.interpretation import render_interpretation
from dashboard_sections.method_note import render_method_note
from dashboard_sections.method_disagreement import render_method_disagreement
from dashboard_sections.posts_disagree import render_posts_disagree

# ============================================================
#  CSS — PRIMARY BUTTONS
# ============================================================

st.markdown("""
<style>
[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] div button,
button[data-testid="stBaseButton-secondary"] {
    background-color: #0d6efd !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stDownloadButton"] div button:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #0b5ed7 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  HEATMAP HELPERS
# ============================================================

def plot_confusion_matrix_heatmap(cm_df, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_normalized_confusion_matrix(cm_df, title="Normalized Confusion Matrix"):
    if cm_df is None or cm_df.empty:
        cm_norm = pd.DataFrame()
    else:
        cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", ax=ax, vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig
# ============================================================
#  LOADERS — Logistic Regression evaluation files
# ============================================================
# Aceste funcții încarcă metricele LR, matricea de confuzie și raportul.
# Dacă fișierele lipsesc, returnează DataFrame-uri goale pentru a evita crash-uri.

def load_lr_evaluation():
    # Încarcă metricele LR din JSON
    try:
        with open(LR_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    # Încarcă matricea de confuzie LR
    try:
        cm_raw = pd.read_csv(LR_CM_PATH, index_col=0)
    except Exception:
        cm_raw = pd.DataFrame()

    # Încarcă classification report LR
    try:
        report = pd.read_csv(LR_REPORT_PATH)
    except Exception:
        report = pd.DataFrame()

    return metrics, cm_raw, report


# ============================================================
#  LOADERS — Manual validation files
# ============================================================
# Încarcă:
# - sample-ul manual
# - metricele manuale
# - matricea de confuzie manuală
# - classification report manual

def load_manual_validation():
    # Încarcă sample-ul manual
    try:
        manual_sample = pd.read_csv(MANUAL_SAMPLE_PATH)
    except Exception:
        manual_sample = pd.DataFrame()

    # Încarcă metricele manuale
    try:
        with open(MANUAL_METRICS_PATH, "r", encoding="utf-8") as f:
            manual_metrics = json.load(f)
    except Exception:
        manual_metrics = {}

    # Încarcă matricea de confuzie manuală
    try:
        manual_cm = pd.read_csv(MANUAL_CM_PATH)
    except Exception:
        manual_cm = pd.DataFrame()

    # Încarcă classification report manual
    try:
        manual_report = pd.read_csv(MANUAL_REPORT_PATH)
    except Exception:
        manual_report = pd.DataFrame()

    return manual_sample, manual_metrics, manual_cm, manual_report
def render_dashboard(base_df_full, method_name, company, limit_rows, refresh):
    lang = st.session_state["lang"].lower()[:2]  # "ro" sau "en"

    METHODS = {
        "Logistic Regression 3-class balanced": "lr_3class_balanced",
        "VADER": "vader",
        "Deep Learning Transformer": "deep_learning_transformer",
    }
    method = METHODS[method_name]

    # ============================================================
    #  SUMMARY — JOIN între 3 tabele (corect)
    # ============================================================

    df = safe_read_df("""
        SELECT 
            c.company_id,
            c.name AS company_name,
            r.method,
            r.label,
            r.score
        FROM reputation.sentiment_result r
        JOIN reputation.mention m ON r.mention_id = m.mention_id
        JOIN reputation.companies c ON m.company_id = c.company_id
    """)

    # Guard: ensure we have data and the expected columns before filtering
    if df is None or df.empty or "method" not in df.columns:
        st.warning("No data available for dashboard (missing 'method' column).")
        return

    df = df[df["method"] == method]

    if company != "All":
        df = df[df["company_name"] == company]

    summary = (
        df.groupby(["company_name"])
        .agg(
            positives=("label", lambda x: (x == "positive").sum()),
            neutrals=("label", lambda x: (x == "neutral").sum()),
            negatives=("label", lambda x: (x == "negative").sum()),
            avg_score=("score", "mean"),
            total_mentions=("label", "count")   # pentru KPIs
        )
        .reset_index()
    )

    # coloane pentru UI (pe vechi)
    summary["mentions"] = summary["total_mentions"]
    summary["pct_negative"] = (summary["negatives"] / summary["mentions"]) * 100
    summary["method"] = method_name

    # adăugăm company_id
    company_ids = df[["company_name", "company_id"]].drop_duplicates()
    summary = summary.merge(company_ids, on="company_name", how="left")
  
    # ============================================================
    #  LOAD DISAGREEMENTS — SQL (identic cu dashboard-ul vechi)
    # ============================================================

    if method_name == "VADER":

        disagree_company = safe_read_df("""
            SELECT *
            FROM reputation.v_company_method_disagreement
        """)

        disagreements = safe_read_df("""
            SELECT *
            FROM reputation.v_sentiment_disagreements
        """)

    elif method_name == "Deep Learning Transformer":

        disagree_company = safe_read_df("""
            SELECT *
            FROM reputation.v_company_lr_dl_disagreement
        """)

        disagreements = safe_read_df("""
            SELECT *
            FROM reputation.v_lr_dl_sentiment_disagreements
        """)

    else:
        disagree_company = pd.DataFrame()
        disagreements = pd.DataFrame()

    # ============================================================
    #  FILTRARE DUPĂ COMPANIE
    # ============================================================

    if company != "All":
        if not disagree_company.empty and "company_name" in disagree_company.columns:
            disagree_company = disagree_company[
                disagree_company["company_name"] == company
            ]

        if not disagreements.empty and "company_name" in disagreements.columns:
            disagreements = disagreements[
                disagreements["company_name"] == company
            ]

    # ============================================================
    #  DEBUG — vezi exact ce vine din SQL
    # ============================================================
    """
    st.markdown("### 🔍 DEBUG — SQL DISAGREEMENTS (operanzi)")

    if not disagree_company.empty:
        st.write("DEBUG — disagree_company (primele 10 rânduri):")
        st.dataframe(disagree_company.head(10), use_container_width=True)
    else:
        st.warning("DEBUG — disagree_company este gol!")
    """
    # ============================================================
    #  CALCUL pct_diff — cu debug complet
    # ============================================================

    if not disagree_company.empty:

        if "different_mentions" in disagree_company.columns and "total_mentions" in disagree_company.columns:

            total_different = disagree_company["different_mentions"].sum()
            total_posts = disagree_company["total_mentions"].sum()

            # DEBUG — operanzi
            #st.write("🔍 DEBUG — different_mentions (sum):", total_different)
            #st.write("🔍 DEBUG — total_mentions (sum):", total_posts)

            if total_posts > 0:
                pct_diff = float((total_different / total_posts) * 100)
            else:
                pct_diff = 0.0

            #st.success(f"🔍 DEBUG — pct_diff calculat: {pct_diff:.2f}%")

        elif "pct_different" in disagree_company.columns:

            pct_diff = float(disagree_company["pct_different"].mean())
            #st.success(f"🔍 DEBUG — pct_different (mean): {pct_diff:.2f}%")

        else:
            pct_diff = 0.0
            #st.error("⚠️ DEBUG — disagree_company nu are coloanele necesare!")

    else:
        pct_diff = 0.0
        #st.error("⚠️ DEBUG — disagree_company este gol, pct_diff = 0")

    # ============================================================
    #  RECALCUL KPIs
    # ============================================================

    kpis = compute_kpis(summary)

    # ============================================================
    #  INTRO + KPIs
    # ============================================================

    # --- INTRO CARD ---
    st.markdown(
        f"""
        <div class="card">
            <h3 style="margin-top: 0;">Method: {method_name}</h3>
            <h2 class="section-title">{t("dashboard_intro_title", lang)}</h2>
            <p>{t("dashboard_intro_description", lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --- OVERVIEW CARD ---
    st.markdown(
        f"""
        <div class="card">
            <h2 class="section-title">{t("dashboard_overview_title", lang)}</h2>
            <p>{t("dashboard_overview_description", lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("dashboard_total_mentions", lang), kpis["total_mentions"])
    c2.metric(t("dashboard_avg_score", lang), f"{kpis['avg_score']:.4f}")
    c3.metric(t("dashboard_pct_negative", lang), f"{kpis['pct_negative']:.2f}%")

    if method_name == "VADER":
        c4.metric(t("dashboard_pct_disagreement_vader", lang), f"{pct_diff:.2f}%")
    elif method_name == "Deep Learning Transformer":
        c4.metric(t("dashboard_pct_disagreement_dl", lang), f"{pct_diff:.2f}%")
    else:
        c4.metric(t("dashboard_pct_disagreement", lang), "—")


    # --- NOTE (small text) ---
    note_html = t("dashboard_overview_note", lang).replace("\n", "<br>")

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: #666; font-style: italic; margin-top: -5px; line-height: 1.35;">
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )




    # ============================================================
    #  SENTIMENT DISTRIBUTION — tabel + bar chart
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_sentiment_distribution_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    method_distribution = safe_read_df("""
        SELECT method,
               SUM(CASE WHEN label='positive' THEN 1 ELSE 0 END) AS positive,
               SUM(CASE WHEN label='neutral' THEN 1 ELSE 0 END) AS neutral,
               SUM(CASE WHEN label='negative' THEN 1 ELSE 0 END) AS negative,
               COUNT(*) AS total
        FROM reputation.sentiment_result
        GROUP BY method
        ORDER BY method;
    """)

    if not method_distribution.empty:
        st.dataframe(method_distribution, use_container_width=True)
        st.bar_chart(method_distribution.set_index("method")[["positive", "neutral", "negative"]])
    else:
        st.info(t("dashboard_no_data_available", lang))

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    #  SENTIMENT BREAKDOWN — metrici dinamice
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_sentiment_breakdown_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    pos = summary["positives"].sum()
    neg = summary["negatives"].sum()
    neu = summary["neutrals"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric(t("dashboard_positive_label", lang), pos)
    c2.metric(t("dashboard_negative_label", lang), neg)
    c3.metric(t("dashboard_neutral_label", lang), neu)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    #  COMPANY SUMMARY — tabel + buton download
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_company_summary_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    ordered_summary = reorder_summary_columns(summary)
    # formatare cu 2 zecimale
    if "avg_score" in ordered_summary.columns:
        ordered_summary["avg_score"] = ordered_summary["avg_score"].round(2)

    if "pct_negative" in ordered_summary.columns:
        ordered_summary["pct_negative"] = ordered_summary["pct_negative"].round(2)

    st.dataframe(ordered_summary, use_container_width=True)

    summary_csv = ordered_summary.to_csv(index=False).encode("utf-8-sig")
    summary_b64 = base64.b64encode(summary_csv).decode()

    st.markdown("""
        <style>
        .primary-download-summary {
            background-color: #0d6efd;
            color: white !important;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            margin-top: 12px;
        }
        .primary-download-summary:hover {
            background-color: #0b5ed7;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<a class="primary-download-summary" '
        f'href="data:file/csv;base64,{summary_b64}" '
        f'download="summary_{method}.csv">'
        f'⬇ {t("dashboard_download_summary_button", lang)}</a>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    #  CHARTS — mentions, pct negative, avg score
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_charts_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_mentions_chart(summary)
    with c2:
        render_pct_negative_chart(summary)
    with c3:
        render_avg_score_chart(summary)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    #  INTERPRETATION — text generat
    # ============================================================

    #render_interpretation(kpis, method_name, lang)
    render_interpretation(kpis, method_name, lang, company, pct_diff)





    # ============================================================
    #  METHOD NOTE — explicații despre metodă
    # ============================================================

    render_method_note(method_name, lang)


    # ============================================================
    #  DISAGREEMENT TABLES — posts + summary
    # ============================================================

    if method_name in ["VADER", "Deep Learning Transformer"]:
        render_method_disagreement(disagree_company, method_name, lang)
        render_posts_disagree(disagreements, method_name, limit_rows, lang)

    # ============================================================
    #  MOST NEGATIVE MENTIONS — top 30
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_most_negative_mentions_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    negative_examples = safe_read_df("""
        SELECT 
            c.name AS company_name,
            m.title,
            m.content,
            r.score,
            m.published_at
        FROM reputation.sentiment_result r
        JOIN reputation.mention m ON r.mention_id = m.mention_id
        JOIN reputation.companies c ON m.company_id = c.company_id
        WHERE r.method = %s
        ORDER BY r.score ASC
        LIMIT 30
    """, params=(method,))

    if company != "All":
        negative_examples = negative_examples[
            negative_examples["company_name"] == company
        ]

    if not negative_examples.empty:
        st.dataframe(negative_examples, use_container_width=True)
    else:
        st.info(t("dashboard_no_negative_examples", lang))

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    #  MODEL EVALUATION — LR, VADER, TRANSFORMER
    # ============================================================

    from dashboard_sections.model_evaluation_helpers import (
        lr_note_en, lr_note_ro,
        vader_note_en, vader_note_ro,
        transformer_note_en, transformer_note_ro,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_model_evaluation_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    # Normalize language
    lang = lang.lower()

    # ------------------------------------------------------------
    # LOGISTIC REGRESSION
    # ------------------------------------------------------------
    if method_name == "Logistic Regression 3-class balanced":

        text = lr_note_ro() if lang == "ro" else lr_note_en()

        st.markdown(
            f"<p style='color:#1f4e79; font-size:15px;'>{text}</p>",
            unsafe_allow_html=True,
        )

        lr_metrics, lr_cm_raw, lr_report = load_lr_evaluation()

        if lr_metrics:
            cards = [
                {"label": "Accuracy", "value": f"{lr_metrics.get('accuracy', 0):.4f}"},
                {"label": "Precision macro", "value": f"{lr_metrics.get('precision_macro', 0):.4f}"},
                {"label": "Recall macro", "value": f"{lr_metrics.get('recall_macro', 0):.4f}"},
                {"label": "F1 macro", "value": f"{lr_metrics.get('f1_macro', 0):.4f}"},
            ]
            render_metrics_grid(cards, columns=4)

        if not lr_cm_raw.empty:
            clean_lr_cm = lr_cm_raw.select_dtypes(include=["number"]).iloc[:3, :3]

            st.markdown("### Logistic Regression confusion matrices")
            st.markdown(
                "<p style='color:#1f4e79; font-size:15px;'>"
                "The raw matrix shows the number of predictions, while the normalized matrix shows percentages per real class."
                "</p>",
                unsafe_allow_html=True,
            )

            col_raw, col_norm = st.columns(2)

            with col_raw:
                st.markdown("#### Raw confusion matrix")
                st.pyplot(plot_confusion_matrix_heatmap(clean_lr_cm))

            with col_norm:
                st.markdown("#### Normalized confusion matrix")
                st.pyplot(plot_normalized_confusion_matrix(clean_lr_cm))

        if not lr_report.empty:
            st.markdown("### Classification report")
            st.dataframe(lr_report, use_container_width=True)


    # ------------------------------------------------------------
    # VADER
    # ------------------------------------------------------------
    elif method_name == "VADER":

        text = vader_note_ro() if lang == "ro" else vader_note_en()

        st.markdown(
            f"<p style='color:#1f4e79; font-size:15px;'>{text}</p>",
            unsafe_allow_html=True,
        )


    # ------------------------------------------------------------
    # TRANSFORMER
    # ------------------------------------------------------------
    elif method_name == "Deep Learning Transformer":

        text = transformer_note_ro() if lang == "ro" else transformer_note_en()

        st.markdown(
            f"<p style='color:#1f4e79; font-size:15px;'>{text}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


    # ============================================================
    #  MANUAL VALIDATION — sample + metrics + confusion matrix
    # ============================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("dashboard_manual_validation_title", lang)}</h2>',
        unsafe_allow_html=True,
    )

    manual_sample, manual_metrics, manual_cm, manual_report = load_manual_validation()

    if method in manual_metrics:
        mm = manual_metrics[method]

        cards = [
            {"label": t("dashboard_sample_size_label", lang), "value": mm.get("sample_size", 0)},
            {"label": t("dashboard_accuracy_label", lang), "value": f"{mm.get('accuracy', 0):.4f}"},
            {"label": t("dashboard_f1_macro_label", lang), "value": f"{mm.get('f1_macro', 0):.4f}"},
        ]
        render_metrics_grid(cards, columns=3)

        manual_cm_df = get_manual_cm_for_method(MANUAL_CM_PATH, method)

        if not manual_cm_df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"### {t('dashboard_raw_confusion_matrix_title', lang)}")
                st.pyplot(plot_confusion_matrix_heatmap(manual_cm_df))

            with c2:
                st.markdown(f"### {t('dashboard_normalized_confusion_matrix_title', lang)}")
                st.pyplot(plot_normalized_confusion_matrix(manual_cm_df))
        else:
            st.info(t("dashboard_no_manual_confusion_matrix", lang))

        if not manual_report.empty:
            st.markdown(f"### {t('dashboard_manual_classification_report_title', lang)}")

            report_df = manual_report.copy()

            if "method" in report_df.columns:
                report_df = report_df[report_df["method"] == method]

            st.dataframe(report_df, use_container_width=True)

        # ============================================================
        #  MANUAL SAMPLE TABLE + DOWNLOAD
        # ============================================================

        st.markdown(f"### {t('dashboard_manual_sample_title', lang)}")

        if not manual_sample.empty:
            manual_df_export = manual_sample.copy()

            manual_df_export["manual_label"] = (
                manual_df_export["manual_label"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            manual_df_export = manual_df_export[
                manual_df_export["manual_label"].isin(["negative", "neutral", "positive"])
            ]

            if company != "All" and "company_name" in manual_df_export.columns:
                manual_df_export = manual_df_export[
                    manual_df_export["company_name"] == company
                ]

            pred_col_map = {
                "lr_3class_balanced": "lr_label",
                "vader": "vader_label",
                "deep_learning_transformer": "dl_label",
            }
            selected_pred_col = pred_col_map.get(method)

            if selected_pred_col and selected_pred_col in manual_df_export.columns:
                manual_df_export["is_correct"] = (
                    manual_df_export["manual_label"] == manual_df_export[selected_pred_col]
                )

                display_cols = [
                    "mention_id",
                    "company_name",
                    "text",
                    "manual_label",
                    selected_pred_col,
                    "is_correct",
                ]
                existing_display_cols = [c for c in display_cols if c in manual_df_export.columns]
                display_df = manual_df_export[existing_display_cols].head(50)

                def highlight_correct(row):
                    if row["is_correct"]:
                        return ["background-color: #d4edda"] * len(row)
                    else:
                        return ["background-color: #f8d7da"] * len(row)

                styled_df = display_df.style.apply(highlight_correct, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                csv_bytes = manual_df_export.to_csv(index=False).encode("utf-8-sig")
                b64 = base64.b64encode(csv_bytes).decode()

                st.markdown("""
                    <style>
                    .primary-download {
                        background-color: #0d6efd;
                        color: white !important;
                        padding: 0.6rem 1.2rem;
                        border-radius: 6px;
                        font-weight: 600;
                        text-decoration: none;
                        display: inline-block;
                        margin-top: 12px;
                    }
                    .primary-download:hover {
                        background-color: #0b5ed7;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(
                    f'<a class="primary-download" href="data:file/csv;base64,{b64}" '
                    f'download="manual_validation_{method}.csv">⬇ Download manual validation sample</a>',
                    unsafe_allow_html=True
                )

            else:
                st.info("No prediction column available for the selected method in manual sample.")
        else:
            st.info(t("dashboard_no_manual_sample_data", lang))

    st.markdown("</div>", unsafe_allow_html=True)


