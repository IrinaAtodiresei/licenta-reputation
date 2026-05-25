import streamlit as st
import pandas as pd
from translations import t


def render_tab5(base_df: pd.DataFrame, rows_slider: int):

    lang = st.session_state["lang"]

    # ============================
    # GLOBAL CSS
    # ============================
    st.markdown("""
    <style>
    .scroll-table {
        height: 500px;
        overflow-y: scroll;
        border: 1px solid #ddd;
        padding: 4px;
        border-radius: 6px;
        background: white;
    }
    .scroll-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .scroll-table th, .scroll-table td {
        padding: 6px 8px;
        border-bottom: 1px solid #eee;
        vertical-align: top;
    }
    .scroll-table tr:hover {
        background-color: #f7f7f7;
    }
    .truncate {
        max-width: 320px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============================
    # INTRO CARD
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("tab5_title", lang)}</h2>',
        unsafe_allow_html=True
    )
    st.write(t("tab5_description", lang))
    st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    # ROWS PER PAGE — INDEPENDENT DE SIDEBAR
    # ============================
    if "tab5_rows_per_page" not in st.session_state:
        st.session_state.tab5_rows_per_page = 10  # default REAL, nu 50

    rows_per_page = st.slider(
        t("tab5_rows_per_page", lang),
        min_value=10,
        max_value=200,
        value=st.session_state.tab5_rows_per_page,
        key="tab5_rows_per_page"
    )

    # ============================
    # PAGINATION STATE
    # ============================
    if "proof_page" not in st.session_state:
        st.session_state.proof_page = 0

    total_rows = len(base_df)
    page_size = rows_per_page
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    # ============================
    # PAGINATION BUTTONS
    # ============================
    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button(t("tab5_prev_button", lang), type="primary",
                     disabled=st.session_state.proof_page == 0):
            st.session_state.proof_page -= 1
            st.rerun()

    with col_page:
        st.markdown(
            f"**{t('tab5_page_label', lang).format(page=st.session_state.proof_page + 1, total=total_pages)}**"
        )

    with col_next:
        if st.button(t("tab5_next_button", lang), type="primary",
                     disabled=st.session_state.proof_page >= total_pages - 1):
            st.session_state.proof_page += 1
            st.rerun()

    # ============================
    # CURRENT PAGE DATA
    # ============================
    start = st.session_state.proof_page * page_size
    end = start + page_size
    page_df = base_df.iloc[start:end].copy()

    # Fix newline display
    page_df["text"] = page_df["text"].astype(str).replace("\n", " ", regex=False)

    # Trunchiere text
    page_df["text"] = page_df["text"].apply(lambda t: f'<div class="truncate">{t}</div>')

    st.caption(
        t("tab5_showing_rows", lang).format(
            start=start + 1,
            end=min(end, total_rows),
            total=total_rows
        )
    )

    # ============================
    # SELECTBOX FOR ROW SELECTION
    # ============================
    st.markdown(f"### {t('tab5_select_row_title', lang)}")

    options = {
        idx: f"{idx} — {page_df.loc[idx, 'company']} — {page_df.loc[idx, 'author']}"
        for idx in page_df.index
    }

    selected = st.selectbox(
        t("tab5_select_row_label", lang),
        options.keys(),
        format_func=lambda x: options[x]
    )

    # ============================
    # SCROLLABLE TABLE
    # ============================
    html_table = page_df[
        ["company", "author", "text", "created_utc"]
    ].to_html(escape=False, index=True)

    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    # ============================
    # INSPECT SELECTED ROW
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("tab5_inspect_title", lang)}</h2>',
        unsafe_allow_html=True
    )

    row = base_df.loc[selected]

    st.json({
        "mention_id": int(row.get("mention_id", selected)),
        "external_id": row.get("external_id", ""),
        "reddit_object_type": row.get("reddit_object_type", ""),
        "reddit_object_id": row.get("reddit_object_id", ""),
        "author": row.get("author", ""),
        "published_at": row.get("created_utc", ""),
        "collected_at": row.get("collected_at", ""),
        "company": row.get("company", ""),
        "subreddit": row.get("subreddit", ""),
        "text": row.get("text", ""),
        "reddit_url": row.get("reddit_url", "")
    })

    st.markdown('</div>', unsafe_allow_html=True)
