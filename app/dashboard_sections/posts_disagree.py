import streamlit as st
from translations import t

def render_posts_disagree(disagreements, method_name, limit_rows, lang):
    if disagreements.empty:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)

    title = (
        t("dashboard_posts_disagree_lr_vader", lang)
        if method_name == "VADER"
        else t("dashboard_posts_disagree_lr_dl", lang)
    )

    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)

    if method_name == "Deep Learning Transformer":
        keep_cols = [
            "mention_id", "company_name", "title", "author", "published_at",
            "lr_label", "lr_score", "dl_label", "dl_score"
        ]
    else:
        keep_cols = [
            "mention_id", "company_name", "title", "author", "published_at",
            "lr_label", "lr_score", "vader_label", "vader_score"
        ]

    existing = [c for c in keep_cols if c in disagreements.columns]

    disagreements = disagreements[existing].sort_values(
        "published_at", ascending=False
    ).head(limit_rows)

    st.dataframe(disagreements, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
