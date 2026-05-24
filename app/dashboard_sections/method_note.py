import streamlit as st
from translations import t
from dashboard_sections.method_note_helpers import (
    generate_method_note,
    generate_method_note_ro,
)

def render_method_note(method_name, lang):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"<h2 class='section-title'>{t('dashboard_method_note_title', lang)}</h2>",
        unsafe_allow_html=True,
    )

    if lang == "ro":
        text = generate_method_note_ro(method_name)
    else:
        text = generate_method_note(method_name)

    st.info(text)
    st.markdown("</div>", unsafe_allow_html=True)
