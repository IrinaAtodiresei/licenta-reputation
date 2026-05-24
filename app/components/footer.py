from datetime import datetime
import streamlit as st

from components.layout import render_base_styles


def render_footer(custom_text: str = None) -> None:
    render_base_styles()
    year = datetime.now().year
    footer_text = custom_text or "Built with Streamlit · Machine Learning · Groq · LangChain"
    st.markdown(
        f"""
        <footer class="app-footer">
            <span>Reputation Dashboard — © {year}</span>
            <span>{footer_text}</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )
