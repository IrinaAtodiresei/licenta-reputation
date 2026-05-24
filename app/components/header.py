import os
import base64
import streamlit as st

from components.layout import render_base_styles


def render_header() -> None:
    """
    Global header component:
    - Blue background (#0A66C2)
    - Logo with gradient fade background
    - White title and subtitle
    - Responsive flex layout
    """
    render_base_styles()

    # Load and encode logo
    logo_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
    )
    try:
        logo_b64 = base64.b64encode(open(logo_path, "rb").read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="app-header__logo" />'
    except FileNotFoundError:
        logo_html = ""

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header__logo-wrapper">
                {logo_html}
            </div>
            <div class="app-header__content">
                <h1 class="app-header__title">Reputation & Sentiment Dashboard</h1>
                <p class="app-header__subtitle">
                    Monitor reputația și sentimentul online pentru companii tehnologice
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
