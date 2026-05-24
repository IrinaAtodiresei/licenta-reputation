import streamlit as st

from components.layout import render_base_styles


def kpi_card(label: str, value: str, help_text: str = None, icon: str = None) -> None:
    render_base_styles()

    icon_html = f'<span class="app-kpi-icon">{icon}</span>' if icon else ""
    help_html = f'<span class="app-kpi-tooltip" title="{help_text}">?</span>' if help_text else ""

    st.markdown(
        f"""
        <div class="app-kpi-card">
            <div style="display:flex; align-items:center; gap: 8px; margin-bottom: 8px;">
                {icon_html}
                <div class="app-kpi-label">{label}{help_html}</div>
            </div>
            <p class="app-kpi-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
