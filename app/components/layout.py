import os
import streamlit as st


def render_base_styles() -> None:
    """
    Loads global CSS from assets/styles.css exactly once.
    Prevents duplicate injection and layout instability.
    """
    if st.session_state.get("_components_styles_loaded", False):
        return

    css_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    )

    try:
        with open(css_path, "r", encoding="utf-8") as css_file:
            css = css_file.read()

        # Inject CSS once
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"[layout] Global styles file not found: {css_path}")

    st.session_state["_components_styles_loaded"] = True


def section(title: str) -> None:
    render_base_styles()
    st.markdown(
        f"""
        <div class="app-card">
            <h2 class="section-title">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(content_function) -> None:
    render_base_styles()
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)


def two_columns(left_fn, right_fn, ratio=(1, 1)) -> None:
    render_base_styles()
    col_left, col_right = st.columns(ratio, gap="large")
    with col_left:
        left_fn()
    with col_right:
        right_fn()


def three_columns(fn1, fn2, fn3, ratio=(1, 1, 1)) -> None:
    render_base_styles()
    col1, col2, col3 = st.columns(ratio, gap="large")
    with col1:
        fn1()
    with col2:
        fn2()
    with col3:
        fn3()
