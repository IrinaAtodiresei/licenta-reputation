import streamlit as st
import os
import base64

def render_home():

    # ============================================================
    # 1) MARCHEAZĂ BODY CA LANDING PAGE (pentru CSS)
    # ============================================================
    st.markdown("""
    <script>
        document.body.classList.add('landing-page');
    </script>
    """, unsafe_allow_html=True)

    # ============================================================
    # 2) ASCUNDE DOAR SIDEBAR-UL ȘI HEADER-UL STREAMLIT
    #    (NU FOLOSEȘTE display:none global)
    # ============================================================
    st.markdown("""
    <style>
        body.landing-page [data-testid="stSidebar"] {display: none !important;}
        body.landing-page header {visibility: hidden !important;}
        body.landing-page footer {visibility: hidden !important;}
    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # 3) CUSTOM CSS LANDING PAGE
    # ============================================================
    st.markdown("""
    <style>

      /* HEADER */
      .header-box {
        background: white;
        padding: 32px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-radius: 12px;
        margin-bottom: 24px;
      }
      .header-box img { height: 64px; }
      .header-box h1 { margin: 0; font-size: 36px; color: #FF8A00; }
      .header-box p { margin: 4px 0 0 0; font-size: 16px; color: #FFC300; }

      /* HERO */
      .hero {
        text-align: center;
        padding: 80px 32px;
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 40px;
      }
      .hero h2 { font-size: 36px; margin-bottom: 16px; }
      .hero p {
        font-size: 18px;
        color: #D1D5DB;
        line-height: 1.6;
        max-width: 700px;
        margin: 0 auto;
      }

      /* BUTTON */
      div.stButton > button {
        width: 100%;
        background-color: #FF8A00 !important;
        color: white !important;
        padding: 14px 32px !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
        cursor: pointer !important;
        transition: 0.2s !important;
      }
      div.stButton > button:hover {
        background-color: #e67a00 !important;
      }

      /* FEATURES */
      .feature-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 6px solid;
      }

      /* FOOTER */
      .footer-box {
        background: #F0F0F0;
        color: #444;
        padding: 32px;
        margin-top: 60px;
        text-align: center;
        border-top: 3px solid #FF8A00;
        border-radius: 12px;
      }

    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # 4) HEADER LANDING PAGE
    # ============================================================
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    col_logo, col_text = st.columns([0.15, 0.85])
    with col_logo:
        st.image(logo_path, width=64)
    with col_text:
        st.markdown("""
        <h1 style="margin: 0; font-size: 36px; color: #FF8A00;">Reputation Dashboard</h1>
        <p style="margin: 4px 0 0 0; font-size: 16px; color: #FFC300;">
            Analiză reputațională · AI Insights · Marketing Intelligence
        </p>
        """, unsafe_allow_html=True)

    # ============================================================
    # 5) HERO SECTION
    # ============================================================
    st.markdown("""
    <div class="hero">
      <h2>🧠 Analiză reputațională asistată de AI</h2>
      <p>
        Această aplicație analizează reputația online a marilor companii tehnologice
        pe baza discuțiilor publice de pe Reddit.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 6) BUTTON TO ENTER APP
    # ============================================================
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Explorează analiza reputației", use_container_width=True):
            st.session_state.entered = True
            st.rerun()

    # ============================================================
    # 7) FEATURES
    # ============================================================
    st.markdown("<h2 style='text-align: center; font-size: 28px; color: #007BFF;'>Funcționalități principale</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left-color:#FFA800;">
          <h3>📊 Dashboard ML</h3>
          <p>Vizualizări interactive pentru analiza sentimentului.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left-color:#FFC300;">
          <h3>🧠 AI Insights</h3>
          <p>Interpretare semantică generată de LLM-uri.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card" style="border-left-color:#007BFF;">
          <h3>📂 Proof of Source</h3>
          <p>Insight-uri susținute de exemple reale din dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # 8) FOOTER
    # ============================================================
    footer_html_logo = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(f"""
    <div class="footer-box">
      <img src="data:image/png;base64,{footer_html_logo}" width="40" />
      <div style="font-size:15px; margin-bottom:6px;">
        Reputation Dashboard — Licență 2026 — <strong>Irina Atodiresei</strong>
      </div>
      <div style="margin-bottom:6px;">
        🔗 <a href="https://github.com/IrinaAtodiresei/licenta-reputation" target="_blank">GitHub Repository</a>
      </div>
      <div style="font-size:13px; color:#666;">
        Built with Streamlit · Machine Learning · LLM · Groq · LangChain
      </div>
    </div>
    """, unsafe_allow_html=True)
