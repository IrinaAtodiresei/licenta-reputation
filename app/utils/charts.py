"""Chart rendering functions using new design system."""
import streamlit as st
import pandas as pd


def render_metric_card(label: str, value: str, subtitle: str = ""):
    """Render a single metric card with modern design."""
    st.markdown(
        f"""
        <div style="
            padding: 16px 18px;
            margin-bottom: 16px;
            border-radius: 14px;
            background-color: #f8fafc;
            border: 1px solid #e5e7eb;
            min-height: 105px;
        ">
            <div style="
                font-size: 0.85rem;
                color: #6b7280;
                margin-bottom: 8px;
            ">
                {label}
            </div>
            <div style="
                font-size: 2rem;
                font-weight: 600;
                color: #111827;
                line-height: 1.1;
            ">
                {value}
            </div>
            {f'<div style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metrics_grid(metrics: list, columns: int = 4):
    """Render multiple metric cards in a grid layout."""
    cols = st.columns(columns)
    for index, metric in enumerate(metrics):
        col = cols[index % columns]
        with col:
            render_metric_card(metric["label"], metric["value"], metric.get("subtitle", ""))


def render_mentions_chart(summary: pd.DataFrame):
    """Render mentions by company bar chart."""
    st.markdown('<div class="rd-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">📊 Mentions by company</h3>', unsafe_allow_html=True)
    
    if not summary.empty and "company_name" in summary.columns and "total_mentions" in summary.columns:
        mentions_chart = summary[["company_name", "total_mentions"]].sort_values("total_mentions", ascending=False)
        st.bar_chart(mentions_chart.set_index("company_name"))
    else:
        st.info("No data available for mentions chart.")
    
    st.markdown('</div>', unsafe_allow_html=True)



def render_pct_negative_chart(summary: pd.DataFrame):
    """Render negative sentiment percentage by company."""
    st.markdown('<div class="rd-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">⚠️ Negative sentiment % by company</h3>', unsafe_allow_html=True)
    
    if not summary.empty and "company_name" in summary.columns and "negatives" in summary.columns and "total_mentions" in summary.columns:
        df = summary.copy()
        df["pct_negative"] = (df["negatives"] / df["total_mentions"]) * 100
        negative_chart = df[["company_name", "pct_negative"]].sort_values("pct_negative", ascending=False)
        st.bar_chart(negative_chart.set_index("company_name"))
    else:
        st.info("No data available for negative sentiment chart.")
    
    st.markdown('</div>', unsafe_allow_html=True)



def render_avg_score_chart(summary: pd.DataFrame):
    """Render average score by company."""
    st.markdown('<div class="rd-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">📈 Average confidence score by company</h3>', unsafe_allow_html=True)
    
    if not summary.empty and "company_name" in summary.columns and "avg_score" in summary.columns:
        score_chart = summary[["company_name", "avg_score"]].sort_values("avg_score", ascending=False)
        st.bar_chart(score_chart.set_index("company_name"))
    else:
        st.info("No data available for average score chart.")
    
    st.markdown('</div>', unsafe_allow_html=True)



def render_sentiment_distribution_chart(summary: pd.DataFrame):
    """Render stacked bar chart of sentiment distribution."""
    st.markdown('<div class="rd-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">Sentiment distribution</h3>', unsafe_allow_html=True)
    
    if not summary.empty and all(col in summary.columns for col in ["company_name", "positives", "neutrals", "negatives"]):
        chart_df = summary[["company_name", "positives", "neutrals", "negatives"]].set_index("company_name")
        st.bar_chart(chart_df)
    else:
        st.info("No data available for sentiment distribution.")
    
    st.markdown('</div>', unsafe_allow_html=True)
