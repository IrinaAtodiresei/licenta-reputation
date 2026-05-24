"""KPI computation functions for the dashboard."""
import pandas as pd


def compute_kpis(summary: pd.DataFrame) -> dict:
    """
    Compute key KPIs from summary.
    (Disagreement is NOT computed here — it is computed in the Dashboard tab,
     exactly like in the old application.)
    """
    if summary.empty:
        return {
            "total_mentions": 0,
            "avg_score": 0.0,
            "pct_negative": 0.0,
        }

    # ============================
    # TOTAL MENTIONS
    # ============================
    total_mentions = int(summary["total_mentions"].sum())

    # ============================
    # AVG SCORE (weighted)
    # ============================
    if total_mentions > 0 and "avg_score" in summary.columns and "total_mentions" in summary.columns:
        avg_score = float((summary["avg_score"] * summary["total_mentions"]).sum() / total_mentions)
    else:
        avg_score = float(summary["avg_score"].mean()) if "avg_score" in summary.columns else 0.0

    # ============================
    # % NEGATIVE (weighted)
    # ============================
    if total_mentions > 0 and "negatives" in summary.columns:
        total_negatives = int(summary["negatives"].sum())
        pct_negative = float((total_negatives / total_mentions) * 100)
    else:
        pct_negative = 0.0

    return {
        "total_mentions": total_mentions,
        "avg_score": avg_score,
        "pct_negative": pct_negative,
    }


def compute_sentiment_counts(summary: pd.DataFrame) -> dict:
    """Compute total positive, negative, neutral counts from summary."""
    if summary.empty:
        return {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        }

    positive_count = int(summary["positives"].sum()) if "positives" in summary.columns else 0
    negative_count = int(summary["negatives"].sum()) if "negatives" in summary.columns else 0
    neutral_count = int(summary["neutrals"].sum()) if "neutrals" in summary.columns else 0

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
    }
