"""Filter and data organization functions."""
import pandas as pd


def filter_by_company(summary: pd.DataFrame, disagree_company: pd.DataFrame, disagreements: pd.DataFrame, company: str):
    """Apply company filter to all dataframes."""
    if company != "All":
        summary = summary[summary["company_name"] == company]
        if not disagree_company.empty and "company_name" in disagree_company.columns:
            disagree_company = disagree_company[disagree_company["company_name"] == company]
        if not disagreements.empty and "company_name" in disagreements.columns:
            disagreements = disagreements[disagreements["company_name"] == company]
    return summary, disagree_company, disagreements


def reorder_summary_columns(summary):
    desired_order = [
        "company_id",
        "company_name",
        "method",
        "mentions",
        "avg_score",
        "positives",
        "negatives",
        "neutrals",
        "pct_negative",
       # "total_mentions",
    ]

    existing_cols = [c for c in desired_order if c in summary.columns]

    if "mentions" in summary.columns:
        return summary[existing_cols].sort_values("mentions", ascending=False)
    else:
        return summary[existing_cols]




def prepare_disagreement_display(disagreements: pd.DataFrame, method: str = "lr_vader") -> pd.DataFrame:
    """Prepare disagreement data for display."""
    if disagreements.empty:
        return disagreements
    
    if method == "lr_dl":
        keep_cols = [
            "mention_id",
            "company_name",
            "title",
            "author",
            "published_at",
            "lr_label",
            "lr_score",
            "dl_label",
            "dl_score",
        ]
    else:  # lr_vader or default
        keep_cols = [
            "mention_id",
            "company_name",
            "title",
            "author",
            "published_at",
            "lr_label",
            "lr_score",
            "vader_label",
            "vader_score",
        ]
    
    existing_cols = [col for col in keep_cols if col in disagreements.columns]
    
    if not existing_cols:
        return disagreements
    
    return disagreements[existing_cols].sort_values("published_at", ascending=False)


def get_company_list(summary: pd.DataFrame) -> list:
    """Extract unique company names from summary."""
    if summary.empty or "company_name" not in summary.columns:
        return []
    
    companies = summary["company_name"].dropna().unique().tolist()
    return sorted(companies)


def apply_filters(data: dict, company: str, disagree_checkbox: bool = False) -> dict:
    """Apply all filters to dashboard data."""
    summary, disagree_company, disagreements = filter_by_company(
        data.get("summary", pd.DataFrame()),
        data.get("disagree_company", pd.DataFrame()),
        data.get("disagreements", pd.DataFrame()),
        company
    )
    
    if disagree_checkbox and not disagreements.empty:
        # Filter to show only disagreement records
        pass  # Already filtered by method above
    
    return {
        "summary": summary,
        "disagree_company": disagree_company,
        "disagreements": disagreements,
    }
