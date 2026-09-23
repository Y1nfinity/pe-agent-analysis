"""
analysis.py
-----------
Purpose:
    Provide empirical summaries and descriptive analytics for Private Equity exits.
    Generates:
        - Exit counts and trends over time
        - Proportions by exit category
        - Vintage-level analyses
        - Quality/missingness checks by year

Usage Example:
    from scripts_restructured import analysis as an

    df = an.ensure_year_column(categorized_df)
    counts = an.counts_by_year(df)
    breakdown = an.counts_by_year_and_category(df)
    vintage = an.vintage_summary(df)
    quality = an.quality_table(df)
"""

from __future__ import annotations
import pandas as pd
from typing import Optional

# -----------------------------------------------------------------------------
# Year Creation
# -----------------------------------------------------------------------------
def ensure_year_column(df):
    """Extract a reliable year from any available date column."""
    date_cols = [c for c in ["close_date", "deal_date", "exit_date", "announcement_date"] if c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # Choose first available date column per row
    df["year"] = df[date_cols].bfill(axis=1).iloc[:, 0].dt.year
    return df

# -----------------------------------------------------------------------------
# Core Summary Functions
# -----------------------------------------------------------------------------
def counts_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return annual exit counts.
    Columns: ['year', 'count']
    """
    if "year" not in df.columns:
        df = ensure_year_column(df)
    return df.groupby("year").size().rename("count").reset_index()

def counts_by_year_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual counts and shares by exit category.
    Columns: ['year', 'exit_category', 'count', 'share']
    """
    if "year" not in df.columns:
        df = ensure_year_column(df)
    if "exit_category" not in df.columns:
        raise KeyError("exit_category not found. Run assign_exit_category() first.")
    grouped = df.groupby(["year", "exit_category"]).size().rename("count").reset_index()
    totals = grouped.groupby("year")["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals
    return grouped

def vintage_summary(df: pd.DataFrame, vintage_col: str = "vintage") -> pd.DataFrame:
    """
    Summarize exit distributions by fund vintage if available.
    Columns: ['vintage', 'exit_category', 'count', 'share']
    """
    if vintage_col not in df.columns:
        raise KeyError(f"Expected column '{vintage_col}' not found in dataset.")
    if "exit_category" not in df.columns:
        raise KeyError("exit_category not found. Run assign_exit_category() first.")
    g = df.groupby([vintage_col, "exit_category"]).size().rename("count").reset_index()
    g["share"] = g.groupby(vintage_col)["count"].transform(lambda x: x / x.sum())
    return g

# -----------------------------------------------------------------------------
# Quality Checks
# -----------------------------------------------------------------------------
def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a small table summarizing:
        - null rates by key column
        - number of exits per year
    """
    if "year" not in df.columns:
        df = ensure_year_column(df)
    key_cols = ["deal_id", "company_id", "company_name", "exit_category", "deal_size_usd_m"]
    present_cols = [c for c in key_cols if c in df.columns]
    result = {}
    for c in present_cols:
        result[f"{c}_null_rate"] = df[c].isna().mean()
    result["n_exits_total"] = len(df)
    if "year" in df.columns:
        result["n_exits_by_year"] = df.groupby("year").size().to_dict()
    return pd.Series(result, name="quality_summary")

# -----------------------------------------------------------------------------
# Combined Reporter
# -----------------------------------------------------------------------------
def build_summary_package(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build a dictionary of all key descriptive summaries in one step.
    Returns:
        {
            'counts_by_year': ...,
            'counts_by_year_and_category': ...,
            'vintage_summary': ...,
            'quality_table': ...
        }
    """
    df = ensure_year_column(df)
    results = {
        "counts_by_year": counts_by_year(df),
        "counts_by_year_and_category": counts_by_year_and_category(df),
        "quality_table": quality_table(df).to_frame().T,
    }
    if "vintage" in df.columns:
        results["vintage_summary"] = vintage_summary(df)
    return results
