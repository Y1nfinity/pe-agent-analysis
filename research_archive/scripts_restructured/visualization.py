"""
visualization.py
----------------
Purpose:
    Produce publication-ready charts from analysis outputs:
        - Bar plots for annual exit counts
        - Stacked bars for exit mix (by year, by category)
        - Optional share-based figures (for percentages)
        - Minimal matplotlib dependency

Rules:
    * No seaborn
    * One chart per figure
    * No manual color choices
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def _prep_outfile(outfile: Optional[str]) -> Optional[Path]:
    """Ensure output directory exists if saving."""
    if outfile:
        path = Path(outfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return None


# -----------------------------------------------------------------------------
# 1. Bar chart — exits by year
# -----------------------------------------------------------------------------
def plot_exits_by_year(df_counts: pd.DataFrame, title: str = "Private Equity Exits by Year",
                       outfile: Optional[str] = None) -> None:
    """
    Parameters
    ----------
    df_counts : pd.DataFrame
        Must contain columns ['year', 'count']
    """
    plt.figure()
    plt.bar(df_counts["year"], df_counts["count"])
    plt.xlabel("Year")
    plt.ylabel("Number of Exits")
    plt.title(title)
    if outfile:
        outpath = _prep_outfile(outfile)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 2. Stacked bar chart — exit mix by category (counts)
# -----------------------------------------------------------------------------
def plot_exit_mix_by_category(df_mix: pd.DataFrame, title: str = "Exit Mix by Year (Counts)",
                              outfile: Optional[str] = None) -> None:
    """
    Parameters
    ----------
    df_mix : pd.DataFrame
        Must contain columns ['year', 'exit_category', 'count']
    """
    pivot = df_mix.pivot(index="year", columns="exit_category", values="count").fillna(0)
    plt.figure()
    bottom = None
    for col in pivot.columns:
        if bottom is None:
            plt.bar(pivot.index, pivot[col].values, label=col)
            bottom = pivot[col].values
        else:
            plt.bar(pivot.index, pivot[col].values, bottom=bottom, label=col)
            bottom = bottom + pivot[col].values
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    if outfile:
        outpath = _prep_outfile(outfile)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 3. Stacked bar chart — exit mix by category (shares)
# -----------------------------------------------------------------------------
def plot_exit_share_by_category(df_mix: pd.DataFrame, title: str = "Exit Mix by Year (Shares)",
                                outfile: Optional[str] = None) -> None:
    """
    Parameters
    ----------
    df_mix : pd.DataFrame
        Must contain columns ['year', 'exit_category', 'share']
    """
    pivot = df_mix.pivot(index="year", columns="exit_category", values="share").fillna(0)
    plt.figure()
    bottom = None
    for col in pivot.columns:
        if bottom is None:
            plt.bar(pivot.index, pivot[col].values, label=col)
            bottom = pivot[col].values
        else:
            plt.bar(pivot.index, pivot[col].values, bottom=bottom, label=col)
            bottom = bottom + pivot[col].values
    plt.xlabel("Year")
    plt.ylabel("Share of Exits")
    plt.title(title)
    plt.legend()
    if outfile:
        outpath = _prep_outfile(outfile)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 4. Bar chart — take-private vs private-to-private share
# -----------------------------------------------------------------------------
def plot_take_private_vs_private_to_private(df_mix: pd.DataFrame,
                                            title: str = "Take-Private vs Private-to-Private Exits",
                                            outfile: Optional[str] = None) -> None:
    """
    Highlights the comparative trend between take-private and private-to-private exits.
    Expects 'exit_category' and 'share' columns.
    """
    subset = df_mix[df_mix["exit_category"].isin(["take_private", "private_to_private"])]
    pivot = subset.pivot(index="year", columns="exit_category", values="share").fillna(0)
    plt.figure()
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.xlabel("Year")
    plt.ylabel("Share of Exits")
    plt.title(title)
    plt.legend()
    if outfile:
        outpath = _prep_outfile(outfile)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
