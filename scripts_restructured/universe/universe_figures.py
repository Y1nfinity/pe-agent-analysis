"""
universe_figures.py
===================

STEP 3 of the Universe Pipeline.

Loads the universe panels created in STEP 2 (universe_panels.py)
and produces high-quality figures for:

    1. Universe Entry-Year Analysis
    2. Universe Exit-Year Analysis
    3. Universe Deal-Year Activity

Inputs:
    outputs/panels_universe/*.csv

Outputs:
    outputs/figures_universe/*.png
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================================
# Helpers
# =====================================================================

def stacked_bar(df, index, columns, title, outfile):
    """
    Create a stacked bar chart: counts (not normalized)
    """
    plt.figure(figsize=(13, 7))
    df.set_index(index)[columns].plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel(index)
    plt.ylabel("Count")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


def stacked_bar_pct(df, index, columns, title, outfile):
    """
    Create a stacked bar chart: percentages (normalized)
    """
    plt.figure(figsize=(13, 7))
    data = df.set_index(index)[columns]
    pct = data.div(data.sum(axis=1), axis=0) * 100
    pct.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel(index)
    plt.ylabel("Percent")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


def line_plot(df, x, y, title, outfile):
    plt.figure(figsize=(13, 6))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


# =====================================================================
# MAIN PLOTTING PIPE
# =====================================================================

def run_universe_figures(panel_dir: Path, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all three panels
    entry = pd.read_csv(panel_dir / "universe_entry_year_panel.csv")
    exit_ = pd.read_csv(panel_dir / "universe_exit_year_panel.csv")
    deal  = pd.read_csv(panel_dir / "universe_deal_year_panel.csv")

    # ==============================================================
    # 1. ENTRY-YEAR FIGURES
    # ==============================================================

    line_plot(
        entry,
        "entry_year", "N_deals",
        "Universe — Deal Counts by Entry Year",
        fig_dir / "EntryYear_DealCounts.png"
    )

    line_plot(
        entry,
        "entry_year", "frac_exits",
        "Universe — Fraction That Are Exits (Entry-Year View)",
        fig_dir / "EntryYear_FracExits.png"
    )

    # identify exit categories automatically
    exit_cols = [c for c in entry.columns if c not in
                 ["entry_year", "N_deals", "N_exits", "frac_exits"]]

    if exit_cols:
        stacked_bar(
            entry,
            index="entry_year",
            columns=exit_cols,
            title="Universe — Exit Composition by Entry Year (Counts)",
            outfile=fig_dir / "EntryYear_ExitBreak_Counts.png"
        )

        stacked_bar_pct(
            entry,
            index="entry_year",
            columns=exit_cols,
            title="Universe — Exit Composition by Entry Year (Percent)",
            outfile=fig_dir / "EntryYear_ExitBreak_Percent.png"
        )

    # ==============================================================
    # 2. EXIT-YEAR FIGURES
    # ==============================================================

    line_plot(
        exit_,
        "exit_year", "N_exits",
        "Universe — Exit Counts by Exit Year",
        fig_dir / "ExitYear_NExits.png"
    )

    line_plot(
        exit_,
        "exit_year", "exit_value_usd_m",
        "Universe — Exit Value by Exit Year",
        fig_dir / "ExitYear_ExitValue_USDmm.png"
    )

    exit_cols2 = [c for c in exit_.columns if c not in
                  ["exit_year", "N_exits", "exit_value_usd_m"]]

    if exit_cols2:
        stacked_bar(
            exit_,
            index="exit_year",
            columns=exit_cols2,
            title="Universe — Exit Composition (Counts)",
            outfile=fig_dir / "ExitYear_ExitBreak_Counts.png"
        )

        stacked_bar_pct(
            exit_,
            index="exit_year",
            columns=exit_cols2,
            title="Universe — Exit Composition (Percent)",
            outfile=fig_dir / "ExitYear_ExitBreak_Percent.png"
        )

    # ==============================================================
    # 3. DEAL-YEAR (ACTIVITY) FIGURES
    # ==============================================================

    line_plot(
        deal,
        "transaction_year", "N_deals",
        "Universe — Deal Activity by Year (Counts)",
        fig_dir / "DealYear_Activity_Counts.png"
    )

    line_plot(
        deal,
        "transaction_year", "total_value_usd_m",
        "Universe — Deal Activity by Year (USD mm)",
        fig_dir / "DealYear_Activity_Value.png"
    )

    deal_cols = [c for c in deal.columns if c not in
                 ["transaction_year", "N_deals", "total_value_usd_m"]]

    if deal_cols:
        stacked_bar(
            deal,
            index="transaction_year",
            columns=deal_cols,
            title="Universe — Deal Type Composition (Counts)",
            outfile=fig_dir / "DealYear_TypeBreak_Counts.png"
        )

        stacked_bar_pct(
            deal,
            index="transaction_year",
            columns=deal_cols,
            title="Universe — Deal Type Composition (Percent)",
            outfile=fig_dir / "DealYear_TypeBreak_Percent.png"
        )

    print("✅ Universe Figures Saved:", fig_dir.resolve())


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    PANEL_DIR = ROOT / "outputs" / "panels_universe"
    FIG_DIR   = ROOT / "outputs" / "figures_universe"

    run_universe_figures(PANEL_DIR, FIG_DIR)
