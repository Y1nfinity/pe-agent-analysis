"""
figures_entry_exit.py — STEP 4 (FINAL)
======================================
Generates all charts for:

1. Universe entry-year panels
2. Take-private entry-year panels
3. Take-private exit-year panels
4. Exit composition (count, %, value)
5. Holding periods
6. Universe vs TP comparison

Inputs:
    outputs/panels/*.csv
Outputs:
    outputs/figures/*.png
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")


# =============================================================================
# Helpers
# =============================================================================

def _savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {path}")


# =============================================================================
# 1. Universe Figures
# =============================================================================

def plot_universe_panels(univ: pd.DataFrame, out: Path):

    # Entries per year
    plt.figure(figsize=(12,5))
    plt.plot(univ["entry_year"], univ["N_entries"], marker="o")
    plt.title("Universe — Number of Entries per Year")
    plt.xlabel("Year")
    plt.ylabel("Entries")
    _savefig(out / "Figure_Universe_Entries.png")

    # Exits per year
    plt.figure(figsize=(12,5))
    plt.plot(univ["entry_year"], univ["N_exits"], marker="o")
    plt.title("Universe — Number of Exits per Year")
    plt.xlabel("Year")
    plt.ylabel("Exits")
    _savefig(out / "Figure_Universe_Exits.png")

    # Exit fraction
    plt.figure(figsize=(12,5))
    plt.plot(univ["entry_year"], univ["frac_exited"], marker="o")
    plt.title("Universe — Exit Rate (Exits / Entries)")
    plt.xlabel("Year")
    plt.ylabel("Exit Rate")
    _savefig(out / "Figure_Universe_ExitRate.png")


# =============================================================================
# 2. Take-Private Entry-Year Figures
# =============================================================================

def plot_tp_entry_year_panels(tp: pd.DataFrame, out: Path):

    # Entries
    plt.figure(figsize=(12,5))
    plt.plot(tp["entry_year"], tp["N_entries"], marker="o")
    plt.title("Take-Private — Entries per Year")
    plt.xlabel("Entry Year")
    plt.ylabel("Entries")
    _savefig(out / "TP_EntryYear_Entries.png")

    # Exits
    plt.figure(figsize=(12,5))
    plt.plot(tp["entry_year"], tp["N_exits"], marker="o")
    plt.title("Take-Private — Exits per Entry Year")
    plt.xlabel("Entry Year")
    plt.ylabel("Exits")
    _savefig(out / "TP_EntryYear_Exits.png")

    # Exit composition (counts)
    exit_types = [c for c in tp.columns if c not in [
        "entry_year", "N_entries", "N_exits", "entry_value_usd_m",
        "exit_value_usd_m", "frac_exited_by_count", "frac_exited_by_value",
        "hp_mean", "hp_std"
    ]]

    comp = tp[["entry_year"] + exit_types].set_index("entry_year")

    comp.plot(kind="bar", stacked=True, figsize=(14,7))
    plt.title("TP — Exit Composition by Entry Year (Counts)")
    plt.ylabel("Exit Count")
    _savefig(out / "TP_EntryYear_ExitBreak_Counts.png")

    # Composition (%)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct.plot(kind="bar", stacked=True, figsize=(14,7))
    plt.title("TP — Exit Composition by Entry Year (%)")
    plt.ylabel("Percent")
    _savefig(out / "TP_EntryYear_ExitBreak_Percent.png")


# =============================================================================
# 3. Take-Private Exit-Year Figures
# =============================================================================

def plot_tp_exit_year_panels(tp: pd.DataFrame, out: Path):

    # Exit counts
    plt.figure(figsize=(12,5))
    plt.plot(tp["exit_year"], tp["N_exits"], marker="o")
    plt.title("TP — Exits per Exit Year")
    plt.xlabel("Exit Year")
    plt.ylabel("Exits")
    _savefig(out / "TP_ExitYear_ExitCounts.png")

    # Exit values
    plt.figure(figsize=(12,5))
    plt.plot(tp["exit_year"], tp["exit_value_usd_m"], marker="o")
    plt.title("TP — Exit Value per Exit Year ($mm)")
    plt.xlabel("Exit Year")
    plt.ylabel("Exit Value ($mm)")
    _savefig(out / "TP_ExitYear_ExitValues.png")

    # Exit composition (counts)
    comp = tp.set_index("exit_year").drop(columns=[
        "N_exits", "exit_value_usd_m", "hp_mean", "hp_std"
    ])

    comp.plot(kind="bar", stacked=True, figsize=(14,7))
    plt.title("TP — Exit Composition by Exit Year (Counts)")
    _savefig(out / "TP_ExitYear_ExitBreak_Counts.png")

    # Composition (%)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct.plot(kind="bar", stacked=True, figsize=(14,7))
    plt.title("TP — Exit Composition by Exit Year (%)")
    _savefig(out / "TP_ExitYear_ExitBreak_Percent.png")


# =============================================================================
# 4 + 5. Exit Composition (counts, percent, value)
# =============================================================================

def plot_exit_composition_full(pairs: pd.DataFrame, out: Path):

    df = pairs.copy()
    df = df[df["exit_date"].notna()]

    # Count composition
    plt.figure(figsize=(12,6))
    df["exit_category"].value_counts().plot(kind="bar")
    plt.title("Exit Composition (Counts)")
    plt.xlabel("Exit Type")
    plt.ylabel("Count")
    _savefig(out / "ExitComposition_Counts.png")

    # Percent composition
    plt.figure(figsize=(12,6))
    (df["exit_category"].value_counts(normalize=True)*100).plot(kind="bar")
    plt.title("Exit Composition (%)")
    plt.ylabel("Percent")
    _savefig(out / "ExitComposition_Percent.png")

    # Value-weighted
    dfv = df[df["deal_size_usd_m_exit"].notna()]
    if not dfv.empty:
        plt.figure(figsize=(12,6))
        dfv.groupby("exit_category")["deal_size_usd_m_exit"].sum().plot(kind="bar")
        plt.title("Exit Composition by Deal Value ($mm)")
        plt.ylabel("Total Deal Value ($mm)")
        _savefig(out / "ExitComposition_Value.png")


# =============================================================================
# 6. Holding Period Figures
# =============================================================================

def plot_holding_periods(pairs: pd.DataFrame, out: Path):

    df = pairs[pairs["exit_date"].notna()]

    # Boxplot across exit types
    plt.figure(figsize=(14,7))
    sns.boxplot(data=df, x="exit_category", y="holding_period_years")
    plt.title("Holding Period by Exit Category")
    plt.xticks(rotation=45)
    _savefig(out / "HoldingPeriod_Boxplot.png")

    # Holding period by exit year
    df["exit_year"] = df["exit_date"].dt.year
    plt.figure(figsize=(14,6))
    plt.plot(df.groupby("exit_year")["holding_period_years"].mean(), marker="o")
    plt.title("Average Holding Period by Exit Year")
    _savefig(out / "HoldingPeriod_ByExitYear.png")


# =============================================================================
# 7. Universe vs Take-Private Comparison
# =============================================================================

def plot_universe_vs_tp(univ: pd.DataFrame, tp: pd.DataFrame, out: Path):

    merged = univ.merge(tp, on="entry_year", how="inner", suffixes=("_univ", "_tp"))

    # Entries comparison
    plt.figure(figsize=(12,6))
    plt.plot(merged["entry_year"], merged["N_entries_univ"], label="Universe")
    plt.plot(merged["entry_year"], merged["N_entries_tp"], label="Take-Private")
    plt.legend()
    plt.title("Entries — Universe vs Take-Private")
    _savefig(out / "Universe_vs_TP_Entries.png")

    # Exit rate comparison
    plt.figure(figsize=(12,6))
    plt.plot(merged["entry_year"], merged["frac_exited_univ"], label="Universe")
    plt.plot(merged["entry_year"], merged["frac_exited_by_count"], label="Take-Private")
    plt.legend()
    plt.title("Exit Rate — Universe vs Take-Private")
    _savefig(out / "Universe_vs_TP_ExitRate.png")

    # Holding periods
    plt.figure(figsize=(12,6))
    plt.plot(merged["entry_year"], merged["hp_mean_univ"], label="Universe")
    plt.plot(merged["entry_year"], merged["hp_mean_tp"], label="Take-Private")
    plt.legend()
    plt.title("Holding Period — Universe vs Take-Private")
    _savefig(out / "Universe_vs_TP_HoldingPeriods.png")


# =============================================================================
# Main driver
# =============================================================================

def run_all_figures(panel_dir: Path, pairs_path: Path, outdir: Path):

    univ = pd.read_csv(panel_dir / "universe_entry_year_panel.csv")
    tp_entry = pd.read_csv(panel_dir / "tp_entry_year_panel.csv")
    tp_exit = pd.read_csv(panel_dir / "tp_exit_year_panel.csv")
    pairs = pd.read_parquet(pairs_path)

    outdir.mkdir(parents=True, exist_ok=True)

    plot_universe_panels(univ, outdir)
    plot_tp_entry_year_panels(tp_entry, outdir)
    plot_tp_exit_year_panels(tp_exit, outdir)
    plot_exit_composition_full(pairs, outdir)
    plot_holding_periods(pairs, outdir)
    plot_universe_vs_tp(univ, tp_entry, outdir)

    print("🎉 STEP 4 graphics complete.")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    panel_dir = ROOT / "outputs" / "panels"
    pairs_path = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    figs = ROOT / "outputs" / "figures"

    run_all_figures(panel_dir, pairs_path, figs)
