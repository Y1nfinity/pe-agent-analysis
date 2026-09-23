"""
universe_panels.py
===================

STEP 2 of the Universe Pipeline.

Creates:
    1. Universe Entry-Year Panel
    2. Universe Exit-Year Panel
    3. Universe Deal-Year Activity Panel

Inputs:
    data/clean/pitchbook_master_clean.parquet

Outputs:
    outputs/panels_universe/*.csv
    outputs/figures_universe/*.png
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# LOAD
# =============================================================================

def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find clean PitchBook master: {path}")
    return pd.read_parquet(path)


# =============================================================================
# PANEL 1 — UNIVERSE ENTRY-YEAR VIEW
# =============================================================================

def build_universe_entry_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For all entries (every deal):
        - N deals
        - N deals categorized as "exits" (based on exit_category)
        - Breakdown by exit_category
        - Fraction exits
    """

    df = df.copy()
    df["entry_year"] = df["transaction_year"]

    # What counts as an "exit"?
    # Anything that has an exit_category that is NOT "other"
    df["is_exit"] = df["exit_category"].ne("other")

    grouped = df.groupby("entry_year")

    out = pd.DataFrame({
        "N_deals": grouped.size(),
        "N_exits": grouped["is_exit"].sum(),
    })

    out["frac_exits"] = out["N_exits"] / out["N_deals"]

    # Breakdown by exit_category
    breakdown = (
        df[df["is_exit"]]
        .groupby(["entry_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    return out.join(breakdown, how="left").fillna(0).reset_index()


# =============================================================================
# PANEL 2 — UNIVERSE EXIT-YEAR VIEW
# =============================================================================

def build_universe_exit_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For all "exit deals":
        - N exits
        - Breakdown by exit_category
        - exit_value_usd_m
    """

    df = df[df["exit_category"].ne("other")].copy()
    df["exit_year"] = df["transaction_year"]

    grouped = df.groupby("exit_year")

    out = pd.DataFrame({
        "N_exits": grouped.size(),
        "exit_value_usd_m": grouped["deal_size_usd_m"].sum(),
    })

    breakdown = (
        df.groupby(["exit_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    return out.join(breakdown, how="left").reset_index()


# =============================================================================
# PANEL 3 — DEAL-YEAR ACTIVITY (UNIVERSE)
# =============================================================================

def build_universe_deal_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For all deals by transaction year:
        - N deals
        - Sum deal value
        - breakdown by deal_type
    """

    df = df.copy()
    grouped = df.groupby("transaction_year")

    out = pd.DataFrame({
        "N_deals": grouped.size(),
        "total_value_usd_m": grouped["deal_size_usd_m"].sum(),
    })

    # deal_type breakdown
    breakdown = (
        df.groupby(["transaction_year", "deal_type"])
        .size()
        .unstack(fill_value=0)
    )

    return out.join(breakdown, how="left").reset_index()


# =============================================================================
# FIGURES
# =============================================================================

def plot_basic_line(df, x, y, title, outpath):
    plt.figure(figsize=(12,6))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_all_figures(entry, exit_, deal, outdir: Path):

    plot_basic_line(
        entry, "entry_year", "N_deals",
        "Universe — Deal Counts by Entry Year",
        outdir / "Universe_EntryYear_NDeals.png"
    )

    plot_basic_line(
        entry, "entry_year", "frac_exits",
        "Universe — Fraction of Deals that are Exits (Entry-Year View)",
        outdir / "Universe_EntryYear_FracExits.png"
    )

    plot_basic_line(
        exit_, "exit_year", "N_exits",
        "Universe — Exit Counts by Exit Year",
        outdir / "Universe_ExitYear_NExits.png"
    )

    plot_basic_line(
        exit_, "exit_year", "exit_value_usd_m",
        "Universe — Exit Value by Exit Year",
        outdir / "Universe_ExitYear_ExitValue.png"
    )

    plot_basic_line(
        deal, "transaction_year", "N_deals",
        "Universe — Total Deals per Year",
        outdir / "Universe_DealYear_NDeals.png"
    )


# =============================================================================
# DRIVER
# =============================================================================

def run_universe_panels(master_path: Path, panel_dir: Path, fig_dir: Path):

    df = load_master(master_path)

    entry_panel = build_universe_entry_year_panel(df)
    exit_panel  = build_universe_exit_year_panel(df)
    deal_panel  = build_universe_deal_year_panel(df)

    panel_dir.mkdir(parents=True, exist_ok=True)

    entry_panel.to_csv(panel_dir / "universe_entry_year_panel.csv", index=False)
    exit_panel.to_csv(panel_dir / "universe_exit_year_panel.csv", index=False)
    deal_panel.to_csv(panel_dir / "universe_deal_year_panel.csv", index=False)

    save_all_figures(entry_panel, exit_panel, deal_panel, fig_dir)

    print("✅ Universe Panels & Figures Saved")
    print("   Panels:", panel_dir.resolve())
    print("   Figures:", fig_dir.resolve())


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    MASTER = ROOT / "data" / "clean" / "pitchbook_master_clean.parquet"
    PANELS = ROOT / "outputs" / "panels_universe"
    FIGS   = ROOT / "outputs" / "figures_universe"

    run_universe_panels(MASTER, PANELS, FIGS)
