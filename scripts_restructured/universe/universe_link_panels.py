"""
universe_panels.py
==================

Builds annual ENTRY-YEAR and EXIT-YEAR panels for the FULL universe.

Input:
    data/clean/universe_entry_exit_pairs.parquet

Output:
    outputs/panels_universe/
        universe_entry_year_panel.csv
        universe_exit_year_panel.csv

Panels mirror the take-private version:
    • Counts per year
    • Dollarized sizes
    • Exit breakdown (count + value)
    • Holding periods (mean + std)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Load Clean Universe Pairs
# ============================================================

def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing universe entry–exit pairs: {path.resolve()}")
    return pd.read_parquet(path)


# ============================================================
# ENTRY-YEAR PANEL (Full Universe)
# ============================================================

def build_universe_entry_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each entry_year:
        N_entries
        entry_value_total
        N_exited
        exit_value_total
        exit breakdown (count + $)
        frac_exited_by_count
        frac_exited_by_value
        holding_period stats
    """

    df = df.copy()
    df["entry_year"] = df["deal_date_entry"].dt.year
    df["exited"] = df["deal_date_exit"].notna()

    grouped = df.groupby("entry_year")

    N_entries = grouped.size()
    entry_val = grouped["deal_size_usd_m_entry"].sum()

    N_exited = grouped["exited"].sum()
    exit_val = grouped["deal_size_usd_m_exit"].sum()

    # Exit breakdown by category
    exit_break_count = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category_exit"])
        .size()
        .unstack(fill_value=0)
    )

    exit_break_value = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category_exit"])["deal_size_usd_m_exit"]
        .sum()
        .unstack(fill_value=0)
    )

    hp_mean = grouped["holding_period_years"].mean()
    hp_std  = grouped["holding_period_years"].std()

    out = pd.DataFrame({
        "N_entries": N_entries,
        "entry_value_usd_m": entry_val,
        "N_exited": N_exited,
        "exit_value_usd_m": exit_val,
        "frac_exited_by_count": N_exited / N_entries,
        "frac_exited_by_value": exit_val / entry_val.replace(0, np.nan),
        "hp_mean": hp_mean,
        "hp_std": hp_std,
    })

    out = out.join(exit_break_count, rsuffix="_count")
    out = out.join(exit_break_value, rsuffix="_value")

    return out.reset_index()


# ============================================================
# EXIT-YEAR PANEL (Full Universe)
# ============================================================

def build_universe_exit_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each exit_year:
        N_exits
        exit_value
        exit breakdown (count + $)
        holding period stats
    """

    df = df.copy()
    df = df[df["deal_date_exit"].notna()]  # exits only

    df["exit_year"] = df["deal_date_exit"].dt.year

    grouped = df.groupby("exit_year")

    N_exits = grouped.size()
    exit_val = grouped["deal_size_usd_m_exit"].sum()

    exit_break_count = (
        df.groupby(["exit_year", "exit_category_exit"])
        .size()
        .unstack(fill_value=0)
    )

    exit_break_value = (
        df.groupby(["exit_year", "exit_category_exit"])["deal_size_usd_m_exit"]
        .sum()
        .unstack(fill_value=0)
    )

    hp_mean = grouped["holding_period_years"].mean()
    hp_std  = grouped["holding_period_years"].std()

    out = pd.DataFrame({
        "N_exits": N_exits,
        "exit_value_usd_m": exit_val,
        "hp_mean": hp_mean,
        "hp_std": hp_std,
    })

    out = out.join(exit_break_count, rsuffix="_count")
    out = out.join(exit_break_value, rsuffix="_value")

    return out.reset_index()


# ============================================================
# MASTER DRIVER
# ============================================================

def run_universe_panels(pairs_path: Path, outdir: Path) -> None:

    df = load_pairs(pairs_path)

    outdir.mkdir(parents=True, exist_ok=True)

    panel_entry = build_universe_entry_year_panel(df)
    panel_exit  = build_universe_exit_year_panel(df)

    panel_entry.to_csv(outdir / "universe_entry_year_panel.csv", index=False)
    panel_exit.to_csv(outdir / "universe_exit_year_panel.csv", index=False)

    print("✅ Universe panels saved:", outdir.resolve())


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    PAIRS = ROOT / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    OUT   = ROOT / "outputs" / "panels_universe"

    run_universe_panels(PAIRS, OUT)
