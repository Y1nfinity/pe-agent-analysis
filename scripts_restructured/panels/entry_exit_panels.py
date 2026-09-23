"""
entry_exit_panels.py — STEP 3 (UPDATED FINAL VERSION)
=====================================================
Creates the three core analytical panels used in reporting:

A. Take-private sample — entry-year view
B. Take-private sample — exit-year view
C. Entire universe — entry-year view

Inputs:
    data/clean/entry_exit_pairs.parquet

Outputs:
    outputs/panels/*.csv
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# Load canonical linked dataset
# =============================================================================

def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"entry_exit_pairs not found: {path.resolve()}")
    return pd.read_parquet(path)


# =============================================================================
# A. TAKE–PRIVATE: ENTRY-YEAR PANEL
# =============================================================================

def build_tp_entry_year_panel(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df["entry_type"].str.contains("public", case=False, na=False)].copy()

    df["entry_year"] = df["entry_date"].dt.year
    df["exited"] = df["exit_date"].notna()

    grouped = df.groupby("entry_year")

    N_entries = grouped.size()
    N_exits   = grouped["exited"].sum()

    entry_val = grouped["deal_size_usd_m_entry"].sum()
    exit_val  = grouped["deal_size_usd_m_exit"].sum()

    exit_break_count = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    exit_break_value = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category"])["deal_size_usd_m_exit"]
        .sum()
        .unstack(fill_value=0)
    )

    hp_mean = grouped["holding_period_years"].mean()
    hp_std  = grouped["holding_period_years"].std()

    out = pd.DataFrame({
        "N_entries": N_entries,
        "N_exits": N_exits,
        "entry_value_usd_m": entry_val,
        "exit_value_usd_m": exit_val,
        "frac_exited_by_count": N_exits / N_entries,
        "frac_exited_by_value": exit_val / entry_val.replace(0, np.nan),
        "hp_mean": hp_mean,
        "hp_std": hp_std,
    })

    out = out.join(exit_break_count, rsuffix="_count")
    out = out.join(exit_break_value, rsuffix="_value")

    return out.reset_index()


# =============================================================================
# B. TAKE–PRIVATE: EXIT-YEAR PANEL
# =============================================================================

def build_tp_exit_year_panel(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df["entry_type"].str.contains("public", case=False, na=False)].copy()
    df = df[df["exit_date"].notna()].copy()

    df["exit_year"] = df["exit_date"].dt.year

    grouped = df.groupby("exit_year")

    N_exits  = grouped.size()
    exit_val = grouped["deal_size_usd_m_exit"].sum()

    exit_break_count = (
        df.groupby(["exit_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    exit_break_value = (
        df.groupby(["exit_year", "exit_category"])["deal_size_usd_m_exit"]
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


# =============================================================================
# C. UNIVERSE: ENTRY-YEAR PANEL (UPDATED)
# =============================================================================

def build_universe_entry_year_panel(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["entry_year"] = df["entry_date"].dt.year
    df["exited"] = df["exit_date"].notna()

    # Ensure consistent exit value column
    if "deal_size_usd_m_exit" not in df.columns:
        df["deal_size_usd_m_exit"] = np.nan

    grouped = df.groupby("entry_year")

    N_entries = grouped.size()
    N_exits   = grouped["exited"].sum()

    exit_val = grouped["deal_size_usd_m_exit"].sum()

    exit_break = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    hp_mean = grouped["holding_period_years"].mean()
    hp_std  = grouped["holding_period_years"].std()

    out = pd.DataFrame({
        "N_entries": N_entries,
        "N_exits": N_exits,
        "frac_exited": N_exits / N_entries,
        "frac_exited_univ": N_exits / N_entries,  # REQUIRED FIX
        "exit_value_usd_m": exit_val,
        "hp_mean": hp_mean,
        "hp_std": hp_std,
    })

    out = out.join(exit_break, rsuffix="_count")

    return out.reset_index()


# =============================================================================
# MAIN DRIVER
# =============================================================================

def run_all_panels(pairs_path: Path, outdir: Path) -> None:

    df = load_pairs(pairs_path)
    outdir.mkdir(parents=True, exist_ok=True)

    tp_entry = build_tp_entry_year_panel(df)
    tp_exit  = build_tp_exit_year_panel(df)
    univ     = build_universe_entry_year_panel(df)

    tp_entry.to_csv(outdir / "tp_entry_year_panel.csv", index=False)
    tp_exit.to_csv(outdir / "tp_exit_year_panel.csv", index=False)
    univ.to_csv(outdir / "universe_entry_year_panel.csv", index=False)

    print("✅ STEP 3 Panels saved to:", outdir.resolve())


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    pairs = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    out   = ROOT / "outputs" / "panels"
    run_all_panels(pairs, out)
