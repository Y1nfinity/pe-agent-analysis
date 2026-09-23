"""
analyze_yearly_activity.py
==========================

STEP 5 EXTENSION — YEARLY INDEXED ANALYSIS (1995–2025)

Produces four yearly panels:
    1. Take-Private Entry-Year Panel
    2. Take-Private Exit-Year Panel
    3. Universe Deal-Year Activity Panel (NEW)
    4. Universe Exit-Year Activity Panel (NEW)

Inputs:
    data/clean/entry_exit_pairs.parquet
    data/clean/pitchbook_clean.parquet  (must contain all transactions)

Outputs:
    outputs/yearly/*.csv
    outputs/yearly_figures/*.png
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

YEAR_MIN = 1995
YEAR_MAX = 2025


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_pairs(pairs_path: Path) -> pd.DataFrame:
    if not pairs_path.exists():
        raise FileNotFoundError(f"entry_exit_pairs not found: {pairs_path}")
    return pd.read_parquet(pairs_path)


def load_universe(universe_path: Path) -> pd.DataFrame:
    if not universe_path.exists():
        raise FileNotFoundError(f"PitchBook universe not found: {universe_path}")
    return pd.read_parquet(universe_path)


# -----------------------------------------------------------------------------
# Helper: restrict to year range
# -----------------------------------------------------------------------------

def clamp_years(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df[df[col].dt.year.between(YEAR_MIN, YEAR_MAX)]
    return df


# -----------------------------------------------------------------------------
# 1. Take-Private Entry-Year Analysis
# -----------------------------------------------------------------------------

def build_tp_entry_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["entry_type"].str.contains("public", case=False, na=False)].copy()

    df = clamp_years(df, "entry_date")
    df["entry_year"] = df["entry_date"].dt.year
    df["exited"] = df["exit_date"].notna()

    group = df.groupby("entry_year")

    out = pd.DataFrame({
        "N_entries": group.size(),
        "N_exits": group["exited"].sum(),
        "entry_value_usd_m": group["deal_size_usd_m_entry"].sum(),
        "exit_value_usd_m": group["deal_size_usd_m_exit"].sum(),
        "hp_mean": group["holding_period_years"].mean(),
        "hp_std": group["holding_period_years"].std(),
    })

    out["frac_exited_by_count"] = out["N_exits"] / out["N_entries"]
    out["frac_exited_by_value"] = out["exit_value_usd_m"] / out["entry_value_usd_m"].replace(0, np.nan)

    # breakdowns
    exit_break = (
        df[df["exited"]]
        .groupby(["entry_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    out = out.join(exit_break, how="left").fillna(0)

    return out.reset_index()


# -----------------------------------------------------------------------------
# 2. Take-Private Exit-Year Analysis
# -----------------------------------------------------------------------------

def build_tp_exit_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["entry_type"].str.contains("public", case=False, na=False)].copy()
    df = clamp_years(df, "exit_date")
    df = df[df["exit_date"].notna()]

    df["exit_year"] = df["exit_date"].dt.year

    group = df.groupby("exit_year")

    out = pd.DataFrame({
        "N_exits": group.size(),
        "exit_value_usd_m": group["deal_size_usd_m_exit"].sum(),
        "hp_mean": group["holding_period_years"].mean(),
        "hp_std": group["holding_period_years"].std(),
    })

    exit_break = (
        df.groupby(["exit_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    out = out.join(exit_break, how="left").fillna(0)

    return out.reset_index()


# -----------------------------------------------------------------------------
# 3. Universe Deal-Year Activity Panel (NEW)
# -----------------------------------------------------------------------------

def build_universe_deal_year(df_univ: pd.DataFrame) -> pd.DataFrame:
    df = df_univ.copy()
    df = clamp_years(df, "transaction_date")

    df["deal_year"] = df["transaction_date"].dt.year

    group = df.groupby("deal_year")

    out = pd.DataFrame({
        "N_deals": group.size(),
        "total_value_usd_m": group["deal_size_usd_m"].sum(),
    })

    # breakdown by deal type
    if "deal_type_clean" in df.columns:
        deal_break = (
            df.groupby(["deal_year", "deal_type_clean"])
            .size()
            .unstack(fill_value=0)
        )
        out = out.join(deal_break, how="left").fillna(0)

    # share of take-private
    out["share_take_private"] = (
        df[df["deal_type_clean"].str.contains("public-to-private", case=False, na=False)]
        .groupby(df["deal_year"])
        .size()
        .reindex(out.index, fill_value=0) / out["N_deals"]
    )

    return out.reset_index()


# -----------------------------------------------------------------------------
# 4. Universe Exit-Year Activity Panel (NEW)
# -----------------------------------------------------------------------------

def build_universe_exit_year(df_univ: pd.DataFrame) -> pd.DataFrame:
    if "exit_date" not in df_univ.columns:
        raise ValueError("Universe file must contain exit_date column.")

    df = clamp_years(df_univ, "exit_date")
    df = df[df["exit_date"].notna()]
    df["exit_year"] = df["exit_date"].dt.year

    group = df.groupby("exit_year")

    out = pd.DataFrame({
        "N_exits": group.size(),
        "exit_value_usd_m": group["deal_size_usd_m"].sum(),
    })

    if "exit_category" in df.columns:
        exit_break = (
            df.groupby(["exit_year", "exit_category"])
            .size()
            .unstack(fill_value=0)
        )
        out = out.join(exit_break, how="left").fillna(0)

    return out.reset_index()


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_all(path_pairs: Path, path_univ: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(path_pairs)
    univ  = load_universe(path_univ)

    tp_entry = build_tp_entry_year(pairs)
    tp_exit  = build_tp_exit_year(pairs)
    univ_deal = build_universe_deal_year(univ)
    univ_exit = build_universe_exit_year(univ)

    tp_entry.to_csv(outdir / "tp_entry_year.csv", index=False)
    tp_exit.to_csv(outdir / "tp_exit_year.csv", index=False)
    univ_deal.to_csv(outdir / "universe_deal_year.csv", index=False)
    univ_exit.to_csv(outdir / "universe_exit_year.csv", index=False)

    print("✅ Yearly panels saved to:", outdir.resolve())


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    pairs_path = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    universe_path = ROOT / "data" / "clean" / "pitchbook_clean.parquet"  # YOU MUST CONFIRM THIS FILE NAME
    outdir = ROOT / "outputs" / "yearly"

    run_all(pairs_path, universe_path, outdir)
