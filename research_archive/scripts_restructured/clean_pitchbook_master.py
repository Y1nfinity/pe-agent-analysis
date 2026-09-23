"""
clean_pitchbook_master.py
==========================

STEP 1 of the Universe Pipeline.

Takes:
    data/raw/pitchbook_export.xlsx

Produces:
    data/clean/pitchbook_master_clean.parquet

Features:
    • Cleans company names
    • Standardizes date → transaction_year
    • Ensures deal_size_usd_m numeric
    • Filters to 1995–2025 inclusive
    • Applies exit categorization (EXIT PIPE)
    • Creates unified schema for the Universe pipeline
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Local imports (you confirmed OK)
# -----------------------------
from scripts_restructured.utilities.name_cleaning import clean_company_name
from scripts_restructured.exit_categorization import categorize_exits


# ============================================================
# Load Raw
# ============================================================

def load_pitchbook_raw(path: Path) -> pd.DataFrame:
    """Load Excel export from PitchBook."""
    if not path.exists():
        raise FileNotFoundError(f"Missing raw pitchbook export: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=0)
    return df


# ============================================================
# PitchBook-specific exit mapping
# ============================================================

def map_pitchbook_exit_category(deal_type: str) -> str:
    """Lightweight exit categorization for PitchBook universe pipeline."""
    if not isinstance(deal_type, str):
        return "other"

    t = deal_type.lower()

    if "ipo" in t:
        return "ipo"
    if "merger" in t or "acquisition" in t:
        return "mna_exit"
    if "secondary" in t:
        return "secondary_buyout"
    if "take-private" in t or "public to private" in t:
        return "take_private"
    if "add-on" in t:
        return "add_on"
    if "recap" in t:
        return "recapitalization"
    if "divestiture" in t:
        return "corporate_divestiture"
    if "growth" in t:
        return "growth_equity"
    if "pipe" in t:
        return "pipe"
    if "distressed" in t:
        return "distressed"
    if "management buyout" in t or "management buy-in" in t:
        return "management_buyout"

    return "other"


# ============================================================
# Cleaning Functions
# ============================================================

def clean_pitchbook(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for PitchBook master."""

    df = df.copy()

    # --- 1. Clean names ---------------------------------------------------
    df["company_name_clean"] = (
        df["Companies"]
        .astype(str)
        .fillna("")
        .map(clean_company_name)
    )

    # --- 2. Standardize deal date -----------------------------------------
    df["deal_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    df["transaction_year"] = df["deal_date"].dt.year

    # --- 3. Filter to 1995–2025 -------------------------------------------
    df = df[df["transaction_year"].between(1995, 2025, inclusive="both")]

    # --- 4. Deal size ------------------------------------------------------
    df["deal_size_usd_m"] = pd.to_numeric(df["Deal Size"], errors="coerce")

    # --- 5. Apply exit categorization (using Deal Type) --------------------
    df["exit_category"] = df["Deal Type"].apply(map_pitchbook_exit_category)

    # --- 6. Final minimal schema -------------------------------------------
    df_clean = df[[
        "Deal ID",
        "company_name_clean",
        "deal_date",
        "transaction_year",
        "Deal Type",
        "exit_category",
        "deal_size_usd_m",
        "Deal Size Status",
        "Investor Funds",
        "Investors",
        "HQ Location",
    ]].rename(columns={
        "Deal ID": "deal_id",
        "Deal Type": "deal_type",
    })

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


# ============================================================
# PUBLIC DRIVER
# ============================================================

def run_clean_pitchbook(raw_path: Path, out_path: Path):
    print("🔹 Loading PitchBook raw...")
    raw = load_pitchbook_raw(raw_path)

    print("🔹 Cleaning...")
    clean = clean_pitchbook(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)

    print(f"✅ PitchBook master cleaned and saved to:")
    print(f"   {out_path.resolve()}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    RAW  = ROOT / "data" / "raw" / "pitchbook_export.xlsx"
    OUT  = ROOT / "data" / "clean" / "pitchbook_master_clean.parquet"

    run_clean_pitchbook(RAW, OUT)
