"""
01_clean_pitchbook_master.py
============================

Cleans the full PitchBook universe export into a standardized schema.

Reads:
    data/raw/pitchbook_export.xlsx

Writes:
    data/clean/pitchbook_master_clean.parquet

This output is a required input for 04_chart_trend_lines.py (used to derive
the Non-Take-Private universe for the "Total" trend line).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ============================================================
# Company name cleaning
# ============================================================

CORPORATE_SUFFIXES = [
    r"\binc\b", r"\binc.\b", r"\bcorp\b", r"\bcorp.\b", r"\bcorporation\b",
    r"\bltd\b", r"\bltd.\b", r"\bco\b", r"\bco.\b", r"\bcompany\b",
    r"\bplc\b", r"\bholdings\b", r"\bholding\b", r"\bgroup\b",
    r"\bsa\b", r"\bag\b", r"\bse\b", r"\bllc\b", r"\bgmbh\b",
    r"\bpartners\b", r"\btechnologies\b", r"\btechnology\b",
]
SUFFIX_PATTERN = re.compile("|".join(CORPORATE_SUFFIXES), flags=re.IGNORECASE)


@lru_cache(maxsize=50000)
def clean_company_name(raw: str) -> str:
    """Standardize a company name for reliable cross-dataset matching."""
    if raw is None:
        return ""

    name = str(raw).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"http\S+|www\.\S+", " ", name)
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = SUFFIX_PATTERN.sub(" ", name)
    name = re.sub(r"\d+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# PitchBook-specific exit mapping
# ============================================================

def map_pitchbook_exit_category(deal_type: str) -> str:
    """Lightweight exit categorization for the universe pipeline."""
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
# Cleaning pipeline
# ============================================================

def load_pitchbook_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw pitchbook export: {path.resolve()}")
    return pd.read_excel(path, sheet_name=0)


def clean_pitchbook(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["company_name_clean"] = (
        df["Companies"].astype(str).fillna("").map(clean_company_name)
    )

    df["deal_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    df["transaction_year"] = df["deal_date"].dt.year

    df = df[df["transaction_year"].between(1995, 2025, inclusive="both")]

    df["deal_size_usd_m"] = pd.to_numeric(df["Deal Size"], errors="coerce")

    df["exit_category"] = df["Deal Type"].apply(map_pitchbook_exit_category)

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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "pitchbook_export.xlsx"
    out_path = root / "data" / "clean" / "pitchbook_master_clean.parquet"

    print("Loading PitchBook raw export...")
    raw = load_pitchbook_raw(raw_path)

    print("Cleaning...")
    clean = clean_pitchbook(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)

    print(f"Saved: {out_path.resolve()}")
    print(f"Rows: {len(clean):,}")


if __name__ == "__main__":
    main()
