from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
import sys
import os

# Add parent directory to path to allow importing utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.name_cleaning import clean_company_name


# NOTE: This script assumes you have specific cleaning functions for
# your TP dataset (public_2_private_all.xlsx).
# I have adapted the structure to match the standard pipeline.

# =============================================================================
# Helpers
# =============================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def prepare_entry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adapts TP raw data to standard format"""
    df = df.copy()
    # Normalize column names - Adjust these based on your actual TP Excel file
    if 'Companies' in df.columns:
        df['company_name_clean'] = df['Companies'].astype(str).map(clean_company_name)
    elif 'Company' in df.columns:
        df['company_name_clean'] = df['Company'].astype(str).map(clean_company_name)

    # Date handling
    if 'Deal Date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['Deal Date'], errors='coerce')

    # Ensure ID
    if 'Deal ID' in df.columns:
        df['deal_id'] = df['Deal ID']

    return df


def prepare_exit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adapts Universe data to act as exit pool"""
    df = df.copy()
    df['company_name_clean'] = df['Companies'].astype(str).map(clean_company_name)
    df['exit_date'] = pd.to_datetime(df['Deal Date'], errors='coerce')
    df['deal_id'] = df['Deal ID']
    return df


def _filter_exits(entries, exits):
    exits = exits.copy()
    if "deal_id" in entries and "deal_id" in exits:
        exits = exits[~exits["deal_id"].isin(entries["deal_id"])]
    # Assuming we filter out TP deals from the exit pool to avoid self-matching
    # if "exit_category" in exits:
    #    exits = exits[exits["exit_category"] != "take_private"]
    return exits


def _earliest_exit(df):
    return (
        df.sort_values("exit_date")
        .groupby(["company_name_clean", "entry_date"], as_index=False)
        .head(1)
    )


def _safe_merge(entries, exits):
    """
    Merge without auto-suffixing.
    Entry and exit columns are already explicitly named.
    """
    return entries.merge(exits, on="company_name_clean", how="inner")


# =============================================================================
# Matching Strategy
# =============================================================================

def exact_match(entries, exits):
    merged = _safe_merge(entries, exits)
    merged = merged[merged.exit_date > merged.entry_date]

    if merged.empty:
        return merged

    merged = _earliest_exit(merged)
    merged["match_type"] = "exact"
    merged["match_score"] = None
    return merged


def fuzzy_match(entries, exits, min_score=90):
    exit_names = exits.company_name_clean.dropna().unique()
    if len(exit_names) == 0:
        return pd.DataFrame()

    matches = []
    for name in entries.company_name_clean.dropna().unique():
        best = process.extractOne(name, exit_names, scorer=fuzz.WRatio)
        if best and best[1] >= min_score:
            matches.append((name, best[0], best[1]))

    if not matches:
        return pd.DataFrame()

    match_df = pd.DataFrame(matches, columns=["entry_name_clean", "exit_name_clean", "match_score"])

    merged = (
        entries.merge(match_df, left_on="company_name_clean", right_on="entry_name_clean")
        .merge(exits, left_on="exit_name_clean", right_on="company_name_clean")
    )

    # FIX: restore canonical firm name from the ENTRY side
    merged["company_name_clean"] = merged["entry_name_clean"]

    merged = merged[merged.exit_date > merged.entry_date]
    if merged.empty:
        return merged

    merged = _earliest_exit(merged)
    merged["match_type"] = "fuzzy"

    return merged


def smart_match(entries, exits, min_score=90):
    exact = exact_match(entries, exits)
    used = exact.company_name_clean.unique() if not exact.empty else []

    fuzzy = fuzzy_match(entries[~entries.company_name_clean.isin(used)],
                        exits, min_score)

    df = pd.concat([exact, fuzzy], ignore_index=True, sort=False)

    if df.empty:
        return df

    df["holding_period_years"] = (
                                         df.exit_date - df.entry_date
                                 ).dt.days / 365.25

    return df


# =============================================================================
# Public API
# =============================================================================

def link_entries_exits(entries_raw, exits_raw):
    entries = prepare_entry_data(entries_raw)
    exits = prepare_exit_data(exits_raw)

    # Note: categorize_exits is external in your original,
    # ensure 'exits' has the 'exit_category' column here or upstream

    exits = _filter_exits(entries, exits)
    return smart_match(entries, exits)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    # Adjusted paths to match standard structure
    entries_raw = load_raw_data(ROOT / "data/raw/public_2_private_all.xlsx")
    exits_raw = load_raw_data(ROOT / "data/raw/pitchbook_export.xlsx")

    pairs = link_entries_exits(entries_raw, exits_raw)

    out = ROOT / "data/clean/tp_entry_exit_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)

    print("✅ Saved:", out)
    print(pairs.head())