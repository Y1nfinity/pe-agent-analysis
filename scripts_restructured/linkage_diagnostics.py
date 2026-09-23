"""
entry_exit_linking.py (FULL RAPIDFUZZ REWRITE)
==============================================

This module links entries (public→private deals) to exits.
Uses:
    - deterministic matching
    - exact matching
    - RapidFuzz fuzzy fallback
All fuzzywuzzy code has been removed.
"""

from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

from scripts_restructured.data_cleaning import (
    prepare_entry_data,
    prepare_exit_data,
    load_raw_data
)
from scripts_restructured.exit_categorization import categorize_exits


# =============================================================================
# RAPIDFUZZ FUZZY MATCHER
# =============================================================================

def fuzzy_match(entries: pd.DataFrame,
                exits: pd.DataFrame,
                min_score: int = 90) -> pd.DataFrame:
    """
    RapidFuzz fuzzy matching:
    - extremely fast
    - no recursion
    - no C-level segmentation faults like fuzzywuzzy
    """

    if entries.empty or exits.empty:
        return pd.DataFrame()

    exit_names = exits["company_name_clean"].dropna().unique().tolist()

    matches = []
    for entry_name in entries["company_name_clean"].dropna().unique():
        best = process.extractOne(
            entry_name,
            exit_names,
            scorer=fuzz.WRatio,
            score_cutoff=min_score
        )
        if best:
            exit_name, score, _ = best
            matches.append((entry_name, exit_name, score))

    if not matches:
        return pd.DataFrame()

    fuzzy_df = pd.DataFrame(matches,
                            columns=["entry_name_clean", "exit_name_clean", "match_score"])

    # Merge fuzzy candidates into exits
    exits_fz = exits.merge(
        fuzzy_df,
        left_on="company_name_clean",
        right_on="exit_name_clean",
        how="inner"
    )

    # Merge into entries
    merged = entries.merge(
        exits_fz,
        left_on="company_name_clean",
        right_on="entry_name_clean",
        suffixes=("_entry", "_exit")
    )

    # chronological filter
    merged = merged[merged["exit_date"] > merged["entry_date"]]

    if merged.empty:
        return merged

    # keep earliest exit
    merged = (
        merged.sort_values(["company_name_clean", "entry_date", "exit_date"])
              .groupby(["company_name_clean", "entry_date"])
              .head(1)
              .reset_index(drop=True)
    )

    merged["match_type"] = "fuzzy"

    return merged


# =============================================================================
# EXACT MATCH LOGIC
# =============================================================================

def exact_match(entries: pd.DataFrame,
                exits: pd.DataFrame) -> pd.DataFrame:

    merged = entries.merge(
        exits,
        on="company_name_clean",
        suffixes=("_entry", "_exit"),
        how="inner"
    )

    merged = merged[merged["exit_date"] > merged["entry_date"]]

    if merged.empty:
        return merged

    merged = (
        merged.sort_values(["company_name_clean", "entry_date", "exit_date"])
              .groupby(["company_name_clean", "entry_date"])
              .head(1)
              .reset_index(drop=True)
    )

    merged["match_type"] = "exact"

    return merged


# =============================================================================
# SMART MATCHER
# =============================================================================

def link_entries_exits(entries: pd.DataFrame,
                       exits: pd.DataFrame,
                       min_score: int = 90) -> pd.DataFrame:

    print("  🔸 Running EXACT match...")
    exact = exact_match(entries, exits)

    matched_names = set(exact["company_name_clean"])
    remaining_entries = entries[~entries["company_name_clean"].isin(matched_names)]

    print(f"     → Exact matched: {len(exact)}")
    print(f"     → Fuzzy candidates remaining: {len(remaining_entries)}")

    print("  🔸 Running FUZZY match (RapidFuzz)...")
    fuzzy = fuzzy_match(remaining_entries, exits, min_score=min_score)

    print(f"     → Fuzzy matched: {len(fuzzy)}")

    # concat both types
    out = pd.concat([exact, fuzzy], ignore_index=True)

    # holding periods
    out["holding_period_years"] = (
        (out["exit_date"] - out["entry_date"]).dt.days / 365.25
    )

    return out.reset_index(drop=True)


# =============================================================================
# EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    CLEAN_DIR = DATA_DIR / "clean"

    print("🔹 Loading raw datasets...")
    entries_raw = load_raw_data(RAW_DIR / "public_2_private_all.xlsx")
    exits_raw = load_raw_data(RAW_DIR / "pitchbook_export.xlsx")

    print("🔹 Preparing datasets...")
    entries = prepare_entry_data(entries_raw)
    exits = prepare_exit_data(exits_raw)
    exits = categorize_exits(exits)

    print("🔹 Running SMART linking...")
    pairs = link_entries_exits(entries, exits, min_score=90)

    out_path = CLEAN_DIR / "entry_exit_pairs.parquet"
    pairs.to_parquet(out_path, index=False)
    print(f"✅ Saved entry_exit_pairs.parquet → {out_path}")
