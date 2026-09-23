"""
entry_exit_linking.py — FINAL, CORRECTED
=======================================
SMART matching without column collisions.
"""

from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

from scripts_restructured.data_cleaning import load_raw_data, prepare_entry_data, prepare_exit_data
from scripts_restructured.exit_categorization import categorize_exits


# =============================================================================
# Helpers
# =============================================================================

def _filter_exits(entries, exits):
    exits = exits.copy()
    if "deal_id" in entries and "deal_id" in exits:
        exits = exits[~exits["deal_id"].isin(entries["deal_id"])]
    if "exit_category" in exits:
        exits = exits[exits["exit_category"] != "take_private"]
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
    used = exact.company_name_clean.unique()

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
    exits   = prepare_exit_data(exits_raw)

    exits = categorize_exits(exits)
    exits = _filter_exits(entries, exits)

    return smart_match(entries, exits)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]

    entries_raw = load_raw_data(ROOT / "data/raw/public_2_private_all.xlsx")
    exits_raw   = load_raw_data(ROOT / "data/raw/pitchbook_export.xlsx")

    pairs = link_entries_exits(entries_raw, exits_raw)

    out = ROOT / "data/clean/entry_exit_pairs.parquet"
    pairs.to_parquet(out, index=False)

    print("✅ Saved:", out)
    print(pairs.head())
