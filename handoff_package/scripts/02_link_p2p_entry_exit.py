"""
02_link_p2p_entry_exit.py
=========================

Links take-private (public-to-private) entries to their eventual exit events,
enforcing that a matched exit must occur strictly after the entry date, and
keeping active/unrealized deals (no exit yet) rather than dropping them.

Reads:
    data/raw/pitchbook_public_2_private_all.xlsx
    data/raw/pitchbook_exit_completed.xlsx

Writes:
    data/clean/p2p_linked_master.csv

This output is a required input for:
    03_chart_tp_nested.py
    04_chart_trend_lines.py
    06_chart_holding_period_se.py
"""

import re
from pathlib import Path

import pandas as pd


def clean_company_name(name) -> str:
    """cleaned company names to improve match rates."""
    if pd.isna(name):
        return ""

    name = str(name).lower()
    name = re.sub(r"\(.*?\)", "", name)  # PitchBook ticker markers, e.g. "(NAS: XYZ)"

    suffixes = [r"\binc\b", r"\bcorp\b", r"\bcorporation\b", r"\blp\b",
                r"\bllc\b", r"\bgroup\b", r"\bplc\b", r"\bco\b"]
    for s in suffixes:
        name = re.sub(s, "", name)

    name = re.sub(r"[^a-z0-9 ]", "", name)
    return " ".join(name.split())


def process_p2p_matching() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "raw"
    output_dir = root / "data" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    df_entry = pd.read_excel(data_dir / "pitchbook_public_2_private_all.xlsx")
    df_exit = pd.read_excel(data_dir / "pitchbook_exit_completed.xlsx")

    rename_map = {
        "Companies": "company_name",
        "Deal Date": "deal_date",
        "Deal Size": "deal_size",
        "Deal Type": "deal_type",
    }
    df_entry = df_entry.rename(columns=rename_map)
    df_exit = df_exit.rename(columns=rename_map)

    df_entry["deal_date"] = pd.to_datetime(df_entry["deal_date"], errors="coerce")
    df_exit["deal_date"] = pd.to_datetime(df_exit["deal_date"], errors="coerce")

    print("Cleaning company names for matching...")
    df_entry["company_name_clean"] = df_entry["company_name"].apply(clean_company_name)
    df_exit["company_name_clean"] = df_exit["company_name"].apply(clean_company_name)

    # Unique ID per entry so we can track individual deals through the merge
    df_entry["entry_uid"] = range(len(df_entry))

    print("Matching exits to entries...")

    # 1. Inner merge to find all candidate exits per entry (by company)
    potential_matches = pd.merge(
        df_entry[["entry_uid", "company_name_clean", "deal_date"]],
        df_exit,
        on="company_name_clean",
        how="inner",
        suffixes=("_entry", "_exit"),
    )

    # 2. Keep only exits that happened AFTER entry
    valid_candidates = potential_matches[
        potential_matches["deal_date_exit"] > potential_matches["deal_date_entry"]
    ].copy()

    # 3. Keep the first exit that occurred after entry
    valid_candidates = valid_candidates.sort_values(by=["entry_uid", "deal_date_exit"])
    best_exits = valid_candidates.drop_duplicates(subset=["entry_uid"], keep="first")

    # 4. Suffix entry columns with _entry (except join/match keys)
    entry_cols_to_rename = {
        col: f"{col}_entry" for col in df_entry.columns
        if col not in ["entry_uid", "company_name_clean"]
    }
    df_entry_suffixed = df_entry.rename(columns=entry_cols_to_rename)

    exit_data_to_join = best_exits.drop(columns=["company_name_clean", "deal_date_entry"])
    exit_rename_map = {
        col: f"{col}_exit" if not col.endswith("_exit") and col != "entry_uid" else col
        for col in exit_data_to_join.columns
    }
    exit_data_to_join = exit_data_to_join.rename(columns=exit_rename_map)

    # 5. Left-join exit data back onto the FULL entry universe, so entries
    #    with no matched exit (active/unrealized deals) are still kept.
    final_output = pd.merge(
        df_entry_suffixed,
        exit_data_to_join,
        on="entry_uid",
        how="left",
    )

    if "deal_date_exit" in final_output.columns and "deal_date_entry" in final_output.columns:
        final_output["holding_period_years"] = (
            final_output["deal_date_exit"] - final_output["deal_date_entry"]
        ).dt.days / 365.25

    total_entries = len(df_entry)
    matches_found = final_output["deal_date_exit"].notna().sum()

    print("-" * 40)
    print(f"Original entry universe: {total_entries}")
    print(f"Matches linked:          {matches_found}")
    print("-" * 40)

    out_path = output_dir / "p2p_linked_master.csv"
    final_output.to_csv(out_path, index=False)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    process_p2p_matching()
