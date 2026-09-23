import pandas as pd
import numpy as np
from pathlib import Path
import re

def clean_company_name(name):
    """
    Aggressively cleans company names to improve match rates.
    """
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # 1. Remove PitchBook specific ticker markers: (NAS: XYZ), (NYS: XYZ)
    name = re.sub(r'\(.*?\)', '', name)
    # 2. Remove common legal suffixes
    suffixes = [r'\binc\b', r'\bcorp\b', r'\bcorporation\b', r'\blp\b', r'\bllc\b', r'\bgroup\b', r'\bplc\b', r'\bco\b']
    for s in suffixes:
        name = re.sub(s, '', name)
    # 3. Final cleanup: remove special chars and extra spaces
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return " ".join(name.split())

def process_p2p_matching():
    # --- PATH SETUP ---
    CURRENT_FILE = Path(__file__).resolve()
    # Adjusting based on your structure: assumes script is in /scripts/subfolder/
    ROOT = CURRENT_FILE.parents[2]

    DATA_DIR = ROOT / "data" / "raw"
    OUTPUT_DIR = ROOT / "data" / "clean"

    print(f"Project Root identified as: {ROOT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    # Loading Excel files as per your environment setup
    df_entry = pd.read_excel(DATA_DIR / "pitchbook_public_2_private_all.xlsx")
    df_exit = pd.read_excel(DATA_DIR / "pitchbook_exit_completed.xlsx")

    # Standardize Column Names BEFORE suffixing
    rename_map = {
        'Companies': 'company_name',
        'Deal Date': 'deal_date',
        'Deal Size': 'deal_size',
        'Deal Type': 'deal_type'
    }
    df_entry = df_entry.rename(columns=rename_map)
    df_exit = df_exit.rename(columns=rename_map)

    # Convert Dates to datetime objects
    df_entry['deal_date'] = pd.to_datetime(df_entry['deal_date'], errors='coerce')
    df_exit['deal_date'] = pd.to_datetime(df_exit['deal_date'], errors='coerce')

    print("Cleaning company names for matching...")
    df_entry['company_name_clean'] = df_entry['company_name'].apply(clean_company_name)
    df_exit['company_name_clean'] = df_exit['company_name'].apply(clean_company_name)

    # Add unique ID to Entry set to ensure we track individual deals through the merge
    df_entry['entry_uid'] = range(len(df_entry))

    print("Matching Exits to Entries...")

    # 1. Perform internal merge to find candidates
    # We use suffixes here to handle overlapping column names during the merge process
    potential_matches = pd.merge(
        df_entry[['entry_uid', 'company_name_clean', 'deal_date']],
        df_exit,
        on='company_name_clean',
        how='inner',
        suffixes=('_entry', '_exit')
    )

    # 2. Filter: Exit must be chronologically after Entry
    valid_candidates = potential_matches[
        potential_matches['deal_date_exit'] > potential_matches['deal_date_entry']
    ].copy()

    # 3. Pick the FIRST exit that occurred after the entry date
    valid_candidates = valid_candidates.sort_values(by=['entry_uid', 'deal_date_exit'])
    best_exits = valid_candidates.drop_duplicates(subset=['entry_uid'], keep='first')

    # 4. Final Suffixing Logic:
    # Rename df_entry columns to have _entry
    # (Excluding 'entry_uid' which is our join key, and 'company_name_clean' for matching)
    entry_cols_to_rename = {col: f"{col}_entry" for col in df_entry.columns
                            if col not in ['entry_uid', 'company_name_clean']}
    df_entry_suffixed = df_entry.rename(columns=entry_cols_to_rename)

    # Prepare the exit columns (already have _exit from step 1, but we ensure consistency)
    # We drop company_name_clean and deal_date_entry from best_exits to avoid duplication
    exit_data_to_join = best_exits.drop(columns=['company_name_clean', 'deal_date_entry'])

    # Ensure all columns in exit_data_to_join (except entry_uid) have _exit
    exit_rename_map = {col: f"{col}_exit" if not col.endswith('_exit') and col != 'entry_uid' else col
                       for col in exit_data_to_join.columns}
    exit_data_to_join = exit_data_to_join.rename(columns=exit_rename_map)

    # 5. Join exit data back to the full entry universe
    final_output = pd.merge(
        df_entry_suffixed,
        exit_data_to_join,
        on='entry_uid',
        how='left'
    )

    # Calculate Holding Period using the new suffixed names
    if 'deal_date_exit' in final_output.columns and 'deal_date_entry' in final_output.columns:
        final_output['holding_period_years'] = (
            final_output['deal_date_exit'] - final_output['deal_date_entry']
        ).dt.days / 365.25

    # Stats for console output
    total_entries = len(df_entry)
    matches_found = final_output['deal_date_exit'].notna().sum()

    print("-" * 40)
    print(f"Original Entry Universe: {total_entries}")
    print(f"Matches Linked:          {matches_found}")
    print("-" * 40)

    # Save to CSV
    save_path = OUTPUT_DIR / "p2p_linked_master.csv"
    final_output.to_csv(save_path, index=False)
    print(f"Success! Linked file with _entry and _exit suffixes saved to: {save_path}")

if __name__ == "__main__":
    process_p2p_matching()