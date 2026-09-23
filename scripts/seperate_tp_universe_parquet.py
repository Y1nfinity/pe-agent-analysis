import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# --- Configuration ---
# Setting the base data directory explicitly based on user's structure:
DATA_CLEAN_DIR = Path(r'C:\Users\azhao\PycharmProjects\PEAgent\data\clean')

# Input Files
UNIVERSE_FILE_PATH = DATA_CLEAN_DIR / 'universe_entry_exit_pairs.parquet'
TP_FILE_PATH = DATA_CLEAN_DIR / 'tp_entry_exit_pairs.parquet'

# Output File
NON_TP_FILE_PATH = DATA_CLEAN_DIR / 'non_tp_entry_exit_pairs.parquet'


# --- Core Anti-Join Logic ---

def standardize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names for reliable merging and downstream analysis.
    The goal is to ensure the merge keys are consistent:
    ('company_name_clean', 'entry_date', 'deal_size_usd_m_entry')
    """
    rename_map = {}

    # 1. Entry Date (Canonical: entry_date)
    for old_col in ["deal_date_entry", "Deal_Date", "Entry_Date", "deal_date"]:
        if old_col in df.columns and "entry_date" not in df.columns:
            rename_map[old_col] = "entry_date"
            break

    # 2. Exit Date (Canonical: exit_date)
    for old_col in ["deal_date_exit", "Exit_Date", "exit_date_raw"]:
        if old_col in df.columns and "exit_date" not in df.columns:
            rename_map[old_col] = "exit_date"
            break

    # 3. Entry Deal Size (Canonical: deal_size_usd_m_entry)
    for old_col in ["deal_size_usd_m", "V_entry", "Deal Size_x"]:
        if old_col in df.columns and "deal_size_usd_m_entry" not in df.columns:
            rename_map[old_col] = "deal_size_usd_m_entry"
            break

    # 4. Company Identifier (Canonical: company_name_clean)
    if "Companies" in df.columns and "company_name_clean" not in df.columns:
        df['company_name_clean'] = df['Companies'].astype(str)
    elif "Company_Name" in df.columns and "company_name_clean" not in df.columns:
        df['company_name_clean'] = df['Company_Name']

    df = df.rename(columns=rename_map)

    # Convert dates to datetime objects if present
    for col in ['entry_date', 'exit_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')

    return df


def generate_non_take_private_data():
    """
    Performs the anti-join (set subtraction) operation:
    Non-TP Deals = Universe Deals - TP Deals
    """
    print(f"--- Starting Anti-Join Process ---")
    print(f"Base data directory: {DATA_CLEAN_DIR}")

    # --- Load and Standardize DataFrames ---

    # Load Universe
    print(f"Attempting to load Universe from: {UNIVERSE_FILE_PATH}")
    try:
        df_universe = pd.read_parquet(UNIVERSE_FILE_PATH)
        df_universe = standardize_df_columns(df_universe)
        print(f"✅ Universe deals loaded successfully: {len(df_universe)} records.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Universe file NOT FOUND at the expected path: {UNIVERSE_FILE_PATH}")
        print("Please ensure the file exists and the path is correct.")
        sys.exit(1)

    # Load Take-Private (TP)
    print(f"Attempting to load Take-Private from: {TP_FILE_PATH}")
    try:
        df_tp = pd.read_parquet(TP_FILE_PATH)
        df_tp = standardize_df_columns(df_tp)
        print(f"✅ Take-Private deals loaded successfully: {len(df_tp)} records.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Take-Private file NOT FOUND at the expected path: {TP_FILE_PATH}")
        print("Please ensure the file exists and the path is correct.")
        sys.exit(1)

    # --- Anti-Join Logic (Subtract TP deals from Universe deals) ---

    # Matching columns must define a unique transaction:
    match_cols = ['company_name_clean', 'entry_date', 'deal_size_usd_m_entry']

    # Use only the columns that successfully standardized in both DataFrames
    common_cols = [col for col in match_cols if col in df_universe.columns and col in df_tp.columns]

    if len(common_cols) < 2:
        print(f"Error: Could not find enough common standardized columns for matching. Found: {common_cols}")
        print("Required for reliable match: company_name_clean, entry_date, deal_size_usd_m_entry.")
        sys.exit(1)

    print(f"Matching on keys: {common_cols}")

    # 1. Create a unique identifier string for each row in the TP DataFrame
    df_tp['__key'] = df_tp[common_cols].astype(str).agg('|'.join, axis=1)
    tp_keys = set(df_tp['__key'])

    # 2. Create the same unique identifier in the UNIVERSE DataFrame
    df_universe['__key'] = df_universe[common_cols].astype(str).agg('|'.join, axis=1)

    # 3. Filter the Universe DataFrame: keep rows where the key is NOT in the TP set
    print(f"\nTotal records in Universe: {len(df_universe)}")
    print(f"Total unique records identified as Take-Private: {len(tp_keys)}")

    df_non_tp = df_universe[~df_universe['__key'].isin(tp_keys)].copy()

    # Clean up the temporary key column
    df_non_tp = df_non_tp.drop(columns=['__key'])

    # --- Save Result ---

    print(f"Resulting Non-Take-Private deals (Universe - TP): {len(df_non_tp)}")

    if len(df_non_tp) == len(df_universe):
        print("Warning: Non-TP set size equals Universe size. Check your matching keys.")

    df_non_tp.to_parquet(NON_TP_FILE_PATH, index=False)
    print(f"✅ FINAL OUTPUT: Successfully created and saved non-take-private data to {NON_TP_FILE_PATH}")
    print(f"--- Anti-Join Process Complete ---")


if __name__ == '__main__':
    generate_non_take_private_data()