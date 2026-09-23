import pandas as pd
import numpy as np
from pathlib import Path
import re


def clean_company_name(name):
    if pd.isna(name): return ""
    name = str(name).lower()
    name = re.sub(r'\(.*?\)', '', name)
    suffixes = [r'\binc\b', r'\bcorp\b', r'\bcorporation\b', r'\blp\b', r'\bllc\b', r'\bgroup\b', r'\bplc\b', r'\bco\b']
    for s in suffixes:
        name = re.sub(s, '', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return " ".join(name.split())


def find_project_root() -> Path:
    """
    Finds the true project root by looking for the 'data' directory.
    This handles cases where the script is deeply nested.
    """
    current = Path(__file__).resolve().parent
    # Iterate up the directory tree
    while current.parent != current:
        if (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (directory containing 'data' folder).")


def process_p2p_matching():
    try:
        root = find_project_root()
        DATA_DIR = root / "data" / "raw"
        OUTPUT_DIR = root / "data" / "clean"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Project Root Detected: {root}")
        print(f"Looking for data in: {DATA_DIR}")

        # Load P2P Entry and Exit Data
        # Using sheet_name="Sheet1" as per user requirements
        df_entry = pd.read_excel(DATA_DIR / "pitchbook_public_2_private_all.xlsx", sheet_name="Sheet1")
        df_exit = pd.read_excel(DATA_DIR / "pitchbook_exit_completed.xlsx", sheet_name="Sheet1")

        rename_map = {
            'Companies': 'company_name',
            'Deal Date': 'deal_date',
            'Deal Size': 'deal_size'
        }
        df_entry = df_entry.rename(columns=rename_map)
        df_exit = df_exit.rename(columns=rename_map)

        df_entry['deal_date'] = pd.to_datetime(df_entry['deal_date'], errors='coerce')
        df_exit['deal_date'] = pd.to_datetime(df_exit['deal_date'], errors='coerce')

        print("Cleaning company names and merging datasets...")
        df_entry['company_name_clean'] = df_entry['company_name'].apply(clean_company_name)
        df_exit['company_name_clean'] = df_exit['company_name'].apply(clean_company_name)

        linked_df = pd.merge(
            df_entry,
            df_exit,
            on='company_name_clean',
            how='left',
            suffixes=('_entry', '_exit')
        )

        # Logical filter: Exit must be after Entry or be NaN (Active)
        final_output = linked_df[
            (linked_df['deal_date_exit'] > linked_df['deal_date_entry']) |
            (linked_df['deal_date_exit'].isna())
            ].copy()

        # Calculate holding period
        final_output['holding_period'] = (final_output['deal_date_exit'] - final_output[
            'deal_date_entry']).dt.days / 365.25

        save_path = OUTPUT_DIR / "p2p_linked_master.csv"
        final_output.to_csv(save_path, index=False)

        print("-" * 30)
        print(f"Success! Final records: {len(final_output)}")
        print(f"File saved to: {save_path}")
        print("-" * 30)

    except Exception as e:
        print(f"LINKER FAILED: {str(e)}")


if __name__ == "__main__":
    process_p2p_matching()