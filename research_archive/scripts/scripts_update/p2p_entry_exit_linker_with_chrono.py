import pandas as pd
import numpy as np
from pathlib import Path
import re


def clean_company_name(name):
    """
    Aggressively cleans company names to improve match rates.
    Removes: Tickers (NAS: AAPL), Legal Suffixes (Inc, Corp), and extra whitespace.
    """
    if pd.isna(name):
        return ""
    name = str(name).lower()

    # 1. Remove PitchBook specific ticker markers: (NAS: XYZ), (NYS: XYZ), etc.
    name = re.sub(r'\(.*?\)', '', name)

    # 2. Remove common legal suffixes
    suffixes = [r'\binc\b', r'\bcorp\b', r'\bcorporation\b', r'\blp\b', r'\bllc\b', r'\bgroup\b', r'\bplc\b', r'\bco\b']
    for s in suffixes:
        name = re.sub(s, '', name)

    # 3. Final cleanup: remove special chars and extra spaces
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return " ".join(name.split())


def process_p2p_matching():
    # Setup Paths
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / "raw"
    OUTPUT_DIR = ROOT / "data" / "clean"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    # Load P2P Entry Data
    df_entry = pd.read_excel(DATA_DIR / "pitchbook_public_2_private_all.xlsx", sheet_name="Sheet1")
    # Load Exit Data
    df_exit = pd.read_excel(DATA_DIR / "pitchbook_exit_completed.xlsx", sheet_name="Sheet1")

    # Standardize Column Names
    rename_map = {
        'Companies': 'company_name',
        'Deal Date': 'deal_date',
        'Deal Size': 'deal_size',
        'Deal Type': 'deal_type'
    }
    df_entry = df_entry.rename(columns=rename_map)
    df_exit = df_exit.rename(columns=rename_map)

    # Convert Dates
    df_entry['deal_date'] = pd.to_datetime(df_entry['deal_date'], errors='coerce')
    df_exit['deal_date'] = pd.to_datetime(df_exit['deal_date'], errors='coerce')

    # Apply Aggressive Cleaning
    print("Cleaning company names...")
    df_entry['company_name_clean'] = df_entry['company_name'].apply(clean_company_name)
    df_exit['company_name_clean'] = df_exit['company_name'].apply(clean_company_name)

    # --- FIXING OVER-PAIRING ---
    print("Linking Entry to Exit records with chronological constraints...")

    # 1. Perform merge (initially many-to-many)
    linked_df = pd.merge(
        df_entry,
        df_exit,
        on='company_name_clean',
        how='left',
        suffixes=('_entry', '_exit')
    )

    # 2. Filtering: Only keep Exits that happened AFTER the Entry
    # Rows with NaN 'deal_date_exit' are still kept (Active deals)
    valid_mask = (linked_df['deal_date_exit'] > linked_df['deal_date_entry']) | (linked_df['deal_date_exit'].isna())
    linked_df = linked_df[valid_mask].copy()

    # 3. Resolve Over-pairing: For each specific Entry event, find the NEAREST Exit
    # Sort by Entry ID (or original name/date) and then by Exit Date
    linked_df = linked_df.sort_values(by=['company_name_entry', 'deal_date_entry', 'deal_date_exit'])

    # Drop duplicate entries, keeping the FIRST exit that occurred after entry
    # This ensures one P2P deal doesn't link to three different future exits
    final_output = linked_df.drop_duplicates(subset=['company_name_entry', 'deal_date_entry'], keep='first').copy()

    # Calculate Holding Period (in years)
    final_output['holding_period'] = (final_output['deal_date_exit'] - final_output['deal_date_entry']).dt.days / 365.25

    # Summary Statistics
    total_p2p = len(df_entry)
    matches_found = final_output['deal_date_exit'].notna().sum()

    print("-" * 30)
    print(f"P2P Entry Universe: {total_p2p}")
    print(f"Unique Exits Linked: {matches_found}")
    print(f"Match Rate:          {(matches_found / total_p2p):.2%}")
    print("-" * 30)

    # Save Results
    final_output.to_csv(OUTPUT_DIR / "p2p_linked_master.csv", index=False)
    print(f"Success! File saved to {OUTPUT_DIR}/p2p_linked_master.csv")


if __name__ == "__main__":
    process_p2p_matching()