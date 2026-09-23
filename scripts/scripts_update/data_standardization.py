import pandas as pd
from pathlib import Path
import re


def clean_company_name(name):
    """Standardizes company names for robust linking."""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove punctuation and special characters
    name = re.sub(r'[^a-z0-9\s]', '', name)
    # Remove common corporate suffixes to improve matching
    suffixes = [r'\binc\b', r'\bcorp\b', r'\bcorporation\b', r'\blp\b', r'\bllc\b', r'\bplc\b', r'\bgroup\b']
    for suffix in suffixes:
        name = re.sub(suffix, '', name)
    return " ".join(name.split())


def standardize_file(file_path: Path, output_path: Path, min_size_m: float = 0.0):
    """
    Standardizes PitchBook exports (Excel or CSV).
    Handles variations in column naming across different PB export types.
    """
    if not file_path.exists():
        print(f"⚠️  Skipping: {file_path.name} (File not found in data/raw)")
        return None

    print(f"Processing: {file_path.name}...")

    # 1. Load data based on extension
    try:
        if file_path.suffix.lower() == '.xlsx':
            # Note: defaults to the first sheet.
            # If your data is on a specific sheet, use pd.read_excel(file_path, sheet_name='SheetName')
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return None

    # 2. Comprehensive Column Mapping
    # This addresses the 'KeyError' by looking for all common PitchBook variations
    rename_map = {
        'Companies': 'company_name_raw',
        'Company Name': 'company_name_raw',
        'Target Name': 'company_name_raw',
        'Deal Date': 'deal_date',
        'Deal Size': 'deal_size_m',
        'Deal Type': 'deal_type',
        'Deal ID': 'deal_id',
        'Industry': 'industry',
        'Primary Industry Code': 'industry_code'
    }

    # Only rename if the column exists in the current file
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3. Verification: Ensure we have the core columns needed for linking
    if 'company_name_raw' not in df.columns:
        print(f"  ❌ Error: Could not identify a company name column.")
        print(f"     Found columns: {df.columns.tolist()[:5]}...")
        return None

    # 4. Canonical Company Name (The key for 'entry_exit_linker.py')
    df['company_name_clean'] = df['company_name_raw'].apply(clean_company_name)

    # 5. Date Conversion
    if 'deal_date' in df.columns:
        df['deal_date'] = pd.to_datetime(df['deal_date'], errors='coerce')
        df['deal_year'] = df['deal_date'].dt.year

    # 6. Size Conversion & Filtering
    if 'deal_size_m' in df.columns:
        df['deal_size_m'] = pd.to_numeric(df['deal_size_m'], errors='coerce')
        if min_size_m > 0:
            initial_count = len(df)
            df = df[df['deal_size_m'] >= min_size_m].copy()
            print(f"  Applied >${min_size_m}M filter: {initial_count} -> {len(df)} rows.")

    # 7. Save to Parquet
    df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved to: {output_path.name}")
    return df


def run_standardization():
    # ROOT is C:\Users\azhao\PycharmProjects\PEAgent
    # Path(__file__) is in .../scripts/scripts_update/data_standardization.py
    # .parents[2] goes up: 1 (scripts_update) -> 2 (scripts) -> 3 (PEAgent)
    # Actually, parents[1] is scripts_update, parents[2] is scripts.
    # To get to PEAgent, we need parents[2] if we count 0 as the directory of the file.
    # Let's use an absolute check for clarity:

    ROOT = Path(__file__).resolve().parents[2]
    RAW_DIR = ROOT / "data" / "raw"
    CLEAN_DIR = ROOT / "data" / "clean"
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # These filenames match exactly what we found in your data/raw folder
    tasks = [
        # (Raw Filename, Output Filename, Min Size Filter)
        ("pitchbook_deals_completed.xlsx", "entry_universe_100m.parquet", 100.0),
        ("pitchbook_exit_completed.xlsx", "exit_events_all.parquet", 0.0),
        ("pitchbook_public_2_private_all.xlsx", "tp_universe_standard.parquet", 0.0)
    ]

    print(f"Looking for data in: {RAW_DIR}")

    for raw_name, clean_name, floor in tasks:
        standardize_file(RAW_DIR / raw_name, CLEAN_DIR / clean_name, min_size_m=floor)


if __name__ == "__main__":
    run_standardization()