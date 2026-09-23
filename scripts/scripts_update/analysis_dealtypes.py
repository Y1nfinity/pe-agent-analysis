import pandas as pd
from pathlib import Path


def find_project_root() -> Path:
    """Finds the 'PEAgent' root directory by searching for data/scripts folders."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent
    fallback = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
    return fallback if fallback.exists() else Path(__file__).resolve().parents[2]


def identify_exit_types():
    # 1. Setup Paths
    ROOT = find_project_root()
    INPUT_FILE = ROOT / "data" / "clean" / "p2p_linked_master_updated.xlsx"

    if not INPUT_FILE.exists():
        csv_fallback = INPUT_FILE.with_suffix('.csv')
        if csv_fallback.exists():
            INPUT_FILE = csv_fallback
        else:
            print(f"ERROR: File not found at {INPUT_FILE}")
            return

    # 2. Load Data
    try:
        if INPUT_FILE.suffix.lower() == '.xlsx':
            df = pd.read_excel(INPUT_FILE, engine='openpyxl')
        else:
            df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 3. Filter for Exited Deals only
    # We only care about the deal types of the 'exit' leg of the transaction
    exited_df = df.dropna(subset=['deal_date_exit']).copy()

    # Identify the deal type columns for the exit
    # Assuming the columns are named deal_type_1_exit, deal_type_2_exit, etc.
    # or similar based on your schema. Adjust names if they differ.
    exit_type_cols = ['deal_type_1_exit', 'deal_type_2_exit', 'deal_type_3_exit']

    # Check if columns exist, if not, try to find columns with 'type' and 'exit'
    actual_cols = [c for c in exit_type_cols if c in exited_df.columns]

    if not actual_cols:
        print("Could not find specific 'exit' type columns. Searching for alternatives...")
        actual_cols = [c for c in exited_df.columns if 'type' in c.lower() and 'exit' in c.lower()]

    if not actual_cols:
        print(f"Available columns: {exited_df.columns.tolist()}")
        return

    print(f"Analyzing combinations for columns: {actual_cols}\n")

    # 4. Group and Count Combinations
    combinations = exited_df.groupby(actual_cols, dropna=False).size().reset_index(name='count')
    combinations = combinations.sort_values('count', ascending=False)

    print("--- UNIQUE EXIT TYPE COMBINATIONS ---")
    print(combinations.to_string(index=False))

    # 5. Summary Statistics
    total_exits = len(exited_df)
    print(f"\nTotal realized exits: {total_exits}")


if __name__ == "__main__":
    identify_exit_types()