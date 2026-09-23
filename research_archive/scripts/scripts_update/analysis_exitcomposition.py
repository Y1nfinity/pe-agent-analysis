import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def find_project_root() -> Path:
    """Finds the 'PEAgent' root directory by searching for data/scripts folders."""
    current = Path(__file__).resolve().parent
    # Look up to 5 levels up for the project root
    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent

    # Fallback to specific user path if found
    fallback = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
    if fallback.exists():
        return fallback

    # Final fallback: 2 levels up from scripts/scripts_update
    return Path(__file__).resolve().parents[2]


def categorize_exit(row):
    """
    Consolidates complex deal type combinations into clean exit categories.
    Adjusted to check common column naming variations.
    """
    # Try multiple possible column names for deal types
    t1 = str(row.get('deal_type_exit') or row.get('deal_type_1_exit') or '').strip()
    t2 = str(row.get('Deal Type 2_exit') or row.get('deal_type_2_exit') or '').strip()
    t3 = str(row.get('Deal Type 3_exit') or row.get('deal_type_3_exit') or '').strip()

    combined = (t1 + " " + t2 + " " + t3).lower()

    # 1. IPOs & Public Listings
    if 'ipo' in combined or 'reverse merger' in combined or 'listing' in combined:
        return 'Public Listing / IPO'

    # 2. Secondary Buyouts (Sale to another PE firm)
    if 'secondary' in combined or 'management buyout' in combined:
        return 'Secondary Buyout (SBO)'

    # 3. Trade Sales / Strategic M&A (Sale to a company)
    if 'merger/acquisition' in t1.lower() or 'merger of equals' in combined or 'strategic' in combined:
        return 'Trade Sale (Strategic)'

    # 4. Standard Buyouts / Others
    if 'buyout/lbo' in t1.lower() or 'recapitalization' in combined:
        return 'Other PE Sale / Recap'

    return 'Other / Miscellaneous'


def create_exit_graphic():
    # 1. Setup Paths using dynamic root detection
    ROOT = find_project_root()
    INPUT_FILE = ROOT / "data" / "clean" / "p2p_linked_master.xlsx"

    print(f"Project Root identified as: {ROOT}")

    if not INPUT_FILE.exists():
        # Check for CSV fallback as per reference script
        csv_fallback = INPUT_FILE.with_suffix('.csv')
        if csv_fallback.exists():
            INPUT_FILE = csv_fallback
        else:
            print(f"ERROR: File not found at {INPUT_FILE}")
            print(f"Please check if the file exists in: {INPUT_FILE.parent}")
            return

    print(f"Attempting to load: {INPUT_FILE}")

    # 2. Load with explicit engine for .xlsx
    try:
        if INPUT_FILE.suffix.lower() == '.xlsx':
            # Use openpyxl for Excel files to avoid format determination errors
            df = pd.read_excel(INPUT_FILE, engine='openpyxl')
        else:
            df = pd.read_csv(INPUT_FILE)

        if df.empty:
            print("The file was loaded but appears to be empty.")
            return

    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 3. Filter for realized exits
    exit_date_col = 'deal_date_exit'
    if exit_date_col not in df.columns:
        print(f"Warning: '{exit_date_col}' not found. Analyzing all rows.")
        exited_df = df.copy()
    else:
        exited_df = df.dropna(subset=[exit_date_col]).copy()

    if exited_df.empty:
        print("No realized exits (non-null exit dates) found in the data.")
        return

    # 4. Apply Taxonomy
    exited_df['exit_category'] = exited_df.apply(categorize_exit, axis=1)

    # 5. Prepare Plot Data
    counts = exited_df['exit_category'].value_counts()
    percentages = (counts / counts.sum() * 100).round(1)

    # 6. Visualization
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    # Colors: Professional Mako palette
    colors = sns.color_palette("mako", len(counts))
    bars = plt.barh(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.9)

    # Add numeric and percentage labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        category_name = counts.index[i]
        pct = percentages[category_name]
        label = f' {int(width)} ({pct}%)'
        plt.text(width, bar.get_y() + bar.get_height() / 2, label, va='center', fontweight='bold')

    plt.title('Private Equity Exit Landscape: Deal Composition', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Deals', fontsize=12)
    plt.ylabel('Exit Strategy', fontsize=12)
    plt.gca().invert_yaxis()

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # Save output to project-relative path
    output_folder = ROOT / "scripts" / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "exit_landscape_chart.png"

    plt.savefig(output_path, dpi=300)
    print(f"\nAnalysis complete. Total exits categorized: {len(exited_df)}")
    print(f"Graphic saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    create_exit_graphic()