import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_project_root() -> Path:
    """Finds the project root by searching for the 'data' directory."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "data").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parents[2]


def create_nested_p2p_chart():
    # 1. Setup Paths
    ROOT = find_project_root()
    INPUT_FILE = ROOT / "data" / "clean" / "p2p_linked_master.csv"
    OUTPUT_DIR = ROOT / "reports" / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Error: Could not find {INPUT_FILE}. Please run the matching script first.")
        return

    # 2. Load and Prepare Data
    df = pd.read_csv(INPUT_FILE)
    df['deal_date_entry'] = pd.to_datetime(df['deal_date_entry'])
    df['entry_year'] = df['deal_date_entry'].dt.year

    # Define Exited flag
    df['is_exited'] = df['deal_date_exit'].notna()

    # 3. Aggregate Data by Year
    # We need: Total Count, Exited Count, Total Value, Exited Value
    yearly_stats = df.groupby('entry_year').agg(
        total_count=('company_name_clean', 'count'),
        exited_count=('is_exited', 'sum'),
        total_value=('deal_size_entry', 'sum'),
        exited_value=('deal_size_entry',
                      lambda x: df.loc[x.index, 'deal_size_entry'][df.loc[x.index, 'is_exited']].sum())
    ).reset_index()

    # Filter out years with no data or very old data if necessary
    yearly_stats = yearly_stats[yearly_stats['entry_year'] >= 2000]

    years = yearly_stats['entry_year']
    x = np.arange(len(years))
    width = 0.35  # width of the groups

    # 4. Initialize Plot
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    # --- LEFT AXIS: BLUE BARS (COUNTS) ---
    # Outline: Total Entries
    rects1_total = ax1.bar(x - width / 2, yearly_stats['total_count'], width,
                           edgecolor='navy', color='none', linewidth=1.5, label='Total Entries (N)')

    # Filled: Exited Entries
    rects1_exited = ax1.bar(x - width / 2, yearly_stats['exited_count'], width,
                            color='navy', alpha=0.6, label='% Exited (N)')

    # --- RIGHT AXIS: RED BARS (VALUE) ---
    # Outline: Total Value
    rects2_total = ax2.bar(x + width / 2, yearly_stats['total_value'], width,
                           edgecolor='darkred', color='none', linewidth=1.5, label='Total Entry Value ($M)')

    # Filled: Exited Value
    rects2_exited = ax2.bar(x + width / 2, yearly_stats['exited_value'], width,
                            color='darkred', alpha=0.6, label='% Exited (Value $M)')

    # 5. Data Labels (Percentages)
    def add_labels(ax, rects_filled, rects_total, is_blue=True):
        for i, (fill, total) in enumerate(zip(rects_filled, rects_total)):
            h_fill = fill.get_height()
            h_total = total.get_height()

            if h_total > 0:
                pct = (h_fill / h_total) * 100
                # Placement at 90% of filled bar height
                label_y = h_fill * 0.9 if h_fill > (h_total * 0.1) else h_fill + (h_total * 0.02)

                ax.text(fill.get_x() + fill.get_width() / 2., label_y,
                        f'{pct:.0f}%',
                        ha='center', va='top', color='white' if h_fill > (h_total * 0.1) else 'black',
                        fontweight='bold', fontsize=9)

    add_labels(ax1, rects1_exited, rects1_total)
    add_labels(ax2, rects2_exited, rects2_total)

    # 6. Formatting and Styling
    ax1.set_xlabel('Entry Year', fontweight='bold')
    ax1.set_ylabel('Number of Deals (Count)', color='navy', fontweight='bold')
    ax2.set_ylabel('Entry Value ($ Million)', color='darkred', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(years.astype(int), rotation=90)

    plt.title('P2P Entry Cohort Analysis: Realization Rates by Year', fontsize=16, fontweight='bold', pad=20)

    # Combined Legend at bottom left
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Reordering legend to match your preference: Exited Filled -> Total Outline
    ax1.legend(lines1[::-1] + lines2[::-1], labels1[::-1] + labels2[::-1],
               loc='upper center', bbox_to_anchor=(0.15, -0.15), ncol=2, frameon=False)

    plt.tight_layout()

    # 7. Save and Show
    save_path = OUTPUT_DIR / "p2p_cohort_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Ensure styles are clean
    plt.style.use('seaborn-v0_8-whitegrid')
    create_nested_p2p_chart()