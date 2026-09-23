import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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


def analyze_holding_periods():
    # 1. Setup Paths (Matching your exact method)
    ROOT = find_project_root()
    INPUT_FILE = ROOT / "data" / "clean" / "p2p_linked_master.xlsx"
    FIGURE_DIR = ROOT / "reports" / "figures"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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

    # 3. Data Cleaning & Holding Period Calculation
    # Ensure date columns are datetime objects
    df['deal_date_entry'] = pd.to_datetime(df['deal_date_entry'], errors='coerce')
    df['deal_date_exit'] = pd.to_datetime(df['deal_date_exit'], errors='coerce')

    # Filter for realized (exited) deals only
    exited_df = df.dropna(subset=['deal_date_entry', 'deal_date_exit']).copy()

    # Calculate holding period in years
    exited_df['hold_period_years'] = (exited_df['deal_date_exit'] - exited_df['deal_date_entry']).dt.days / 365.25

    # Sanity filter: Remove negative periods or outliers > 20 years
    exited_df = exited_df[(exited_df['hold_period_years'] > 0) & (exited_df['hold_period_years'] < 10)]
    exited_df['exit_year'] = exited_df['deal_date_exit'].dt.year

    # 4. Visualization
    # Set a clean, professional style
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Distribution of Holding Periods
    sns.histplot(exited_df['hold_period_years'], bins=15, kde=True, color='#1e4b8a', ax=ax1)
    median_val = exited_df['hold_period_years'].median()
    mean_val = exited_df['hold_period_years'].mean()

    ax1.axvline(median_val, color='#a51c30', linestyle='--', label=f'Median: {median_val:.1f} yrs')
    ax1.set_title('Distribution of P2P Holding Periods', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Years from Entry to Exit')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Subplot 2: Average Holding Period Trend by Exit Year
    # Filter for years with a meaningful number of exits (e.g., >= 2)
    yearly_avg = exited_df.groupby('exit_year')['hold_period_years'].agg(['mean', 'count'])
    yearly_avg = yearly_avg[yearly_avg['count'] >= 2]

    ax2.plot(yearly_avg.index, yearly_avg['mean'], marker='o', color='#1e4b8a', linewidth=2, markersize=8)
    ax2.fill_between(yearly_avg.index, yearly_avg['mean'] - 0.5, yearly_avg['mean'] + 0.5, color='#1e4b8a', alpha=0.1)

    ax2.set_title('Avg Holding Period Trend (by Exit Year)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Year of Exit')
    ax2.set_ylabel('Average Years Held')
    ax2.set_xticks(yearly_avg.index[::2])  # Show every 2nd year for readability
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 5. Save and Close
    save_path = FIGURE_DIR / "p2p_holding_period_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Success! Analysis saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    analyze_holding_periods()