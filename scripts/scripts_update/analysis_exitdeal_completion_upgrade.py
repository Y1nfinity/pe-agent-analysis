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
        print(f"Error: Could not find {INPUT_FILE}.")
        return

    # 2. Load and Prepare
    df = pd.read_csv(INPUT_FILE)
    df['deal_date_entry'] = pd.to_datetime(df['deal_date_entry'], errors='coerce')
    df['deal_date_exit'] = pd.to_datetime(df['deal_date_exit'], errors='coerce')
    df = df.dropna(subset=['deal_date_entry']).copy()
    df['entry_year'] = df['deal_date_entry'].dt.year
    df['is_exited'] = df['deal_date_exit'].notna()

    # 3. Aggregate
    yearly_stats = df.groupby('entry_year').apply(
        lambda x: pd.Series({
            'total_count': x['company_name_clean'].count(),
            'exited_count': x['is_exited'].sum(),
            'total_value': x['deal_size_entry'].sum(),
            'exited_value': x.loc[x['is_exited'], 'deal_size_entry'].sum()
        })
    ).reset_index()

    # Restrict years: 2000 to 2022
    yearly_stats = yearly_stats[(yearly_stats['entry_year'] >= 2000) &
                                (yearly_stats['entry_year'] <= 2022)]

    yearly_stats['pct_n'] = (yearly_stats['exited_count'] / yearly_stats['total_count']) * 100
    yearly_stats['pct_v'] = np.where(yearly_stats['total_value'] > 0,
                                     (yearly_stats['exited_value'] / yearly_stats['total_value']) * 100, 0)

    years = yearly_stats['entry_year']
    x = np.arange(len(years))

    width = 0.42
    x_count = x - width / 2
    x_value = x + width / 2

    # 4. Initialize Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    custom_blue = "#6666b2"
    custom_red = "#b96666"

    rects1 = ax1.bar(x_count, yearly_stats['pct_n'], width, color=custom_blue, alpha=1.0,
                     label='% Exited (Count)', zorder=3)
    rects2 = ax2.bar(x_value, yearly_stats['pct_v'], width, color=custom_red, alpha=1.0,
                     label='% Exited (Value)', zorder=3)

    # 5. Data Labels (Wrapped and Clipped)
    def add_wrapped_labels(ax, rects):
        for rect in rects:
            height = rect.get_height()
            # Only label if height is significantly above 0 to keep the bottom clean
            if not np.isfinite(height) or height < 0.5:
                continue

            label_text = f"{int(height)}\n%"
            x_pos = rect.get_x() + rect.get_width() / 2.0
            y_pos = height - 1.2 if height > 8 else height + 0.3

            txt = ax.text(
                x_pos, y_pos, label_text,
                ha='center', va='top' if height > 8 else 'bottom',
                fontsize=9, fontweight='bold', color='black', zorder=10
            )
            txt.set_clip_path(rect)

    add_wrapped_labels(ax1, rects1)
    add_wrapped_labels(ax2, rects2)

    # 6. Formatting
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.margins(y=0)  # Flush with axes

    ax1.set_xlabel('Entry Year', fontweight='bold', labelpad=10)
    ax1.set_ylabel('% Exited (Count)', color=custom_blue, fontweight='bold')
    ax2.set_ylabel('% Exited (Value)', color=custom_red, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(years.astype(int), rotation=90)

    # Grid pushed to background and softened
    ax1.grid(axis='y', linestyle='-', alpha=0.2, zorder=0)
    ax2.grid(False)

    # Solid baseline at 0
    ax1.axhline(0, color='black', linewidth=1, zorder=4)

    plt.title('P2P Entry Cohorts (2000-2022): Realization Rates by Year',
              fontsize=16, fontweight='bold', pad=30)

    # --- UPDATED LEGEND POSITION: TOP RIGHT ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Adding a white background (frameon=True) so gridlines don't cross the text
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', frameon=True, shadow=True, fancybox=True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_nested_p2p_chart()