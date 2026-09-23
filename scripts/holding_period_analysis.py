import pandas as pd
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Any
from pathlib import Path
# New imports for plotting
import matplotlib.pyplot as plt
import numpy as np

# Ensure you have the 'calculate_holding_periods.py' file available in the same directory
from calculate_holding_periods import calculate_holding_period

# --- Stage 1: Data Cleaning Utilities (Integrated from Project Overview) ---
CORPORATE_SUFFIXES = [
    r"inc\b", r"\bco\b", r"\bcorp\b", r"\bcorporation\b", r"\bcompany\b",
    r"\bltd\b", r"\bllc\b", r"\bsa\b", r"\bna\b", r"\bas\b", r"\bplc\b",
    r"\bgmbh\b", r"\bpartners\b", r"\btechnologies\b", r"\btechnology\b",
    r"\band\b", r"\bholdings\b"
]
SUFFIX_PATTERN = re.compile("|".join(CORPORATE_SUFFIXES), flags=re.IGNORECASE)


@lru_cache(maxsize=50000)
def clean_company_name(raw: str) -> str:
    """
    Clean company names into a standardized canonical format for matching.
    """
    if raw is None:
        return ""

    # Convert to string and lowercase
    name = str(raw).strip().lower()

    # Normalize unicode (e.g., accented characters) and strip non-ASCII
    name = unicodedata.normalize("NFKD", name).encode('ascii', 'ignore').decode('utf-8')

    # Remove URLs or web-style identifiers
    name = re.sub(r"http\S+|www\.\S+", " ", name)

    # Remove ticker symbols in parentheses, e.g. "Apple (AAPL)"
    name = re.sub(r"\([^)]*\)", " ", name)

    # Remove punctuation entirely
    name = re.sub(r"[^\w\s]", " ", name)

    # Remove corporate suffixes
    name = SUFFIX_PATTERN.sub(" ", name)

    # Remove digits and collapse excessive whitespace
    name = re.sub(r"\d+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# -----------------------------------------------------------------------------


# --- Stage 2A: Loading Data from Structured File ---
def load_deals_from_parquet(root: Path) -> pd.DataFrame:
    """
    Loads the cleaned, linked universe entry/exit pairs (the full dataset
    that includes holding period information).

    Args:
        root: The project root Path object.

    Returns:
        A DataFrame with the essential deal columns, or raises FileNotFoundError.
    """
    p = root / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Clean universe pairs not found: {p}")

    df = pd.read_parquet(p)
    print(f"Successfully loaded {len(df)} deals from {p}")

    # --- Normalize column names (must align with expected format below) ---
    rename_map = {}
    if "deal_date_entry" in df.columns:
        rename_map["deal_date_entry"] = "entry_date"
    if "deal_date_exit" in df.columns:
        rename_map["deal_date_exit"] = "exit_date"

    df = df.rename(columns=rename_map)

    # Ensure required columns are datetime objects
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors='coerce')
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors='coerce')

    # Filter only deals that have exited (non-null exit_date)
    df_exited = df[df["exit_date"].notna()].copy()

    # Check for the required cleaning column, or create a simplified one if missing
    if 'company_name_clean' not in df_exited.columns:
        print("Warning: 'company_name_clean' column not found in Parquet. Running full cleaning pipeline.")
        # Attempt to clean a standard name column if possible
        if 'Companies' in df_exited.columns:
            df_exited['company_name_clean'] = df_exited['Companies'].apply(clean_company_name)
        elif 'company_name' in df_exited.columns:
            df_exited['company_name_clean'] = df_exited['company_name'].apply(clean_company_name)
        else:
            print("ERROR: Cannot perform full cleaning without a source name column.")

    return df_exited


# --- Stage 2B: Integrated Preprocessing Pipeline ---
def load_and_preprocess_deals(root: Path) -> pd.DataFrame:
    """
    Attempts to load data from the clean Parquet file. If it fails,
    it uses hardcoded dummy data for execution.
    """
    print(f"\n--- Stage 2: Attempting to Load Deals and Preprocess ---")

    try:
        # Load from the structured, clean dataset
        df_exited = load_deals_from_parquet(root)

        # Check if holding periods are already calculated (they should be in the clean file)
        if 'holding_years' not in df_exited.columns:
            print("Holding period columns missing. Calculating periods now.")
            df_exited['holding_days'], df_exited['holding_years'] = calculate_holding_period(
                df_exited['entry_date'],
                df_exited['exit_date']
            )
        else:
            print("Holding periods found in data, skipping calculation.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}. Using hardcoded dummy data for execution.")

        # Fallback to hardcoded dummy data
        data = {
            'Company Name': ['A-Holdings, Inc.', 'B-Tech LLC', 'C-Consulting', 'D-Logistics Co.', 'E-Retail Inc.',
                             'F-FinTech', 'G-Energy Holdings'],
            'Deal Date (Entry)': ['2015-01-01', '2016-06-01', '2017-03-01', '2018-10-15', '2019-01-01', '2019-12-01',
                                  '2020-05-01'],
            'Deal Date (Exit)': ['2019-05-15', '2021-12-31', '2023-03-01', None, '2023-01-01', '2024-06-01',
                                 '2024-05-01'],
            'Deal Status': ['Exited', 'Exited', 'Exited', 'Active', 'Exited', 'Exited', 'Exited'],
        }
        df = pd.DataFrame(data)

        # Apply necessary preprocessing steps to the dummy data
        df['company_name_clean'] = df['Company Name'].apply(clean_company_name)
        df['entry_date'] = pd.to_datetime(df['Deal Date (Entry)'], errors='coerce')
        df['exit_date'] = pd.to_datetime(df['Deal Date (Exit)'], errors='coerce')
        df_exited = df[df['Deal Status'] == 'Exited'].copy()

        df_exited['holding_days'], df_exited['holding_years'] = calculate_holding_period(
            df_exited['entry_date'],
            df_exited['exit_date']
        )
        print("Holding periods calculated successfully on dummy data.")

    # Final necessary step before analysis
    df_exited['exit_year'] = df_exited['exit_date'].dt.year
    print(f"Filtered to {len(df_exited)} exited deals.")
    print(f"Sample cleaned names: {df_exited['company_name_clean'].head(2).tolist()}...")

    return df_exited


# --- Stage 3: Statistical Analysis ---
def analyze_holding_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates descriptive statistics for holding periods, grouped by exit year.
    The resulting DataFrame retains numeric types for plotting.

    Args:
        df: A pandas DataFrame containing 'exit_year' and 'holding_years'.

    Returns:
        A DataFrame summarizing the holding period statistics by exit year.
    """
    print("\n--- Stage 3: Starting Holding Period Statistical Analysis ---")

    # Define the aggregations needed: Quartiles, Mean, and Standard Deviation
    aggregations = {
        'holding_years': [
            'count',
            'mean',
            'std',
            lambda x: x.quantile(0.25),  # Q1
            'median',  # Q2
            lambda x: x.quantile(0.75)  # Q3
        ]
    }

    # Group by Exit Year and apply the aggregations
    holding_stats = df.groupby('exit_year').agg(aggregations)

    # Flatten the MultiIndex columns for readability
    holding_stats.columns = [
        '_'.join(col).strip() for col in holding_stats.columns.values
    ]

    # Rename the quartile columns
    holding_stats = holding_stats.rename(columns={
        'holding_years_<lambda_0>': 'holding_years_Q1',
        'holding_years_median': 'holding_years_Q2_Median',
        'holding_years_<lambda_1>': 'holding_years_Q3'
    })

    # Note: We return the dataframe with NUMERIC values for plotting,
    # and only format the copy that is printed to console in the main block.
    return holding_stats.reset_index()


# --- Stage 4: Visualization ---
def plot_holding_period_trends(summary_df: pd.DataFrame, root: Path,
                               title: str = "PE Holding Period Trends by Exit Year"):
    """
    Generates a line chart showing the median holding period trend
    with a shaded band representing the interquartile range (Q1 to Q3).

    Args:
        summary_df: DataFrame output from analyze_holding_periods.
        root: The project root path for saving the figure.
        title: The title for the plot.
    """
    print("\n--- Stage 4: Generating Holding Period Trend Plot ---")

    # 1. Prepare data (filter out years with too few observations if desired, but we'll plot all)
    # Convert columns back to numeric (needed if the summary was formatted for printing)
    df = summary_df.copy()

    # Filter out years with only 1 deal, as std dev and quartiles are less meaningful
    df = df[df['holding_years_count'] > 1]

    if df.empty:
        print("Warning: Insufficient data (less than 2 deals per year) to generate trend plot.")
        return

    years = df['exit_year']
    median = df['holding_years_Q2_Median']
    q1 = df['holding_years_Q1']
    q3 = df['holding_years_Q3']

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # 3. Plot the Quartile Band (Interquartile Range - IQR)
    ax.fill_between(years, q1, q3,
                    color='lightblue', alpha=0.4,
                    label='Interquartile Range (Q1 to Q3)')

    # 4. Plot the Median Line
    ax.plot(years, median,
            color='darkblue', linewidth=2, marker='o',
            label='Median Holding Period (Years)')

    # 5. Customize
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Exit Year", fontsize=12)
    ax.set_ylabel("Holding Period (Years)", fontsize=12)

    # Set X-ticks to show every year present in the data
    ax.set_xticks(years)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.legend(loc='upper right')

    # Ensure 'outputs/figures' directory exists
    output_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "holding_period_trend.png"

    # 6. Save Plot
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved trend figure → {outfile}")


def plot_holding_period_boxplot(raw_df: pd.DataFrame, root: Path,
                                title: str = "Holding Period Distribution by Exit Year"):
    """
    Generates a Box-and-Whisker plot showing the distribution of holding periods
    for each exit year.

    Args:
        raw_df: The raw DataFrame containing 'exit_year' and 'holding_years'.
        root: The project root path for saving the figure.
        title: The title for the plot.
    """
    print("\n--- Stage 4: Generating Holding Period Box Plot ---")

    # Filter out any NaNs in the holding period just in case
    df = raw_df.dropna(subset=['holding_years', 'exit_year']).copy()

    # Sort years for correct order on the plot
    sorted_years = sorted(df['exit_year'].unique())

    # Prepare data for boxplot: a list of lists/arrays, where each inner list
    # contains the holding years for a specific exit year.
    data_to_plot = [df[df['exit_year'] == year]['holding_years'].values for year in sorted_years]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate the Box Plot
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=[str(int(year)) for year in sorted_years])

    # Customize colors
    for box in bp['boxes']:
        # change outline color
        box.set(color='darkblue', linewidth=2)
        # change fill color
        box.set(facecolor='lightblue')

    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=2)

    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='darkblue', linewidth=2)

    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    # Customize
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Exit Year", fontsize=12)
    ax.set_ylabel("Holding Period (Years)", fontsize=12)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure 'outputs/figures' directory exists
    output_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "holding_period_boxplot.png"

    # Save Plot
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved box plot figure → {outfile}")


if __name__ == '__main__':
    # Determine the project root (assuming the script is inside a 'scripts' or similar directory)
    # This aligns with the path resolution in the plotting example you provided
    ROOT = Path(__file__).resolve().parents[1]

    # Full Pipeline Execution

    # 1. Load, Clean, Preprocess, and Calculate Periods
    deals_data = load_and_preprocess_deals(root=ROOT)

    # 2. Analyze the Exited Deals (Returns numeric DataFrame)
    analysis_results_numeric = analyze_holding_periods(deals_data)

    # 3. Generate Visualizations (Line Chart + Box Plot)
    plot_holding_period_trends(analysis_results_numeric, ROOT)
    plot_holding_period_boxplot(deals_data, ROOT)  # Uses the raw data for the boxplot

    # 4. Print Tabular Results (using a formatted version of the data)
    print("\n--- Final Analysis: Average Holding Period (Years) by Exit Year ---")

    # Create the formatted version for display
    display_results = analysis_results_numeric.copy()
    format_mapping = {
        col: '{:.2f}'.format for col in display_results.columns if col not in ['exit_year', 'holding_years_count']
    }
    for col, fmt in format_mapping.items():
        # Handle nan values during formatting for the console output
        display_results[col] = display_results[col].apply(lambda y: fmt(y) if pd.notna(y) else 'nan')

    display_results['holding_years_count'] = display_results['holding_years_count'].astype(int)
    print(display_results.to_string(index=False))

    print("\n\nPipeline Run Complete. Next steps typically involve linking these results to returns/valuation data.")