"""
entry_exit_analysis.py
======================
Analyzes linked entry–exit pairs from the private equity dataset.

Inputs:
    data/clean/entry_exit_pairs.parquet

Outputs:
    - Summary tables of holding periods and exit outcomes
    - Optional visualizations of holding periods and exit trends

This module is imported by tools scripts (e.g. analyze_entry_exit.py)
and can also be run directly for testing.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure output paths always resolve from project root
ROOT = Path(__file__).resolve().parents[1]
if ROOT.name != "PEAgent":  # fallback if run from subfolder
    ROOT = ROOT.parents[0]

OUTPUT_DIR = ROOT / "outputs" / "entry_exit_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Output directory confirmed: {OUTPUT_DIR.resolve()}")

# ---------------------------------------------------------------------
# 1. Core Analysis Functions
# ---------------------------------------------------------------------
def load_entry_exit_pairs(path: str | Path) -> pd.DataFrame:
    """Load the cleaned entry–exit pairs dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def summarize_holding_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize holding period statistics by exit type.
    """
    df = df.copy()
    df["holding_period_years"] = pd.to_numeric(df["holding_period_years"], errors="coerce")
    summary = (
        df.groupby("exit_type")["holding_period_years"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary["mean"] = summary["mean"].round(2)
    summary["median"] = summary["median"].round(2)
    summary["std"] = summary["std"].round(2)
    return summary


def summarize_outcomes_by_entry_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize exit outcomes (exit_type share) by entry year.
    """
    df = df.copy()
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year
    df = df.dropna(subset=["entry_year", "exit_type"])

    grouped = (
        df.groupby(["entry_year", "exit_type"])
        .size()
        .reset_index(name="count")
    )

    # Calculate share within each entry year
    totals = grouped.groupby("entry_year")["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals

    return grouped


def compute_relisting_rate(df: pd.DataFrame) -> float:
    """Compute the share of exits that are IPOs (i.e., re-listings)."""
    ipo_mask = df["exit_type"].str.lower().str.contains("ipo", na=False)
    return ipo_mask.mean()


# ---------------------------------------------------------------------
# 2. Visualization Functions
# ---------------------------------------------------------------------
def plot_holding_period_distribution(df: pd.DataFrame, outfile=None):
    """Plot histogram of holding periods."""
    plt.figure(figsize=(8, 5))
    df["holding_period_years"].hist(bins=30)
    plt.title("Distribution of Holding Periods (Years)")
    plt.xlabel("Holding Period (Years)")
    plt.ylabel("Number of Deals")
    plt.grid(alpha=0.3)
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()


def plot_exit_outcomes_by_entry_year(df: pd.DataFrame, outfile=None):
    """Plot share of exit types by entry year."""
    pivot = df.pivot(index="entry_year", columns="exit_type", values="share").fillna(0)
    pivot.plot(kind="area", stacked=True, figsize=(10, 6))
    plt.title("Exit Outcomes by Entry Vintage (Share of Exits)")
    plt.xlabel("Entry Year")
    plt.ylabel("Share of Exits")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(alpha=0.3)
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 3. Composite Function to Run Full Analysis
# ---------------------------------------------------------------------
def run_entry_exit_analysis(pairs_path: Path, output_dir: Path) -> dict:
    """
    Run the full entry–exit analysis pipeline and export results.

    Returns a dict of summary DataFrames for further use.
    """
    print("🔹 Loading entry–exit pairs...")
    df = load_entry_exit_pairs(pairs_path)
    print(f"   Loaded {df.shape[0]:,} rows.")

    print("🔹 Summarizing holding periods...")
    holding_summary = summarize_holding_periods(df)
    print(holding_summary.head())

    print("🔹 Summarizing outcomes by entry year...")
    outcome_summary = summarize_outcomes_by_entry_vintage(df)
    print(outcome_summary.head())

    print("🔹 Computing re-listing (IPO) rate...")
    relist_rate = compute_relisting_rate(df)
    print(f"   Re-listing rate: {relist_rate:.2%}")

    # Export outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    holding_summary.to_csv(output_dir / "EntryExit_HoldingPeriods.csv", index=False)
    outcome_summary.to_csv(output_dir / "EntryExit_Outcomes_ByEntryYear.csv", index=False)

    # Visualizations
    print("🔹 Generating visualizations...")
    plot_holding_period_distribution(df, output_dir / "Figure_HoldingPeriods.png")
    plot_exit_outcomes_by_entry_year(outcome_summary, output_dir / "Figure_ExitOutcomes_ByEntryYear.png")

    print(f"✅ Entry–Exit analysis complete. Results saved to {output_dir.resolve()}")
    return {
        "holding_summary": holding_summary,
        "outcome_summary": outcome_summary,
        "relist_rate": relist_rate,
    }


# ============================================================
# ADVANCED EXIT ANALYSIS AND VISUALIZATION MODULE
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Ensure seaborn style consistent across all visuals
sns.set(style="whitegrid")


# ============================================================
# 1️⃣ Utility: Create stacked bar chart (counts or percentages)
# ============================================================
def plot_stacked_bar(
    df: pd.DataFrame,
    index_col: str,
    category_col: str,
    value_col: str = "count",
    normalize: bool = False,
    title: str = "",
    output_path: Path = None,
    annotate: bool = True,
):
    """
    General-purpose stacked bar plot.
    - index_col: column for x-axis (e.g. year or vintage)
    - category_col: column to split stacks (e.g. exit_category)
    - value_col: numeric variable (count or deal_size)
    - normalize: if True, shows percentages
    """
    pivot_df = (
        df.groupby([index_col, category_col])[value_col]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    ax = pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(index_col.capitalize())
    ax.set_ylabel("Percentage (%)" if normalize else "Count")

    # Optional annotations (on top of bars)
    if annotate:
        for i, (idx, row) in enumerate(pivot_df.iterrows()):
            total = row.sum()
            ax.text(
                i,
                total * (1.02 if not normalize else 1.05),
                f"{int(total) if not normalize else 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved figure: {output_path}")
    else:
        plt.show()


# ============================================================
# 2️⃣ EXIT BREAKDOWN TABLES (Counts, Percentages, Deal Size)
# ============================================================
def build_exit_breakdown_tables(df: pd.DataFrame) -> dict:
    """Counts, shares, and deal-value totals by exit type × exit_year."""

    df = df.copy()

    # Ensure exit_year exists
    if "exit_year" not in df.columns:
        if "exit_date" not in df.columns:
            raise KeyError("Dataset must contain exit_year or exit_date.")
        df["exit_year"] = pd.to_datetime(df["exit_date"], errors="coerce").dt.year

    # Basic count version
    counts = (
        df.groupby(["exit_year", "exit_type"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("exit_year")["count"].transform("sum")
    counts["share"] = counts["count"] / totals * 100

    # Deal value if present
    if "deal_size" in df.columns:
        size_weighted = (
            df.groupby(["exit_year", "exit_type"])["deal_size"]
            .sum()
            .reset_index(name="deal_value")
        )
    else:
        size_weighted = pd.DataFrame()

    return {
        "counts": counts,
        "deal_value": size_weighted
    }


# ============================================================
# 3️⃣ TAKE-PRIVATE EXIT ANALYSIS
# ============================================================
def analyze_take_private_exits(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Focused analysis: exits that originated from take-private entries.
    Returns summary by exit_type.
    """
    df = pairs.copy()
    df = df[df["entry_type"].str.lower().str.contains("public to private")]

    summary = (
        df.groupby("exit_type")
        .agg(
            n_exits=("exit_type", "count"),
            mean_holding_years=("holding_period_years", "mean"),
            median_holding_years=("holding_period_years", "median"),
            total_deal_value=("deal_size", "sum") if "deal_size" in df.columns else ("exit_type", "count"),
        )
        .reset_index()
        .sort_values("n_exits", ascending=False)
    )

    summary["share_of_takeprivate_exits"] = (
        summary["n_exits"] / summary["n_exits"].sum() * 100
    )

    return summary


# ============================================================
# 4️⃣ VINTAGE VS EXIT ANALYSIS
# ============================================================
def analyze_vintage_exit_relationship(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Create vintage (entry year) vs exit-type matrix.
    Returns both counts and share-per-vintage.
    """
    pairs = pairs.copy()
    pairs["entry_year"] = pd.to_datetime(pairs["entry_date"]).dt.year

    matrix = (
        pairs.groupby(["entry_year", "exit_type"])
        .size()
        .reset_index(name="count")
    )
    totals = matrix.groupby("entry_year")["count"].transform("sum")
    matrix["share"] = matrix["count"] / totals * 100
    return matrix


# ============================================================
# 5️⃣ HOLDING PERIOD QUALITY ANALYSIS (Mean + Boxplot)
# ============================================================
def analyze_holding_periods(df: pd.DataFrame, output_dir: Path):
    """
    Create summary statistics and boxplot for holding periods by exit type.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    stats = (
        df.groupby("exit_type")["holding_period_years"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    stats.to_csv(output_dir / "HoldingPeriod_By_ExitType.csv", index=False)
    print(f"✅ Saved holding period summary to {output_dir / 'HoldingPeriod_By_ExitType.csv'}")

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="holding_period_years",
        y="exit_type",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 5},
        orient="h",
    )
    plt.title("Holding Period Distribution by Exit Type", fontsize=14)
    plt.xlabel("Years")
    plt.ylabel("Exit Type")
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_HoldingPeriod_By_ExitType.png", dpi=300)
    plt.close()
    print(f"✅ Saved boxplot: {output_dir / 'Figure_HoldingPeriod_By_ExitType.png'}")

    return stats




# ---------------------------------------------------------------------
# 4. Run as Script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    pairs_path = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    output_dir = ROOT / "outputs" / "entry_exit_analysis"

    results = run_entry_exit_analysis(pairs_path, output_dir)