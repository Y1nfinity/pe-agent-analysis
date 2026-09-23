# tools/holding.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def holding_stats(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns descriptive statistics on HoldingYears.
    """
    desc = pairs["HoldingYears"].describe(percentiles=[0.25, 0.5, 0.75])
    return desc.to_frame("HoldingYears")

def plot_holding_distribution(pairs: pd.DataFrame, outdir: str | None = None):
    """
    Creates histogram, KDE, and boxplot for HoldingYears.
    """
    sns.set(style="whitegrid")

    # Histogram + KDE
    plt.figure(figsize=(8, 5))
    sns.histplot(pairs["HoldingYears"], kde=True, bins=40, color="steelblue")
    plt.xlabel("Holding Period (Years)")
    plt.title("Distribution of Holding Periods")
    if outdir:
        plt.savefig(f"{outdir}/holding_hist.png", dpi=300)
    plt.show()

    # Boxplot
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=pairs["HoldingYears"], color="orange")
    plt.xlabel("Holding Period (Years)")
    plt.title("Boxplot of Holding Periods")
    if outdir:
        plt.savefig(f"{outdir}/holding_box.png", dpi=300)
    plt.show()


def categorize_holding_period(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a categorical column for Short/Medium/Long HPs.
    """
    def classify(hp):
        if hp < 2:
            return "Short (<2y)"
        elif hp <= 7:
            return "Medium (2–7y)"
        else:
            return "Long (>7y)"

    pairs = pairs.copy()
    pairs["HP_Category"] = pairs["HoldingYears"].apply(classify)
    return pairs

def segment_by(pairs: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Groups holding periods by a column (e.g., DealSize, Industry, Sponsor).
    Returns summary stats.
    """
    return pairs.groupby(column)["HoldingYears"].describe()

def outlier_analysis(pairs: pd.DataFrame, lower=0.5, upper=15) -> pd.DataFrame:
    """
    Flags extreme short/long holding periods.
    Default: shorter than 0.5 years, longer than 15 years.
    """
    return pairs[(pairs["HoldingYears"] < lower) | (pairs["HoldingYears"] > upper)]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_histogram(pairs: pd.DataFrame, path: Path, bins: int = 40) -> None:
    _ensure_dir(path)
    plt.figure(figsize=(8, 5))
    plt.hist(pairs["HoldingYears"].dropna(), bins=bins)
    plt.xlabel("Holding Period (Years)")
    plt.ylabel("Count")
    plt.title("Distribution of Holding Periods")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_boxplot(pairs: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path)
    plt.figure(figsize=(8, 1.8))
    plt.boxplot(pairs["HoldingYears"].dropna(), vert=False)
    plt.xlabel("Holding Period (Years)")
    plt.title("Holding Periods — Boxplot")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_density(pairs: pd.DataFrame, path: Path) -> None:
    """
    Simple kernel density estimate via pandas (wraps matplotlib),
    avoids extra dependencies.
    """
    _ensure_dir(path)
    plt.figure(figsize=(8, 5))
    pairs["HoldingYears"].dropna().plot(kind="density")
    plt.xlabel("Holding Period (Years)")
    plt.title("Holding Periods — Density")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()