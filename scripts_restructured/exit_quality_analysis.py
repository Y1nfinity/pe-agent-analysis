"""
exit_quality_analysis.py
========================
Group 5: Exit quality analysis

Outputs:
    - Exit quality scores
    - Quality × vintage matrices
    - Quality distribution bar charts
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "exit_quality"
OUT.mkdir(parents=True, exist_ok=True)

QUALITY_MAP = {
    "IPO": 5,
    "Merger/Acquisition": 4,
    "Buyout/LBO": 3,
    "Secondary Buyout": 2,
    "PIPE": 1,
    "Add-on": 0,
    "Corporate Divestiture": 0,
    "Distressed Acquisition": 0,
    "Other": 0
}


def load_pairs():
    return pd.read_parquet(DATA / "entry_exit_pairs.parquet")


# -------------------------------------------------------
# Assign quality score
# -------------------------------------------------------
def assign_quality(df):
    df = df.copy()
    df["quality_score"] = df["exit_type"].map(QUALITY_MAP).fillna(0)
    return df


# -------------------------------------------------------
# Quality distribution
# -------------------------------------------------------
def plot_quality_distribution(df, out):
    counts = df["quality_score"].value_counts().sort_index()
    counts.to_csv(out / "ExitQuality_Distribution.csv")

    plt.figure(figsize=(8,6))
    counts.plot(kind="bar")
    plt.title("Exit Quality Score Distribution")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    plt.savefig(out / "Figure_ExitQuality_Distribution.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Quality × vintage heatmap
# -------------------------------------------------------
def quality_by_vintage(df, out):
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year

    heat = (
        df.groupby(["entry_year", "exit_type"])["quality_score"]
        .mean()
        .unstack()
        .fillna(0)
    )

    plt.figure(figsize=(12,6))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Avg Exit Quality Score by Entry Vintage × Exit Type")
    plt.savefig(out / "Figure_ExitQuality_ByVintageHeatmap.png", dpi=300)
    plt.close()

    heat.to_csv(out / "ExitQuality_ByVintage.csv")


def run_all():
    df = load_pairs()
    df_q = assign_quality(df)
    plot_quality_distribution(df_q, OUT)
    quality_by_vintage(df_q, OUT)
    print(f"✅ Exit quality analysis saved to {OUT}")


if __name__ == "__main__":
    run_all()
