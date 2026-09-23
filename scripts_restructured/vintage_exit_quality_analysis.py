"""
vintage_exit_quality_analysis.py
================================
Advanced analysis of:
    - Exit type share by vintage
    - Holding period heatmap by entry vintage
    - Holding period density by exit type
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "vintage_exit_quality"
OUT.mkdir(parents=True, exist_ok=True)


def load_pairs():
    return pd.read_parquet(DATA / "entry_exit_pairs.parquet")


# --------------------------------------------------------
# 1. Exit type share by vintage (line chart)
# --------------------------------------------------------
def exit_share_by_vintage(df, out):
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year

    counts = (
        df.groupby(["entry_year", "exit_type"])
        .size()
        .reset_index(name="count")
    )

    totals = counts.groupby("entry_year")["count"].transform("sum")
    counts["share"] = counts["count"] / totals

    pivot = counts.pivot(index="entry_year", columns="exit_type", values="share")

    plt.figure(figsize=(12,6))
    pivot.plot(ax=plt.gca(), marker="o")
    plt.title("Exit Type Share by Entry Vintage")
    plt.ylabel("Share")
    plt.xlabel("Entry Year")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(out / "Figure_ExitShare_ByVintage.png", dpi=300)
    plt.close()

    pivot.to_csv(out / "ExitShare_ByVintage.csv")


# --------------------------------------------------------
# 2. Holding period heatmap
# --------------------------------------------------------
def holding_period_heatmap(df, out):
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year

    heat = (
        df.groupby(["entry_year", "exit_type"])["holding_period_years"]
        .mean()
        .unstack()
        .fillna(0)
    )

    plt.figure(figsize=(12,7))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Mean Holding Period (Years) by Entry Vintage × Exit Type")
    plt.tight_layout()
    plt.savefig(out / "Figure_HoldingPeriod_Heatmap.png", dpi=300)
    plt.close()

    heat.to_csv(out / "HoldingPeriod_Heatmap.csv")


# --------------------------------------------------------
# 3. Holding period density plots
# --------------------------------------------------------
def holding_period_density(df, out):
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data=df,
        x="holding_period_years",
        hue="exit_type",
        common_norm=False,
        fill=True,
        alpha=0.4
    )
    plt.title("Density of Holding Periods by Exit Type")
    plt.xlabel("Holding Period (Years)")
    plt.savefig(out / "Figure_HoldingPeriod_Density.png", dpi=300)
    plt.close()


# --------------------------------------------------------
# Run all
# --------------------------------------------------------
def run_all():
    df = load_pairs()
    exit_share_by_vintage(df, OUT)
    holding_period_heatmap(df, OUT)
    holding_period_density(df, OUT)
    print(f"✅ Vintage & holding period analysis saved to: {OUT}")


if __name__ == "__main__":
    run_all()
