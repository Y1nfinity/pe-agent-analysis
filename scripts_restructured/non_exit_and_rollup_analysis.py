"""
non_exit_and_rollup_analysis.py
===============================
Group 6 & 7:
    - Add-on vs Real Exit decomposition
    - Roll-up intensity over time
    - Non-exit / right-censoring analysis
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "non_exit_rollup"
OUT.mkdir(parents=True, exist_ok=True)


def load_pairs():
    return pd.read_parquet(DATA / "entry_exit_pairs.parquet")


# -------------------------------------------------------
# 1. Roll-up (Add-on) share by year
# -------------------------------------------------------
def rollup_share(df, out):
    df["exit_year"] = pd.to_datetime(df["exit_date"]).dt.year

    is_rollup = df["exit_type"].str.contains("Add-on", case=False, na=False)

    shares = df.groupby("exit_year")["exit_type"].apply(
        lambda x: (x.str.contains("Add-on", case=False, na=False)).mean()
    )

    shares.to_csv(out / "RollupShare_ByYear.csv")

    shares.plot(figsize=(10,5), marker="o")
    plt.title("Share of Add-on (Roll-up) Events by Exit Year")
    plt.ylabel("Share")
    plt.xlabel("Year")
    plt.grid(alpha=0.3)
    plt.savefig(out / "Figure_RollupShare_ByYear.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# 2. Histogram of time since entry (non-exits)
# -------------------------------------------------------
def non_exit_time(df, out):
    df_ne = df[df["exit_date"].isna()].copy()
    df_ne["entry_year"] = pd.to_datetime(df_ne["entry_date"]).dt.year

    now = pd.Timestamp("2025-01-01")
    df_ne["time_since_entry"] = (now - pd.to_datetime(df_ne["entry_date"])).dt.days / 365

    plt.figure(figsize=(10,5))
    sns.histplot(df_ne["time_since_entry"], bins=30)
    plt.title("Time Since Entry (Firms with No Recorded Exit)")
    plt.xlabel("Years")
    plt.savefig(out / "Figure_TimeSinceEntry_NonExit.png", dpi=300)
    plt.close()

    df_ne[["entry_date", "time_since_entry"]].to_csv(out / "NonExit_TimeSinceEntry.csv")


# -------------------------------------------------------
# 3. Non-exit vs real exit counts
# -------------------------------------------------------
def exit_vs_nonexit(df, out):
    df["has_exit"] = df["exit_date"].notna()

    counts = df["has_exit"].value_counts()
    counts.to_csv(out / "Exit_vs_NonExit_Counts.csv")

    counts.plot(kind="bar")
    plt.xticks([0,1], ["Exited", "No Exit"], rotation=0)
    plt.title("Counts of Exits vs Non-Exits")
    plt.savefig(out / "Figure_Exit_vs_NonExit_Counts.png", dpi=300)
    plt.close()


def run_all():
    df = load_pairs()
    rollup_share(df, OUT)
    non_exit_time(df, OUT)
    exit_vs_nonexit(df, OUT)
    print(f"✅ Roll-up & non-exit analysis saved to {OUT}")


if __name__ == "__main__":
    run_all()
