# tools/plots_p2p.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# helper to control xtick density
def _apply_year_ticks(ax, years, max_target=20, rotation=0):
    """Reduce crowded xticks for better readability"""
    step = max(1, len(years) // max_target)
    ax.set_xticks(np.arange(0, len(years), step))
    ax.set_xticklabels(years[::step], rotation=rotation, ha="center")

# ---------- NEW FUNCTIONS ---------- #

def stacked_exits_by_vintage_counts(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: number of exits by type and entry year."""
    df = breakdown.groupby(["EntryYear", "ExitCategory"], as_index=False)["exits_n"].sum()
    pivot = df.pivot(index="EntryYear", columns="ExitCategory", values="exits_n").fillna(0)

    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, colormap="tab20", edgecolor="none")
    plt.title("Exit Composition by Vintage (Counts)")
    plt.ylabel("Number of Exits")
    plt.xlabel("Entry Year")
    plt.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exits_by_vintage_counts.png", dpi=220)
    plt.close()


def stacked_exits_by_vintage_spend(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: total entry spend (dollars) by exit type and year."""
    df = breakdown.groupby(["EntryYear", "ExitCategory"], as_index=False)["entry_spend_for_exits"].sum()
    pivot = df.pivot(index="EntryYear", columns="ExitCategory", values="entry_spend_for_exits").fillna(0)

    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, colormap="tab20b", edgecolor="none")
    plt.title("Exit Composition by Vintage (Entry Spend)")
    plt.ylabel("Total Entry Spend ($)")
    plt.xlabel("Entry Year")
    plt.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exits_by_vintage_spend.png", dpi=220)
    plt.close()


def exit_mix_fraction_by_vintage(breakdown: pd.DataFrame, out: Path):
    """100% stacked bar: fraction of exits by type (number-weighted)."""
    df = breakdown.groupby(["EntryYear", "ExitCategory"], as_index=False)["exits_n"].sum()
    pivot = df.pivot(index="EntryYear", columns="ExitCategory", values="exits_n").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)  # normalize to 1.0

    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, colormap="Set3", edgecolor="none")
    plt.title("Exit Mix Fraction by Vintage (Number of Exits)")
    plt.ylabel("Fraction of Exits")
    plt.xlabel("Entry Year")
    plt.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exit_mix_fraction_by_vintage.png", dpi=220)
    plt.close()


def heatmap_irr_by_exit_vintage(breakdown: pd.DataFrame, out: Path):
    """Heatmap: average IRR proxy by exit type and entry year."""
    df = breakdown.copy()
    pivot = df.pivot(index="ExitCategory", columns="EntryYear", values="avg_irr")

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="RdYlGn", annot=False, linewidths=0.3, cbar_kws={"label": "Average IRR (proxy)"})
    plt.title("Average IRR by Vintage × Exit Type")
    plt.xlabel("Entry Year")
    plt.ylabel("Exit Type")
    plt.tight_layout()
    plt.savefig(out / "heatmap_avg_irr_by_exit_vintage.png", dpi=220)
    plt.close()


def boxplot_holding_by_exit(per_deal: pd.DataFrame, out: Path):
    """Boxplot of holding period (years) by exit type."""
    df = per_deal.copy()
    df = df[df["HoldingYears"].notna()]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="ExitCategory", y="HoldingYears", palette="tab10")
    plt.title("Holding Period by Exit Type")
    plt.ylabel("Holding Period (Years)")
    plt.xlabel("Exit Type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "holding_boxplot_by_exit.png", dpi=220)
    plt.close()


def boxplot_irr_by_exit(per_deal: pd.DataFrame, out: Path):
    """Boxplot of IRR (proxy) by exit type."""
    df = per_deal.copy()
    df = df[df["IRR_XIRR"].notna()]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="ExitCategory", y="IRR_XIRR", palette="tab10")
    plt.title("IRR Distribution by Exit Type (Proxy)")
    plt.ylabel("IRR (Proxy)")
    plt.xlabel("Exit Type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "irr_boxplot_by_exit.png", dpi=220)
    plt.close()

def bar_paired_unmatched_by_vintage(vintage_overview: pd.DataFrame, breakdown: pd.DataFrame, unmatched_v: pd.DataFrame, out: Path):
    """
    Side-by-side bar: per vintage, paired count vs unmatched count.
    We derive 'paired exits count' as sum of exits_n per vintage (paired),
    and unmatched_n is provided. (This is a coverage view, not total entries.)
    """
    ex = breakdown.groupby("EntryYear", as_index=False)["exits_n"].sum().rename(columns={"exits_n":"paired_exits"})
    un = unmatched_v[["EntryYear","unmatched_n"]].copy()
    merged = ex.merge(un, on="EntryYear", how="outer").fillna(0).sort_values("EntryYear")
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.4
    x = np.arange(len(merged))
    ax.bar(x - width/2, merged["paired_exits"], width, label="Paired exits")
    ax.bar(x + width/2, merged["unmatched_n"], width, label="Unmatched entries")
    ax.set_title("Coverage by Vintage: Paired Exits vs Unmatched Entries")
    ax.set_xlabel("Entry Year")
    ax.set_ylabel("Count")
    ax.set_xticks(x, merged["EntryYear"].astype(int).tolist(), rotation=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

