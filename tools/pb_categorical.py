# tools/pb_categorical.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pathlib import Path


# ------------------------------------------------------------
# Formatting utilities
# ------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def money_fmt():
    return FuncFormatter(lambda x, pos: f"${x/1e6:,.0f}M")

def pct_fmt():
    return FuncFormatter(lambda x, pos: f"{x*100:.0f}%")

def _apply_year_ticks(ax, years, max_target=18, rotation=0):
    """Downsample x-axis labels for readability."""
    years = list(years)
    n = len(years)
    if n <= max_target:
        idxs = list(range(n))
    else:
        step = max(1, int(round(n / max_target)))
        idxs = list(range(0, n, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels([str(int(years[i])) for i in idxs], rotation=rotation, ha="center")
    ax.margins(x=0.01)


# ------------------------------------------------------------
# Core aggregation utilities
# ------------------------------------------------------------

def assign_exit_category(per_deal: pd.DataFrame) -> pd.DataFrame:
    """
    Combine available PitchBook exit fields into a unified 'ExitCategory' column.
    Looks for 'Exit_DealType', 'Exit_DealType2', 'Exit_DealType3'.
    """
    cols = [c for c in ["Exit_DealType", "Exit_DealType2", "Exit_DealType3"] if c in per_deal.columns]

    if not cols:
        per_deal["ExitCategory"] = "Unknown"
        return per_deal

    per_deal["ExitCategory"] = (
        per_deal[cols]
        .astype(str)
        .apply(lambda r: " | ".join(
            [x for x in dict.fromkeys(r) if x and x.lower() != "nan"]), axis=1)
        .replace("", "Unknown")
    )
    return per_deal


def summarize_by_exit_category(per_deal: pd.DataFrame, metric: str = "IRR_XIRR") -> pd.DataFrame:
    """Average metric by exit category."""
    df = per_deal.copy()
    df = df[df["ExitCategory"].notna()]
    grouped = df.groupby("ExitCategory", as_index=False)[metric].mean()
    return grouped


def summarize_by_vintage_exit(breakdown: pd.DataFrame, value_col: str = "exits_n") -> pd.DataFrame:
    """Pivot-style summary by (EntryYear x ExitCategory)."""
    df = breakdown.copy()
    df["EntryYear"] = df["EntryYear"].astype(int)
    return df.pivot(index="EntryYear", columns="ExitCategory", values=value_col).fillna(0)


# ------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------

def plot_exit_counts_by_vintage(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: number of exits by ExitCategory across vintages."""
    pivot = summarize_by_vintage_exit(breakdown, value_col="exits_n")
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Exit Composition by Vintage (Counts)")
    ax.set_ylabel("Number of Exits")
    _apply_year_ticks(ax, pivot.index, rotation=0)
    ax.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exits_by_vintage_counts.png", dpi=220)
    plt.close()


def plot_exit_spend_by_vintage(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: entry spend of exited deals by ExitCategory."""
    pivot = summarize_by_vintage_exit(breakdown, value_col="entry_spend_for_exits")
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20b")
    ax.set_title("Exit Composition by Vintage (Entry Spend)")
    ax.set_ylabel("Entry Spend (USD Millions)")
    ax.yaxis.set_major_formatter(money_fmt())
    _apply_year_ticks(ax, pivot.index, rotation=0)
    ax.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exits_by_vintage_spend.png", dpi=220)
    plt.close()


def plot_exit_mix_fraction(breakdown: pd.DataFrame, out: Path):
    """Normalized 100% stacked bar: exit-type fractions by vintage."""
    pivot = summarize_by_vintage_exit(breakdown, value_col="exits_n")
    pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_title("Exit Mix Fraction by Vintage")
    ax.set_ylabel("Fraction of Exits")
    _apply_year_ticks(ax, pivot.index, rotation=0)
    ax.yaxis.set_major_formatter(pct_fmt())
    ax.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "exit_mix_fraction_by_vintage.png", dpi=220)
    plt.close()


def plot_box_metric_by_exit(per_deal: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path):
    """Boxplot of any numeric metric by ExitCategory."""
    df = per_deal[["ExitCategory", metric]].dropna()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="ExitCategory", y=metric, palette="tab10", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Exit Type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / f"box_{metric}_by_exit.png", dpi=220)
    plt.close()


def plot_heatmap_avg_metric(breakdown: pd.DataFrame, metric_col: str, out: Path):
    """Heatmap of average metric by (ExitCategory × EntryYear)."""
    df = breakdown.copy()
    if metric_col not in df.columns:
        print(f"⚠️ {metric_col} not found in breakdown.")
        return
    pivot = df.pivot(index="ExitCategory", columns="EntryYear", values=metric_col)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="RdYlGn", linewidths=0.3, cbar_kws={"label": f"Average {metric_col}"})
    plt.title(f"Average {metric_col} by Vintage × Exit Type")
    plt.xlabel("Entry Year")
    plt.ylabel("Exit Type")
    plt.tight_layout()
    plt.savefig(out / f"heatmap_avg_{metric_col}_by_exit_vintage.png", dpi=220)
    plt.close()


# ------------------------------------------------------------
# Paired vs Unmatched Coverage Visuals
# ------------------------------------------------------------

def plot_paired_unmatched_by_vintage(breakdown: pd.DataFrame, unmatched_v: pd.DataFrame, out: Path):
    """Side-by-side bars showing paired vs unmatched counts per vintage."""
    paired = breakdown.groupby("EntryYear", as_index=False)["exits_n"].sum().rename(columns={"exits_n": "paired"})
    merged = paired.merge(unmatched_v, on="EntryYear", how="outer").fillna(0).sort_values("EntryYear")
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["paired"], width, label="Paired exits")
    ax.bar(x + width / 2, merged["unmatched_n"], width, label="Unmatched entries")
    ax.set_title("Coverage by Vintage: Paired vs Unmatched Entries")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    _apply_year_ticks(ax, merged["EntryYear"], rotation=0)
    plt.tight_layout()
    plt.savefig(out / "paired_vs_unmatched_by_vintage.png", dpi=220)
    plt.close()
