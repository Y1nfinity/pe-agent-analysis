# scripts/plot_overview.py
from __future__ import annotations

from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ---------- Helpers ----------

def money_fmt():
    return FuncFormatter(lambda x, pos: f"${x:,.0f}")

def pct_fmt():
    return FuncFormatter(lambda x, pos: f"{x*100:.0f}%")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_exit_category(per_deal: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Choose which PB label to use as 'ExitCategory' for plotting:
      [analysis]
      exit_category = "Exit_DealType" | "Exit_DealType2" | "Exit_DealType3" | "combined"
    Defaults to Exit_DealType.
    """
    mode = (cfg.get("analysis", {}) or {}).get("exit_category") or "Exit_DealType"
    mode = str(mode).strip().lower()
    if mode == "exit_dealtype2" and "Exit_DealType2" in per_deal.columns:
        s = per_deal["Exit_DealType2"].astype(str)
    elif mode == "exit_dealtype3" and "Exit_DealType3" in per_deal.columns:
        s = per_deal["Exit_DealType3"].astype(str)
    elif mode == "combined" and {"Exit_DealType","Exit_DealType2","Exit_DealType3"}.issubset(per_deal.columns):
        s = (
            per_deal[["Exit_DealType","Exit_DealType2","Exit_DealType3"]]
            .astype(str)
            .apply(lambda r: " | ".join([x for x in dict.fromkeys(r) if x and x.lower() != "nan"]), axis=1)
        )
    else:
        s = per_deal.get("Exit_DealType", pd.Series(index=per_deal.index, dtype="object")).astype(str)
    return s.str.strip().str.lower().replace({"": "unknown", "nan": "unknown"})

def read_tables(root: Path, cfg: dict):
    tables_dir = root / cfg["outputs"]["tables_dir"]
    per_deal = pd.read_csv(tables_dir / "per_deal_realized_metrics.csv", parse_dates=["Entry_DealDate","Exit_DealDate"])
    breakdown = pd.read_csv(tables_dir / "vintage_exit_breakdown.csv")
    vint_over = pd.read_csv(tables_dir / "vintage_overview_entries_spend.csv")
    unmatched_v = pd.read_csv(tables_dir / "vintage_unmatched_fractions.csv")
    return per_deal, breakdown, vint_over, unmatched_v

# ---------- Plots ----------

def stacked_exits_by_vintage_counts(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: exits count per vintage, stacked by exit category."""
    df = breakdown.copy()
    df["EntryYear"] = df["EntryYear"].astype(int)
    pivot = df.pivot(index="EntryYear", columns="ExitCategory", values="exits_n").fillna(0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title("Exits by Vintage (Counts)")
    ax.set_xlabel("Entry Year")
    ax.set_ylabel("Number of Exits")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def stacked_exits_by_vintage_spend(breakdown: pd.DataFrame, out: Path):
    """Stacked bar: entry spend of exited deals per vintage, stacked by exit category."""
    df = breakdown.copy()
    df["EntryYear"] = df["EntryYear"].astype(int)
    pivot = df.pivot(index="EntryYear", columns="ExitCategory", values="entry_spend_for_exits").fillna(0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title("Exits by Vintage (Entry Spend of Exited Deals)")
    ax.set_xlabel("Entry Year")
    ax.set_ylabel("Entry Spend (USD)")
    ax.yaxis.set_major_formatter(money_fmt())
    ax.legend(title="Exit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def boxplot_metric_by_exit(per_deal: pd.DataFrame, exit_cat: pd.Series, metric: str, out: Path, title: str, ylabel: str, yfmt=None):
    """Boxplot of a per-deal metric (e.g., IRR_XIRR, HoldingYears) by exit category."""
    df = per_deal.copy()
    df["ExitCategory"] = exit_cat
    df = df[df[metric].notna()]
    if df.empty:
        # create an empty placeholder figure to avoid silent failures
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out, dpi=200)
        plt.close()
        return
    order = df["ExitCategory"].value_counts().index.tolist()
    data = [df.loc[df["ExitCategory"] == cat, metric].values for cat in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Exit Type")
    ax.set_ylabel(ylabel)
    if yfmt:
        ax.yaxis.set_major_formatter(yfmt)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def line_avg_metric_by_vintage(per_deal: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str, yfmt=None):
    """Line chart of average metric by EntryYear (e.g., mean IRR)."""
    df = per_deal[["EntryYear", metric]].dropna().copy()
    if df.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out, dpi=200)
        plt.close()
        return
    grp = df.groupby("EntryYear", as_index=False)[metric].mean().sort_values("EntryYear")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grp["EntryYear"], grp[metric], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Entry Year")
    ax.set_ylabel(ylabel)
    if yfmt:
        ax.yaxis.set_major_formatter(yfmt)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out, dpi=200)
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

def heatmap_avg_metric(per_deal: pd.DataFrame, exit_cat: pd.Series, metric: str, out: Path, title: str):
    """
    Heatmap-like table using imshow for average metric by (EntryYear x ExitCategory).
    (Matplotlib only, no seaborn.)
    """
    df = per_deal.copy()
    df["ExitCategory"] = exit_cat
    df = df[["EntryYear","ExitCategory",metric]].dropna()
    if df.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out, dpi=200)
        plt.close()
        return

    piv = df.pivot_table(index="EntryYear", columns="ExitCategory", values=metric, aggfunc="mean")
    row_labels = piv.index.astype(int).tolist()
    col_labels = piv.columns.tolist()
    data = piv.values

    fig, ax = plt.subplots(figsize=(max(8, 0.7*len(col_labels)), max(5, 0.4*len(row_labels))))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Exit Type")
    ax.set_ylabel("Entry Year")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=20, ha="right")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric, rotation=90)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

# ---------- Main ----------

def main():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())
    out_dir = root / "artifacts" / "plots"
    ensure_dir(out_dir)

    per_deal, breakdown, vint_over, unmatched_v = read_tables(root, cfg)

    # Build exit category for per-deal plots
    exit_cat = pick_exit_category(per_deal, cfg)

    # 1) Composition & coverage
    stacked_exits_by_vintage_counts(breakdown, out_dir / "exits_by_vintage_counts.png")
    stacked_exits_by_vintage_spend(breakdown, out_dir / "exits_by_vintage_spend.png")
    bar_paired_unmatched_by_vintage(vint_over, breakdown, unmatched_v, out_dir / "paired_vs_unmatched_by_vintage.png")

    # 2) Performance
    boxplot_metric_by_exit(
        per_deal, exit_cat, metric="IRR_XIRR",
        out=out_dir / "irr_boxplot_by_exit.png",
        title="IRR by Exit Type",
        ylabel="IRR (annualized)",
    )
    boxplot_metric_by_exit(
        per_deal, exit_cat, metric="HoldingYears",
        out=out_dir / "holding_boxplot_by_exit.png",
        title="Holding Period by Exit Type",
        ylabel="Holding Period (years)",
    )
    line_avg_metric_by_vintage(
        per_deal, metric="IRR_XIRR",
        out=out_dir / "avg_irr_by_vintage.png",
        title="Average IRR by Vintage",
        ylabel="IRR (annualized)",
    )

    # 3) Heatmap (optional, useful for spotting thin samples)
    heatmap_avg_metric(
        per_deal, exit_cat, metric="IRR_XIRR",
        out=out_dir / "heatmap_avg_irr_by_vintage_exit.png",
        title="Average IRR by Vintage × Exit Type",
    )

    print("Saved plots to:", out_dir)

if __name__ == "__main__":
    main()
