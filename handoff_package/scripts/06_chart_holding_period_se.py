"""
06_chart_holding_period_se.py
==============================

Produces p2p_avg_holding_period_trend_se.png: average holding period by exit
year, with a +/-1 standard error band.

Reads:
    data/clean/p2p_linked_master.csv   (see 02_link_p2p_entry_exit.py)

Writes:
    output/figures/p2p_avg_holding_period_trend_se.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXCEL_BLUE = "#4472C4"
EXCEL_BLUE_LIGHT = "#8FAADC"
GRID_COLOR = "#EFEFEF"


def _apply_house_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Garamond", "EB Garamond", "Cormorant Garamond", "Times New Roman", "DejaVu Serif"],
        "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 1.0,
        "grid.color": GRID_COLOR, "grid.linestyle": "-", "grid.linewidth": 0.8,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })


def _finish_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def load_p2p_linked_master(root: Path) -> pd.DataFrame:
    clean_dir = root / "data" / "clean"
    for name, reader in (
        ("p2p_linked_master.xlsx", lambda p: pd.read_excel(p, engine="openpyxl")),
        ("p2p_linked_master.csv", pd.read_csv),
        ("p2p_linked_master.parquet", pd.read_parquet),
    ):
        path = clean_dir / name
        if path.exists():
            print(f"Loaded: {path}")
            return reader(path)

    print(f"ERROR: Could not find p2p_linked_master.* in {clean_dir}")
    return pd.DataFrame()


def prepare_holding_period_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["deal_date_entry", "deal_date_exit"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    df = df.copy()
    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["deal_date_exit"] = pd.to_datetime(df["deal_date_exit"], errors="coerce")

    exited_df = df.dropna(subset=["deal_date_entry", "deal_date_exit"]).copy()
    exited_df["hold_period_years"] = (
        exited_df["deal_date_exit"] - exited_df["deal_date_entry"]
    ).dt.days / 365.25

    exited_df = exited_df[
        (exited_df["hold_period_years"] > 0) & (exited_df["hold_period_years"] < 10)
    ].copy()
    exited_df["exit_year"] = exited_df["deal_date_exit"].dt.year
    return exited_df


def build_yearly_stats(exited_df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    yearly_stats = (
        exited_df.groupby("exit_year")["hold_period_years"]
        .agg(["mean", "std", "count"]).sort_index()
    )
    yearly_stats = yearly_stats[yearly_stats["count"] >= min_count].copy()
    if yearly_stats.empty:
        return yearly_stats

    yearly_stats["std"] = yearly_stats["std"].fillna(0)
    yearly_stats["se"] = yearly_stats["std"] / np.sqrt(yearly_stats["count"])
    return yearly_stats


def plot_holding_period_se_band(stats_df: pd.DataFrame, filename: str, figure_dir: Path):
    if stats_df.empty:
        print(f"Skipping {filename}: no yearly stats available.")
        return

    _apply_house_style()

    fig, ax = plt.subplots(figsize=(16, 9))
    x = stats_df.index.to_numpy()
    mean = stats_df["mean"].to_numpy()
    lower = np.maximum(mean - stats_df["se"].to_numpy(), 0)
    upper = mean + stats_df["se"].to_numpy()

    _finish_axes(ax)

    ax.fill_between(x, lower, upper, color=EXCEL_BLUE_LIGHT, alpha=0.35,
                     label="±1 Standard Error", zorder=5)
    ax.plot(x, mean, marker="o", linewidth=3, markersize=8, color=EXCEL_BLUE,
            label="Average Holding Period", zorder=10)

    for year, value in zip(x, mean):
        ax.text(year, value + 0.12, f"{value:.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=EXCEL_BLUE, zorder=20)

    ax.set_title("Average Holding Period by Exit Year (±1 Standard Error)",
                 fontsize=19, fontweight="bold", pad=30)
    ax.set_xlabel("Exit Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Years Held", fontsize=15, fontweight="bold", labelpad=12)

    if len(x) > 1:
        ax.set_xticks(x[::2])
    else:
        ax.set_xticks(x)

    y_max = max(float(np.nanmax(upper)), float(np.nanmax(mean))) * 1.12
    ax.set_ylim(0, y_max)

    leg = ax.legend(loc="upper right", frameon=True, facecolor="white",
                     framealpha=1.0, edgecolor="#DDDDDD")
    leg.set_zorder(100)

    plt.tight_layout()
    save_path = figure_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = load_p2p_linked_master(root)
    if df.empty:
        print("No data loaded. Stopping.")
        return

    exited_df = prepare_holding_period_data(df)
    if exited_df.empty:
        print("No valid exited deals after filtering. Stopping.")
        return

    yearly_stats = build_yearly_stats(exited_df, min_count=2)
    if yearly_stats.empty:
        print("No exit years meet the minimum count threshold.")
        return

    plot_holding_period_se_band(yearly_stats, "p2p_avg_holding_period_trend_se.png", figure_dir)


if __name__ == "__main__":
    main()
