from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATHING
# ============================================================

def find_project_root() -> Path:
    """Finds the PEAgent root directory by searching for data/scripts folders."""
    current = Path(__file__).resolve().parent

    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent

    fallback = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
    return fallback if fallback.exists() else Path(__file__).resolve().parents[2]


# ============================================================
# 2. GARAMOND HOUSE STYLE
# ============================================================

EXCEL_BLUE = "#4472C4"
EXCEL_BLUE_LIGHT = "#8FAADC"
GRID_COLOR = "#EFEFEF"


def _apply_house_style():
    """
    Garamond-based Excel-style chart formatting.
    """
    plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Garamond",
                "EB Garamond",
                "Cormorant Garamond",
                "Times New Roman",
                "DejaVu Serif",
            ],

            # Garamond reads slightly smaller/lighter, so use modestly larger sizes.
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,

            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",

            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,

            "grid.color": GRID_COLOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,

            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def _finish_axes(ax):
    """
    Excel-style cleanup:
    - horizontal gridlines only
    - no top/right spines
    - clean black left/bottom axes
    """
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


# ============================================================
# 3. DATA LOADING + PREP
# ============================================================

def load_p2p_linked_master(root: Path) -> pd.DataFrame:
    input_file = root / "data" / "clean" / "p2p_linked_master.xlsx"

    if not input_file.exists():
        csv_fallback = input_file.with_suffix(".csv")
        parquet_fallback = input_file.with_suffix(".parquet")

        if csv_fallback.exists():
            input_file = csv_fallback
        elif parquet_fallback.exists():
            input_file = parquet_fallback
        else:
            print(f"❌ ERROR: File not found at {input_file}")
            return pd.DataFrame()

    try:
        if input_file.suffix.lower() == ".xlsx":
            df = pd.read_excel(input_file, engine="openpyxl")
        elif input_file.suffix.lower() == ".csv":
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_file)
        else:
            print(f"❌ ERROR: Unsupported file type: {input_file}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return pd.DataFrame()

    print(f"✅ Loaded: {input_file}")
    return df


def prepare_holding_period_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["deal_date_entry", "deal_date_exit"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df.copy()

    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["deal_date_exit"] = pd.to_datetime(df["deal_date_exit"], errors="coerce")

    exited_df = df.dropna(subset=["deal_date_entry", "deal_date_exit"]).copy()

    exited_df["hold_period_years"] = (
        exited_df["deal_date_exit"] - exited_df["deal_date_entry"]
    ).dt.days / 365.25

    exited_df = exited_df[
        (exited_df["hold_period_years"] > 0)
        & (exited_df["hold_period_years"] < 10)
    ].copy()

    exited_df["exit_year"] = exited_df["deal_date_exit"].dt.year

    return exited_df


def build_yearly_stats(exited_df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    yearly_stats = (
        exited_df
        .groupby("exit_year")["hold_period_years"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )

    yearly_stats = yearly_stats[yearly_stats["count"] >= min_count].copy()

    if yearly_stats.empty:
        return yearly_stats

    yearly_stats["std"] = yearly_stats["std"].fillna(0)
    yearly_stats["se"] = yearly_stats["std"] / np.sqrt(yearly_stats["count"])
    yearly_stats["ci95"] = 1.96 * yearly_stats["se"]

    return yearly_stats


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_holding_period_band(
    stats_df: pd.DataFrame,
    band_type: str,
    title_suffix: str,
    filename: str,
    figure_dir: Path,
    line_color: str = EXCEL_BLUE,
    band_color: str = EXCEL_BLUE_LIGHT,
):
    """
    band_type options:
        - "std"   : mean ± standard deviation
        - "se"    : mean ± standard error
        - "95ci"  : mean ± 1.96 * standard error
    """
    if stats_df.empty:
        print(f"⚠️ Skipping {filename}: no yearly stats available.")
        return

    _apply_house_style()

    fig, ax = plt.subplots(figsize=(16, 9))

    x = stats_df.index.to_numpy()
    mean = stats_df["mean"].to_numpy()

    if band_type == "std":
        lower = mean - stats_df["std"].to_numpy()
        upper = mean + stats_df["std"].to_numpy()
        band_label = "±1 Standard Deviation"
    elif band_type == "se":
        lower = mean - stats_df["se"].to_numpy()
        upper = mean + stats_df["se"].to_numpy()
        band_label = "±1 Standard Error"
    elif band_type == "95ci":
        lower = mean - stats_df["ci95"].to_numpy()
        upper = mean + stats_df["ci95"].to_numpy()
        band_label = "95% Confidence Interval"
    else:
        raise ValueError("band_type must be one of: 'std', 'se', '95ci'")

    # Prevent shaded band from going below zero.
    lower = np.maximum(lower, 0)

    _finish_axes(ax)

    # Shaded band
    ax.fill_between(
        x,
        lower,
        upper,
        color=band_color,
        alpha=0.35,
        label=band_label,
        zorder=5,
    )

    # Mean line
    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=3,
        markersize=8,
        color=line_color,
        label="Average Holding Period",
        zorder=10,
    )

    # Data labels
    for year, value in zip(x, mean):
        ax.text(
            year,
            value + 0.12,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=line_color,
            zorder=20,
        )

    ax.set_title(
        f"Average Holding Period by Exit Year ({title_suffix})",
        fontsize=19,
        fontweight="bold",
        pad=30,
    )

    ax.set_xlabel(
        "Exit Year",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )

    ax.set_ylabel(
        "Average Years Held",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )

    if len(x) > 1:
        ax.set_xticks(x[::2])
    else:
        ax.set_xticks(x)

    # Give labels and band a little breathing room.
    y_max = max(float(np.nanmax(upper)), float(np.nanmax(mean))) * 1.12
    ax.set_ylim(0, y_max)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )
    leg.set_zorder(100)

    plt.tight_layout()

    save_path = figure_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    plt.close()


# ============================================================
# 5. MAIN
# ============================================================

def make_avg_holding_period_trend():
    ROOT = find_project_root()

    FIGURE_DIR = ROOT / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_p2p_linked_master(ROOT)

    if df.empty:
        print("❌ No data loaded. Stopping.")
        return

    exited_df = prepare_holding_period_data(df)

    if exited_df.empty:
        print("❌ No valid exited deals after filtering. Stopping.")
        return

    yearly_stats = build_yearly_stats(exited_df, min_count=2)

    if yearly_stats.empty:
        print("❌ No exit years meet the minimum count threshold.")
        return

    print("\n========== HOLDING PERIOD TREND CHECK ==========")
    print(f"Exited deals used:       {len(exited_df):,}")
    print(f"Exit years plotted:      {len(yearly_stats):,}")
    print(f"Median holding period:   {exited_df['hold_period_years'].median():.2f} years")
    print(f"Mean holding period:     {exited_df['hold_period_years'].mean():.2f} years")
    print("================================================\n")

    plot_holding_period_band(
        yearly_stats,
        band_type="std",
        title_suffix="±1 Standard Deviation",
        filename="p2p_avg_holding_period_trend_std.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    plot_holding_period_band(
        yearly_stats,
        band_type="se",
        title_suffix="±1 Standard Error",
        filename="p2p_avg_holding_period_trend_se.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    plot_holding_period_band(
        yearly_stats,
        band_type="95ci",
        title_suffix="95% Confidence Interval",
        filename="p2p_avg_holding_period_trend_95ci.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    print("✅ Done: created STD, SE, and 95% CI versions.")


if __name__ == "__main__":
    make_avg_holding_period_trend()