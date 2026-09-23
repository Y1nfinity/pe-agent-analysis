import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
import numpy as np


def find_project_root() -> Path:
    """Finds the 'PEAgent' root directory by searching for data/scripts folders."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent
    fallback = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
    return fallback if fallback.exists() else Path(__file__).resolve().parents[2]


def weighted_stats(group):
    """
    Compute weighted holding-period statistics within one exit_year group.
    Weights = deal_size_entry
    """
    x = group["hold_period_years"].to_numpy(dtype=float)
    w = group["deal_size_entry"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]

    if len(x) == 0:
        return pd.Series({
            "weighted_mean": np.nan,
            "weighted_std": np.nan,
            "n_obs": 0,
            "weight_sum": 0.0,
            "n_eff": np.nan,
            "weighted_se": np.nan,
            "weighted_ci95": np.nan,
        })

    weighted_mean = np.average(x, weights=w)

    # Weighted population variance/std
    weighted_var = np.average((x - weighted_mean) ** 2, weights=w)
    weighted_std = np.sqrt(weighted_var)

    # Effective sample size for weighted SE / CI
    weight_sum = w.sum()
    weight_sq_sum = np.sum(w ** 2)
    n_eff = (weight_sum ** 2) / weight_sq_sum if weight_sq_sum > 0 else np.nan

    weighted_se = weighted_std / np.sqrt(n_eff) if n_eff > 1 else np.nan
    weighted_ci95 = 1.96 * weighted_se if pd.notna(weighted_se) else np.nan

    return pd.Series({
        "weighted_mean": weighted_mean,
        "weighted_std": weighted_std,
        "n_obs": len(x),
        "weight_sum": weight_sum,
        "n_eff": n_eff,
        "weighted_se": weighted_se,
        "weighted_ci95": weighted_ci95,
    })


def plot_holding_period_band(stats_df, band_type, title_suffix, filename, figure_dir,
                             line_color="#4472C4", band_color="#8FAADC"):
    """
    band_type options:
        - 'std'   : weighted mean ± weighted standard deviation
        - 'se'    : weighted mean ± weighted standard error
        - '95ci'  : weighted mean ± 1.96 * weighted standard error
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x = stats_df.index
    mean = stats_df["weighted_mean"]

    if band_type == "std":
        lower = mean - stats_df["weighted_std"]
        upper = mean + stats_df["weighted_std"]
    elif band_type == "se":
        lower = mean - stats_df["weighted_se"]
        upper = mean + stats_df["weighted_se"]
    elif band_type == "95ci":
        lower = mean - stats_df["weighted_ci95"]
        upper = mean + stats_df["weighted_ci95"]
    else:
        raise ValueError("band_type must be one of: 'std', 'se', '95ci'")

    # Line
    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2.5,
        markersize=6,
        color=line_color,
    )

    # Shaded band
    ax.fill_between(
        x,
        lower,
        upper,
        color=band_color,
        alpha=0.35,
    )

    # Titles and labels
    ax.set_title(f"Value-Weighted Average Holding Period by Exit Year ({title_suffix})", fontweight="bold")
    ax.set_xlabel("Exit Year")
    ax.set_ylabel("Value-Weighted Average Years Held")

    # Grid
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.grid(axis="x", visible=False)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tick formatting
    if len(x) > 1:
        ax.set_xticks(x[::2])
    else:
        ax.set_xticks(x)

    plt.tight_layout()

    save_path = figure_dir / filename
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def make_value_weighted_holding_period_trend():
    # ----------------------------
    # Excel-style visual settings
    # ----------------------------
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 11
    rcParams["axes.titlesize"] = 13
    rcParams["axes.labelsize"] = 11

    # Excel default colors
    EXCEL_BLUE = "#4472C4"
    EXCEL_BLUE_LIGHT = "#8FAADC"

    # 1) Paths
    ROOT = find_project_root()
    xlsx_file = ROOT / "data" / "clean" / "p2p_linked_master_updated.xlsx"
    csv_file = ROOT / "data" / "clean" / "p2p_linked_master_updated.csv"
    INPUT_FILE = xlsx_file if xlsx_file.exists() else csv_file

    FIGURE_DIR = ROOT / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found in {ROOT / 'data' / 'clean'}")
        return

    # 2) Load data
    if INPUT_FILE.suffix.lower() == ".xlsx":
        df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    else:
        df = pd.read_csv(INPUT_FILE)

    # 3) Holding period calculation
    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["deal_date_exit"] = pd.to_datetime(df["deal_date_exit"], errors="coerce")
    df["deal_size_entry"] = pd.to_numeric(df["deal_size_entry"], errors="coerce")

    exited_df = df.dropna(subset=["deal_date_entry", "deal_date_exit", "deal_size_entry"]).copy()

    exited_df["hold_period_years"] = (
        (exited_df["deal_date_exit"] - exited_df["deal_date_entry"]).dt.days / 365.25
    )

    exited_df = exited_df[
        (exited_df["hold_period_years"] > 0) &
        (exited_df["hold_period_years"] < 10) &
        (exited_df["deal_size_entry"] > 0)
    ].copy()

    exited_df["exit_year"] = exited_df["deal_date_exit"].dt.year

    # 4) Aggregate with value weights
    yearly_stats = (
        exited_df.groupby("exit_year")
        .apply(weighted_stats)
        .sort_index()
    )

    # Optional minimum threshold:
    # keep years with at least 2 observations and positive aggregate weight
    yearly_stats = yearly_stats[
        (yearly_stats["n_obs"] >= 2) &
        (yearly_stats["weight_sum"] > 0)
    ]

    if yearly_stats.empty:
        print("No exit years meet the minimum threshold after weighting.")
        return

    # Fill missing dispersion values if needed
    yearly_stats["weighted_std"] = yearly_stats["weighted_std"].fillna(0)

    # 5) Create all three plots
    plot_holding_period_band(
        yearly_stats,
        band_type="std",
        title_suffix="±1 Weighted Standard Deviation",
        filename="p2p_value_weighted_holding_period_trend_std.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    plot_holding_period_band(
        yearly_stats,
        band_type="se",
        title_suffix="±1 Weighted Standard Error",
        filename="p2p_value_weighted_holding_period_trend_se.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    plot_holding_period_band(
        yearly_stats,
        band_type="95ci",
        title_suffix="95% Weighted Confidence Interval",
        filename="p2p_value_weighted_holding_period_trend_95ci.png",
        figure_dir=FIGURE_DIR,
        line_color=EXCEL_BLUE,
        band_color=EXCEL_BLUE_LIGHT,
    )

    print("Done: created value-weighted STD, SE, and 95% CI versions.")


if __name__ == "__main__":
    make_value_weighted_holding_period_trend()