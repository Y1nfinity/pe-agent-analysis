"""
deal_size_analysis.py
=====================

High-quality visualizations for exit deal sizes:

    1) Log distribution of exit deal sizes
    2) Deal size by exit type (violin + boxplots)
    3) Deal size vs. holding periods
    4) Pareto concentration curve of exit value

Primary column:
    - deal_size_usd_m_exit (with robust fallback detection)

Outputs:
    - PNG figures saved under outputs/deal_size_analysis
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------

# Hybrid aesthetic: white background, soft grids, fixed palette
PALETTE = [
    "#1f4e79",  # navy
    "#2f855a",  # green
    "#d97706",  # amber
    "#b91c1c",  # red
    "#4b5563",  # steel/gray blue
    "#7c3aed",  # violet
]

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.9,
)

sns.set_palette(PALETTE)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "deal_size_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_pairs(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load entry–exit pairs from parquet.

    Parameters
    ----------
    path : Path, optional
        Override default path if needed.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = DATA / "entry_exit_pairs.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    print(f"✅ Loaded entry–exit pairs from: {path} (n={len(df):,})")
    return df


# ---------------------------------------------------------------------
# Robust deal-size column detection
# ---------------------------------------------------------------------


def get_deal_size_column(df: pd.DataFrame) -> str:
    """
    Detect the exit deal-size column in a robust way.

    Priority:
        1. deal_size_usd_m_exit
        2. exit_size_usd_m (legacy)
        3. post_valuation_usd_m_exit

    Fallback:
        - Any numeric column whose name contains "size" but NOT:
            - "entry"
            - "status"

    Returns
    -------
    str
        Name of the chosen deal-size column.

    Raises
    ------
    KeyError
        If no usable column is found.
    """
    preferred = [
        "deal_size_usd_m_exit",
        "exit_size_usd_m",
        "post_valuation_usd_m_exit",
    ]

    for col in preferred:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().sum() > 0:
                print(f"✅ Using deal size column: {col}")
                return col

    numeric_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if "size" in col_lower and "entry" not in col_lower and "status" not in col_lower:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0:
                numeric_candidates.append(col)

    if numeric_candidates:
        best = numeric_candidates[0]
        df[best] = pd.to_numeric(df[best], errors="coerce")
        print(f"⚠️ Using fallback numeric deal size column: {best}")
        return best

    raise KeyError(
        "❌ No usable exit deal size column found.\n"
        "Expected one of:\n"
        "  - deal_size_usd_m_exit\n"
        "  - exit_size_usd_m\n"
        "  - post_valuation_usd_m_exit\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ---------------------------------------------------------------------
# Helper formatters & utilities
# ---------------------------------------------------------------------


def _thousands_formatter(x, pos):
    """Format large integer counts with thousands separators."""
    return f"{int(x):,}"


def _millions_formatter(x, pos):
    """Format USD millions with thousands separators, no decimals."""
    return f"{x:,.0f}"


def _filter_positive_sizes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Keep only rows with strictly positive, non-null deal sizes.

    Returns a copy to avoid SettingWithCopy warnings.
    """
    mask = df[col].notna() & (df[col] > 0)
    out = df.loc[mask].copy()
    print(f"   ↪ Filtered to {len(out):,} rows with positive {col}.")
    return out


def _add_log_size(df: pd.DataFrame, col: str, log_col: str = "log_size") -> pd.DataFrame:
    """
    Add a base-10 log column of deal sizes.

    Mutates df in place and returns it (for chaining).
    """
    df[log_col] = np.log10(df[col])
    return df


# ---------------------------------------------------------------------
# 1. Log distribution of deal sizes
# ---------------------------------------------------------------------


def plot_log_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Histogram + KDE of log10 exit deal sizes, with a secondary axis
    showing the original scale in USD millions.
    """
    col = get_deal_size_column(df)
    df = _filter_positive_sizes(df, col)
    if df.empty:
        print("⚠️ No positive deal sizes available for log distribution.")
        return

    df = _add_log_size(df, col)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["log_size"], bins=40, kde=True)
    plt.title("Distribution of Exit Deal Sizes (Log Scale)")
    plt.xlabel("Deal Size (USD Millions, log10)")
    plt.ylabel("Number of Exits")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_thousands_formatter))

    ax = plt.gca()
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: 10 ** x, lambda v: np.log10(v)),
    )
    secax.set_xlabel("Deal Size (USD Millions)")
    secax.xaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    plt.tight_layout()
    out_path = out_dir / "1_Log_Distribution_Deal_Size.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved log distribution plot → {out_path}")


# ---------------------------------------------------------------------
# 2. Violin + Boxplot by exit type
# ---------------------------------------------------------------------


def plot_deal_size_by_exit(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Violin and box plots of log10 exit deal sizes by exit_type.

    Requires:
        - exit_type column in df.
    """
    col = get_deal_size_column(df)
    df = _filter_positive_sizes(df, col)

    if "exit_type" not in df.columns:
        print("⚠️ Column 'exit_type' missing; skipping by-exit-type plots.")
        return

    df = df[df["exit_type"].notna()].copy()
    if df.empty:
        print("⚠️ No rows with non-null exit_type; skipping by-exit-type plots.")
        return

    df = _add_log_size(df, col)

    # Order exit types by median log size for readability
    order = (
        df.groupby("exit_type")["log_size"]
          .median()
          .sort_values()
          .index
    )

    # ----- Violin plot -----
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="exit_type",
        y="log_size",
        order=order,
        inner="quartile",
        cut=0,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("Exit Deal Sizes by Exit Type (Violin, Log Scale)")
    plt.ylabel("Deal Size (USD Millions, log10)")
    plt.xlabel("Exit Type")

    ax = plt.gca()
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda y: 10 ** y, lambda v: np.log10(v)),
    )
    secax.set_ylabel("Deal Size (USD Millions)")
    secax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    plt.tight_layout()
    out_path = out_dir / "2_Deal_Size_By_Exit_Type_Violin.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved violin plot by exit type → {out_path}")

    # ----- Boxplot -----
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="exit_type",
        y="log_size",
        order=order,
        fliersize=2,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("Exit Deal Sizes by Exit Type (Boxplot, Log Scale)")
    plt.ylabel("Deal Size (USD Millions, log10)")
    plt.xlabel("Exit Type")

    ax = plt.gca()
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda y: 10 ** y, lambda v: np.log10(v)),
    )
    secax.set_ylabel("Deal Size (USD Millions)")
    secax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    plt.tight_layout()
    out_path = out_dir / "3_Deal_Size_By_Exit_Type_Box.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved boxplot by exit type → {out_path}")


# ---------------------------------------------------------------------
# 3. Deal size vs holding period scatter
# ---------------------------------------------------------------------


def plot_size_vs_holding(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plot of log10 exit deal size vs holding_period_years.

    If 'exit_type' exists, it is used as the hue.
    """
    col = get_deal_size_column(df)
    df = _filter_positive_sizes(df, col)

    if "holding_period_years" not in df.columns:
        print("⚠️ Column 'holding_period_years' missing; skipping size vs. holding plot.")
        return

    df = df[
        df["holding_period_years"].notna()
        & (df["holding_period_years"] >= 0)
    ].copy()

    if df.empty:
        print("⚠️ No valid (deal size, holding period) pairs; skipping scatter.")
        return

    df = _add_log_size(df, col)

    hue = "exit_type" if "exit_type" in df.columns else None

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="holding_period_years",
        y="log_size",
        hue=hue,
        alpha=0.6,
        s=30,
    )
    plt.xlabel("Holding Period (Years)")
    plt.ylabel("Deal Size (USD Millions, log10)")
    plt.title("Deal Size vs Holding Period")

    ax = plt.gca()
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda y: 10 ** y, lambda v: np.log10(v)),
    )
    secax.set_ylabel("Deal Size (USD Millions)")
    secax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    # Move legend outside if present
    if ax.get_legend() is not None:
        plt.legend(
            title="Exit Type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    plt.tight_layout()
    out_path = out_dir / "4_Deal_Size_vs_Holding_Period.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved deal size vs. holding period scatter → {out_path}")


# ---------------------------------------------------------------------
# 4. Pareto concentration curve
# ---------------------------------------------------------------------


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Pareto concentration curve: cumulative share of total exit value
    as a function of the fraction of deals (sorted largest to smallest).
    """
    col = get_deal_size_column(df)
    df = _filter_positive_sizes(df, col)

    if df.empty:
        print("⚠️ No positive deal sizes; skipping Pareto curve.")
        return

    sizes = df[col].sort_values(ascending=False).values
    total = sizes.sum()
    if total <= 0:
        print("⚠️ Non-positive total exit value; skipping Pareto curve.")
        return

    cum_share = np.cumsum(sizes) / total
    n = len(sizes)
    frac = np.arange(1, n + 1) / n

    plt.figure(figsize=(9, 6))
    plt.plot(frac, cum_share, linewidth=2)
    plt.title("Pareto Concentration of Exit Deal Value")
    plt.xlabel("Fraction of Deals (sorted largest to smallest)")
    plt.ylabel("Cumulative Share of Total Exit Value")
    plt.grid(alpha=0.3)

    # 80/20 marker
    idx_80 = np.searchsorted(cum_share, 0.8)
    if idx_80 < len(frac):
        x80 = frac[idx_80]
        y80 = cum_share[idx_80]
        plt.axhline(0.8, linestyle="--", linewidth=1)
        plt.axvline(x80, linestyle="--", linewidth=1)
        plt.text(
            x80,
            0.82,
            f"~{x80 * 100:.1f}% of deals → 80% of value",
            ha="right",
            va="bottom",
        )

    plt.tight_layout()
    out_path = out_dir / "5_Pareto_Deal_Value.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved Pareto concentration plot → {out_path}")


# ---------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------


def run_all() -> None:
    df = load_pairs()

    print("🔹 Running Deal Size Analysis...")
    plot_log_distribution(df, OUT)
    plot_deal_size_by_exit(df, OUT)
    plot_size_vs_holding(df, OUT)
    plot_pareto(df, OUT)
    print(f"✅ All deal-size figures saved to: {OUT}")


if __name__ == "__main__":
    run_all()
