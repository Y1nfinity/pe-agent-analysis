# tools/exit_quality.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Quality mapping ----------
# We map from the baseline ExitTypeLabel (you already have: IPO, Sale to strategic, Sale to another sponsor, Sale to continuation fund, Other/Unknown)
QUALITY_MAP = {
    "IPO": "High",
    "Sale to strategic": "Low",
    "Sale to another sponsor": "Low",
    "Sale to continuation fund": "Poor",
}
# All other labels fall into "Other"
DEFAULT_QUALITY = "Other"

def add_exit_quality(df: pd.DataFrame, exit_type_col: str = "ExitTypeLabel") -> pd.DataFrame:
    out = df.copy()
    out["ExitQuality"] = out[exit_type_col].map(QUALITY_MAP).fillna(DEFAULT_QUALITY)
    return out

# ---------- Aggregations ----------
def _agg_by_year_category(
    df: pd.DataFrame,
    year_col: str,
    cat_col: str,
    size_col: str = "Entry_DealSize",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (counts_by_year, size_by_year).
    counts: rows = (year, category), col Count
    size:   rows = (year, category), col TotalSize (sum of Entry_DealSize)
    """
    counts = df.groupby([year_col, cat_col]).size().reset_index(name="Count")
    size = df.groupby([year_col, cat_col])[size_col].sum(min_count=1).reset_index(name="TotalSize")
    return counts, size

def _to_fraction(df: pd.DataFrame, year_col: str, value_col: str, cat_col: str) -> pd.DataFrame:
    """Add Fraction column per year (value / sum(value) in that year)."""
    out = df.copy()
    totals = out.groupby(year_col)[value_col].transform("sum")
    out["Fraction"] = np.where(totals.gt(0), out[value_col] / totals, 0.0)
    return out

# ---------- Plotters ----------
def _stacked_fraction_bars(
    df_frac: pd.DataFrame, year_col: str, cat_col: str, value_col: str, out_path: Path, title: str, y_label: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_frac.pivot_table(index=year_col, columns=cat_col, values=value_col, aggfunc="sum", fill_value=0.0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11,5))
    ax.set_xlabel("Vintage Year"); ax.set_ylabel(y_label); ax.set_title(title)
    ax.legend(title=cat_col, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def _grouped_absolute_bars(
    df_abs: pd.DataFrame, year_col: str, cat_col: str, value_col: str, out_path: Path, title: str, y_label: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_abs.pivot_table(index=year_col, columns=cat_col, values=value_col, aggfunc="sum", fill_value=0).sort_index()
    years = pivot.index.to_numpy()
    labels = list(pivot.columns)
    n = len(labels)
    x = np.arange(len(years))
    width = 0.75 / max(1, n)

    fig, ax = plt.subplots(figsize=(11,5))
    for i, lab in enumerate(labels):
        ax.bar(x + i*width - (n-1)*width/2, pivot[lab].to_numpy(), width, label=lab)
    ax.set_xticks(x, years)
    ax.set_xlabel("Vintage Year"); ax.set_ylabel(y_label); ax.set_title(title)
    ax.legend(title=cat_col, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

# ---------- Public API: build tables + charts ----------
def build_and_plot_quality_panels(
    pairs_p2p: pd.DataFrame,
    out_tables: Path,
    out_plots: Path,
    year_col: str = "Vintage",         # you set Vintage = EntryYear earlier
    exit_type_col: str = "ExitTypeLabel",
    size_col: str = "Entry_DealSize",
    exclude_missing_size: bool = False,
    label_prefix: str = "p2p_exit_quality",
) -> dict:
    """
    Creates tables & charts for ExitQuality and baseline ExitType.
    Returns dict of created file paths by key.
    """
    out = {}
    df = pairs_p2p.copy()

    # Optionally filter out missing size (for size-weighted views)
    if exclude_missing_size:
        df = df[df[size_col].notna() & (df[size_col] > 0)].copy()

    # --- A) Exit Quality buckets ---
    df_q = add_exit_quality(df, exit_type_col=exit_type_col)

    # counts & size aggregates
    q_counts, q_size = _agg_by_year_category(df_q, year_col, "ExitQuality", size_col=size_col)
    q_counts_frac = _to_fraction(q_counts, year_col, "Count", "ExitQuality")
    q_size_frac   = _to_fraction(q_size,   year_col, "TotalSize", "ExitQuality")

    # save tables
    p_counts_csv = out_tables / f"{label_prefix}_counts.csv"
    p_size_csv   = out_tables / f"{label_prefix}_size.csv"
    q_counts.to_csv(p_counts_csv, index=False); out["quality_counts_csv"] = p_counts_csv
    q_size.to_csv(p_size_csv,     index=False); out["quality_size_csv"]   = p_size_csv

    # plots – stacked fractions
    _stacked_fraction_bars(
        q_size_frac, year_col, "ExitQuality", "Fraction",
        out_plots / f"{label_prefix}_fraction_by_size.png",
        "Fraction of Entry Spend by Exit Quality", "Fraction of Total Entry Spend"
    ); out["quality_fraction_by_size_png"] = out_plots / f"{label_prefix}_fraction_by_size.png"

    _stacked_fraction_bars(
        q_counts_frac, year_col, "ExitQuality", "Fraction",
        out_plots / f"{label_prefix}_fraction_by_count.png",
        "Fraction of Companies by Exit Quality", "Fraction of Companies"
    ); out["quality_fraction_by_count_png"] = out_plots / f"{label_prefix}_fraction_by_count.png"

    # plots – grouped absolutes
    _grouped_absolute_bars(
        q_size, year_col, "ExitQuality", "TotalSize",
        out_plots / f"{label_prefix}_abs_size.png",
        "Total Entry Spend by Exit Quality", "Entry Spend (USD)"
    ); out["quality_abs_size_png"] = out_plots / f"{label_prefix}_abs_size.png"

    _grouped_absolute_bars(
        q_counts, year_col, "ExitQuality", "Count",
        out_plots / f"{label_prefix}_abs_count.png",
        "Company Count by Exit Quality", "Companies (count)"
    ); out["quality_abs_count_png"] = out_plots / f"{label_prefix}_abs_count.png"

    # --- B) Baseline exit types (original labels you already have in ExitTypeLabel) ---
    t_counts, t_size = _agg_by_year_category(df, year_col, exit_type_col, size_col=size_col)
    t_counts_frac = _to_fraction(t_counts, year_col, "Count", exit_type_col)
    t_size_frac   = _to_fraction(t_size,   year_col, "TotalSize", exit_type_col)

    # save tables
    p_t_counts_csv = out_tables / f"{label_prefix}_types_counts.csv"
    p_t_size_csv   = out_tables / f"{label_prefix}_types_size.csv"
    t_counts.to_csv(p_t_counts_csv, index=False); out["types_counts_csv"] = p_t_counts_csv
    t_size.to_csv(p_t_size_csv,     index=False); out["types_size_csv"]   = p_t_size_csv

    # plots – stacked fractions
    _stacked_fraction_bars(
        t_size_frac, year_col, exit_type_col, "Fraction",
        out_plots / f"{label_prefix}_types_fraction_by_size.png",
        "Fraction of Entry Spend by Exit Type", "Fraction of Total Entry Spend"
    ); out["types_fraction_by_size_png"] = out_plots / f"{label_prefix}_types_fraction_by_size.png"

    _stacked_fraction_bars(
        t_counts_frac, year_col, exit_type_col, "Fraction",
        out_plots / f"{label_prefix}_types_fraction_by_count.png",
        "Fraction of Companies by Exit Type", "Fraction of Companies"
    ); out["types_fraction_by_count_png"] = out_plots / f"{label_prefix}_types_fraction_by_count.png"

    # plots – grouped absolutes
    _grouped_absolute_bars(
        t_size, year_col, exit_type_col, "TotalSize",
        out_plots / f"{label_prefix}_types_abs_size.png",
        "Total Entry Spend by Exit Type", "Entry Spend (USD)"
    ); out["types_abs_size_png"] = out_plots / f"{label_prefix}_types_abs_size.png"

    _grouped_absolute_bars(
        t_counts, year_col, exit_type_col, "Count",
        out_plots / f"{label_prefix}_types_abs_count.png",
        "Company Count by Exit Type", "Companies (count)"
    ); out["types_abs_count_png"] = out_plots / f"{label_prefix}_types_abs_count.png"

    return out
