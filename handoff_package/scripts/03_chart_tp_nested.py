"""
03_chart_tp_nested.py
=====================

Produces TP_Nested.png: total vs. realized (exited) counts and entry value
for the Take-Private (P2P) universe, by entry-year vintage.

Reads:
    data/raw/pitchbook_public_2_private_all.xlsx   (take-private entries)
    data/clean/p2p_linked_master.csv               (entry->exit pairs; see 02_link_p2p_entry_exit.py)

Writes:
    output/figures/TP_Nested.png
    output/figures/TP_Summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YEAR_MIN: Final[int] = 2000
YEAR_MAX: Final[int] = 2025

EXCEL_BLUE: Final[str] = "#4472C4"
EXCEL_RED: Final[str] = "#C00000"

REQUIRED_ENTRY_COLUMNS: Final[set[str]] = {
    "company_name_clean", "entry_date", "deal_size_usd_m_entry",
}
REQUIRED_PAIR_COLUMNS: Final[set[str]] = {
    "company_name_clean", "entry_date", "exit_date",
}


# ============================================================
# Loading + column standardization
# ============================================================

def clean_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype("string").str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Exit Date": "exit_date", "Exit_Date": "exit_date",
        "deal_date_exit": "exit_date",
        "Entry Date": "entry_date", "Entry_Date": "entry_date",
        "deal_date_entry": "entry_date", "Deal Date": "entry_date",
        "deal_date": "entry_date",
        "Deal Size": "deal_size_usd_m_entry",
        "deal_size_usd_m": "deal_size_usd_m_entry",
        "V_entry": "deal_size_usd_m_entry",
        "Companies": "company_name_clean", "Company": "company_name_clean",
        "Company Name": "company_name_clean",
    }
    applicable = {s: t for s, t in rename_map.items() if s in df.columns and t not in df.columns}
    return df.rename(columns=applicable).copy()


def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "entry_date" in df.columns:
        df["entry_date"] = normalize_date(df["entry_date"])
    if "exit_date" in df.columns:
        df["exit_date"] = normalize_date(df["exit_date"])
    if "company_name_clean" in df.columns:
        df["company_name_clean"] = clean_name(df["company_name_clean"])
    if "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    return df


def load_data_flexible(root: Path, folder: str, filename_base: str) -> pd.DataFrame:
    """Load the first available parquet, Excel, or CSV version of a dataset."""
    folder_path = root / "data" / folder

    for extension in (".parquet", ".xlsx", ".csv"):
        path = folder_path / f"{filename_base}{extension}"
        if not path.exists():
            continue

        if extension == ".parquet":
            df = pd.read_parquet(path)
        elif extension == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)

        df = standardize_columns(df)
        df = correct_data_types(df)
        print(f"Loaded: {path}  ({len(df):,} rows)")
        return df

    print(f"WARNING: No data file found for {folder_path / filename_base}")
    return pd.DataFrame()


def require_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{dataset_name} is missing required columns: {sorted(missing)}")


# ============================================================
# Matching + summary
# ============================================================

def prepare_pairs_for_merge(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest exit after entry when multiple exits are linked."""
    if pairs.empty:
        return pd.DataFrame(columns=["company_name_clean", "entry_date", "exit_date"])

    require_columns(pairs, REQUIRED_PAIR_COLUMNS, "Entry-exit pair dataset")

    pairs = pairs.copy()
    pairs = pairs.dropna(subset=["company_name_clean", "entry_date", "exit_date"])
    pairs = pairs.loc[pairs["company_name_clean"].ne("")]
    pairs = pairs.loc[pairs["exit_date"] > pairs["entry_date"]]

    return (
        pairs.sort_values(["company_name_clean", "entry_date", "exit_date"])
        .drop_duplicates(subset=["company_name_clean", "entry_date"], keep="first")
        [["company_name_clean", "entry_date", "exit_date"]]
        .reset_index(drop=True)
    )


def build_summary(entries: pd.DataFrame, pairs: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """Annual entry counts, exited counts, values, and realization rates."""
    require_columns(entries, REQUIRED_ENTRY_COLUMNS, "Entry dataset")

    entries = entries.copy()
    entries["entry_date"] = normalize_date(entries["entry_date"])
    entries["company_name_clean"] = clean_name(entries["company_name_clean"])
    entries["deal_size_usd_m_entry"] = pd.to_numeric(entries["deal_size_usd_m_entry"], errors="coerce")
    entries = entries.dropna(subset=["entry_date"])

    prepared_pairs = prepare_pairs_for_merge(pairs)

    if prepared_pairs.empty:
        merged = entries.copy()
        merged["exit_date"] = pd.NaT
    else:
        merged = entries.merge(
            prepared_pairs, on=["company_name_clean", "entry_date"],
            how="left", validate="many_to_one",
        )

    merged["has_exit"] = (
        merged["exit_date"].notna() & merged["entry_date"].notna()
        & (merged["exit_date"] > merged["entry_date"])
    )
    merged["entry_year"] = merged["entry_date"].dt.year
    merged = merged.loc[merged["entry_year"].between(year_min, year_max, inclusive="both")].copy()

    summary = (
        merged.groupby("entry_year", as_index=False)
        .agg(
            N_total=("company_name_clean", "size"),
            N_exited=("has_exit", "sum"),
            V_total=("deal_size_usd_m_entry", lambda s: s.sum(min_count=1)),
        )
        .sort_values("entry_year").reset_index(drop=True)
    )

    exited_value = (
        merged.loc[merged["has_exit"]]
        .groupby("entry_year")["deal_size_usd_m_entry"]
        .sum(min_count=1).rename("V_exited").reset_index()
    )
    summary = summary.merge(exited_value, on="entry_year", how="left")
    summary["V_exited"] = summary["V_exited"].fillna(0.0)
    summary["N_total"] = summary["N_total"].astype("int64")
    summary["N_exited"] = summary["N_exited"].astype("int64")

    summary["pct_N"] = np.where(summary["N_total"] > 0, (summary["N_exited"] / summary["N_total"]) * 100, 0.0)
    summary["pct_V"] = np.where(
        summary["V_total"].notna() & summary["V_total"].ne(0),
        (summary["V_exited"] / summary["V_total"]) * 100, 0.0,
    )

    return summary[["entry_year", "N_total", "N_exited", "V_total", "V_exited", "pct_N", "pct_V"]]


# ============================================================
# Chart styling + rendering
# ============================================================

def apply_house_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Garamond", "Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.titlesize": 16, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.title_fontsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 1.0,
        "grid.color": "#EFEFEF", "grid.linestyle": "-", "grid.linewidth": 0.8,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })


def finish_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_nested_labels(ax: plt.Axes, filled_bars, totals: pd.Series) -> None:
    for bar, total in zip(filled_bars, totals):
        filled_height = float(bar.get_height())
        total_height = float(total) if pd.notna(total) else np.nan

        if not np.isfinite(filled_height) or not np.isfinite(total_height) or filled_height <= 0 or total_height <= 0:
            continue

        percentage = (filled_height / total_height) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            filled_height + max(total_height * 0.01, 0.01),
            f"{int(round(percentage))}\n%",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="black", zorder=20,
        )


def plot_nested_style(summary: pd.DataFrame, title: str, output_file: Path, scale_to_billions: bool) -> None:
    """Total and realized counts and values as nested bars."""
    apply_house_style()

    years = summary["entry_year"].astype(int).tolist()
    x = np.arange(len(summary))
    width = 0.42

    scale = 1000.0 if scale_to_billions else 1.0
    unit_label = "($B)" if scale_to_billions else "($M)"

    fig, count_axis = plt.subplots(figsize=(16, 9))
    value_axis = count_axis.twinx()

    finish_axes(count_axis)
    value_axis.grid(False)
    value_axis.spines["top"].set_visible(False)

    count_axis.bar(x - width / 2, summary["N_total"], width, edgecolor=EXCEL_BLUE,
                   color="none", linewidth=1.6, zorder=10, label="Total Entries (N)")
    exited_count_bars = count_axis.bar(x - width / 2, summary["N_exited"], width,
                                        color=EXCEL_BLUE, alpha=0.65, zorder=11, label="Exited (N)")

    scaled_total_value = summary["V_total"] / scale
    scaled_exited_value = summary["V_exited"] / scale

    value_axis.bar(x + width / 2, scaled_total_value, width, edgecolor=EXCEL_RED,
                    color="none", linewidth=1.6, zorder=10, label=f"Total Value {unit_label}")
    exited_value_bars = value_axis.bar(x + width / 2, scaled_exited_value, width,
                                        color=EXCEL_RED, alpha=0.55, zorder=11, label=f"Realized {unit_label}")

    add_nested_labels(count_axis, exited_count_bars, summary["N_total"])
    add_nested_labels(value_axis, exited_value_bars, scaled_total_value)

    count_axis.set_xlabel("Entry Year (Vintage)", fontweight="bold")
    count_axis.set_ylabel("Number of Entries", fontweight="bold")
    value_axis.set_ylabel(f"Entry Value {unit_label}", fontweight="bold")
    count_axis.set_xticks(x)
    count_axis.set_xticklabels(years, rotation=90)

    handles_1, labels_1 = count_axis.get_legend_handles_labels()
    handles_2, labels_2 = value_axis.get_legend_handles_labels()
    count_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right",
                       bbox_to_anchor=(1.0, 1.12), ncol=2, facecolor="white",
                       framealpha=1.0, edgecolor="#DDDDDD")

    count_axis.set_title(title, fontweight="bold", pad=75)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    take_private_entries = load_data_flexible(root, "raw", "pitchbook_public_2_private_all")
    take_private_pairs = load_data_flexible(root, "clean", "p2p_linked_master")

    require_columns(take_private_entries, REQUIRED_ENTRY_COLUMNS, "Take-private entry dataset")

    take_private_summary = build_summary(take_private_entries, take_private_pairs, YEAR_MIN, YEAR_MAX)

    summary_path = figure_dir / "TP_Summary.csv"
    take_private_summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    nested_path = figure_dir / "TP_Nested.png"
    plot_nested_style(
        take_private_summary,
        "Take-Private (P2P): Volume & Realization",
        nested_path,
        scale_to_billions=False,
    )
    print(f"Generated: {nested_path}")


if __name__ == "__main__":
    main()
