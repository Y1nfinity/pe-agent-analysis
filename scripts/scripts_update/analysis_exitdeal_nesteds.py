from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

YEAR_MIN: Final[int] = 2000
YEAR_MAX: Final[int] = 2025

EXCEL_BLUE: Final[str] = "#4472C4"
EXCEL_RED: Final[str] = "#C00000"

REQUIRED_ENTRY_COLUMNS: Final[set[str]] = {
    "company_name_clean",
    "entry_date",
    "deal_size_usd_m_entry",
}

REQUIRED_PAIR_COLUMNS: Final[set[str]] = {
    "company_name_clean",
    "entry_date",
    "exit_date",
}


# ============================================================
# 2. PATHING AND DATA LOADING
# ============================================================


def find_project_root() -> Path:
    """
    Return the PEAgent project root.

    Expected script location:
        PEAgent/scripts/scripts_update/analysis_exitdeal_nesteds.py
    """
    script_path = Path(__file__).resolve()

    try:
        return script_path.parents[2]
    except IndexError as exc:
        raise RuntimeError(
            f"Could not determine the project root from script path: {script_path}"
        ) from exc


def clean_name(series: pd.Series) -> pd.Series:
    """Normalize company names to lowercase alphanumeric strings."""
    return (
        series.fillna("")
        .astype("string")
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def normalize_date(series: pd.Series) -> pd.Series:
    """Convert values to normalized pandas datetimes with invalid values as NaT."""
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def make_entry_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable company/date key.

    This uses vectorized string concatenation instead of row-wise
    ``agg("|".join, axis=1)``, which can fail when a row contains a float or NaN.
    """
    required = {"company_name_clean", "entry_date"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            "Cannot create an entry key because these columns are missing: "
            f"{sorted(missing)}"
        )

    company = clean_name(df["company_name_clean"])
    entry_date = (
        normalize_date(df["entry_date"])
        .dt.strftime("%Y-%m-%d")
        .fillna("")
        .astype("string")
    )

    return company.str.cat(entry_date, sep="|")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common source columns to the names used by this analysis."""
    rename_map = {
        "Exit Date": "exit_date",
        "Exit_Date": "exit_date",
        "deal_date_exit": "exit_date",
        "Entry Date": "entry_date",
        "Entry_Date": "entry_date",
        "deal_date_entry": "entry_date",
        "Deal Date": "entry_date",
        "deal_date": "entry_date",
        "Deal Size": "deal_size_usd_m_entry",
        "deal_size_usd_m": "deal_size_usd_m_entry",
        "V_entry": "deal_size_usd_m_entry",
        "Companies": "company_name_clean",
        "Company": "company_name_clean",
        "Company Name": "company_name_clean",
    }

    applicable_map = {
        source: target
        for source, target in rename_map.items()
        if source in df.columns and target not in df.columns
    }

    return df.rename(columns=applicable_map).copy()


def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the columns used by the matching and aggregation logic."""
    df = df.copy()

    if "entry_date" in df.columns:
        df["entry_date"] = normalize_date(df["entry_date"])

    if "exit_date" in df.columns:
        df["exit_date"] = normalize_date(df["exit_date"])

    if "company_name_clean" in df.columns:
        df["company_name_clean"] = clean_name(df["company_name_clean"])

    if "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(
            df["deal_size_usd_m_entry"], errors="coerce"
        )

    if "entry_year" in df.columns:
        df["entry_year"] = pd.to_numeric(df["entry_year"], errors="coerce")

    return df


def load_data_flexible(
    root: Path,
    folder: str,
    filename_base: str,
) -> pd.DataFrame:
    """
    Load the first available parquet, Excel, or CSV version of a dataset.

    Search order:
        1. .parquet
        2. .xlsx
        3. .csv
    """
    folder_path = root / "data" / folder

    for extension in (".parquet", ".xlsx", ".csv"):
        path = folder_path / f"{filename_base}{extension}"

        if not path.exists():
            continue

        try:
            if extension == ".parquet":
                df = pd.read_parquet(path)
            elif extension == ".xlsx":
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load data file: {path}") from exc

        df = standardize_columns(df)
        df = correct_data_types(df)

        print(f"Loaded: {path}")
        print(f"  Rows: {len(df):,} | Columns: {len(df.columns):,}")
        return df

    print(
        "WARNING: No data file found for "
        f"{folder_path / filename_base} "
        "(.parquet, .xlsx, or .csv)."
    )
    return pd.DataFrame()


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when a loaded dataset lacks required columns."""
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            f"{dataset_name} is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ============================================================
# 3. ENTRY AND EXIT MATCHING
# ============================================================


def derive_non_take_private_entries(
    universe_entries: pd.DataFrame,
    take_private_entries: pd.DataFrame,
) -> pd.DataFrame:
    """Remove take-private entries from the complete PE entry universe."""
    universe_entries = universe_entries.copy()
    universe_entries["__key"] = make_entry_key(universe_entries)

    if take_private_entries.empty:
        print("No take-private entry file was loaded; using the full universe as Non-TP.")
        return universe_entries.drop(columns="__key")

    take_private_entries = take_private_entries.copy()
    take_private_entries["__key"] = make_entry_key(take_private_entries)

    # Empty company/date keys should not be used to exclude universe rows.
    valid_tp_keys = set(
        take_private_entries.loc[
            take_private_entries["company_name_clean"].ne("")
            & take_private_entries["entry_date"].notna(),
            "__key",
        ]
    )

    non_tp_entries = universe_entries.loc[
        ~universe_entries["__key"].isin(valid_tp_keys)
    ].copy()

    excluded_count = len(universe_entries) - len(non_tp_entries)
    print(f"Take-private keys found: {len(valid_tp_keys):,}")
    print(f"Universe rows classified as Take-Private: {excluded_count:,}")
    print(f"Universe rows classified as Non-TP: {len(non_tp_entries):,}")

    return non_tp_entries.drop(columns="__key")


def prepare_pairs_for_merge(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare entry-exit pairs for matching.

    When multiple valid exits exist for the same company and entry date,
    the earliest exit after entry is retained.
    """
    if pairs.empty:
        return pd.DataFrame(columns=["company_name_clean", "entry_date", "exit_date"])

    require_columns(pairs, REQUIRED_PAIR_COLUMNS, "Entry-exit pair dataset")

    pairs = pairs.copy()
    pairs = pairs.dropna(subset=["company_name_clean", "entry_date", "exit_date"])
    pairs = pairs.loc[pairs["company_name_clean"].ne("")]
    pairs = pairs.loc[pairs["exit_date"] > pairs["entry_date"]]

    return (
        pairs.sort_values(
            ["company_name_clean", "entry_date", "exit_date"],
            ascending=[True, True, True],
        )
        .drop_duplicates(
            subset=["company_name_clean", "entry_date"],
            keep="first",
        )
        [["company_name_clean", "entry_date", "exit_date"]]
        .reset_index(drop=True)
    )


# ============================================================
# 4. SUMMARY CALCULATION
# ============================================================


def empty_summary() -> pd.DataFrame:
    """Return an empty dataframe with the expected summary schema."""
    return pd.DataFrame(
        columns=[
            "entry_year",
            "N_total",
            "N_exited",
            "V_total",
            "V_exited",
            "pct_N",
            "pct_V",
        ]
    )


def build_summary(
    entries: pd.DataFrame,
    pairs: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """Calculate annual entry counts, exited counts, values, and realization rates."""
    if entries.empty:
        return empty_summary()

    require_columns(entries, REQUIRED_ENTRY_COLUMNS, "Entry dataset")

    entries = entries.copy()
    entries["entry_date"] = normalize_date(entries["entry_date"])
    entries["company_name_clean"] = clean_name(entries["company_name_clean"])
    entries["deal_size_usd_m_entry"] = pd.to_numeric(
        entries["deal_size_usd_m_entry"], errors="coerce"
    )

    # Rows without a usable entry date cannot be assigned to a vintage.
    entries = entries.dropna(subset=["entry_date"])

    prepared_pairs = prepare_pairs_for_merge(pairs)

    if prepared_pairs.empty:
        merged = entries.copy()
        merged["exit_date"] = pd.NaT
    else:
        merged = entries.merge(
            prepared_pairs,
            on=["company_name_clean", "entry_date"],
            how="left",
            validate="many_to_one",
        )

    merged["has_exit"] = (
        merged["exit_date"].notna()
        & merged["entry_date"].notna()
        & (merged["exit_date"] > merged["entry_date"])
    )

    merged["entry_year"] = merged["entry_date"].dt.year
    merged = merged.loc[
        merged["entry_year"].between(year_min, year_max, inclusive="both")
    ].copy()

    if merged.empty:
        return empty_summary()

    summary = (
        merged.groupby("entry_year", as_index=False)
        .agg(
            N_total=("company_name_clean", "size"),
            N_exited=("has_exit", "sum"),
            V_total=("deal_size_usd_m_entry", lambda s: s.sum(min_count=1)),
        )
        .sort_values("entry_year")
        .reset_index(drop=True)
    )

    exited_value = (
        merged.loc[merged["has_exit"]]
        .groupby("entry_year")["deal_size_usd_m_entry"]
        .sum(min_count=1)
        .rename("V_exited")
        .reset_index()
    )

    summary = summary.merge(exited_value, on="entry_year", how="left")
    summary["V_exited"] = summary["V_exited"].fillna(0.0)

    summary["N_total"] = summary["N_total"].astype("int64")
    summary["N_exited"] = summary["N_exited"].astype("int64")

    summary["pct_N"] = np.where(
        summary["N_total"] > 0,
        (summary["N_exited"] / summary["N_total"]) * 100,
        0.0,
    )

    summary["pct_V"] = np.where(
        summary["V_total"].notna() & summary["V_total"].ne(0),
        (summary["V_exited"] / summary["V_total"]) * 100,
        0.0,
    )

    return summary[
        [
            "entry_year",
            "N_total",
            "N_exited",
            "V_total",
            "V_exited",
            "pct_N",
            "pct_V",
        ]
    ]


def combine_summaries(*summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Combine summaries by entry year.

    This avoids adding dataframes by row index, which can misalign values when
    TP and Non-TP summaries contain different sets of years.
    """
    usable = [
        summary[["entry_year", "N_total", "N_exited", "V_total", "V_exited"]].copy()
        for summary in summaries
        if not summary.empty
    ]

    if not usable:
        return empty_summary()

    combined = pd.concat(usable, ignore_index=True)
    combined = (
        combined.groupby("entry_year", as_index=False)
        .agg(
            N_total=("N_total", "sum"),
            N_exited=("N_exited", "sum"),
            V_total=("V_total", lambda s: s.sum(min_count=1)),
            V_exited=("V_exited", lambda s: s.sum(min_count=1)),
        )
        .sort_values("entry_year")
        .reset_index(drop=True)
    )

    combined["pct_N"] = np.where(
        combined["N_total"] > 0,
        (combined["N_exited"] / combined["N_total"]) * 100,
        0.0,
    )

    combined["pct_V"] = np.where(
        combined["V_total"].notna() & combined["V_total"].ne(0),
        (combined["V_exited"] / combined["V_total"]) * 100,
        0.0,
    )

    return combined


# ============================================================
# 5. CHART STYLING
# ============================================================


def apply_house_style() -> None:
    """Apply a consistent Excel-like chart style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Garamond", "Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.color": "#EFEFEF",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    """Keep horizontal gridlines and remove the top and right borders."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_percentage_labels(ax: plt.Axes, bars) -> None:
    """Add whole-number percentage labels above bars."""
    for bar in bars:
        height = float(bar.get_height())

        if not np.isfinite(height) or height < 0.5:
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{int(round(height))}\n%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=20,
        )


def add_nested_labels(
    ax: plt.Axes,
    filled_bars,
    totals: pd.Series,
) -> None:
    """Label each filled nested bar with its percentage of the outlined total."""
    for bar, total in zip(filled_bars, totals):
        filled_height = float(bar.get_height())
        total_height = float(total) if pd.notna(total) else np.nan

        if (
            not np.isfinite(filled_height)
            or not np.isfinite(total_height)
            or filled_height <= 0
            or total_height <= 0
        ):
            continue

        percentage = (filled_height / total_height) * 100

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            filled_height + max(total_height * 0.01, 0.01),
            f"{int(round(percentage))}\n%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
            zorder=20,
        )


# ============================================================
# 6. VISUALIZATION FUNCTIONS
# ============================================================


def plot_percentage_style(
    summary: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """Plot realization percentages by count and entry value."""
    apply_house_style()

    years = summary["entry_year"].astype(int).tolist()
    x = np.arange(len(summary))
    width = 0.42

    fig, count_axis = plt.subplots(figsize=(16, 9))
    value_axis = count_axis.twinx()

    finish_axes(count_axis)
    value_axis.grid(False)
    value_axis.spines["top"].set_visible(False)

    count_bars = count_axis.bar(
        x - width / 2,
        summary["pct_N"],
        width,
        color=EXCEL_BLUE,
        label="% Exited (by Count)",
        zorder=10,
    )

    value_bars = value_axis.bar(
        x + width / 2,
        summary["pct_V"],
        width,
        color=EXCEL_RED,
        label="% Exited (by Value $M)",
        zorder=10,
    )

    add_percentage_labels(count_axis, count_bars)
    add_percentage_labels(value_axis, value_bars)

    for axis in (count_axis, value_axis):
        axis.set_ylim(0, 105)
        axis.spines["top"].set_visible(False)

    count_axis.set_xlabel("Entry Year (Vintage)", fontweight="bold")
    count_axis.set_ylabel("Exited Entries (%)", fontweight="bold")
    value_axis.set_ylabel("Exited Entry Value (%)", fontweight="bold")

    count_axis.set_xticks(x)
    count_axis.set_xticklabels(years, rotation=90)

    handles_1, labels_1 = count_axis.get_legend_handles_labels()
    handles_2, labels_2 = value_axis.get_legend_handles_labels()
    count_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )

    count_axis.set_title(title, fontweight="bold", pad=75)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nested_style(
    summary: pd.DataFrame,
    title: str,
    output_file: Path,
    scale_to_billions: bool,
) -> None:
    """Plot total and realized counts and values as nested bars."""
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

    count_axis.bar(
        x - width / 2,
        summary["N_total"],
        width,
        edgecolor=EXCEL_BLUE,
        color="none",
        linewidth=1.6,
        zorder=10,
        label="Total Entries (N)",
    )

    exited_count_bars = count_axis.bar(
        x - width / 2,
        summary["N_exited"],
        width,
        color=EXCEL_BLUE,
        alpha=0.65,
        zorder=11,
        label="Exited (N)",
    )

    scaled_total_value = summary["V_total"] / scale
    scaled_exited_value = summary["V_exited"] / scale

    value_axis.bar(
        x + width / 2,
        scaled_total_value,
        width,
        edgecolor=EXCEL_RED,
        color="none",
        linewidth=1.6,
        zorder=10,
        label=f"Total Value {unit_label}",
    )

    exited_value_bars = value_axis.bar(
        x + width / 2,
        scaled_exited_value,
        width,
        color=EXCEL_RED,
        alpha=0.55,
        zorder=11,
        label=f"Realized {unit_label}",
    )

    add_nested_labels(count_axis, exited_count_bars, summary["N_total"])
    add_nested_labels(value_axis, exited_value_bars, scaled_total_value)

    count_axis.set_xlabel("Entry Year (Vintage)", fontweight="bold")
    count_axis.set_ylabel("Number of Entries", fontweight="bold")
    value_axis.set_ylabel(f"Entry Value {unit_label}", fontweight="bold")

    count_axis.set_xticks(x)
    count_axis.set_xticklabels(years, rotation=90)

    handles_1, labels_1 = count_axis.get_legend_handles_labels()
    handles_2, labels_2 = value_axis.get_legend_handles_labels()
    count_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )

    count_axis.set_title(title, fontweight="bold", pad=75)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. OUTPUT HELPERS
# ============================================================


def save_summary_tables(
    report_directory: Path,
    total_summary: pd.DataFrame,
    take_private_summary: pd.DataFrame,
    non_take_private_summary: pd.DataFrame,
) -> None:
    """Save the three summary tables as CSV files for validation."""
    summary_directory = report_directory / "summary_tables"
    summary_directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "Total_Summary.csv": total_summary,
        "TP_Summary.csv": take_private_summary,
        "NonTP_Summary.csv": non_take_private_summary,
    }

    for filename, summary in outputs.items():
        path = summary_directory / filename
        summary.to_csv(path, index=False)
        print(f"Saved summary: {path}")


def print_summary_status(name: str, summary: pd.DataFrame) -> None:
    """Print a compact status line for a calculated summary."""
    if summary.empty:
        print(f"{name}: no observations in the selected year range.")
        return

    total_entries = int(summary["N_total"].sum())
    exited_entries = int(summary["N_exited"].sum())
    print(
        f"{name}: {total_entries:,} entries, "
        f"{exited_entries:,} matched exits, "
        f"{len(summary):,} vintages."
    )


# ============================================================
# 8. MAIN DRIVER
# ============================================================


def main() -> None:
    root = find_project_root()
    report_directory = root / "reports"
    figure_directory = report_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exit Deal Nested Analysis")
    print("=" * 60)
    print(f"Project root: {root}")
    print(f"Entry-year range: {YEAR_MIN}-{YEAR_MAX}")
    print()

    # Load the same four source datasets used by the original script.
    universe_master = load_data_flexible(
        root,
        "clean",
        "pitchbook_master_clean",
    )
    take_private_entries = load_data_flexible(
        root,
        "raw",
        "pitchbook_public_2_private_all",
    )
    take_private_pairs = load_data_flexible(
        root,
        "clean",
        "p2p_linked_master",
    )
    non_take_private_pairs = load_data_flexible(
        root,
        "clean",
        "non_tp_entry_exit_pairs",
    )

    if universe_master.empty:
        raise FileNotFoundError(
            "Could not load pitchbook_master_clean from the project's data/clean folder."
        )

    require_columns(universe_master, REQUIRED_ENTRY_COLUMNS, "Universe master")

    if not take_private_entries.empty:
        require_columns(
            take_private_entries,
            REQUIRED_ENTRY_COLUMNS,
            "Take-private entry dataset",
        )

    print()
    non_take_private_entries = derive_non_take_private_entries(
        universe_master,
        take_private_entries,
    )

    print()
    take_private_summary = build_summary(
        take_private_entries,
        take_private_pairs,
        YEAR_MIN,
        YEAR_MAX,
    )

    non_take_private_summary = build_summary(
        non_take_private_entries,
        non_take_private_pairs,
        YEAR_MIN,
        YEAR_MAX,
    )

    total_summary = combine_summaries(
        take_private_summary,
        non_take_private_summary,
    )

    print_summary_status("Entire Universe", total_summary)
    print_summary_status("Take-Private", take_private_summary)
    print_summary_status("Non-Take-Private", non_take_private_summary)

    save_summary_tables(
        report_directory,
        total_summary,
        take_private_summary,
        non_take_private_summary,
    )

    chart_tasks = [
        (
            total_summary,
            "Entire Universe (All PE)",
            "Total",
            True,
        ),
        (
            take_private_summary,
            "Take-Private (P2P)",
            "TP",
            False,
        ),
        (
            non_take_private_summary,
            "Non-Take-Private (Other PE)",
            "NonTP",
            True,
        ),
    ]

    print()
    for summary, title, filename_prefix, scale_to_billions in chart_tasks:
        if summary.empty:
            print(f"Skipped charts for {filename_prefix}: summary is empty.")
            continue

        percentage_path = figure_directory / f"{filename_prefix}_Percentage.png"
        nested_path = figure_directory / f"{filename_prefix}_Nested.png"

        plot_percentage_style(
            summary,
            f"{title}: Realization Rates",
            percentage_path,
        )
        plot_nested_style(
            summary,
            f"{title}: Volume & Realization",
            nested_path,
            scale_to_billions,
        )

        print(f"Generated: {percentage_path}")
        print(f"Generated: {nested_path}")

    print()
    print("Analysis complete.")
    print(f"Figures saved to: {figure_directory.resolve()}")
    print(f"Summary tables saved to: {(report_directory / 'summary_tables').resolve()}")


if __name__ == "__main__":
    main()