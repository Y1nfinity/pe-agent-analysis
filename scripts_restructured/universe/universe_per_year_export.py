"""
universe_per_year_export.py
===========================

Parallel to per_year_export.py but applied to the full PitchBook universe.

Input:
    data/clean/pitchbook_master_clean.parquet

Output:
    outputs/per_year_universe/
        entry_year_based/YYYY/
        exit_year_based/YYYY/
        index.csv
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Load clean PitchBook universe
# ------------------------------------------------------------
def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Figure helpers
# ------------------------------------------------------------
def save_hist(series, title, outfile):
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=20)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_bar(df, x, y, title, outfile):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_stacked(df, index_col, category_col, value_col, title, outfile):
    if df.empty:
        return

    pivot = df.pivot_table(
        index=index_col,
        columns=category_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(index_col)
    ax.set_ylabel(value_col)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)




# ------------------------------------------------------------
# ENTRY YEAR PANEL
# ------------------------------------------------------------
def build_entry_year_panel(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["transaction_year"] == year].copy()


def export_entry_year(df: pd.DataFrame, outdir: Path, year: int):

    year_dir = outdir / "entry_year_based" / str(year)
    fig_dir = year_dir / "figures"
    year_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(year_dir / "data.csv", index=False)

    summary = {
        "N_deals": len(df),
        "deal_value_total": df["deal_size_usd_m"].sum(skipna=True),
        "deal_value_mean": df["deal_size_usd_m"].mean(skipna=True),
        "deal_value_median": df["deal_size_usd_m"].median(skipna=True),
    }

    pd.DataFrame([summary]).to_csv(year_dir / "summary.csv", index=False)

    # Histogram
    save_hist(
        df["deal_size_usd_m"].dropna(),
        f"Deal Size Distribution — Entry Year {year}",
        fig_dir / "deal_size_dist.png"
    )

    # Composition (deal type)
    save_stacked(
        df,
        index_col="transaction_year",
        category_col="exit_category",
        value_col="deal_size_usd_m",
        title=f"Composition by Exit Category — Entry Year {year}",
        outfile=fig_dir / "composition_value.png"
    )


# ------------------------------------------------------------
# EXIT YEAR PANEL
# ------------------------------------------------------------
def build_exit_year_panel(df: pd.DataFrame, year: int) -> pd.DataFrame:
    # Exit-type deals = those classified with non-null exit_category
    return df[(df["transaction_year"] == year) & df["exit_category"].notna()].copy()


def export_exit_year(df: pd.DataFrame, outdir: Path, year: int):

    year_dir = outdir / "exit_year_based" / str(year)
    fig_dir = year_dir / "figures"
    year_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(year_dir / "data.csv", index=False)

    summary = {
        "N_exits": len(df),
        "exit_value_total": df["deal_size_usd_m"].sum(skipna=True),
        "exit_value_mean": df["deal_size_usd_m"].mean(skipna=True),
        "exit_value_median": df["deal_size_usd_m"].median(skipna=True),
    }

    pd.DataFrame([summary]).to_csv(year_dir / "summary.csv", index=False)

    # Exit composition
    save_stacked(
        df,
        index_col="transaction_year",
        category_col="exit_category",
        value_col="deal_size_usd_m",
        title=f"Exit Composition — Exit Year {year}",
        outfile=fig_dir / "exit_composition_value.png"
    )


# ------------------------------------------------------------
# INDEX FILE
# ------------------------------------------------------------
def write_index(rows, outdir: Path):
    pd.DataFrame(rows).to_csv(outdir / "index.csv", index=False)


# ------------------------------------------------------------
# MASTER DRIVER
# ------------------------------------------------------------
def run_universe_per_year(universe_path: Path, outdir: Path, year_min=1995, year_max=2025):

    df = load_universe(universe_path)
    outdir.mkdir(parents=True, exist_ok=True)

    index_rows = []

    for y in range(year_min, year_max + 1):

        df_entry = build_entry_year_panel(df, y)
        df_exit  = build_exit_year_panel(df, y)

        if not df_entry.empty:
            export_entry_year(df_entry, outdir, y)

        if not df_exit.empty:
            export_exit_year(df_exit, outdir, y)

        index_rows.append({
            "year": y,
            "has_entry_year": not df_entry.empty,
            "has_exit_year": not df_exit.empty,
            "entry_count": len(df_entry),
            "exit_count": len(df_exit),
            "entry_folder": str(outdir / "entry_year_based" / str(y)),
            "exit_folder": str(outdir / "exit_year_based" / str(y)),
        })

    write_index(index_rows, outdir)
    print(f"✅ Universe per-year export complete → {outdir.resolve()}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]
    UNIVERSE = ROOT / "data" / "clean" / "pitchbook_master_clean.parquet"
    OUT      = ROOT / "outputs" / "per_year_universe"

    run_universe_per_year(UNIVERSE, OUT)
