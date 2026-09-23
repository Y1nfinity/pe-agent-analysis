"""
per_year_export.py
==================

Creates YEAR-BY-YEAR data + figures for both:
    • Entry-year based panels
    • Exit-year based panels

Enhancements (per user request):
    (1) Dollarized entry values per year
    (2) Dollarized exit values per year (both views)
    (3) Holding-period distribution histograms
    (4) A metadata index file of all exported years

Outputs structure:
    outputs/per_year/
        entry_year_based/YYYY/
            data.csv
            summary.csv
            figures/*.png
        exit_year_based/YYYY/
            data.csv
            summary.csv
            figures/*.png
        index.csv
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Load pairs
# ------------------------------------------------------------
def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Figure helpers
# ------------------------------------------------------------
def save_hist(series, title, outfile):
    if series.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=20)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def save_bar(df, x, y, title, outfile):
    if df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(axis="y")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def save_stacked(df, category_col, value_col, title, outfile):
    """
    Replaces pivot-based stacked bar with category-level aggregation.
    Works for entry-year and exit-year single-year panels.
    """
    if df.empty:
        return

    agg = (
        df.groupby(category_col)[value_col]
          .sum()
          .sort_values(ascending=False)
    )

    plt.figure(figsize=(9,5))
    plt.bar(agg.index, agg.values)
    plt.title(title)
    plt.xlabel(category_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()



# ------------------------------------------------------------
# ENTRY YEAR PANEL
# ------------------------------------------------------------
def build_entry_year_panel(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df_y = df[df["entry_date"].dt.year == year].copy()
    if df_y.empty:
        return df_y
    df_y["exited"] = df_y["exit_date"].notna()
    return df_y


def export_entry_year(df: pd.DataFrame, outdir: Path, year: int):

    year_dir = outdir / "entry_year_based" / str(year)
    fig_dir = year_dir / "figures"
    year_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(year_dir / "data.csv", index=False)

    # ---------- SUMMARY ----------
    summary = {
        "N_entries": len(df),
        "N_exits": df["exited"].sum(),
        "entry_value_total": df["deal_size_usd_m_entry"].sum(skipna=True),
        "entry_value_exited": df.loc[df["exited"], "deal_size_usd_m_entry"].sum(skipna=True),
        "entry_value_mean": df["deal_size_usd_m_entry"].mean(skipna=True),
        "entry_value_median": df["deal_size_usd_m_entry"].median(skipna=True),
        "exit_value_total": df["deal_size_usd_m_exit"].sum(skipna=True),
        "exit_value_mean": df["deal_size_usd_m_exit"].mean(skipna=True),
        "exit_value_median": df["deal_size_usd_m_exit"].median(skipna=True),
        "frac_exited_by_count": df["exited"].mean(),
        "frac_exited_by_value":
            df.loc[df["exited"], "deal_size_usd_m_entry"].sum(skipna=True)
            / df["deal_size_usd_m_entry"].sum(skipna=True)
            if df["deal_size_usd_m_entry"].sum(skipna=True) > 0
            else None,
    }

    pd.DataFrame([summary]).to_csv(year_dir / "summary.csv", index=False)

    # ---------- FIGURES ----------
    # Holding-period histogram (only exited)
    hp = df.loc[df["exited"], "holding_period_years"].dropna()
    save_hist(hp, f"Holding Period Distribution — Entry Year {year}",
              fig_dir / "holding_period_dist.png")

    # Exit composition (stacked bar)
    df_exit = df[df["exited"]]
    if not df_exit.empty:
        save_stacked(
            df_exit,
            category_col="exit_category",
            value_col="deal_size_usd_m_exit",
            title=f"Exit Composition (Value) — Entry Year {year}",
            outfile=fig_dir / "exit_composition_value.png"
        )


# ------------------------------------------------------------
# EXIT YEAR PANEL
# ------------------------------------------------------------
def build_exit_year_panel(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["exit_date"].dt.year == year].copy()


def export_exit_year(df: pd.DataFrame, outdir: Path, year: int):

    year_dir = outdir / "exit_year_based" / str(year)
    fig_dir = year_dir / "figures"
    year_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(year_dir / "data.csv", index=False)

    summary = {
        "N_exits": len(df),
        "exit_value_total": df["deal_size_usd_m_exit"].sum(skipna=True),
        "exit_value_mean": df["deal_size_usd_m_exit"].mean(skipna=True),
        "exit_value_median": df["deal_size_usd_m_exit"].median(skipna=True),
        "hp_mean": df["holding_period_years"].mean(skipna=True),
        "hp_median": df["holding_period_years"].median(skipna=True),
    }

    pd.DataFrame([summary]).to_csv(year_dir / "summary.csv", index=False)

    # ---------- FIGURES ----------
    # holding-period histogram
    hp = df["holding_period_years"].dropna()
    save_hist(hp, f"Holding Period Distribution — Exit Year {year}",
              fig_dir / "holding_period_dist.png")

    # Exit composition (value)
    save_stacked(
    df,
    category_col="exit_category",
    value_col="deal_size_usd_m_exit",
    title=f"Exit Composition (Value) — Exit Year {year}",
    outfile=fig_dir / "exit_composition_value.png"
    )


# ------------------------------------------------------------
# INDEX FILE
# ------------------------------------------------------------
def write_index(index_rows, outdir: Path):
    idx = pd.DataFrame(index_rows)
    idx.to_csv(outdir / "index.csv", index=False)


# ------------------------------------------------------------
# MASTER DRIVER
# ------------------------------------------------------------
def run_per_year_export(pairs_path: Path, outdir: Path, year_min=1995, year_max=2025):

    df = load_pairs(pairs_path)

    index_rows = []

    for year in range(year_min, year_max + 1):

        df_entry = build_entry_year_panel(df, year)
        df_exit  = build_exit_year_panel(df, year)

        if not df_entry.empty:
            export_entry_year(df_entry, outdir, year)

        if not df_exit.empty:
            export_exit_year(df_exit, outdir, year)

        index_rows.append({
            "year": year,
            "has_entry_year_data": not df_entry.empty,
            "has_exit_year_data": not df_exit.empty,
            "entry_count": len(df_entry),
            "exit_count": len(df_exit),
            "entry_year_folder": str(outdir / "entry_year_based" / str(year)),
            "exit_year_folder": str(outdir / "exit_year_based" / str(year)),
        })

    write_index(index_rows, outdir)
    print(f"✅ Per-year export complete → {outdir.resolve()}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]
    CLEAN = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    OUT   = ROOT / "outputs" / "per_year"

    run_per_year_export(CLEAN, OUT)
