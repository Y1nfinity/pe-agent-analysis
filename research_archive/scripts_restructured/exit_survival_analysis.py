"""
exit_survival_analysis.py (Upgraded Formatting)
==============================================

Market-wide exit diagnostics:
    1. Survivorship-style curve (1 - ECDF of holding periods)
    2. Exit likelihood by entry vintage
    3. Entry vs Exit count comparison

Inputs:
    data/clean/entry_exit_pairs.parquet

Outputs:
    outputs/entry_exit_survival/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------
# Paths and style
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "entry_exit_survival"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.9
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "DejaVu Sans"
})


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _percent_fmt(x, pos):
    return f"{x * 100:.0f}%"


def _thousands_fmt(x, pos):
    return f"{int(x):,}"


# --------------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------------
def load_pairs() -> pd.DataFrame:
    path = DATA / "entry_exit_pairs.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)

    # Basic sanity: ensure dates and holding periods exist
    if "holding_period_years" not in df.columns:
        raise KeyError("Expected column 'holding_period_years' in entry_exit_pairs.")
    if "entry_date" not in df.columns or "exit_date" not in df.columns:
        raise KeyError("Expected 'entry_date' and 'exit_date' in entry_exit_pairs.")

    return df


# --------------------------------------------------------------------
# 1. Survivorship-style curve
# --------------------------------------------------------------------
def plot_survivorship_curve(df: pd.DataFrame, out: Path) -> None:
    """
    Plots 1 - ECDF of holding periods for *exited* deals.
    Note: true KM with censoring would require non-exited entries as well.
    """
    df = df.copy()
    df = df[df["holding_period_years"].notna() & (df["holding_period_years"] >= 0)]

    if df.empty:
        print("⚠️ No valid holding periods; skipping survivorship curve.")
        return

    holding = np.sort(df["holding_period_years"].values)
    n = len(holding)

    # ECDF
    ecdf = np.arange(1, n + 1) / n
    survival = 1 - ecdf  # fraction not yet exited at each holding

    plt.figure(figsize=(9, 6))
    plt.step(holding, survival, where="post", linewidth=2)
    plt.xlabel("Holding Period (Years)")
    plt.ylabel("Survival (Fraction Not Yet Exited)")
    plt.title("Survivorship of Exited PE Deals (1 - ECDF)")
    plt.grid(alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_percent_fmt))
    plt.tight_layout()
    plt.savefig(out / "Figure_SurvivorshipCurve.png")
    plt.close()


# --------------------------------------------------------------------
# 2. Exit likelihood by entry vintage
# --------------------------------------------------------------------
def exit_likelihood_by_vintage(df: pd.DataFrame, out: Path) -> None:
    """
    Exit likelihood by entry year, based on *pairs only*:
        exits_per_entry_year / unique_companies_per_entry_year
    Note: true likelihood would require all entries (exited + non-exited).
    """
    df = df.copy()
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year

    # Unique entries per vintage (within linked pairs universe)
    entries_per_year = (
        df.groupby("entry_year")["company_name_clean"]
          .nunique()
          .rename("n_entries")
    )

    exits_per_year = (
        df.groupby("entry_year")["exit_date"]
          .count()
          .rename("n_exits")
    )

    likelihood = (exits_per_year / entries_per_year).fillna(0.0)

    # Save numeric table (fraction + percent)
    out_df = pd.DataFrame({
        "entry_year": likelihood.index,
        "n_entries": entries_per_year.reindex(likelihood.index),
        "n_exits": exits_per_year.reindex(likelihood.index),
        "exit_fraction": likelihood.values,
        "exit_percent": likelihood.values * 100,
    })
    out_df.to_csv(out / "ExitLikelihood_ByVintage.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(likelihood.index, likelihood.values, marker="o", linewidth=2)
    ax.set_title("Exit Likelihood by Entry Vintage (Linked Pairs Only)")
    ax.set_xlabel("Entry Year")
    ax.set_ylabel("Exit Likelihood")
    ax.yaxis.set_major_formatter(FuncFormatter(_percent_fmt))
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / "Figure_ExitLikelihood_ByVintage.png")
    plt.close()


# --------------------------------------------------------------------
# 3. Entry vs Exit counts per year
# --------------------------------------------------------------------
def entry_vs_exit_counts(df: pd.DataFrame, out: Path) -> None:
    """
    Compare counts of entries and exits by calendar year, within the linked pairs.
    """
    df = df.copy()
    df["entry_year"] = pd.to_datetime(df["entry_date"]).dt.year
    df["exit_year"] = pd.to_datetime(df["exit_date"]).dt.year

    entry_counts = (
        df.groupby("entry_year")["company_name_clean"]
          .nunique()
          .rename("entries")
    )
    exit_counts = (
        df.groupby("exit_year")["company_name_clean"]
          .nunique()
          .rename("exits")
    )

    comp = pd.concat([entry_counts, exit_counts], axis=1).fillna(0)
    comp = comp.astype(int)

    comp_out = comp.reset_index().rename(columns={"index": "year"})
    comp_out.to_csv(out / "Entry_vs_Exit_Counts.csv", index=False)

    # Plot
    years = comp.index.astype(int)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    width = 0.4
    ax.bar(years - width / 2, comp["entries"], width=width, label="Entries")
    ax.bar(years + width / 2, comp["exits"], width=width, label="Exits")

    ax.set_title("Entries vs Exits Over Time (Linked Pairs)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Companies")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_fmt))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "Figure_Entry_vs_Exit_Counts.png")
    plt.close()


# --------------------------------------------------------------------
# Run all
# --------------------------------------------------------------------
def run_all() -> None:
    df = load_pairs()
    print("🔹 Running exit survival diagnostics...")
    plot_survivorship_curve(df, OUT)
    exit_likelihood_by_vintage(df, OUT)
    entry_vs_exit_counts(df, OUT)
    print(f"✅ Exit survival diagnostics saved to: {OUT}")


if __name__ == "__main__":
    run_all()
