"""
Exit Completion by Entry Year — Full Universe Only

For each entry year:
    • Blue solid  = exited count
    • Blue outline = total count
    • % exited (count) inside white region

    • Red solid   = exited entry $ value
    • Red outline = total entry $ value
    • % exited (value) inside white region
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ===============================================================
# Load entry-exit dataset
# ===============================================================
def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairs file: {path}")
    return pd.read_parquet(path)


# ===============================================================
# Build entry-year panel with corrected exit logic
# ===============================================================
def build_entryyear_exit_completion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Extract entry year ---
    df["entry_year"] = df["entry_date"].dt.year

    # --- Correct exit logic ---
    CENSOR = pd.Timestamp("2025-12-31")
    df["exited"] = (
        df["exit_date"].notna()
        & (df["exit_date"] > df["entry_date"])
        & (df["exit_date"] <= CENSOR)
    )

    # --- Group by entry year ---
    grouped = df.groupby("entry_year")

    N_entries = grouped.size()
    N_exited = grouped["exited"].sum()

    entry_value_total = grouped["deal_size_usd_m_entry"].sum()
    entry_value_exited = (
        df[df["exited"]]
        .groupby("entry_year")["deal_size_usd_m_entry"]
        .sum()
    )

    panel = pd.DataFrame({
        "N_entries": N_entries,
        "N_exited": N_exited,
        "entry_value_total": entry_value_total,
        "entry_value_exited": entry_value_exited
    }).fillna(0)

    panel["pct_exited_N"] = panel["N_exited"] / panel["N_entries"].replace(0, np.nan)
    panel["pct_exited_val"] = (
        panel["entry_value_exited"] / panel["entry_value_total"].replace(0, np.nan)
    )

    return panel.reset_index()


# ===============================================================
# Plot
# ===============================================================
def plot_exit_completion(df: pd.DataFrame, title: str, outfile: Path):

    years = df["entry_year"]
    x = np.arange(len(years))

    # --- Count bars ---
    exited_N = df["N_exited"]
    total_N = df["N_entries"]

    # --- Dollar bars ---
    exited_val = df["entry_value_exited"]
    total_val = df["entry_value_total"]

    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twinx()

    width = 0.35

    # =====================================
    # BLUE — Counts
    # =====================================
    # solid exited
    ax1.bar(
        x - width/2,
        exited_N,
        width,
        color="royalblue",
        label="Exited (Count)"
    )

    # outline total
    ax1.bar(
        x - width/2,
        total_N,
        width,
        facecolor="none",
        edgecolor="royalblue",
        linewidth=2,
        label="Total (Count)"
    )

    # =====================================
    # RED — Dollar value
    # =====================================
    ax2.bar(
        x + width/2,
        exited_val,
        width,
        color="firebrick",
        label="Exited (Value)"
    )

    ax2.bar(
        x + width/2,
        total_val,
        width,
        facecolor="none",
        edgecolor="firebrick",
        linewidth=2,
        label="Total (Value)"
    )

    # =====================================
    # % LABELS (in the white region)
    # =====================================
    for i, row in df.iterrows():

        # Count %
        pctN = row["pct_exited_N"]
        if not np.isnan(pctN):
            yN = row["N_exited"] + 0.5 * (row["N_entries"] - row["N_exited"])
            ax1.text(
                x[i] - width/2,
                yN,
                f"{pctN:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="black"
            )

        # Value %
        pctV = row["pct_exited_val"]
        if not np.isnan(pctV):
            yV = row["entry_value_exited"] + 0.5 * (
                row["entry_value_total"] - row["entry_value_exited"]
            )
            ax2.text(
                x[i] + width/2,
                yV,
                f"{pctV:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="black"
            )

    # =====================================
    # Formatting
    # =====================================
    ax1.set_xlabel("Entry Year")
    ax1.set_ylabel("Deal Count")
    ax2.set_ylabel("Deal Value (USD millions)")
    ax1.set_title(title, fontsize=18, weight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=90)

    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Saved → {outfile}")


# ===============================================================
# Driver
# ===============================================================
def run_all(pairs_path: Path, outdir: Path):

    df = load_pairs(pairs_path)

    universe_panel = build_entryyear_exit_completion(df)

    plot_exit_completion(
        universe_panel,
        "Exit Completion by Entry Year — Full Universe",
        outdir / "EntryYear_ExitCompletion_Universe.png"
    )


# ===============================================================
# CLI
# ===============================================================
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]
    PAIRS = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"
    OUTDIR = ROOT / "outputs" / "figures"

    run_all(PAIRS, OUTDIR)
