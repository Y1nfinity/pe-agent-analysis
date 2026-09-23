"""
entry_exit_completion_bars_tp.py
=================================

Creates entry-year completion charts for the **take-private sample only**.

Blue = % exited (count)
Red  = % exited (value)

Input:
    data/clean/entry_exit_pairs.parquet

Output:
    outputs/figures/TP_EntryCompletion.png
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Load take-private entry–exit pairs
# ---------------------------------------------------------
def load_tp_pairs(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"TP pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    # Normalize column names (just in case)
    rename_map = {}
    if "deal_size_usd_m_entry" not in df.columns and "deal_size_usd_m" in df.columns:
        rename_map["deal_size_usd_m"] = "deal_size_usd_m_entry"

    df = df.rename(columns=rename_map)

    # Validate required fields
    required = ["entry_date", "exit_date", "deal_size_usd_m_entry"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["entry_year"] = df["entry_date"].dt.year
    df["has_exit"] = df["exit_date"].notna()

    return df


# ---------------------------------------------------------
# Summarize TP sample
# ---------------------------------------------------------
def summarize(df: pd.DataFrame, year_min=2000, year_max=2025):
    rows = []

    for year in range(year_min, year_max + 1):
        df_y = df[df["entry_year"] == year]

        if df_y.empty:
            continue

        N = len(df_y)
        N_ex = df_y["has_exit"].sum()

        V_total = df_y["deal_size_usd_m_entry"].sum(skipna=True)
        V_ex = df_y.loc[df_y["has_exit"], "deal_size_usd_m_entry"].sum(skipna=True)

        rows.append({
            "entry_year": year,
            "N": N,
            "N_ex": N_ex,
            "pct_N": N_ex / N if N else 0,
            "V_total": V_total,
            "V_ex": V_ex,
            "pct_V": V_ex / V_total if V_total else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Plotting function
# ---------------------------------------------------------
def plot_completion(df: pd.DataFrame, outpath: Path, title: str):
    plt.figure(figsize=(18, 8))

    years = df["entry_year"]
    x = range(len(years))

    # Bars
    plt.bar(x, df["pct_N"], width=0.4, color="blue", alpha=0.7, label="% Exited (N)")
    plt.bar([i + 0.4 for i in x], df["pct_V"], width=0.4, color="red", alpha=0.6, label="% Exited ($)")

    # Labels
    for i, y in enumerate(df["pct_N"]):
        plt.text(i, y + 0.02, f"{y:.0%}", ha="center", fontsize=8, color="blue")

    for i, y in enumerate(df["pct_V"]):
        plt.text(i + 0.4, y + 0.02, f"{y:.0%}", ha="center", fontsize=8, color="red")

    plt.xticks([i + 0.2 for i in x], years, rotation=90)
    plt.ylim(0, 1.15)
    plt.ylabel("Percent Exited")
    plt.title(title)
    plt.legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✅ Saved figure → {outpath}")


# ---------------------------------------------------------
# MASTER DRIVER
# ---------------------------------------------------------
def run_tp_completion(year_min=2000, year_max=2025):
    ROOT = Path(__file__).resolve().parents[2]

    print("🔹 Loading TP entry-exit pairs…")
    df_tp = load_tp_pairs(ROOT)

    print("🔹 Summarizing...")
    df_sum = summarize(df_tp, year_min=year_min, year_max=year_max)

    OUT = ROOT / "outputs" / "figures" / "TP_EntryCompletion.png"

    print("🔹 Plotting…")
    plot_completion(df_sum, OUT, title="Take-Private Deals — Entry-Year Completion (% Exited by Count and Value)")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    run_tp_completion(2000, 2025)
