"""
DEBUG VERSION — tp_vs_universe_entryyear.py

Adds:
    • Debug prints for verifying counts
    • Better validation checks
    • Improved bar layout: side-by-side (blue = count, red = value)
    • Vertical x-axis labels
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_panel(path: Path) -> pd.DataFrame:
    print(f"🔹 Loading panel: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    print(f"   → {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("\n")
    return df


def debug_print(title, df):
    print("\n" + "="*80)
    print(f"DEBUG — {title}")
    print("="*80)
    if df is None or df.empty:
        print("❌ DataFrame is EMPTY")
    else:
        print(df.head(20))
        print(df.describe(include='all'))
    print("="*80 + "\n")


def plot_tp_vs_universe(tp_panel, univ_panel, outdir: Path):

    # -------------------------------------------------------------
    # STEP 1 — Renaming according to actual existing panel scripts
    # -------------------------------------------------------------
    tp = tp_panel.rename(columns={
        "N_entries": "N_entries_tp",
        "entry_value_usd_m": "entry_value_tp"
    })

    univ = univ_panel.rename(columns={
        "N_entries": "N_entries_univ"
    })

    debug_print("TP PANEL (renamed)", tp)
    debug_print("UNIVERSE PANEL (renamed)", univ)

    # -------------------------------------------------------------
    # STEP 2 — Merge
    # -------------------------------------------------------------
    merged = pd.merge(tp, univ, on="entry_year", how="inner")

    debug_print("MERGED PANEL", merged)

    if merged.empty:
        print("❌ No overlapping years — cannot plot.")
        return

    # -------------------------------------------------------------
    # STEP 3 — Validate counts
    # -------------------------------------------------------------
    print("\n🔍 VALIDATION CHECK")
    for _, row in merged.iterrows():
        y = int(row["entry_year"])
        tp_n = row["N_entries_tp"]
        univ_n = row["N_entries_univ"]
        print(f"Year {y}: TP={tp_n}, Univ={univ_n}, Share={tp_n/univ_n if univ_n else None}")

    # -------------------------------------------------------------
    # FIX percentages
    # -------------------------------------------------------------
    merged["pct_count"] = merged["N_entries_tp"] / merged["N_entries_univ"].replace(0, pd.NA)

    total_tp_value = merged["entry_value_tp"].sum()
    if total_tp_value > 0:
        merged["pct_value"] = merged["entry_value_tp"] / total_tp_value
    else:
        merged["pct_value"] = 0

    # -------------------------------------------------------------
    # STEP 4 — Plot
    # -------------------------------------------------------------
    years = merged["entry_year"].astype(int)
    n = len(years)
    x = range(n)

    width = 0.4  # side-by-side bars

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # -------- BLUE BAR: TP COUNT --------
    ax1.bar(
        [i - width/2 for i in x],
        merged["N_entries_tp"],
        width=width,
        color="steelblue",
        label="TP Deal Count"
    )
    ax1.set_ylabel("Count (TP Deals)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Add % labels for counts
    for i, y, pct in zip(x, merged["N_entries_tp"], merged["pct_count"]):
        if pd.notna(pct):
            ax1.text(i - width/2, y + 0.5, f"{pct:.1%}", color="blue", ha="center")

    # -------- RED BAR: TP DOLLAR VALUE --------
    ax2 = ax1.twinx()
    ax2.bar(
        [i + width/2 for i in x],
        merged["entry_value_tp"],
        width=width,
        color="indianred",
        label="TP Dollar Volume"
    )
    ax2.set_ylabel("Dollar Volume (USD mm)", color="indianred")
    ax2.tick_params(axis="y", labelcolor="indianred")

    # Value % labels
    for i, v, pct in zip(x, merged["entry_value_tp"], merged["pct_value"]):
        if pd.notna(pct):
            ax2.text(i + width/2, v + 0.5, f"{pct:.1%}", color="darkred", ha="center")

    # -------- Titles & labels --------
    plt.title("Take-Private Activity vs Universe — Entry Year View (Counts & Dollar Volume)")

    plt.xticks(x, years, rotation=90)  # vertical labeling

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "TP_vs_Universe_EntryYear_DEBUG.png"
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"\n✅ Saved debug figure → {outfile}\n")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]

    tp_path   = ROOT / "outputs" / "panels" / "tp_entry_year_panel.csv"
    univ_path = ROOT / "outputs" / "panels" / "universe_entry_year_panel.csv"
    OUT       = ROOT / "outputs" / "figures"

    tp = load_panel(tp_path)
    univ = load_panel(univ_path)

    plot_tp_vs_universe(tp, univ, OUT)
