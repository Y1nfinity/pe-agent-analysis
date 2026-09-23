from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Helper: Plot TP vs Universe (Entry Year)
# =============================================================================
def plot_tp_vs_universe_entry(tp: pd.DataFrame, univ: pd.DataFrame, outfile: Path):

    # --- rename columns to avoid collisions ---
    tp = tp.copy().rename(columns={
        "N_entries": "N_entries_tp",
        "entry_value_usd_m": "entry_value_tp"
    })

    univ = univ.copy().rename(columns={
        "N_entries": "N_entries_univ",
        "entry_value_usd_m": "entry_value_univ"
    })

    # --- merge ---
    merged = pd.merge(tp, univ, on="entry_year", how="inner")

    # --- compute percentages ---
    merged["pct_entry_count"] = merged["N_entries_tp"] / merged["N_entries_univ"]
    merged["pct_entry_value"] = merged["entry_value_tp"] / merged["entry_value_univ"]

    # --- filter years ---
    merged = merged[(merged["entry_year"] >= 2000) & (merged["entry_year"] <= 2025)]

    # =============================================================================
    # PLOT
    # =============================================================================
    fig, ax1 = plt.subplots(figsize=(16, 8))

    x = merged["entry_year"]
    width = 0.35

    # ------------------------ LEFT Y-AXIS: COUNTS ------------------------
    ax1.bar(x - width/2, merged["N_entries_tp"],
            width=width, label="TP Entries (Count)", color="#2C7BE5")  # blue

    ax1.set_ylabel("Number of Take-Private Entries", fontsize=12)
    ax1.set_xlabel("Entry Year", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, rotation=90)

    # Place % labels above blue bars
    for i, (xpos, p) in enumerate(zip(x - width/2, merged["pct_entry_count"])):
        ax1.text(xpos, merged["N_entries_tp"].iloc[i] + 1,
                 f"{p:.1%}", ha="center", va="bottom", fontsize=9, color="black")

    # ------------------------ RIGHT Y-AXIS: USD VALUE ------------------------
    ax2 = ax1.twinx()

    ax2.bar(x + width/2, merged["entry_value_tp"],
            width=width, label="TP Entry Value ($M)", color="#E63757")  # red

    ax2.set_ylabel("TP Entry Value (USD millions)", fontsize=12)

    # Place % labels above red bars
    for i, (xpos, p) in enumerate(zip(x + width/2, merged["pct_entry_value"])):
        ax2.text(xpos, merged["entry_value_tp"].iloc[i] + 1,
                 f"{p:.1%}", ha="center", va="bottom", fontsize=9, color="black")

    # ------------------------ TITLE & LEGEND ------------------------
    fig.suptitle("Take-Private vs Universe — Entry Year Comparison\nCounts + Dollar Values (2000–2025)",
                 fontsize=16)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"✅ Saved TP vs Universe ENTRY-YEAR figure → {outfile}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    tp_path   = ROOT / "outputs" / "panels" / "tp_entry_year_panel.csv"
    univ_path = ROOT / "outputs" / "panels_universe" / "universe_entry_year_panel.csv"
    out       = ROOT / "outputs" / "figures" / "TP_vs_Universe_EntryYear.png"

    print("🔹 Loading TP entry-year panel:", tp_path)
    print("🔹 Loading Universe entry-year panel:", univ_path)

    tp = pd.read_csv(tp_path)
    univ = pd.read_csv(univ_path)

    plot_tp_vs_universe_entry(tp, univ, out)
