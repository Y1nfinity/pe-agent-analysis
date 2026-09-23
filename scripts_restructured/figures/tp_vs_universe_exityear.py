import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_tp_vs_universe_exits(tp_exit_panel, univ_exit_panel, outpath):
    print("TP columns:", tp_exit_panel.columns.tolist())
    print("UNIV columns:", univ_exit_panel.columns.tolist())

    # ---- Rename for merge clarity ----
    tp = tp_exit_panel.rename(columns={
        "exit_year": "year",
        "N_exits": "N_exits_tp",
        "exit_value_usd_m": "exit_value_tp"
    })

    univ = univ_exit_panel.rename(columns={
        "exit_year": "year",
        "N_exits": "N_exits_univ",
        "exit_value_usd_m": "exit_value_univ"
    })

    # ---- Merge ----
    df = pd.merge(tp, univ, on="year", how="inner")

    # ---- Compute shares ----
    df["pct_exits"] = df["N_exits_tp"] / df["N_exits_univ"]
    df["pct_value"] = df["exit_value_tp"] / df["exit_value_univ"]

    # ---- Sort by year ----
    df = df.sort_values("year")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(18, 8))

    years = df["year"].astype(int)
    x = range(len(years))
    width = 0.4

    # Left axis → counts
    ax1.bar([i - width/2 for i in x], df["N_exits_tp"],
            width=width, label="TP Exit Count", color="steelblue")
    ax1.set_ylabel("TP Exit Count")

    # Annotate % inside blue bars
    for i, v in enumerate(df["N_exits_tp"]):
        pct = df["pct_exits"].iloc[i]
        ax1.text(i - width/2, v + (v * 0.02), f"{pct:.1%}",
                 ha="center", va="bottom", fontsize=9, color="black")

    # Right axis → dollar values
    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], df["exit_value_tp"],
            width=width, label="TP Exit Value", color="firebrick")
    ax2.set_ylabel("TP Exit Value (USD millions)")

    # Annotate % inside red bars
    for i, v in enumerate(df["exit_value_tp"]):
        pct = df["pct_value"].iloc[i]
        ax2.text(i + width/2, v + (v * 0.02), f"{pct:.1%}",
                 ha="center", va="bottom", fontsize=9, color="black")

    # X-axis formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close()

    print(f"✅ Saved TP vs Universe EXIT-YEAR chart → {outpath}")





if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    tp_path = ROOT / "outputs" / "panels" / "tp_exit_year_panel.csv"
    univ_path = ROOT / "outputs" / "panels_universe" / "universe_exit_year_panel.csv"
    out = ROOT / "outputs" / "figures" / "TP_vs_Universe_ExitYear.png"

    tp = pd.read_csv(tp_path)
    univ = pd.read_csv(univ_path)

    plot_tp_vs_universe_exits(tp, univ, out)
