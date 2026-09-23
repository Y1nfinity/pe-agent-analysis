"""
take_private_flow_analysis.py
=============================
Analysis of take-private entries and their exit pathways.

Outputs:
    Sankey/alluvial flow diagram
    Exit composition for TP subset
    Deal value over time (TP only)
    Holding period ECDF
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "take_private_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def load_pairs():
    return pd.read_parquet(DATA / "entry_exit_pairs.parquet")


# -------------------------------------------------------
# Identify take-private entries
# -------------------------------------------------------
def filter_take_privates(df):
    mask = df["entry_type"].str.lower().str.contains(
        r"(?:public\s*to\s*private|take[-\s]*private|p2p)",
        na=False,
    )

    return df.loc[mask].copy()


# -------------------------------------------------------
# 1. Sankey / Alluvial Diagram
# -------------------------------------------------------
def plot_sankey(df, out_file):
    entry_node = "Public to Private"
    exit_types = df["exit_type"].fillna("Unknown").unique().tolist()

    labels = [entry_node] + exit_types
    source = []
    target = []
    value = []

    for etype in exit_types:
        source.append(0)
        target.append(labels.index(etype))
        value.append((df["exit_type"] == etype).sum())

    fig = go.Figure(go.Sankey(
        node=dict(label=labels),
        link=dict(source=source, target=target, value=value)
    ))

    fig.update_layout(title="Take-Private → Exit Flow", font_size=12)
    fig.write_html(out_file)


# -------------------------------------------------------
# 2. Exit composition bar chart (TP only)
# -------------------------------------------------------
def plot_exit_composition(df, out):
    comp = df["exit_type"].dropna().value_counts()

    if comp.empty:
        print("⚠️ No exit_type values found for take-private deals; skipping exit composition plot.")
        return

    comp.to_csv(out / "TP_Exit_Composition.csv")

    plt.figure(figsize=(10, 5))
    comp.plot(kind="bar")

    plt.figure(figsize=(10,5))
    comp.plot(kind="bar")
    plt.title("Exit Types for Take-Privates")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out / "Figure_TP_Exit_Composition.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# 3. TP deal value over time
# -------------------------------------------------------
def tp_deal_value(df, out):
    if "deal_size" not in df.columns:
        return

    df["exit_year"] = pd.to_datetime(df["exit_date"]).dt.year
    agg = df.groupby("exit_year")["deal_size"].sum()
    agg.to_csv(out / "TP_DealValue_ByYear.csv")

    plt.figure(figsize=(10,5))
    agg.plot(marker="o")
    plt.title("Deal Value (Take-Privates Only)")
    plt.xlabel("Exit Year")
    plt.ylabel("Total Deal Size (USD)")
    plt.grid(alpha=0.3)
    plt.savefig(out / "Figure_TP_DealValue_ByYear.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# 4. Holding period ECDF
# -------------------------------------------------------
def plot_ecdf(df, out):
    holding = np.sort(df["holding_period_years"])
    y = np.arange(1, len(holding)+1) / len(holding)

    plt.figure(figsize=(8,5))
    plt.plot(holding, y)
    plt.title("ECDF of Holding Periods (Take-Privates)")
    plt.xlabel("Holding Period (Years)")
    plt.ylabel("Cumulative Probability")
    plt.grid(alpha=0.3)
    plt.savefig(out / "Figure_TP_Holding_ECDF.png", dpi=300)
    plt.close()


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
def run_all():
    df = load_pairs()
    tp = filter_take_privates(df)

    plot_sankey(tp, OUT / "Figure_TP_Sankey.html")
    plot_exit_composition(tp, OUT)
    tp_deal_value(tp, OUT)
    plot_ecdf(tp, OUT)

    print(f"✅ Take private analysis done: {OUT}")


if __name__ == "__main__":

    run_all()
