# tools/p2p_vintage.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- Exit-type mapping (professor categories) --------
def map_exit_type(row: pd.Series) -> str:
    d1 = (row.get("Exit_DealType") or "").strip().lower()
    d2 = (row.get("Exit_DealType2") or "").strip().lower()
    d3 = (row.get("Exit_DealType3") or "").strip().lower()
    synopsis = (row.get("Exit_DealSynopsis") or "").strip().lower()

    if d1 == "ipo":
        return "IPO"

    if d1 == "merger/acquisition":
        # SBO tag
        if "secondary buyout" in d2 or "secondary buyout" in d3:
            return "Sale to another sponsor"
        # Continuation fund cues
        if ("continuation" in d2) or ("continuation" in d3) or ("continuation" in synopsis):
            return "Sale to continuation fund"
        return "Sale to strategic"

    # fallback
    if "continuation" in (d2 + " " + d3 + " " + synopsis):
        return "Sale to continuation fund"
    return "Other/Unknown"

# Keep these 4 by default (drop "Other/Unknown")
ALLOWED_EXIT_TYPES = [
    "IPO",
    "Sale to strategic",
    "Sale to another sponsor",
    "Sale to continuation fund",
]

def restrict_exit_types(df: pd.DataFrame, allowed: list[str] = ALLOWED_EXIT_TYPES) -> pd.DataFrame:
    """Filter to the allowed exit categories."""
    return df[df["ExitTypeLabel"].isin(allowed)].copy()

# -------- IRR proxy (single cash-flow: entry -> exit) --------
def compute_irr_proxy(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    m = (
        out["HoldingYears"].notna() & (out["HoldingYears"] > 0) &
        out["Entry_DealSize"].notna() & (out["Entry_DealSize"] > 0) &
        out["Exit_DealSize"].notna()  & (out["Exit_DealSize"]  > 0)
    )
    out["IRR"] = np.nan
    out.loc[m, "IRR"] = (out.loc[m, "Exit_DealSize"] / out.loc[m, "Entry_DealSize"]) ** (1.0 / out.loc[m, "HoldingYears"]) - 1.0
    return out

# -------- Filter to P2P entries & tag Vintage --------
def filter_p2p_and_vintage(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.copy()
    if "EntryYear" not in p.columns:
        p["EntryYear"] = pd.to_datetime(p["EntryDate"]).dt.year
    p["Vintage"] = p["EntryYear"]
    if "Entry_PublicToPrivate" not in p.columns:
        raise KeyError("Missing 'Entry_PublicToPrivate'. Add it via carry_cols_entry in pair_entry_exit().")
    p = p[p["Entry_PublicToPrivate"] == True].copy()
    p["ExitTypeLabel"] = p.apply(map_exit_type, axis=1)
    return p

# -------- (1)(i) per-vintage: # firms acquired, total going-in spend --------
def vintage_overview_from_entries(entries_p2p: pd.DataFrame) -> pd.DataFrame:
    e = entries_p2p.copy()
    if "EntryYear" not in e.columns:
        e["EntryYear"] = pd.to_datetime(e["DealDate"]).dt.year
    e = e.rename(columns={"EntryYear": "Vintage", "DealSize": "Entry_DealSize"})
    # one row per acquired company within vintage
    ledger = e[["Company","Vintage","Entry_DealSize"]].drop_duplicates()
    out = ledger.groupby("Vintage").agg(
        num_firms_acquired=("Company","nunique"),
        total_going_in_spend=("Entry_DealSize","sum"),
    ).reset_index()
    return out

# -------- Build (ii) tables: exits and non-exits within vintage --------
def build_exit_breakdown_and_nonexit(
    pairs_p2p: pd.DataFrame,
    entries_p2p: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # entries ledger (all P2P acquisitions)
    E = entries_p2p.copy()
    if "EntryYear" not in E.columns:
        E["EntryYear"] = pd.to_datetime(E["DealDate"]).dt.year
    E = E.rename(columns={"EntryYear":"Vintage","DealSize":"Entry_DealSize"})
    E = E[["Company","Vintage","Entry_DealSize"]].drop_duplicates()

    # exited subset (paired)
    X = pairs_p2p[["Company","Vintage","Entry_DealSize","ExitTypeLabel","IRR"]].copy()

    # totals per vintage (for denominators)
    V = E.groupby("Vintage").agg(
        total_firms=("Company","nunique"),
        total_entry_spend=("Entry_DealSize","sum"),
    ).reset_index()

    # (ii)(a)(1)(2): breakdown by exit type
    if not X.empty:
        G = X.groupby(["Vintage","ExitTypeLabel"]).agg(
            firms=("Company","nunique"),
            entry_spend=("Entry_DealSize","sum"),
            avg_irr=("IRR","mean"),
        ).reset_index()
        G = G.merge(V, on="Vintage", how="left")
        G["fraction_by_number"] = G["firms"] / G["total_firms"]
        G["fraction_by_entry_spend"] = G["entry_spend"] / G["total_entry_spend"]
        breakdown = G[[
            "Vintage","ExitTypeLabel","firms","entry_spend","avg_irr","fraction_by_number","fraction_by_entry_spend"
        ]].copy()
    else:
        breakdown = pd.DataFrame(columns=[
            "Vintage","ExitTypeLabel","firms","entry_spend","avg_irr","fraction_by_number","fraction_by_entry_spend"
        ])

    # (ii)(b): non-exited fractions
    XF = X[["Company","Vintage"]].drop_duplicates()
    remaining = E.merge(XF, on=["Company","Vintage"], how="left", indicator=True)
    remaining = remaining[remaining["_merge"]=="left_only"].drop(columns="_merge")
    if not remaining.empty:
        NE = remaining.groupby("Vintage").agg(
            firms_no_exit=("Company","nunique"),
            entry_spend_no_exit=("Entry_DealSize","sum"),
        ).reset_index()
        NE = NE.merge(V, on="Vintage", how="left")
        NE["fraction_by_number_no_exit"] = NE["firms_no_exit"] / NE["total_firms"]
        NE["fraction_by_entry_spend_no_exit"] = NE["entry_spend_no_exit"] / NE["total_entry_spend"]
    else:
        NE = pd.DataFrame(columns=[
            "Vintage","firms_no_exit","entry_spend_no_exit","fraction_by_number_no_exit","fraction_by_entry_spend_no_exit"
        ])
    return breakdown, NE

# -------- Plots required by professor --------
def plot_fraction_bars_by_vintage(df_breakdown: pd.DataFrame, value_col: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_breakdown.pivot_table(index="Vintage", columns="ExitTypeLabel", values=value_col, aggfunc="sum", fill_value=0.0).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10,5))
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Fraction")
    ax.set_title(f"Vintage Fractions by Exit Type — {value_col}")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def plot_irr_by_exit_type_over_time(df_breakdown: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,5))
    for label, sub in df_breakdown.groupby("ExitTypeLabel"):
        sub = sub.sort_values("Vintage")
        ax.plot(sub["Vintage"], sub["avg_irr"], marker="o", label=label)
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Average IRR (exited only)")
    ax.set_title("Average IRR by Exit Type and Vintage")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def plot_exit_counts_stacked(df_breakdown: pd.DataFrame, out_path: Path) -> None:
    """
    Stacked bar chart: absolute firm counts by exit type per vintage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_breakdown.pivot_table(
        index="Vintage", columns="ExitTypeLabel", values="firms", aggfunc="sum", fill_value=0
    ).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10,5))
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Exited firms (count)")
    ax.set_title("Exited P2P Deals by Exit Type — Counts (Stacked)")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_exit_counts_grouped(df_breakdown: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped (side-by-side) bar chart: absolute firm counts by exit type per vintage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_breakdown.pivot_table(
        index="Vintage", columns="ExitTypeLabel", values="firms", aggfunc="sum", fill_value=0
    ).sort_index()

    vintages = pivot.index.to_numpy()
    labels = list(pivot.columns)
    n_types = len(labels)
    x = np.arange(len(vintages))
    width = 0.75 / max(1, n_types)

    fig, ax = plt.subplots(figsize=(10,5))
    for i, label in enumerate(labels):
        ax.bar(x + i*width - (n_types-1)*width/2, pivot[label].to_numpy(), width, label=label)

    ax.set_xticks(x, vintages, rotation=0)
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Exited firms (count)")
    ax.set_title("Exited P2P Deals by Exit Type — Counts (Grouped)")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_exit_spend_stacked(df_breakdown: pd.DataFrame, out_path: Path) -> None:
    """
    Stacked bar chart: absolute entry spend of exited P2P deals, by exit type and vintage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_breakdown.pivot_table(
        index="Vintage", columns="ExitTypeLabel", values="entry_spend", aggfunc="sum", fill_value=0.0
    ).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10,5))
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Entry spend (USD)")
    ax.set_title("Exited P2P Deals — Entry Spend by Exit Type (Stacked)")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_exit_spend_grouped(df_breakdown: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped (side-by-side) bar chart: absolute entry spend by exit type per vintage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df_breakdown.pivot_table(
        index="Vintage", columns="ExitTypeLabel", values="entry_spend", aggfunc="sum", fill_value=0.0
    ).sort_index()

    vintages = pivot.index.to_numpy()
    labels = list(pivot.columns)
    n_types = len(labels)
    x = np.arange(len(vintages))
    width = 0.75 / max(1, n_types)

    fig, ax = plt.subplots(figsize=(10,5))
    for i, label in enumerate(labels):
        ax.bar(x + i*width - (n_types-1)*width/2, pivot[label].to_numpy(), width, label=label)

    ax.set_xticks(x, vintages, rotation=0)
    ax.set_xlabel("Vintage Year"); ax.set_ylabel("Entry spend (USD)")
    ax.set_title("Exited P2P Deals — Entry Spend by Exit Type (Grouped)")
    ax.legend(title="Exit Type", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()