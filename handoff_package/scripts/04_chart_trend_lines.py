"""
04_chart_trend_lines.py
=======================

Produces TP_Trend_Line.png and Total_Trend_Line.png: realization-rate trends
(by count and by entry value) across entry-year vintages.

  - TP_Trend_Line.png:    Take-Private (P2P) universe only.
  - Total_Trend_Line.png: Entire PE universe (Take-Private + Non-Take-Private
                           combined). NOTE: no script in the original project
                           produced this file directly -- it is reconstructed
                           here by combining the TP and Non-TP trend summaries
                           the same way analysis_exitdeal_nesteds.py combines
                           them for its own "Total" chart (sum N_total/N_exited/
                           V_total/V_exited per entry_year, then recompute the
                           percentages).

Reads:
    data/clean/pitchbook_master_clean.parquet      (full PE universe; see 01_clean_pitchbook_master.py)
    data/raw/pitchbook_public_2_private_all.xlsx   (take-private entries)
    data/clean/p2p_linked_master.csv               (TP entry->exit pairs; see 02_link_p2p_entry_exit.py)
    data/clean/non_tp_entry_exit_pairs.parquet      (Non-TP entry->exit pairs; provided as-is, see README)

Writes:
    output/figures/TP_Trend_Line.png
    output/figures/Total_Trend_Line.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCEL_BLUE = "#4472C4"
EXCEL_RED = "#C00000"

Y_RANGE = (2000, 2025)


# ============================================================
# Loading
# ============================================================

def clean_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def make_key(df: pd.DataFrame, name_col: str = "company_name_clean", date_col: str = "entry_date") -> pd.Series:
    """Stable matching key: cleaned_company_name|YYYY-MM-DD."""
    name_part = clean_name(df[name_col]) if name_col in df.columns else pd.Series([""] * len(df), index=df.index)

    if date_col in df.columns:
        date_part = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    else:
        date_part = pd.Series([""] * len(df), index=df.index)

    return name_part + "|" + date_part


def load_universe_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not p.exists():
        print(f"Missing universe file: {p}")
        return pd.DataFrame()

    df = pd.read_parquet(p).copy()

    if "deal_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
    elif "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        print("Universe file missing deal_date / entry_date.")
        return pd.DataFrame()

    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m"], errors="coerce")
    elif "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    else:
        df["deal_size_usd_m_entry"] = np.nan

    if "company_name_clean" not in df.columns:
        print("Universe file missing company_name_clean.")
        return pd.DataFrame()

    df["company_name_clean"] = clean_name(df["company_name_clean"])
    df["entry_year"] = df["entry_date"].dt.year
    return df


def load_tp_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "raw" / "pitchbook_public_2_private_all.xlsx"
    if not p.exists():
        print(f"Missing TP entries file: {p}")
        return pd.DataFrame()

    df = pd.read_excel(p).copy()

    if "Deal Date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    elif "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        print("TP entries file missing Deal Date / entry_date.")
        return pd.DataFrame()

    if "Deal Size" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["Deal Size"], errors="coerce")
    elif "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    else:
        df["deal_size_usd_m_entry"] = np.nan

    if "Companies" in df.columns:
        df["company_name_clean"] = clean_name(df["Companies"])
    elif "company_name_clean" in df.columns:
        df["company_name_clean"] = clean_name(df["company_name_clean"])
    else:
        print("TP entries file missing Companies / company_name_clean.")
        return pd.DataFrame()

    df["entry_year"] = df["entry_date"].dt.year
    return df


def load_exit_pairs(root: Path, filename: str) -> pd.DataFrame:
    p = root / "data" / "clean" / filename
    if not p.exists():
        print(f"Missing exit-pairs file: {p}")
        return pd.DataFrame()

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p).copy()
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p).copy()
    elif p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p).copy()
    else:
        print(f"Unsupported file type: {p}")
        return pd.DataFrame()

    rename_map = {
        "Exit Date": "exit_date", "Exit_Date": "exit_date", "deal_date_exit": "exit_date",
        "Entry Date": "entry_date", "Entry_Date": "entry_date", "deal_date_entry": "entry_date",
        "Deal Date": "entry_date", "deal_date": "entry_date",
        "Company": "company_name_clean", "Company Name": "company_name_clean",
        "Companies": "company_name_clean",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["company_name_clean", "entry_date", "exit_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Exit-pairs file {filename} missing columns: {missing}")
        return pd.DataFrame()

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["company_name_clean"] = clean_name(df["company_name_clean"])
    return df


# ============================================================
# Trend aggregation
# ============================================================

def build_trend_summary(entries: pd.DataFrame, pairs: pd.DataFrame, y_min: int, y_max: int) -> pd.DataFrame:
    """pct_N = N_exited / N_total * 100, pct_V = V_exited / V_total * 100."""
    if entries.empty:
        return pd.DataFrame()

    entries = entries.copy()
    pairs = pairs.copy()

    required_entry_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry"]
    if any(c not in entries.columns for c in required_entry_cols):
        print(f"Entries missing required columns: {required_entry_cols}")
        return pd.DataFrame()

    entries["company_name_clean"] = clean_name(entries["company_name_clean"])
    entries["entry_date"] = pd.to_datetime(entries["entry_date"], errors="coerce")
    entries["deal_size_usd_m_entry"] = pd.to_numeric(entries["deal_size_usd_m_entry"], errors="coerce")
    entries["entry_year"] = entries["entry_date"].dt.year

    key_cols = ["company_name_clean", "entry_date"]

    if not pairs.empty and all(c in pairs.columns for c in key_cols + ["exit_date"]):
        pairs["company_name_clean"] = clean_name(pairs["company_name_clean"])
        pairs["entry_date"] = pd.to_datetime(pairs["entry_date"], errors="coerce")
        pairs["exit_date"] = pd.to_datetime(pairs["exit_date"], errors="coerce")

        pairs_dedup = (
            pairs.dropna(subset=["exit_date"])
            .sort_values("exit_date", ascending=True)
            .drop_duplicates(subset=key_cols, keep="first")
        )
        merged = entries.merge(pairs_dedup[key_cols + ["exit_date"]], on=key_cols, how="left")
    else:
        merged = entries.copy()
        merged["exit_date"] = pd.NaT

    merged["has_exit"] = (
        merged["exit_date"].notna() & merged["entry_date"].notna()
        & (merged["exit_date"] > merged["entry_date"])
    ).fillna(False).astype(bool)

    subset = merged[(merged["entry_year"] >= y_min) & (merged["entry_year"] <= y_max)].copy()
    if subset.empty:
        return pd.DataFrame()

    grp = subset.groupby("entry_year", dropna=True)
    years = sorted(grp.groups.keys())

    df_out = pd.DataFrame({
        "entry_year": years,
        "N_total": grp.size().reindex(years).values,
        "N_exited": grp["has_exit"].sum().reindex(years).values,
        "V_total": grp["deal_size_usd_m_entry"].sum(min_count=1).reindex(years).values,
        "V_exited": (
            subset[subset["has_exit"]]
            .groupby("entry_year")["deal_size_usd_m_entry"]
            .sum(min_count=1).reindex(years).fillna(0).values
        ),
    })

    df_out["pct_N"] = (df_out["N_exited"] / df_out["N_total"].replace(0, np.nan)).fillna(0) * 100
    df_out["pct_V"] = (df_out["V_exited"] / df_out["V_total"].replace(0, np.nan)).fillna(0) * 100

    return df_out.sort_values("entry_year")


def combine_trend_summaries(*summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Combine TP + Non-TP trend summaries into an entire-universe ("Total") summary.

    Sums N_total/N_exited/V_total/V_exited by entry_year, then recomputes
    pct_N/pct_V -- the same approach analysis_exitdeal_nesteds.py uses for its
    "Total" chart, adapted here for the trend-line schema.
    """
    usable = [
        s[["entry_year", "N_total", "N_exited", "V_total", "V_exited"]].copy()
        for s in summaries if not s.empty
    ]
    if not usable:
        return pd.DataFrame()

    combined = pd.concat(usable, ignore_index=True)
    combined = (
        combined.groupby("entry_year", as_index=False)
        .agg(
            N_total=("N_total", "sum"),
            N_exited=("N_exited", "sum"),
            V_total=("V_total", lambda s: s.sum(min_count=1)),
            V_exited=("V_exited", lambda s: s.sum(min_count=1)),
        )
        .sort_values("entry_year").reset_index(drop=True)
    )

    combined["pct_N"] = np.where(combined["N_total"] > 0, (combined["N_exited"] / combined["N_total"]) * 100, 0.0)
    combined["pct_V"] = np.where(
        combined["V_total"].notna() & combined["V_total"].ne(0),
        (combined["V_exited"] / combined["V_total"]) * 100, 0.0,
    )
    return combined


# ============================================================
# Chart styling + rendering
# ============================================================

def _apply_house_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Garamond", "EB Garamond", "Cormorant Garamond", "Times New Roman", "DejaVu Serif"],
        "font.size": 13, "axes.titlesize": 19, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.title_fontsize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 1.0,
        "grid.color": "#EFEFEF", "grid.linestyle": "-", "grid.linewidth": 0.8,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })


def _finish_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trend_line_chart(summary: pd.DataFrame, title: str, outfile: Path, unit: str = "$M"):
    if summary.empty:
        print(f"Skipping chart because summary is empty: {title}")
        return

    _apply_house_style()

    years = summary["entry_year"].tolist()
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(16, 9))
    _finish_axes(ax)

    ax.plot(x, summary["pct_N"], color=EXCEL_BLUE, marker="o", linewidth=3,
            markersize=9, label="% Exited (by Count)", zorder=10)
    ax.plot(x, summary["pct_V"], color=EXCEL_RED, marker="s", linewidth=3,
            markersize=9, label=f"% Exited (by Entry Value {unit})", zorder=10)

    for i, (pn, pv) in enumerate(zip(summary["pct_N"], summary["pct_V"])):
        pn, pv = float(pn), float(pv)

        if abs(pn - pv) < 5:
            n_offset, n_va = 2.5, "bottom"
            v_offset, v_va = -4.5, "top"
        else:
            n_offset = -4.5 if pn > 92 else 2.5
            n_va = "top" if pn > 92 else "bottom"
            v_offset = -4.5 if pv > 92 else 2.5
            v_va = "top" if pv > 92 else "bottom"

        ax.text(i, pn + n_offset, f"{int(round(pn))}%", color=EXCEL_BLUE, ha="center",
                va=n_va, fontweight="bold", fontsize=11, zorder=20)
        ax.text(i, pv + v_offset, f"{int(round(pv))}%", color=EXCEL_RED, ha="center",
                va=v_va, fontweight="bold", fontsize=11, zorder=20)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Entry Year (Vintage)", fontsize=15, fontweight="bold", labelpad=15)
    ax.set_ylabel("Realization Rate (%)", fontsize=15, fontweight="bold", labelpad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(y)}" for y in years], rotation=90)

    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2, frameon=True,
                     facecolor="white", framealpha=1.0, edgecolor="#DDDDDD")
    leg.set_zorder(100)

    ax.set_title(title, fontsize=19, fontweight="bold", pad=75)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    univ = load_universe_entries(root)
    tp_entries = load_tp_entries(root)

    if univ.empty or tp_entries.empty:
        print("Missing required input data. Stopping.")
        return

    univ["__key"] = make_key(univ)
    tp_entries["__key"] = make_key(tp_entries)
    nontp_entries = univ[~univ["__key"].isin(set(tp_entries["__key"]))].copy()

    tp_pairs = load_exit_pairs(root, "p2p_linked_master.csv")
    nontp_pairs = load_exit_pairs(root, "non_tp_entry_exit_pairs.parquet")

    print(f"Universe rows:     {len(univ):,}")
    print(f"TP entry rows:     {len(tp_entries):,}")
    print(f"Non-TP entry rows: {len(nontp_entries):,}")

    tp_sum = build_trend_summary(tp_entries, tp_pairs, *Y_RANGE)
    nontp_sum = build_trend_summary(nontp_entries, nontp_pairs, *Y_RANGE)
    total_sum = combine_trend_summaries(tp_sum, nontp_sum)

    tp_output = figure_dir / "TP_Trend_Line.png"
    total_output = figure_dir / "Total_Trend_Line.png"

    plot_trend_line_chart(tp_sum, "Take-Private (P2P): Realization Trends", tp_output, unit="$M")
    plot_trend_line_chart(total_sum, "Entire Universe (All PE): Realization Trends", total_output, unit="$M")

    print(f"Generated: {tp_output}")
    print(f"Generated: {total_output}")


if __name__ == "__main__":
    main()
