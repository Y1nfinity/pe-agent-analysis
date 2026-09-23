from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATHING & DATA LOADING
# ============================================================

def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_name(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def make_key(df: pd.DataFrame, name_col: str = "company_name_clean", date_col: str = "entry_date") -> pd.Series:
    """
    Build a stable matching key:
        cleaned_company_name|YYYY-MM-DD

    This avoids errors from NaN/NaT/float values and prevents mismatches between
    '2020-01-01' and '2020-01-01 00:00:00'.
    """
    if name_col not in df.columns:
        name_part = pd.Series([""] * len(df), index=df.index)
    else:
        name_part = clean_name(df[name_col])

    if date_col not in df.columns:
        date_part = pd.Series([""] * len(df), index=df.index)
    else:
        date_part = (
            pd.to_datetime(df[date_col], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )

    return name_part + "|" + date_part


def load_universe_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "pitchbook_master_clean.parquet"

    if not p.exists():
        print(f"❌ Missing universe file: {p}")
        return pd.DataFrame()

    df = pd.read_parquet(p).copy()

    # Normalize columns
    if "deal_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
    elif "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        print("❌ Universe file missing deal_date / entry_date.")
        return pd.DataFrame()

    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m"], errors="coerce")
    elif "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    else:
        df["deal_size_usd_m_entry"] = np.nan

    if "company_name_clean" not in df.columns:
        print("❌ Universe file missing company_name_clean.")
        return pd.DataFrame()

    df["company_name_clean"] = clean_name(df["company_name_clean"])
    df["entry_year"] = df["entry_date"].dt.year

    return df


def load_tp_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "raw" / "pitchbook_public_2_private_all.xlsx"

    if not p.exists():
        print(f"❌ Missing TP entries file: {p}")
        return pd.DataFrame()

    df = pd.read_excel(p).copy()

    if "Deal Date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    elif "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        print("❌ TP entries file missing Deal Date / entry_date.")
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
        print("❌ TP entries file missing Companies / company_name_clean.")
        return pd.DataFrame()

    df["entry_year"] = df["entry_date"].dt.year

    return df


def load_exit_pairs(root: Path, filename: str) -> pd.DataFrame:
    p = root / "data" / "clean" / filename

    if not p.exists():
        print(f"⚠️ Missing exit-pairs file: {p}")
        return pd.DataFrame()

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p).copy()
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p).copy()
    elif p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p).copy()
    else:
        print(f"⚠️ Unsupported file type: {p}")
        return pd.DataFrame()

    rename_map = {
        "Exit Date": "exit_date",
        "Exit_Date": "exit_date",
        "deal_date_exit": "exit_date",

        "Entry Date": "entry_date",
        "Entry_Date": "entry_date",
        "deal_date_entry": "entry_date",
        "Deal Date": "entry_date",
        "deal_date": "entry_date",

        "Company": "company_name_clean",
        "Company Name": "company_name_clean",
        "Companies": "company_name_clean",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["company_name_clean", "entry_date", "exit_date"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"⚠️ Exit-pairs file {filename} missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["company_name_clean"] = clean_name(df["company_name_clean"])

    return df


# ============================================================
# 2. TREND AGGREGATION ENGINE
# ============================================================

def build_trend_summary(entries: pd.DataFrame, pairs: pd.DataFrame, y_min: int, y_max: int) -> pd.DataFrame:
    """
    Calculates realization rates:
        pct_N = N_exited / N_total * 100
        pct_V = V_exited / V_total * 100
    """
    if entries.empty:
        return pd.DataFrame()

    entries = entries.copy()
    pairs = pairs.copy()

    required_entry_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry"]
    missing_entry_cols = [c for c in required_entry_cols if c not in entries.columns]

    if missing_entry_cols:
        print(f"⚠️ Entries missing required columns: {missing_entry_cols}")
        print(f"Available columns: {list(entries.columns)}")
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
            pairs
            .dropna(subset=["exit_date"])
            .sort_values("exit_date", ascending=True)
            .drop_duplicates(subset=key_cols, keep="first")
        )

        merged = entries.merge(
            pairs_dedup[key_cols + ["exit_date"]],
            on=key_cols,
            how="left"
        )
    else:
        merged = entries.copy()
        merged["exit_date"] = pd.NaT

    merged["has_exit"] = (
        merged["exit_date"].notna()
        & merged["entry_date"].notna()
        & (merged["exit_date"] > merged["entry_date"])
    ).fillna(False).astype(bool)

    subset = merged[
        (merged["entry_year"] >= y_min)
        & (merged["entry_year"] <= y_max)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    grp = subset.groupby("entry_year", dropna=True)
    years = sorted(grp.groups.keys())

    df_out = pd.DataFrame(
        {
            "entry_year": years,
            "N_total": grp.size().reindex(years).values,
            "N_exited": grp["has_exit"].sum().reindex(years).values,
            "V_total": grp["deal_size_usd_m_entry"].sum(min_count=1).reindex(years).values,
            "V_exited": (
                subset[subset["has_exit"]]
                .groupby("entry_year")["deal_size_usd_m_entry"]
                .sum(min_count=1)
                .reindex(years)
                .fillna(0)
                .values
            ),
        }
    )

    df_out["pct_N"] = (
        df_out["N_exited"] / df_out["N_total"].replace(0, np.nan)
    ).fillna(0) * 100

    df_out["pct_V"] = (
        df_out["V_exited"] / df_out["V_total"].replace(0, np.nan)
    ).fillna(0) * 100

    return df_out.sort_values("entry_year")


# ============================================================
# 3. GARAMOND HOUSE STYLE
# ============================================================

EXCEL_BLUE = "#4472C4"
EXCEL_RED = "#C00000"


def _apply_house_style():
    """
    Garamond-based house style for all line charts.
    """
    plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Garamond",
                "EB Garamond",
                "Cormorant Garamond",
                "Times New Roman",
                "DejaVu Serif",
            ],

            # Garamond reads slightly smaller/lighter, so sizes are bumped up.
            "font.size": 13,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,

            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",

            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,

            "grid.color": "#EFEFEF",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,

            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def _finish_axes(ax):
    """
    Excel-style cleanup:
    - horizontal gridlines only
    - no top/right spines
    - clean white plot area
    """
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 4. LINE CHART VISUALIZATION
# ============================================================

def plot_trend_line_chart(summary: pd.DataFrame, title: str, outfile: Path, unit: str = "$M"):
    if summary.empty:
        print(f"⚠️ Skipping chart because summary is empty: {title}")
        return

    _apply_house_style()

    years = summary["entry_year"].tolist()
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(16, 9))
    _finish_axes(ax)

    # Lines and markers
    ax.plot(
        x,
        summary["pct_N"],
        color=EXCEL_BLUE,
        marker="o",
        linewidth=3,
        markersize=9,
        label="% Exited (by Count)",
        zorder=10,
    )

    ax.plot(
        x,
        summary["pct_V"],
        color=EXCEL_RED,
        marker="s",
        linewidth=3,
        markersize=9,
        label=f"% Exited (by Entry Value {unit})",
        zorder=10,
    )

    # Smart data-label placement
    for i, (pn, pv) in enumerate(zip(summary["pct_N"], summary["pct_V"])):
        pn = float(pn)
        pv = float(pv)

        # If points are close, separate labels vertically.
        if abs(pn - pv) < 5:
            n_offset, n_va = 2.5, "bottom"
            v_offset, v_va = -4.5, "top"
        else:
            # Keep labels inside top of chart.
            n_offset = -4.5 if pn > 92 else 2.5
            n_va = "top" if pn > 92 else "bottom"

            v_offset = -4.5 if pv > 92 else 2.5
            v_va = "top" if pv > 92 else "bottom"

        ax.text(
            i,
            pn + n_offset,
            f"{int(round(pn))}%",
            color=EXCEL_BLUE,
            ha="center",
            va=n_va,
            fontweight="bold",
            fontsize=11,
            zorder=20,
        )

        ax.text(
            i,
            pv + v_offset,
            f"{int(round(pv))}%",
            color=EXCEL_RED,
            ha="center",
            va=v_va,
            fontweight="bold",
            fontsize=11,
            zorder=20,
        )

    # Axes
    ax.set_ylim(0, 100)

    ax.set_xlabel(
        "Entry Year (Vintage)",
        fontsize=15,
        fontweight="bold",
        labelpad=15,
    )

    ax.set_ylabel(
        "Realization Rate (%)",
        fontsize=15,
        fontweight="bold",
        labelpad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(y)}" for y in years], rotation=90)

    # Legend
    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )
    leg.set_zorder(100)

    ax.set_title(
        title,
        fontsize=19,
        fontweight="bold",
        pad=75,
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    ROOT = find_project_root()
    Y_RANGE = (2000, 2025)

    FIG_DIR = ROOT / "reports" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    univ = load_universe_entries(ROOT)
    tp_entries = load_tp_entries(ROOT)

    if univ.empty:
        print("❌ Universe entries are empty. Stopping.")
        return

    if tp_entries.empty:
        print("❌ TP entries are empty. Stopping.")
        return

    # Safe matching keys
    univ["__key"] = make_key(univ)
    tp_entries["__key"] = make_key(tp_entries)

    nontp_entries = univ[~univ["__key"].isin(set(tp_entries["__key"]))].copy()

    tp_pairs = load_exit_pairs(ROOT, "p2p_linked_master.csv")
    nontp_pairs = load_exit_pairs(ROOT, "non_tp_entry_exit_pairs.parquet")

    print("\n========== DATA CHECK ==========")
    print(f"Universe rows:       {len(univ):,}")
    print(f"TP entry rows:       {len(tp_entries):,}")
    print(f"Non-TP entry rows:   {len(nontp_entries):,}")
    print(f"TP pair rows:        {len(tp_pairs):,}")
    print(f"Non-TP pair rows:    {len(nontp_pairs):,}")
    print("================================\n")

    # Build summaries
    tp_sum = build_trend_summary(tp_entries, tp_pairs, *Y_RANGE)
    nontp_sum = build_trend_summary(nontp_entries, nontp_pairs, *Y_RANGE)

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    from datetime import datetime

    print(f"\nRun started: {datetime.now()}")
    print(f"Executing script: {Path(__file__).resolve()}")
    print(f"Project root: {ROOT.resolve()}")
    print(f"Figure folder: {FIG_DIR.resolve()}")
    print(f"Analysis range: {Y_RANGE}")

    print("\nSummary sizes:")
    print(f"TP summary rows: {len(tp_sum)}")
    print(f"Non-TP summary rows: {len(nontp_sum)}")

    print("\nTP summary:")
    print(tp_sum)

    print("\nNon-TP summary:")
    print(nontp_sum)

    tp_output = FIG_DIR / "TP_Trend_Line.png"
    nontp_output = FIG_DIR / "NonTP_Trend_Line.png"

    # Plot TP line chart
    plot_trend_line_chart(
        tp_sum,
        "Take-Private (P2P): Realization Trends",
        tp_output,
        unit="$M",
    )

    # Plot Non-TP line chart
    plot_trend_line_chart(
        nontp_sum,
        "Non-Take-Private: Realization Trends",
        nontp_output,
        unit="$B",
    )

    print("\n========== OUTPUT CHECK ==========")
    print(f"TP graph exists: {tp_output.exists()}")
    print(f"TP graph path:   {tp_output.resolve()}")

    print(f"Non-TP graph exists: {nontp_output.exists()}")
    print(f"Non-TP graph path:   {nontp_output.resolve()}")

    if tp_output.exists():
        print(
            "TP graph modified:",
            datetime.fromtimestamp(tp_output.stat().st_mtime)
        )

    if nontp_output.exists():
        print(
            "Non-TP graph modified:",
            datetime.fromtimestamp(nontp_output.stat().st_mtime)
        )

    print("==================================")
    print("✅ Trend line charts generated.")
    print(f"📁 Saved figures to: {FIG_DIR.resolve()}")



if __name__ == "__main__":
    main()