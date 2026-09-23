from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATHING & ROBUST DATA LOADING
# ============================================================

def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def load_data_flexible(root: Path, folder: str, filename_base: str) -> pd.DataFrame:
    """Checks for .parquet, .xlsx, and .csv versions of a file and standardizes columns."""
    folder_path = root / "data" / folder
    for ext in [".parquet", ".xlsx", ".csv"]:
        p = folder_path / f"{filename_base}{ext}"
        if not p.exists():
            continue

        if ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext == ".xlsx":
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

        # Standardize Column Names Immediately
        rename_map = {
            "Exit Date": "exit_date",
            "Exit_Date": "exit_date",
            "deal_date_exit": "exit_date",
            "Entry Date": "entry_date",
            "Entry_Date": "entry_date",
            "deal_date_entry": "entry_date",
            "Deal Date": "entry_date",
            "deal_date": "entry_date",
            "Deal Size": "deal_size_usd_m_entry",
            "deal_size_usd_m": "deal_size_usd_m_entry",
            "V_entry": "deal_size_usd_m_entry",
            "Companies": "company_name_clean",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Type correction
        if "entry_date" in df.columns:
            df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
        if "exit_date" in df.columns:
            df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
        if "company_name_clean" in df.columns:
            df["company_name_clean"] = clean_name(df["company_name_clean"])

        return df

    return pd.DataFrame()


# ============================================================
# 2. VALIDATED AGGREGATION ENGINE
# ============================================================

def build_summary(entries: pd.DataFrame, pairs: pd.DataFrame, y_min: int, y_max: int) -> pd.DataFrame:
    if entries.empty or "entry_date" not in entries.columns:
        return pd.DataFrame()

    key_cols = ["company_name_clean", "entry_date"]

    # Defensive check for exit_date column in pairs
    if not pairs.empty and "exit_date" in pairs.columns:
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
        merged["exit_date"].notna() & (merged["exit_date"] > merged["entry_date"])
    ).fillna(False).infer_objects().astype(bool)

    if "entry_year" not in merged.columns:
        merged["entry_year"] = merged["entry_date"].dt.year

    subset = merged[(merged["entry_year"] >= y_min) & (merged["entry_year"] <= y_max)].copy()
    if subset.empty:
        return pd.DataFrame()

    grp = subset.groupby("entry_year")
    years = list(grp.groups.keys())

    df_out = pd.DataFrame(
        {
            "entry_year": years,
            "N_total": grp.size().values,
            "N_exited": grp["has_exit"].sum().values,
            "V_total": grp["deal_size_usd_m_entry"].sum(min_count=1).values,
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
    df_out["pct_N"] = (df_out["N_exited"] / df_out["N_total"]) * 100
    df_out["pct_V"] = (df_out["V_exited"] / df_out["V_total"].replace(0, np.nan)).fillna(0) * 100
    return df_out.sort_values("entry_year")


# ============================================================
# 3. VISUALIZATION FUNCTIONS (Times New Roman + Excel Colors)
# ============================================================

# Excel theme colors
EXCEL_BLUE = "#4472C4"
EXCEL_RED = "#C00000"


def _apply_house_style():
    """
    Forces a consistent "Excel-like" look:
    - Times New Roman (best-effort; falls back to other serif fonts if missing)
    - White background
    - Light horizontal gridlines only
    - Black axes/spines
    """
    plt.style.use("default")
    plt.rcParams.update(
        {
            # Font: force serif family + Times New Roman preference
            "font.family": "serif",
            "font.serif": ["Garamond"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            # Backgrounds
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            # Axes + grid
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.color": "#EFEFEF",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            # Keep colors from being influenced by style sheets
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def _finish_axes(ax):
    """Excel-ish cleanup: y-grid only, remove top/right spines, keep plot area centered/clean."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_percentage_style(summary: pd.DataFrame, title: str, outfile: Path):
    _apply_house_style()

    years = summary["entry_year"].tolist()
    x = np.arange(len(summary))
    width = 0.42

    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()

    # Left axis styling
    _finish_axes(ax1)
    ax2.grid(False)

    rects1 = ax1.bar(
        x - width / 2,
        summary["pct_N"],
        width,
        color=EXCEL_BLUE,
        label="% Exited (by Count)",
        zorder=10,
    )
    rects2 = ax2.bar(
        x + width / 2,
        summary["pct_V"],
        width,
        color=EXCEL_RED,
        label="% Exited (by Value $M)",
        zorder=10,
    )

    def add_labels(ax, rects):
        for rect in rects:
            h = float(rect.get_height())
            if h < 0.5:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 0.3,
                f"{int(round(h))}\n%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=20,
            )

    add_labels(ax1, rects1)
    add_labels(ax2, rects2)

    for ax in (ax1, ax2):
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)

    ax1.set_xlabel("Entry Year (Vintage)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(y)}" for y in years], rotation=90)

    # Legend (combined) – Excel-ish, simple
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )

    ax1.set_title(title, fontweight="bold", pad=75)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_nested_style(summary: pd.DataFrame, title: str, outfile: Path, scale_to_billions: bool):
    _apply_house_style()

    years = summary["entry_year"].tolist()
    x = np.arange(len(summary))
    width = 0.42

    scale = 1000.0 if scale_to_billions else 1.0
    unit_label = "($B)" if scale_to_billions else "($M)"

    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()

    _finish_axes(ax1)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    # N (left axis) - outlined total, filled exited
    ax1.bar(
        x - width / 2,
        summary["N_total"],
        width,
        edgecolor=EXCEL_BLUE,
        color="none",
        lw=1.6,
        zorder=10,
        label="Total Entries (N)",
    )
    rects1 = ax1.bar(
        x - width / 2,
        summary["N_exited"],
        width,
        color=EXCEL_BLUE,
        alpha=0.65,
        zorder=11,
        label="Exited (N)",
    )

    # V (right axis) - outlined total, filled realized
    ax2.bar(
        x + width / 2,
        summary["V_total"] / scale,
        width,
        edgecolor=EXCEL_RED,
        color="none",
        lw=1.6,
        zorder=10,
        label=f"Total Value {unit_label}",
    )
    rects2 = ax2.bar(
        x + width / 2,
        summary["V_exited"] / scale,
        width,
        color=EXCEL_RED,
        alpha=0.55,
        zorder=11,
        label=f"Realized {unit_label}",
    )

    def add_nested_top_labels(ax, rects_fill, totals):
        for rect, tot in zip(rects_fill, totals):
            h_fill = float(rect.get_height())
            tot = float(tot)
            if h_fill <= 0 or tot <= 0:
                continue
            pct = (h_fill / tot) * 100
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h_fill + (tot * 0.01),
                f"{int(round(pct))}\n%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=20,
            )

    add_nested_top_labels(ax1, rects1, summary["N_total"])
    add_nested_top_labels(ax2, rects2, summary["V_total"] / scale)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(y)}" for y in years], rotation=90)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#DDDDDD",
    )

    ax1.set_title(title, fontweight="bold", pad=75)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ============================================================
# 4. MAIN DRIVER
# ============================================================

def main():
    ROOT = find_project_root()
    Y_RANGE = (2000, 2022)

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data with standard names
    univ_master = load_data_flexible(ROOT, "clean", "pitchbook_master_clean")
    tp_entries = load_data_flexible(ROOT, "raw", "pitchbook_public_2_private_all")
    tp_pairs = load_data_flexible(ROOT, "clean", "p2p_linked_master")
    nontp_pairs = load_data_flexible(ROOT, "clean", "non_tp_entry_exit_pairs")

    if univ_master.empty:
        print(f"❌ Could not load universe master from {ROOT}")
        return

    # 2. Derive Non-TP entries
    univ_master["__key"] = univ_master[["company_name_clean", "entry_date"]].astype(str).agg("|".join, axis=1)
    if not tp_entries.empty:
        tp_entries["__key"] = (
                tp_entries["company_name_clean"].fillna("").astype(str)
                + "|"
                + tp_entries["entry_date"].fillna("").astype(str)
        )
        nontp_entries = univ_master[~univ_master["__key"].isin(set(tp_entries["__key"]))].copy()
    else:
        nontp_entries = univ_master.copy()

    # 3. Calculate Sub-Summaries
    tp_sum = build_summary(tp_entries, tp_pairs, *Y_RANGE)
    nontp_sum = build_summary(nontp_entries, nontp_pairs, *Y_RANGE)

    # 4. MATH COMBINATION: Total = TP + NonTP
    if not tp_sum.empty and not nontp_sum.empty:
        total_sum = tp_sum.copy()
        raw_cols = ["N_total", "N_exited", "V_total", "V_exited"]
        total_sum[raw_cols] = tp_sum[raw_cols].add(nontp_sum[raw_cols], fill_value=0)
        total_sum["pct_N"] = (total_sum["N_exited"] / total_sum["N_total"]) * 100
        total_sum["pct_V"] = (total_sum["V_exited"] / total_sum["V_total"].replace(0, np.nan)).fillna(0) * 100
    else:
        total_sum = nontp_sum if tp_sum.empty else tp_sum

    # 5. Execute Plotting Tasks
    tasks = [
        (total_sum, "Entire Universe (All PE)", "Total", True),
        (tp_sum, "Take-Private (P2P)", "TP", False),
        (nontp_sum, "Non-Take-Private (Other PE)", "NonTP", True),
    ]

    for summary, title, base, scale in tasks:
        if summary.empty:
            continue
        plot_percentage_style(summary, f"{title}: Realization Rates", fig_dir / f"{base}_Percentage.png")
        plot_nested_style(summary, f"{title}: Volume & Realization", fig_dir / f"{base}_Nested.png", scale)
        print(f"✅ Generated charts for {base}")

    print(f"\n📁 Saved figures to: {fig_dir.resolve()}\n")


if __name__ == "__main__":
    main()