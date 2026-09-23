from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHING
# ============================================================

def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    out_root = root / "outputs" / "pairing report"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, table_dir


# ============================================================
# 2. HOUSE STYLE
# ============================================================

EXCEL_BLUE = "#4472C4"
EXCEL_RED = "#C00000"
EXCEL_GREEN = "#70AD47"
EXCEL_ORANGE = "#ED7D31"


def apply_house_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.color": "#EAEAEA",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def finish_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 3. FLEXIBLE LOADERS
# ============================================================

def load_data_flexible(root: Path, folder: str, filename_base: str) -> pd.DataFrame:
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

        return df

    return pd.DataFrame()


# ============================================================
# 4. STANDARDIZATION
# ============================================================

def clean_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.strip()
    )


def standardize_entries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    rename_map = {
        "Companies": "company_name_clean",
        "company_name": "company_name_clean",
        "Company": "company_name_clean",
        "Entry Date": "entry_date",
        "Entry_Date": "entry_date",
        "Deal Date": "entry_date",
        "deal_date": "entry_date",
        "entry_date": "entry_date",
        "Deal Size": "deal_size_usd_m_entry",
        "deal_size_usd_m": "deal_size_usd_m_entry",
        "V_entry": "deal_size_usd_m_entry",
        "Deal Type": "deal_type",
        "Deal Type 2": "deal_type_2",
        "Deal Type 3": "deal_type_3",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "company_name_clean" not in df.columns:
        raise ValueError("Entries dataset must contain a company/name column.")
    df["company_name_clean"] = clean_name(df["company_name_clean"])

    if "entry_date" not in df.columns:
        raise ValueError("Entries dataset must contain an entry-date column.")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["entry_year"] = df["entry_date"].dt.year

    if "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    else:
        df["deal_size_usd_m_entry"] = np.nan

    for c in ["deal_type", "deal_type_2", "deal_type_3"]:
        if c not in df.columns:
            df[c] = np.nan

    keep_cols = [
        "company_name_clean",
        "entry_date",
        "entry_year",
        "deal_size_usd_m_entry",
        "deal_type",
        "deal_type_2",
        "deal_type_3",
    ]
    return df[keep_cols].copy()


def standardize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    rename_map = {
        "Companies": "company_name_clean",
        "company_name": "company_name_clean",
        "Company": "company_name_clean",
        "company_name_clean": "company_name_clean",
        "Entry Date": "entry_date",
        "Entry_Date": "entry_date",
        "deal_date_entry": "entry_date",
        "entry_date": "entry_date",
        "Exit Date": "exit_date",
        "Exit_Date": "exit_date",
        "deal_date_exit": "exit_date",
        "exit_date": "exit_date",
        "Deal Size": "deal_size_usd_m_entry",
        "V_entry": "deal_size_usd_m_entry",
        "deal_size_usd_m_entry": "deal_size_usd_m_entry",
        "deal_size_usd_m": "deal_size_usd_m_entry",
        "V_exit": "deal_size_usd_m_exit",
        "deal_size_usd_m_exit": "deal_size_usd_m_exit",
        "ExitCategory": "exit_category",
        "exit_category_exit": "exit_category",
        "exit_category": "exit_category",
        "Deal Type": "deal_type",
        "deal_type_entry": "deal_type",
        "holding_period_years": "holding_period_years",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "company_name_clean" not in df.columns:
        raise ValueError("Pairs dataset must contain a company/name column.")
    df["company_name_clean"] = clean_name(df["company_name_clean"])

    if "entry_date" not in df.columns:
        raise ValueError("Pairs dataset must contain an entry-date column.")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    else:
        df["exit_date"] = pd.NaT

    if "deal_size_usd_m_entry" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")
    else:
        df["deal_size_usd_m_entry"] = np.nan

    if "deal_size_usd_m_exit" in df.columns:
        df["deal_size_usd_m_exit"] = pd.to_numeric(df["deal_size_usd_m_exit"], errors="coerce")
    else:
        df["deal_size_usd_m_exit"] = np.nan

    if "holding_period_years" not in df.columns:
        df["holding_period_years"] = (
            (df["exit_date"] - df["entry_date"]).dt.days / 365.25
        )

    if "exit_category" not in df.columns:
        df["exit_category"] = "Unknown"

    if "deal_type" not in df.columns:
        df["deal_type"] = np.nan

    df["entry_year"] = df["entry_date"].dt.year
    df["exit_year"] = df["exit_date"].dt.year

    keep_cols = [
        "company_name_clean",
        "entry_date",
        "entry_year",
        "exit_date",
        "exit_year",
        "deal_size_usd_m_entry",
        "deal_size_usd_m_exit",
        "deal_type",
        "exit_category",
        "holding_period_years",
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols].copy()


# ============================================================
# 5. DIAGNOSTIC BASE TABLE
# ============================================================

def build_pairing_base(entries: pd.DataFrame, pairs: pd.DataFrame, recent_cutoff_year: int = 2020) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()

    key_cols = ["company_name_clean", "entry_date"]

    if pairs.empty:
        base = entries.copy()
        base["exit_date"] = pd.NaT
        base["exit_year"] = np.nan
        base["deal_size_usd_m_exit"] = np.nan
        base["exit_category"] = np.nan
        base["holding_period_years"] = np.nan
    else:
        pairs_dedup = (
            pairs.dropna(subset=["entry_date"])
            .sort_values(["company_name_clean", "entry_date", "exit_date"])
            .drop_duplicates(subset=key_cols, keep="first")
        )

        merge_cols = key_cols + [
            "exit_date",
            "exit_year",
            "deal_size_usd_m_exit",
            "exit_category",
            "holding_period_years",
        ]

        base = entries.merge(
            pairs_dedup[merge_cols],
            on=key_cols,
            how="left",
        )

    base["matched"] = (
        base["exit_date"].notna() & (base["exit_date"] > base["entry_date"])
    ).fillna(False)

    base["unmatched"] = ~base["matched"]

    if "deal_size_usd_m_entry" not in base.columns:
        base["deal_size_usd_m_entry"] = np.nan

    base["entry_size_missing"] = base["deal_size_usd_m_entry"].isna()

    # Vintage censoring diagnostic
    base["recent_vintage_flag"] = base["entry_year"] >= recent_cutoff_year
    base["older_vintage_flag"] = base["entry_year"] < recent_cutoff_year

    base["unmatched_recent_vintage"] = base["unmatched"] & base["recent_vintage_flag"]
    base["unmatched_older_vintage"] = base["unmatched"] & base["older_vintage_flag"]

    # Size buckets
    base["size_bucket"] = pd.cut(
        base["deal_size_usd_m_entry"],
        bins=[-np.inf, 100, 500, 1000, np.inf],
        labels=["<100", "100-500", "500-1000", "1000+"],
    )
    base["size_bucket"] = base["size_bucket"].astype(object).where(
        ~base["deal_size_usd_m_entry"].isna(), "Missing"
    )

    return base


# ============================================================
# 6. SUMMARY TABLES
# ============================================================

def build_overall_summary(base: pd.DataFrame) -> pd.DataFrame:
    total_entries = len(base)
    matched_entries = int(base["matched"].sum())
    unmatched_entries = int(base["unmatched"].sum())

    value_total = base["deal_size_usd_m_entry"].sum(min_count=1)
    value_matched = base.loc[base["matched"], "deal_size_usd_m_entry"].sum(min_count=1)
    value_unmatched = base.loc[base["unmatched"], "deal_size_usd_m_entry"].sum(min_count=1)

    pairing_rate_n = (matched_entries / total_entries * 100) if total_entries else 0.0
    pairing_rate_v = (value_matched / value_total * 100) if pd.notna(value_total) and value_total != 0 else np.nan

    out = pd.DataFrame(
        [
            {"metric": "total_entries", "value": total_entries},
            {"metric": "matched_entries", "value": matched_entries},
            {"metric": "unmatched_entries", "value": unmatched_entries},
            {"metric": "pairing_rate_count_pct", "value": pairing_rate_n},
            {"metric": "entry_value_total_usd_m", "value": value_total},
            {"metric": "entry_value_matched_usd_m", "value": value_matched},
            {"metric": "entry_value_unmatched_usd_m", "value": value_unmatched},
            {"metric": "pairing_rate_value_pct", "value": pairing_rate_v},
            {"metric": "older_vintage_unmatched", "value": int(base["unmatched_older_vintage"].sum())},
            {"metric": "recent_vintage_unmatched", "value": int(base["unmatched_recent_vintage"].sum())},
            {"metric": "median_entry_size_matched", "value": base.loc[base["matched"], "deal_size_usd_m_entry"].median()},
            {"metric": "median_entry_size_unmatched", "value": base.loc[base["unmatched"], "deal_size_usd_m_entry"].median()},
            {"metric": "mean_entry_size_matched", "value": base.loc[base["matched"], "deal_size_usd_m_entry"].mean()},
            {"metric": "mean_entry_size_unmatched", "value": base.loc[base["unmatched"], "deal_size_usd_m_entry"].mean()},
        ]
    )
    return out


def build_pairing_by_year(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    grouped = base.groupby("entry_year")

    out = pd.DataFrame(
        {
            "entry_year": grouped.size().index,
            "total_entries": grouped.size().values,
            "matched_entries": grouped["matched"].sum().values,
            "unmatched_entries": grouped["unmatched"].sum().values,
            "entry_value_total_usd_m": grouped["deal_size_usd_m_entry"].sum(min_count=1).values,
            "entry_value_matched_usd_m": (
                base.loc[base["matched"]]
                .groupby("entry_year")["deal_size_usd_m_entry"]
                .sum(min_count=1)
                .reindex(grouped.size().index)
                .fillna(0)
                .values
            ),
        }
    )

    out["pairing_rate_count_pct"] = out["matched_entries"] / out["total_entries"] * 100
    out["pairing_rate_value_pct"] = (
        out["entry_value_matched_usd_m"] / out["entry_value_total_usd_m"].replace(0, np.nan) * 100
    )

    out["unmatched_recent_vintage"] = (
        base.loc[base["unmatched_recent_vintage"]]
        .groupby("entry_year")
        .size()
        .reindex(out["entry_year"])
        .fillna(0)
        .astype(int)
        .values
    )

    out["unmatched_older_vintage"] = (
        base.loc[base["unmatched_older_vintage"]]
        .groupby("entry_year")
        .size()
        .reindex(out["entry_year"])
        .fillna(0)
        .astype(int)
        .values
    )

    return out.sort_values("entry_year")


def build_pairing_by_type(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty or "deal_type" not in base.columns:
        return pd.DataFrame()

    tmp = base.copy()
    tmp["deal_type"] = tmp["deal_type"].astype(str)

    grouped = tmp.groupby("deal_type")

    out = pd.DataFrame(
        {
            "deal_type": grouped.size().index,
            "total_entries": grouped.size().values,
            "matched_entries": grouped["matched"].sum().values,
            "unmatched_entries": grouped["unmatched"].sum().values,
            "entry_value_total_usd_m": grouped["deal_size_usd_m_entry"].sum(min_count=1).values,
            "entry_value_matched_usd_m": (
                tmp.loc[tmp["matched"]]
                .groupby("deal_type")["deal_size_usd_m_entry"]
                .sum(min_count=1)
                .reindex(grouped.size().index)
                .fillna(0)
                .values
            ),
        }
    )

    out["pairing_rate_count_pct"] = out["matched_entries"] / out["total_entries"] * 100
    out["pairing_rate_value_pct"] = (
        out["entry_value_matched_usd_m"] / out["entry_value_total_usd_m"].replace(0, np.nan) * 100
    )

    out["median_entry_size"] = grouped["deal_size_usd_m_entry"].median().values
    out["missing_size_pct"] = grouped["entry_size_missing"].mean().values * 100

    return out.sort_values(["matched_entries", "total_entries"], ascending=[False, False])


def build_pairing_by_size_bucket(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    grouped = base.groupby("size_bucket", dropna=False)

    out = pd.DataFrame(
        {
            "size_bucket": grouped.size().index.astype(str),
            "total_entries": grouped.size().values,
            "matched_entries": grouped["matched"].sum().values,
            "unmatched_entries": grouped["unmatched"].sum().values,
        }
    )
    out["pairing_rate_count_pct"] = out["matched_entries"] / out["total_entries"] * 100
    return out


def build_exit_type_distribution(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    matched = base.loc[base["matched"]].copy()
    if matched.empty:
        return pd.DataFrame()

    matched["exit_category"] = matched["exit_category"].fillna("Unknown").astype(str)

    out = (
        matched.groupby("exit_category")
        .agg(
            exits_n=("matched", "size"),
            entry_value_usd_m=("deal_size_usd_m_entry", "sum"),
            exit_value_usd_m=("deal_size_usd_m_exit", "sum"),
            mean_holding_period_years=("holding_period_years", "mean"),
            median_holding_period_years=("holding_period_years", "median"),
        )
        .reset_index()
        .sort_values("exits_n", ascending=False)
    )

    total_exits = out["exits_n"].sum()
    total_entry_value = out["entry_value_usd_m"].sum(min_count=1)

    out["fraction_by_number_pct"] = out["exits_n"] / total_exits * 100 if total_exits else np.nan
    out["fraction_by_entry_value_pct"] = (
        out["entry_value_usd_m"] / total_entry_value * 100 if pd.notna(total_entry_value) and total_entry_value != 0 else np.nan
    )

    return out


def build_holding_period_summary(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    matched = base.loc[base["matched"]].copy()
    if matched.empty:
        return pd.DataFrame()

    hp = matched["holding_period_years"].dropna()

    out = pd.DataFrame(
        [
            {"metric": "n_matched_with_holding_period", "value": len(hp)},
            {"metric": "mean_holding_period_years", "value": hp.mean()},
            {"metric": "median_holding_period_years", "value": hp.median()},
            {"metric": "std_holding_period_years", "value": hp.std()},
            {"metric": "p25_holding_period_years", "value": hp.quantile(0.25)},
            {"metric": "p75_holding_period_years", "value": hp.quantile(0.75)},
            {"metric": "min_holding_period_years", "value": hp.min()},
            {"metric": "max_holding_period_years", "value": hp.max()},
        ]
    )
    return out


def build_unmatched_older_vintages(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    out = base.loc[base["unmatched_older_vintage"]].copy()
    out = out.sort_values(["entry_year", "deal_size_usd_m_entry"], ascending=[True, False])
    return out


def build_matched_vs_unmatched_size_summary(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    rows = []
    for label, mask in [("Matched", base["matched"]), ("Unmatched", base["unmatched"])]:
        sub = base.loc[mask, "deal_size_usd_m_entry"]
        rows.append(
            {
                "group": label,
                "n": int(mask.sum()),
                "mean_entry_size_usd_m": sub.mean(),
                "median_entry_size_usd_m": sub.median(),
                "missing_size_pct": base.loc[mask, "entry_size_missing"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 7. SAVE TABLES
# ============================================================

def save_tables(
    overall: pd.DataFrame,
    by_year: pd.DataFrame,
    by_type: pd.DataFrame,
    by_size_bucket: pd.DataFrame,
    exit_types: pd.DataFrame,
    holding_summary: pd.DataFrame,
    unmatched_older: pd.DataFrame,
    matched_vs_unmatched_size: pd.DataFrame,
    base: pd.DataFrame,
    table_dir: Path,
) -> None:
    overall.to_csv(table_dir / "pairing_overall_summary.csv", index=False)
    by_year.to_csv(table_dir / "pairing_by_year.csv", index=False)
    by_type.to_csv(table_dir / "pairing_by_type.csv", index=False)
    by_size_bucket.to_csv(table_dir / "pairing_by_size_bucket.csv", index=False)
    exit_types.to_csv(table_dir / "matched_exit_type_distribution.csv", index=False)
    holding_summary.to_csv(table_dir / "matched_holding_period_summary.csv", index=False)
    unmatched_older.to_csv(table_dir / "unmatched_older_vintages.csv", index=False)
    matched_vs_unmatched_size.to_csv(table_dir / "matched_vs_unmatched_size_summary.csv", index=False)
    base.to_csv(table_dir / "pairing_diagnostic_base.csv", index=False)


# ============================================================
# 8. FIGURES
# ============================================================

def plot_pairing_rate_by_year(by_year: pd.DataFrame, fig_dir: Path) -> None:
    if by_year.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        by_year["entry_year"],
        by_year["pairing_rate_count_pct"],
        linewidth=2.5,
        linestyle="-",
        marker="o",
        label="Pairing Rate (Count)",
    )

    if "pairing_rate_value_pct" in by_year.columns:
        ax.plot(
            by_year["entry_year"],
            by_year["pairing_rate_value_pct"],
            linewidth=2.5,
            linestyle="--",
            marker="s",
            label="Pairing Rate (Value)",
        )

    finish_axes(ax)
    ax.set_title("Pairing Rate by Entry Year", fontweight="bold")
    ax.set_xlabel("Entry Year", fontweight="bold")
    ax.set_ylabel("Percent", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pairing_rate_by_year.png", dpi=300)
    plt.close()


def plot_paired_vs_unmatched_counts_by_year(by_year: pd.DataFrame, fig_dir: Path) -> None:
    if by_year.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    x = np.arange(len(by_year))
    ax.bar(
        x,
        by_year["matched_entries"],
        color=EXCEL_BLUE,
        label="Matched",
    )
    ax.bar(
        x,
        by_year["unmatched_entries"],
        bottom=by_year["matched_entries"],
        color=EXCEL_RED,
        label="Unmatched",
    )

    finish_axes(ax)
    ax.set_title("Matched vs Unmatched Entries by Entry Year", fontweight="bold")
    ax.set_xlabel("Entry Year", fontweight="bold")
    ax.set_ylabel("Number of Entries", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(by_year["entry_year"].astype(int), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "paired_vs_unmatched_counts_by_year.png", dpi=300)
    plt.close()


def plot_pairing_rate_by_type(by_type: pd.DataFrame, fig_dir: Path, top_n: int = 12) -> None:
    if by_type.empty:
        return

    plot_df = by_type.sort_values("total_entries", ascending=False).head(top_n).copy()

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        plot_df["deal_type"].astype(str),
        plot_df["pairing_rate_count_pct"],
        color=EXCEL_BLUE,
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Pairing Rate by Deal Type", fontweight="bold")
    ax.set_xlabel("Deal Type", fontweight="bold")
    ax.set_ylabel("Pairing Rate (%)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "pairing_rate_by_type.png", dpi=300)
    plt.close()


def plot_pairing_rate_by_size_bucket(by_size_bucket: pd.DataFrame, fig_dir: Path) -> None:
    if by_size_bucket.empty:
        return

    order = ["<100", "100-500", "500-1000", "1000+", "Missing"]
    plot_df = by_size_bucket.copy()
    plot_df["sort_order"] = plot_df["size_bucket"].map({v: i for i, v in enumerate(order)})
    plot_df = plot_df.sort_values("sort_order")

    apply_house_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_df["size_bucket"],
        plot_df["pairing_rate_count_pct"],
        color=EXCEL_RED,
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Pairing Rate by Entry Size Bucket", fontweight="bold")
    ax.set_xlabel("Entry Size Bucket ($M)", fontweight="bold")
    ax.set_ylabel("Pairing Rate (%)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "pairing_rate_by_size_bucket.png", dpi=300)
    plt.close()


def plot_holding_period_distribution(base: pd.DataFrame, fig_dir: Path) -> None:
    matched = base.loc[base["matched"], "holding_period_years"].dropna()
    if matched.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        matched,
        bins=25,
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Holding Period Distribution (Matched Deals)", fontweight="bold")
    ax.set_xlabel("Holding Period (Years)", fontweight="bold")
    ax.set_ylabel("Number of Deals", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "holding_period_distribution.png", dpi=300)
    plt.close()


def plot_exit_type_distribution(exit_types: pd.DataFrame, fig_dir: Path, top_n: int = 10) -> None:
    if exit_types.empty:
        return

    plot_df = exit_types.head(top_n).copy()

    apply_house_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        plot_df["exit_category"].astype(str),
        plot_df["fraction_by_number_pct"],
        color=EXCEL_GREEN,
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Exit Type Distribution (Matched Deals)", fontweight="bold")
    ax.set_xlabel("Exit Type", fontweight="bold")
    ax.set_ylabel("Share of Matched Exits (%)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "exit_type_distribution.png", dpi=300)
    plt.close()


def plot_unmatched_recent_vs_older(by_year: pd.DataFrame, fig_dir: Path) -> None:
    if by_year.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(15, 8))

    x = np.arange(len(by_year))
    ax.bar(
        x,
        by_year["unmatched_older_vintage"],
        color=EXCEL_ORANGE,
        label="Unmatched (Older Vintage)",
    )
    ax.bar(
        x,
        by_year["unmatched_recent_vintage"],
        bottom=by_year["unmatched_older_vintage"],
        color=EXCEL_RED,
        label="Unmatched (Recent Vintage)",
    )

    finish_axes(ax)
    ax.set_title("Unmatched Entries: Older vs Recent Vintages", fontweight="bold")
    ax.set_xlabel("Entry Year", fontweight="bold")
    ax.set_ylabel("Number of Unmatched Entries", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(by_year["entry_year"].astype(int), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "unmatched_recent_vs_older_by_year.png", dpi=300)
    plt.close()


def plot_matched_vs_unmatched_median_size(size_summary: pd.DataFrame, fig_dir: Path) -> None:
    if size_summary.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        size_summary["group"],
        size_summary["median_entry_size_usd_m"],
        color=[EXCEL_BLUE, EXCEL_RED],
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Median Entry Size: Matched vs Unmatched", fontweight="bold")
    ax.set_xlabel("Group", fontweight="bold")
    ax.set_ylabel("Median Entry Size ($M)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "matched_vs_unmatched_median_size.png", dpi=300)
    plt.close()


# ============================================================
# 9. REPORT FRAMEWORK
# ============================================================

def write_report_framework(
    overall: pd.DataFrame,
    table_dir: Path,
    fig_dir: Path,
    dataset_label: str,
) -> None:
    metric_map = dict(zip(overall["metric"], overall["value"]))

    total_entries = int(metric_map.get("total_entries", 0))
    matched_entries = int(metric_map.get("matched_entries", 0))
    unmatched_entries = int(metric_map.get("unmatched_entries", 0))
    pairing_rate = float(metric_map.get("pairing_rate_count_pct", 0.0))
    older_unmatched = int(metric_map.get("older_vintage_unmatched", 0))
    recent_unmatched = int(metric_map.get("recent_vintage_unmatched", 0))

    text = f"""# Pairing Report Framework — {dataset_label}

## 1. Executive takeaway
This report evaluates entry–exit pairing quality for the {dataset_label} dataset.

### Headline metrics
- Total entries: **{total_entries:,}**
- Matched entries: **{matched_entries:,}**
- Unmatched entries: **{unmatched_entries:,}**
- Pairing rate by count: **{pairing_rate:.2f}%**
- Older-vintage unmatched entries: **{older_unmatched:,}**
- Recent-vintage unmatched entries: **{recent_unmatched:,}**

## 2. Core questions
1. How many entries pair successfully?
2. Are unmatched deals concentrated in recent vintages, which may simply reflect censoring?
3. Are unmatched deals systematically smaller?
4. Do pairing rates differ by entry type?
5. What do the matched deals imply about exit type and holding period?

## 3. Output locations
- Tables: `outputs/pairing report/tables`
- Figures: `outputs/pairing report/figures`

## 4. Suggested writeup structure

### A. Data and pairing methodology
Describe:
- entry dataset used
- pair dataset used
- merge keys: company + entry date
- treatment of duplicate pairs
- definition of matched vs unmatched
- definition of recent vs older vintages

### B. Overall pairing performance
Use:
- `pairing_overall_summary.csv`

Discuss:
- total matching success
- count-based vs value-based pairing
- the difference between older-vintage and recent-vintage unmatched entries

### C. Pairing by entry year
Use:
- `pairing_by_year.csv`
- `pairing_rate_by_year.png`
- `paired_vs_unmatched_counts_by_year.png`
- `unmatched_recent_vs_older_by_year.png`

Discuss:
- whether low pairing rates are mostly recent-vintage censoring
- whether older cohorts still show substantial unmatched entries

### D. Pairing by type
Use:
- `pairing_by_type.csv`
- `pairing_rate_by_type.png`

Discuss:
- whether pairing is systematically better or worse for certain transaction types

### E. Size-bias analysis
Use:
- `pairing_by_size_bucket.csv`
- `matched_vs_unmatched_size_summary.csv`
- `pairing_rate_by_size_bucket.png`
- `matched_vs_unmatched_median_size.png`

Discuss:
- whether unmatched deals tend to be smaller
- whether missing size itself is concentrated among unmatched entries

### F. Matched-exit interpretation
Use:
- `matched_exit_type_distribution.csv`
- `matched_holding_period_summary.csv`
- `exit_type_distribution.png`
- `holding_period_distribution.png`

Discuss:
- exit-type composition among successfully paired deals
- average and median holding periods
- whether the matched sample looks representative

## 5. Figure checklist
1. `pairing_rate_by_year.png`
2. `paired_vs_unmatched_counts_by_year.png`
3. `unmatched_recent_vs_older_by_year.png`
4. `pairing_rate_by_type.png`
5. `pairing_rate_by_size_bucket.png`
6. `holding_period_distribution.png`
7. `exit_type_distribution.png`
8. `matched_vs_unmatched_median_size.png`

## 6. Table checklist
1. `pairing_overall_summary.csv`
2. `pairing_by_year.csv`
3. `pairing_by_type.csv`
4. `pairing_by_size_bucket.csv`
5. `matched_exit_type_distribution.csv`
6. `matched_holding_period_summary.csv`
7. `matched_vs_unmatched_size_summary.csv`
8. `unmatched_older_vintages.csv`
9. `pairing_diagnostic_base.csv`

## 7. Final interpretation prompts
- Is the pairing gap mostly a recent-vintage issue?
- Do older unmatched deals indicate real incompleteness?
- Is there evidence of small-firm bias in matching?
- How much confidence should we place in matched-sample exit analyses?

"""
    (table_dir.parent / "pairing_report_framework.md").write_text(text, encoding="utf-8")


# ============================================================
# 10. MAIN CONFIG + DRIVER
# ============================================================

def main() -> None:
    root = find_project_root()
    fig_dir, table_dir = ensure_dirs(root)

    print(f"Starting pairing report. Saving outputs to: {fig_dir.parent}")

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    # Default: P2P analysis
    entries_name = "pitchbook_public_2_private_all"
    pairs_name = "p2p_linked_master"
    entries_folder = "raw"
    pairs_folder = "clean"
    dataset_label = "Take-Private / P2P"

    # Alternative examples:
    # entries_name = "pitchbook_master_clean"
    # pairs_name = "non_tp_entry_exit_pairs"
    # entries_folder = "clean"
    # pairs_folder = "clean"
    # dataset_label = "Non-Take-Private / Other PE"

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    entries_raw = load_data_flexible(root, entries_folder, entries_name)
    pairs_raw = load_data_flexible(root, pairs_folder, pairs_name)

    if entries_raw.empty:
        raise FileNotFoundError(
            f"Could not find entries dataset: data/{entries_folder}/{entries_name}.[parquet/xlsx/csv]"
        )
    if pairs_raw.empty:
        raise FileNotFoundError(
            f"Could not find pairs dataset: data/{pairs_folder}/{pairs_name}.[parquet/xlsx/csv]"
        )

    print(f"Loaded entries dataset: {entries_name}")
    print(f"Loaded pairs dataset: {pairs_name}")

    # --------------------------------------------------------
    # STANDARDIZE + BUILD BASE
    # --------------------------------------------------------
    entries = standardize_entries(entries_raw)
    pairs = standardize_pairs(pairs_raw)

    base = build_pairing_base(entries, pairs, recent_cutoff_year=2020)

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------
    overall = build_overall_summary(base)
    by_year = build_pairing_by_year(base)
    by_type = build_pairing_by_type(base)
    by_size_bucket = build_pairing_by_size_bucket(base)
    exit_types = build_exit_type_distribution(base)
    holding_summary = build_holding_period_summary(base)
    unmatched_older = build_unmatched_older_vintages(base)
    matched_vs_unmatched_size = build_matched_vs_unmatched_size_summary(base)

    save_tables(
        overall=overall,
        by_year=by_year,
        by_type=by_type,
        by_size_bucket=by_size_bucket,
        exit_types=exit_types,
        holding_summary=holding_summary,
        unmatched_older=unmatched_older,
        matched_vs_unmatched_size=matched_vs_unmatched_size,
        base=base,
        table_dir=table_dir,
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------
    plot_pairing_rate_by_year(by_year, fig_dir)
    plot_paired_vs_unmatched_counts_by_year(by_year, fig_dir)
    plot_unmatched_recent_vs_older(by_year, fig_dir)
    plot_pairing_rate_by_type(by_type, fig_dir, top_n=12)
    plot_pairing_rate_by_size_bucket(by_size_bucket, fig_dir)
    plot_holding_period_distribution(base, fig_dir)
    plot_exit_type_distribution(exit_types, fig_dir, top_n=10)
    plot_matched_vs_unmatched_median_size(matched_vs_unmatched_size, fig_dir)

    # --------------------------------------------------------
    # FRAMEWORK
    # --------------------------------------------------------
    write_report_framework(
        overall=overall,
        table_dir=table_dir,
        fig_dir=fig_dir,
        dataset_label=dataset_label,
    )

    print("Done.")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to: {table_dir}")
    print(f"Framework saved to: {fig_dir.parent / 'pairing_report_framework.md'}")


if __name__ == "__main__":
    main()