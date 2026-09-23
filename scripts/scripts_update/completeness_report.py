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
    out_root = root / "outputs" / "completeness report"
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
EXCEL_PURPLE = "#7030A0"


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
# 3. LOADERS
# ============================================================

def load_raw_exit_completed(root: Path) -> pd.DataFrame:
    path = root / "data" / "raw" / "pitchbook_exit_completed.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path)


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ============================================================
# 4. TEXT / COLUMN STANDARDIZATION
# ============================================================

def standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Deal Synopsis": "deal_synopsis",
        "Deal Date": "deal_date",
        "Deal Type": "deal_type",
        "Deal Type 2": "deal_type_2",
        "Deal Type 3": "deal_type_3",
        "Deal Size": "deal_size_raw",
        "Companies": "company_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "deal_synopsis" not in df.columns:
        raise ValueError("Column 'Deal Synopsis' not found in raw dataset.")

    df["deal_synopsis"] = df["deal_synopsis"].astype(str)
    df["deal_synopsis_lower"] = df["deal_synopsis"].str.lower()

    if "deal_date" in df.columns:
        df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
        df["deal_year"] = df["deal_date"].dt.year

    if "deal_type" in df.columns:
        df["deal_type"] = df["deal_type"].astype(str)

    if "deal_type_2" in df.columns:
        df["deal_type_2"] = df["deal_type_2"].astype(str)

    if "deal_type_3" in df.columns:
        df["deal_type_3"] = df["deal_type_3"].astype(str)

    if "deal_size_raw" in df.columns:
        df["deal_size_raw"] = df["deal_size_raw"].astype(str)
        cleaned = (
            df["deal_size_raw"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["deal_size_numeric"] = pd.to_numeric(cleaned, errors="coerce")
    else:
        df["deal_size_numeric"] = np.nan

    return df


# ============================================================
# 5. PHRASE FLAGS / COMPLETENESS FLAGS
# ============================================================

def add_completeness_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    phrase_patterns = {
        "undisclosed_sum": "undisclosed sum",
        "undisclosed": "undisclosed",
        "unreported": "unreported",
    }

    for col, phrase in phrase_patterns.items():
        df[col] = df["deal_synopsis_lower"].str.contains(phrase, regex=False, na=False)

    df["has_target_phrase"] = df[["undisclosed_sum", "undisclosed", "unreported"]].any(axis=1)

    def matched_phrase(row) -> str | None:
        if row["undisclosed_sum"]:
            return "undisclosed sum"
        if row["undisclosed"]:
            return "undisclosed"
        if row["unreported"]:
            return "unreported"
        return None

    df["matched_phrase"] = df.apply(matched_phrase, axis=1)

    df["deal_size_missing_numeric"] = df["deal_size_numeric"].isna()

    if "deal_size_raw" in df.columns:
        raw_lower = df["deal_size_raw"].astype(str).str.lower()
        df["deal_size_text_undisclosed"] = raw_lower.str.contains("undisclosed", na=False)
        df["deal_size_text_unreported"] = raw_lower.str.contains("unreported", na=False)
        df["deal_size_text_missing_marker"] = raw_lower.str.contains(
            "undisclosed|unreported|not disclosed|n/a|na",
            na=False,
            regex=True,
        )
    else:
        df["deal_size_text_undisclosed"] = False
        df["deal_size_text_unreported"] = False
        df["deal_size_text_missing_marker"] = False

    return df


# ============================================================
# 6. SUMMARY TABLES
# ============================================================

def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    matched_rows = int(df["has_target_phrase"].sum())
    pct_matched = matched_rows / total_rows * 100 if total_rows else 0.0

    out = pd.DataFrame(
        [
            {"metric": "total_rows", "value": total_rows},
            {"metric": "rows_with_target_phrase", "value": matched_rows},
            {"metric": "pct_rows_with_target_phrase", "value": pct_matched},
            {"metric": "rows_with_undisclosed_sum", "value": int(df["undisclosed_sum"].sum())},
            {"metric": "rows_with_undisclosed", "value": int(df["undisclosed"].sum())},
            {"metric": "rows_with_unreported", "value": int(df["unreported"].sum())},
            {"metric": "rows_missing_numeric_deal_size", "value": int(df["deal_size_missing_numeric"].sum())},
            {
                "metric": "pct_missing_numeric_deal_size",
                "value": (df["deal_size_missing_numeric"].mean() * 100) if total_rows else 0.0,
            },
            {
                "metric": "rows_with_text_missing_marker_in_deal_size",
                "value": int(df["deal_size_text_missing_marker"].sum()),
            },
        ]
    )
    return out


def build_phrase_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phrase_col, label in [
        ("undisclosed_sum", "undisclosed sum"),
        ("undisclosed", "undisclosed"),
        ("unreported", "unreported"),
    ]:
        count = int(df[phrase_col].sum())
        pct = df[phrase_col].mean() * 100 if len(df) else 0.0
        rows.append({"phrase": label, "count": count, "percent_of_rows": pct})
    return pd.DataFrame(rows)


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "deal_year" not in df.columns:
        return pd.DataFrame()

    year_df = (
        df.dropna(subset=["deal_year"])
        .groupby("deal_year")
        .agg(
            total_rows=("has_target_phrase", "size"),
            matched_rows=("has_target_phrase", "sum"),
            missing_numeric_deal_size=("deal_size_missing_numeric", "sum"),
            text_missing_marker_rows=("deal_size_text_missing_marker", "sum"),
        )
        .reset_index()
        .sort_values("deal_year")
    )

    year_df["matched_percent"] = year_df["matched_rows"] / year_df["total_rows"] * 100
    year_df["missing_numeric_percent"] = year_df["missing_numeric_deal_size"] / year_df["total_rows"] * 100
    year_df["text_missing_marker_percent"] = year_df["text_missing_marker_rows"] / year_df["total_rows"] * 100

    return year_df


def build_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "deal_type" not in df.columns:
        return pd.DataFrame()

    type_df = (
        df.groupby("deal_type")
        .agg(
            total_rows=("has_target_phrase", "size"),
            matched_rows=("has_target_phrase", "sum"),
            missing_numeric_deal_size=("deal_size_missing_numeric", "sum"),
            text_missing_marker_rows=("deal_size_text_missing_marker", "sum"),
        )
        .reset_index()
        .sort_values(["matched_rows", "total_rows"], ascending=[False, False])
    )

    type_df["matched_percent"] = type_df["matched_rows"] / type_df["total_rows"] * 100
    type_df["missing_numeric_percent"] = type_df["missing_numeric_deal_size"] / type_df["total_rows"] * 100
    type_df["text_missing_marker_percent"] = type_df["text_missing_marker_rows"] / type_df["total_rows"] * 100

    return type_df


def build_phrase_by_year_type(df: pd.DataFrame) -> pd.DataFrame:
    if "deal_year" not in df.columns or "deal_type" not in df.columns:
        return pd.DataFrame()

    out = (
        df.dropna(subset=["deal_year"])
        .groupby(["deal_year", "deal_type"])
        .agg(
            total_rows=("has_target_phrase", "size"),
            matched_rows=("has_target_phrase", "sum"),
        )
        .reset_index()
    )
    out["matched_percent"] = out["matched_rows"] / out["total_rows"] * 100
    return out.sort_values(["deal_year", "matched_percent"], ascending=[True, False])


def build_missing_vs_phrase_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["phrase_flag"] = np.where(tmp["has_target_phrase"], "has_phrase", "no_phrase")
    tmp["size_missing_flag"] = np.where(tmp["deal_size_missing_numeric"], "missing_numeric_size", "numeric_size_present")

    ct = pd.crosstab(tmp["phrase_flag"], tmp["size_missing_flag"], margins=True)
    return ct.reset_index()


# ============================================================
# 7. SAVE TABLES
# ============================================================

def save_tables(
    overall: pd.DataFrame,
    phrase_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    type_summary: pd.DataFrame,
    phrase_by_year_type: pd.DataFrame,
    crosstab_df: pd.DataFrame,
    matched_rows: pd.DataFrame,
    table_dir: Path,
) -> None:
    overall.to_csv(table_dir / "overall_summary.csv", index=False)
    phrase_summary.to_csv(table_dir / "phrase_summary.csv", index=False)

    if not year_summary.empty:
        year_summary.to_csv(table_dir / "undisclosed_by_year.csv", index=False)

    if not type_summary.empty:
        type_summary.to_csv(table_dir / "undisclosed_by_type.csv", index=False)

    if not phrase_by_year_type.empty:
        phrase_by_year_type.to_csv(table_dir / "undisclosed_by_year_and_type.csv", index=False)

    crosstab_df.to_csv(table_dir / "phrase_vs_missing_size_crosstab.csv", index=False)
    matched_rows.to_csv(table_dir / "all_rows_with_target_phrase.csv", index=False)


# ============================================================
# 8. PLOTS
# ============================================================

def plot_undisclosed_share_by_year(year_summary: pd.DataFrame, fig_dir: Path) -> None:
    if year_summary.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        year_summary["deal_year"],
        year_summary["matched_percent"],
        linewidth=2.5,
        linestyle="-",
        marker="o",
        label="Synopsis contains undisclosed/unreported",
    )

    finish_axes(ax)
    ax.set_title("Share of Deals with Undisclosed Language by Year", fontweight="bold")
    ax.set_xlabel("Deal Year", fontweight="bold")
    ax.set_ylabel("Percent of Deals", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "undisclosed_share_by_year.png", dpi=300)
    plt.close()


def plot_missing_numeric_share_by_year(year_summary: pd.DataFrame, fig_dir: Path) -> None:
    if year_summary.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        year_summary["deal_year"],
        year_summary["missing_numeric_percent"],
        linewidth=2.5,
        linestyle="-",
        marker="o",
        label="Numeric deal size missing",
    )

    finish_axes(ax)
    ax.set_title("Share of Deals with Missing Numeric Deal Size by Year", fontweight="bold")
    ax.set_xlabel("Deal Year", fontweight="bold")
    ax.set_ylabel("Percent of Deals", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "missing_numeric_deal_size_by_year.png", dpi=300)
    plt.close()


def plot_phrase_vs_numeric_missing_by_year(year_summary: pd.DataFrame, fig_dir: Path) -> None:
    if year_summary.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        year_summary["deal_year"],
        year_summary["matched_percent"],
        linewidth=2.5,
        linestyle="-",
        marker="o",
        label="Undisclosed language",
    )
    ax.plot(
        year_summary["deal_year"],
        year_summary["missing_numeric_percent"],
        linewidth=2.5,
        linestyle="--",
        marker="s",
        label="Missing numeric deal size",
    )

    finish_axes(ax)
    ax.set_title("Undisclosed Language vs Missing Numeric Deal Size by Year", fontweight="bold")
    ax.set_xlabel("Deal Year", fontweight="bold")
    ax.set_ylabel("Percent of Deals", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "phrase_vs_missing_numeric_by_year.png", dpi=300)
    plt.close()


def plot_undisclosed_share_by_type(type_summary: pd.DataFrame, fig_dir: Path, top_n: int = 12) -> None:
    if type_summary.empty:
        return

    plot_df = type_summary.sort_values("matched_percent", ascending=False).head(top_n).copy()

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        plot_df["deal_type"].astype(str),
        plot_df["matched_percent"],
        color=EXCEL_BLUE,
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Share of Deals with Undisclosed Language by Deal Type", fontweight="bold")
    ax.set_xlabel("Deal Type", fontweight="bold")
    ax.set_ylabel("Percent of Deals", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "undisclosed_share_by_type.png", dpi=300)
    plt.close()


def plot_counts_by_type(type_summary: pd.DataFrame, fig_dir: Path, top_n: int = 12) -> None:
    if type_summary.empty:
        return

    plot_df = type_summary.sort_values("matched_rows", ascending=False).head(top_n).copy()

    apply_house_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(plot_df))
    width = 0.42

    ax.bar(
        x - width / 2,
        plot_df["total_rows"],
        width,
        color="none",
        edgecolor=EXCEL_RED,
        linewidth=1.5,
        label="Total deals",
    )
    ax.bar(
        x - width / 2,
        plot_df["matched_rows"],
        width,
        color=EXCEL_RED,
        alpha=0.6,
        label="Deals with undisclosed language",
    )

    finish_axes(ax)
    ax.set_title("Counts of Deals with Undisclosed Language by Deal Type", fontweight="bold")
    ax.set_xlabel("Deal Type", fontweight="bold")
    ax.set_ylabel("Number of Deals", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["deal_type"].astype(str), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "undisclosed_counts_by_type.png", dpi=300)
    plt.close()


def plot_phrase_breakdown(phrase_summary: pd.DataFrame, fig_dir: Path) -> None:
    if phrase_summary.empty:
        return

    apply_house_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        phrase_summary["phrase"],
        phrase_summary["count"],
        color=[EXCEL_BLUE, EXCEL_RED, EXCEL_GREEN],
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Frequency of Target Phrases in Deal Synopsis", fontweight="bold")
    ax.set_xlabel("Phrase", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "phrase_frequency.png", dpi=300)
    plt.close()


def plot_size_presence_vs_phrase(df: pd.DataFrame, fig_dir: Path) -> None:
    tmp = pd.DataFrame(
        {
            "group": ["No target phrase", "Has target phrase"],
            "pct_missing_numeric_size": [
                df.loc[~df["has_target_phrase"], "deal_size_missing_numeric"].mean() * 100,
                df.loc[df["has_target_phrase"], "deal_size_missing_numeric"].mean() * 100,
            ],
        }
    )

    apply_house_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        tmp["group"],
        tmp["pct_missing_numeric_size"],
        color=[EXCEL_BLUE, EXCEL_RED],
        edgecolor="black",
        linewidth=0.5,
    )

    finish_axes(ax)
    ax.set_title("Missing Numeric Deal Size: Phrase vs No Phrase", fontweight="bold")
    ax.set_xlabel("Group", fontweight="bold")
    ax.set_ylabel("Percent Missing Numeric Deal Size", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "missing_size_phrase_vs_no_phrase.png", dpi=300)
    plt.close()


def plot_heatmap_year_type(phrase_by_year_type: pd.DataFrame, fig_dir: Path, top_n_types: int = 8) -> None:
    if phrase_by_year_type.empty:
        return

    top_types = (
        phrase_by_year_type.groupby("deal_type")["matched_rows"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_types)
        .index
    )

    plot_df = phrase_by_year_type[phrase_by_year_type["deal_type"].isin(top_types)].copy()

    heat = plot_df.pivot(index="deal_type", columns="deal_year", values="matched_percent").fillna(0)

    apply_house_style()
    fig, ax = plt.subplots(figsize=(16, 7))

    im = ax.imshow(heat.values, aspect="auto")

    ax.set_title("Undisclosed Language Share by Deal Type and Year", fontweight="bold")
    ax.set_xlabel("Deal Year", fontweight="bold")
    ax.set_ylabel("Deal Type", fontweight="bold")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(x)) for x in heat.columns], rotation=90)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(str))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percent of Deals", rotation=90)

    plt.tight_layout()
    plt.savefig(fig_dir / "undisclosed_heatmap_year_type.png", dpi=300)
    plt.close()


# ============================================================
# 9. MARKDOWN FRAMEWORK
# ============================================================

def write_report_framework(
    overall: pd.DataFrame,
    phrase_summary: pd.DataFrame,
    table_dir: Path,
    fig_dir: Path,
) -> None:
    overall_map = dict(zip(overall["metric"], overall["value"]))

    total_rows = int(overall_map.get("total_rows", 0))
    rows_with_phrase = int(overall_map.get("rows_with_target_phrase", 0))
    pct_rows_with_phrase = float(overall_map.get("pct_rows_with_target_phrase", 0.0))

    undisclosed_sum = int(
        phrase_summary.loc[phrase_summary["phrase"] == "undisclosed sum", "count"].iloc[0]
    ) if not phrase_summary.empty and (phrase_summary["phrase"] == "undisclosed sum").any() else 0

    undisclosed = int(
        phrase_summary.loc[phrase_summary["phrase"] == "undisclosed", "count"].iloc[0]
    ) if not phrase_summary.empty and (phrase_summary["phrase"] == "undisclosed").any() else 0

    md = f"""# Completeness Report Framework

## 1. Executive takeaway
This report evaluates the completeness of the PitchBook exit dataset by measuring how often deal descriptions contain undisclosed pricing language and how often deal size is missing in usable numeric form.

### Headline statistics
- Total deals: **{total_rows:,}**
- Deals with target phrase in synopsis: **{rows_with_phrase:,}**
- Share with target phrase: **{pct_rows_with_phrase:.2f}%**
- Rows containing "undisclosed sum": **{undisclosed_sum:,}**
- Rows containing "undisclosed": **{undisclosed:,}**

## 2. Research question
The key question is whether private-market deal data is systematically incomplete, and whether missingness appears to vary by year and by deal type.

## 3. Data
- Source file: `data/raw/pitchbook_exit_completed.xlsx`
- Output tables: `outputs/completeness report/tables`
- Output figures: `outputs/completeness report/figures`

## 4. Suggested report structure

### A. Data and definitions
Describe:
- dataset source
- time range
- deal universe
- fields used
- definition of "undisclosed language"
- definition of "missing numeric deal size"

### B. Main descriptive findings
Use:
- `phrase_summary.csv`
- `overall_summary.csv`

Discuss:
- how large the dataset is
- how common undisclosed language is
- how common missing numeric size is

### C. Time trend
Use:
- `undisclosed_by_year.csv`
- `undisclosed_share_by_year.png`
- `missing_numeric_deal_size_by_year.png`
- `phrase_vs_missing_numeric_by_year.png`

Discuss:
- whether opacity rises over time
- whether phrase-based opacity tracks missing numeric size

### D. Deal-type heterogeneity
Use:
- `undisclosed_by_type.csv`
- `undisclosed_share_by_type.png`
- `undisclosed_counts_by_type.png`
- `undisclosed_heatmap_year_type.png`

Discuss:
- which deal types are most opaque
- whether opacity is concentrated in PE-style transactions

### E. Implications for downstream analysis
Use:
- `phrase_vs_missing_size_crosstab.csv`
- `missing_size_phrase_vs_no_phrase.png`

Discuss:
- whether phrase-based opacity is associated with missing usable size data
- why this matters for exit-value analysis, holding-period analysis, and return calculations

## 5. Figure insertion checklist
1. `phrase_frequency.png`
2. `undisclosed_share_by_year.png`
3. `missing_numeric_deal_size_by_year.png`
4. `phrase_vs_missing_numeric_by_year.png`
5. `undisclosed_share_by_type.png`
6. `undisclosed_counts_by_type.png`
7. `missing_size_phrase_vs_no_phrase.png`
8. `undisclosed_heatmap_year_type.png`

## 6. Table insertion checklist
1. `overall_summary.csv`
2. `phrase_summary.csv`
3. `undisclosed_by_year.csv`
4. `undisclosed_by_type.csv`
5. `undisclosed_by_year_and_type.csv`
6. `phrase_vs_missing_size_crosstab.csv`

## 7. Next analytical extensions
- compare disclosed vs undisclosed deals on observed size
- merge with paired exit datasets
- test whether missingness is concentrated in certain exit pathways
- evaluate whether sample restrictions distort inference

"""

    (table_dir.parent / "completeness_report_framework.md").write_text(md, encoding="utf-8")


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:
    root = find_project_root()
    fig_dir, table_dir = ensure_dirs(root)

    print(f"Starting completeness report. Saving outputs to: {fig_dir.parent}")

    # Load and prepare raw dataset
    raw = load_raw_exit_completed(root)
    raw = standardize_raw_columns(raw)
    raw = add_completeness_flags(raw)

    matched_rows = raw.loc[raw["has_target_phrase"]].copy()

    # Build tables
    overall = build_overall_summary(raw)
    phrase_summary = build_phrase_summary(raw)
    year_summary = build_year_summary(raw)
    type_summary = build_type_summary(raw)
    phrase_by_year_type = build_phrase_by_year_type(raw)
    crosstab_df = build_missing_vs_phrase_crosstab(raw)

    # Save tables
    save_tables(
        overall=overall,
        phrase_summary=phrase_summary,
        year_summary=year_summary,
        type_summary=type_summary,
        phrase_by_year_type=phrase_by_year_type,
        crosstab_df=crosstab_df,
        matched_rows=matched_rows,
        table_dir=table_dir,
    )

    # Make figures
    plot_phrase_breakdown(phrase_summary, fig_dir)
    plot_undisclosed_share_by_year(year_summary, fig_dir)
    plot_missing_numeric_share_by_year(year_summary, fig_dir)
    plot_phrase_vs_numeric_missing_by_year(year_summary, fig_dir)
    plot_undisclosed_share_by_type(type_summary, fig_dir, top_n=12)
    plot_counts_by_type(type_summary, fig_dir, top_n=12)
    plot_size_presence_vs_phrase(raw, fig_dir)
    plot_heatmap_year_type(phrase_by_year_type, fig_dir, top_n_types=8)

    # Write framework
    write_report_framework(
        overall=overall,
        phrase_summary=phrase_summary,
        table_dir=table_dir,
        fig_dir=fig_dir,
    )

    print("Done.")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to: {table_dir}")
    print(f"Framework saved to: {fig_dir.parent / 'completeness_report_framework.md'}")


if __name__ == "__main__":
    main()