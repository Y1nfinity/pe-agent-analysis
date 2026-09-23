"""
analyze_entry_exits.py (REWRITTEN)
==================================
Driver script for running the full entry–exit analysis pipeline.

This script:
    - Loads entry_exit_pairs.parquet (output of entry_exit_linking.py)
    - Runs the core analysis from entry_exit_analysis.py
    - Builds exit breakdown tables (counts, deal value)
    - Produces stacked bar charts (counts, %, deal value)
    - Exports economic value tables
    - Builds the Exit Market Overview master table
    - Adds a match-quality summary using `match_type` (exact / fuzzy)

It is compatible with the new schema:
    exit_type       (fine-grained)
    exit_category   (canonical)
    holding_period_years
    match_type      ("exact", "fuzzy", ...)
    match_score
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = ROOT / "scripts_restructured"
for p in (ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs" / "entry_exit_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# IMPORT MAIN ANALYSIS MODULE
# -------------------------------------------------------------------
import entry_exit_analysis as eea  # scripts_restructured/entry_exit_analysis.py


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def _detect_category_col(df: pd.DataFrame) -> str | None:
    """
    Decide whether to use 'exit_type' or 'exit_category' as the category dimension.
    Priority:
        1) exit_type     (fine-grained)
        2) exit_category (canonical)
    Returns None if neither exists.
    """
    if "exit_type" in df.columns:
        return "exit_type"
    if "exit_category" in df.columns:
        return "exit_category"
    return None


def _safe_print(msg: str) -> None:
    print(msg, flush=True)


# -------------------------------------------------------------------
# LOAD INPUT DATA
# -------------------------------------------------------------------
pairs_path = DATA_DIR / "entry_exit_pairs.parquet"
if not pairs_path.exists():
    raise FileNotFoundError(f"❌ Expected file not found: {pairs_path.resolve()}")

_safe_print(f"🔹 Loading linked pairs: {pairs_path.name}")
pairs = pd.read_parquet(pairs_path)
_safe_print(f"   Loaded {pairs.shape[0]:,} linked entry–exit pairs.\n")

# Quick schema sanity info
required_cols = ["entry_date", "exit_date", "holding_period_years"]
missing = [c for c in required_cols if c not in pairs.columns]
if missing:
    _safe_print(f"⚠️ WARNING: missing expected columns in pairs: {missing}")
else:
    _safe_print("✅ Core columns for holding periods are present.\n")


# -------------------------------------------------------------------
# CORE ANALYSIS FROM MODULE
# -------------------------------------------------------------------
# NOTE: We keep this API unchanged so we don't break entry_exit_analysis.py.
_safe_print("🔹 Running core entry–exit analysis from entry_exit_analysis.py ...")
summaries = eea.run_entry_exit_analysis(pairs_path, OUTPUT_DIR)

# -------------------------------------------------------------------
# EXIT BREAKDOWN TABLES
# -------------------------------------------------------------------
_safe_print("🔹 Building exit breakdown tables...")
exit_tables = eea.build_exit_breakdown_tables(pairs)
exit_counts = exit_tables.get("counts", pd.DataFrame())
deal_value_df = exit_tables.get("deal_value", pd.DataFrame())

# Decide which category column to use for plotting tables
category_col_counts = _detect_category_col(exit_counts)
category_col_value = _detect_category_col(deal_value_df)

if exit_counts.empty:
    _safe_print("⚠️ exit_counts table is empty — skipping count-based stacked bars.\n")
else:
    if category_col_counts is None:
        _safe_print("⚠️ No 'exit_type' or 'exit_category' in exit_counts — skipping count-based stacked bars.\n")
    else:
        # -------------------------------------------------------------------
        # STACKED BARS — COUNTS
        # -------------------------------------------------------------------
        eea.plot_stacked_bar(
            exit_counts,
            index_col="exit_year",
            category_col=category_col_counts,
            value_col="count",
            title=f"Exit Composition by Year (Counts) [{category_col_counts}]",
            output_path=OUTPUT_DIR / "Figure_ExitComposition_Counts.png",
            normalize=False,
        )

        eea.plot_stacked_bar(
            exit_counts,
            index_col="exit_year",
            category_col=category_col_counts,
            value_col="count",
            title=f"Exit Composition by Year (Percentage of Exits) [{category_col_counts}]",
            output_path=OUTPUT_DIR / "Figure_ExitComposition_Percent.png",
            normalize=True,
        )

        _safe_print("✅ Count-based stacked bar charts saved.\n")

# -------------------------------------------------------------------
# STACKED BARS — DEAL VALUE (IF AVAILABLE)
# -------------------------------------------------------------------
if deal_value_df is not None and not deal_value_df.empty:
    if category_col_value is None:
        _safe_print("⚠️ No 'exit_type' or 'exit_category' in deal_value_df — skipping deal-value stacked bars.\n")
    else:
        eea.plot_stacked_bar(
            deal_value_df,
            index_col="exit_year",
            category_col=category_col_value,
            value_col="deal_value",
            title=f"Exit Composition by Deal Value (USD) [{category_col_value}]",
            output_path=OUTPUT_DIR / "Figure_ExitComposition_DealValue.png",
            normalize=False,
        )

        eea.plot_stacked_bar(
            deal_value_df,
            index_col="exit_year",
            category_col=category_col_value,
            value_col="deal_value",
            title=f"Exit Composition by Deal Value (Percent) [{category_col_value}]",
            output_path=OUTPUT_DIR / "Figure_ExitComposition_DealValue_Percent.png",
            normalize=True,
        )

        _safe_print("✅ Deal-value stacked bar charts saved.\n")
else:
    _safe_print("⚠️ No deal_size/deal_value table available — skipping deal-value charts.\n")


# -------------------------------------------------------------------
# TAKE-PRIVATE EXIT ANALYSIS
# -------------------------------------------------------------------
_safe_print("🔹 Analyzing take-private exits...")
tp_summary = eea.analyze_take_private_exits(pairs)
tp_summary.to_csv(OUTPUT_DIR / "TakePrivate_Exit_Composition.csv", index=False)
_safe_print("✅ Take-private exit summary saved.\n")


# -------------------------------------------------------------------
# VINTAGE × EXIT-TYPE ANALYSIS
# -------------------------------------------------------------------
_safe_print("🔹 Analyzing vintage × exit-type relationship...")
vintage_matrix = eea.analyze_vintage_exit_relationship(pairs)
vintage_matrix.to_csv(OUTPUT_DIR / "Vintage_Exit_Matrix.csv", index=False)
_safe_print("✅ Vintage × Exit-type matrix saved.\n")


# -------------------------------------------------------------------
# ECONOMIC-VALUE TABLE EXPORTS
# -------------------------------------------------------------------
if deal_value_df is not None and not deal_value_df.empty:
    # By year + category
    deal_value_df.to_csv(
        OUTPUT_DIR / "Exit_Value_ByYear_Category.csv", index=False
    )

    # Totals by category
    cat_col_for_value = category_col_value or "exit_type"
    cat_value = (
        deal_value_df.groupby(cat_col_for_value, dropna=False)["deal_value"]
        .sum()
        .reset_index()
        .sort_values("deal_value", ascending=False)
        .rename(columns={cat_col_for_value: "exit_type"})
    )
    cat_value["share"] = (
        cat_value["deal_value"] / cat_value["deal_value"].sum() * 100.0
    )
    cat_value.to_csv(OUTPUT_DIR / "Exit_Value_ByCategory.csv", index=False)

    # Totals by year
    year_value = (
        deal_value_df.groupby("exit_year", dropna=False)["deal_value"]
        .sum()
        .reset_index()
    )
    year_value.to_csv(OUTPUT_DIR / "Exit_Value_ByYear.csv", index=False)

    _safe_print("✅ Deal-value economic tables saved.\n")
else:
    _safe_print("⚠️ No deal_value table — cannot export deal-value economic tables.\n")


# -------------------------------------------------------------------
# MATCH QUALITY SUMMARY (NEW, USING match_type)
# -------------------------------------------------------------------
_safe_print("🔹 Summarizing match quality (exact vs fuzzy)...")

if "match_type" in pairs.columns:
    match_quality = (
        pairs["match_type"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis("match_type")
        .reset_index(name="n_pairs")
    )
    match_quality["share"] = (
        match_quality["n_pairs"] / match_quality["n_pairs"].sum() * 100.0
    )
    match_quality.to_csv(OUTPUT_DIR / "EntryExit_MatchQuality.csv", index=False)
    _safe_print("✅ Match-quality summary saved as EntryExit_MatchQuality.csv\n")
else:
    _safe_print("⚠️ Column 'match_type' not found in pairs — skipping match-quality summary.\n")


# -------------------------------------------------------------------
# MASTER EXIT MARKET OVERVIEW
# -------------------------------------------------------------------
_safe_print("🔹 Building Exit Market Overview table...")

holding_summary = summaries.get("holding_summary")

# Total counts + shares (based on exit_counts)
if exit_counts is not None and not exit_counts.empty:
    cat_col_for_counts = category_col_counts or "exit_type"
    total_counts = (
        exit_counts.groupby(cat_col_for_counts, dropna=False)["count"]
        .sum()
        .reset_index()
        .rename(columns={cat_col_for_counts: "exit_type", "count": "n_exits"})
    )
    total_counts["share"] = (
        total_counts["n_exits"] / total_counts["n_exits"].sum() * 100.0
    )
else:
    total_counts = pd.DataFrame(columns=["exit_type", "n_exits", "share"])

# Add deal value totals, if available
if deal_value_df is not None and not deal_value_df.empty:
    cat_col_for_value = category_col_value or "exit_type"
    total_value = (
        deal_value_df.groupby(cat_col_for_value, dropna=False)["deal_value"]
        .sum()
        .reset_index()
        .rename(columns={cat_col_for_value: "exit_type"})
    )
else:
    total_value = pd.DataFrame({"exit_type": total_counts.get("exit_type", []), "deal_value": np.nan})

# Merge pieces into overview
if holding_summary is not None and not holding_summary.empty:
    overview = (
        total_counts
        .merge(holding_summary, on="exit_type", how="left")
        .merge(total_value, on="exit_type", how="left")
        .sort_values("n_exits", ascending=False)
    )
else:
    overview = (
        total_counts
        .merge(total_value, on="exit_type", how="left")
        .sort_values("n_exits", ascending=False)
    )

overview.to_csv(OUTPUT_DIR / "Exit_Market_Overview.csv", index=False)
_safe_print("✅ Exit Market Overview table saved.\n")

# -------------------------------------------------------------------
# FINISHED
# -------------------------------------------------------------------
_safe_print("🎉 All entry–exit analysis complete!")
_safe_print(f"   Results saved to: {OUTPUT_DIR.resolve()}\n")
