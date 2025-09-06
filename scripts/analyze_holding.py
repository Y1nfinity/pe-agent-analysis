# scripts/analyze_holding.py
from __future__ import annotations
from pathlib import Path
import pandas as pd


from tools.io import load_and_clean
from tools.transforms import add_p2p_flag, add_entry_exit_flags
from tools.pairing import pair_entry_exit
from tools.holding import (
    holding_stats,
    categorize_holding_period,
    segment_by,
    outlier_analysis,
    plot_histogram,
    plot_boxplot,
    plot_density,
)

def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config.toml"

    # 1) Load & prepare base events
    df, _ = load_and_clean(cfg)
    df = add_p2p_flag(df)
    df = add_entry_exit_flags(df)

    # 2) Pairing (adjust thresholds if you want)
    pairs, audit = pair_entry_exit(
        df,
        restrict_same_sponsor=False,
        min_hold_days=0,
        max_gap_years=None,
        carry_cols_entry=["DealType", "DealType2", "DealType3", "DealSize", "Sponsor"],
        carry_cols_exit=["DealType", "DealType2", "DealType3", "DealSize", "Sponsor"],
    )
    if pairs.empty:
        print("No paired entries/exits produced. Check your entry/exit classification.")
        return

    print("Total pairs made:", len(pairs))
    print("Unique companies paired:", pairs['Company'].nunique())
    print("Holding years summary:")
    print(pairs['HoldingYears'].describe())

    # 3) Save intermediate paired data
    out_tables = root / "artifacts" / "tables"
    out_plots = root / "artifacts" / "plots"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out_tables / "paired_holding_periods.csv", index=False)
    audit.to_csv(out_tables / "pairing_audit.csv", index=False)

    # 4) First descriptive stats
    basic_stats = holding_stats(pairs)
    basic_stats.to_csv(out_tables / "holding_stats_overall.csv", index=False)

    print("\n=== BASIC HOLDING STATS ===")
    print(basic_stats.to_string(index=False))

    # 5) Plots: histogram, density, boxplot
    plot_histogram(pairs, out_plots / "holding_hist.png", bins=40)
    plot_density(pairs, out_plots / "holding_density.png")
    plot_boxplot(pairs, out_plots / "holding_box.png")

    # 6) Categorize by cutoffs Short/Medium/Long
    pairs_cat = categorize_holding_period(pairs)
    pairs_cat.to_csv(out_tables / "paired_with_categories.csv", index=False)

    # 7) Optional segmentations (guard if columns exist)
    # Deal size bins (quartiles) for a quick view
    if "Exit_DealSize" in pairs_cat.columns:
        q = pairs_cat["Exit_DealSize"].dropna().quantile([0.25, 0.5, 0.75]).to_dict()
        def size_bucket(x):
            if pd.isna(x): return "missing"
            if x <= q[0.25]: return "Q1 (small)"
            if x <= q[0.5]:  return "Q2"
            if x <= q[0.75]: return "Q3"
            return "Q4 (large)"
        pairs_cat["DealSizeBucket"] = pairs_cat["Exit_DealSize"].apply(size_bucket)
        seg_size = segment_by(pairs_cat, "DealSizeBucket")
        seg_size.to_csv(out_tables / "holding_by_dealsize_bucket.csv", index=False)

    # Industry segmentation if you have PrimaryIndustry carried over (adjust as needed)
    if "Exit_DealType" in pairs_cat.columns:
        # Example: exit type acts as a proxy for "exit quality" later
        seg_exit_type = segment_by(pairs_cat, "Exit_DealType")
        seg_exit_type.to_csv(out_tables / "holding_by_exit_type.csv", index=False)

    # Sponsor segmentation (top 20 sponsors by pair count)
    if "Exit_Sponsor" in pairs_cat.columns:
        top_sponsors = (
            pairs_cat["Exit_Sponsor"]
            .value_counts()
            .head(20)
            .index.tolist()
        )
        pairs_top = pairs_cat[pairs_cat["Exit_Sponsor"].isin(top_sponsors)].copy()
        seg_sponsor = segment_by(pairs_top, "Exit_Sponsor")
        seg_sponsor.to_csv(out_tables / "holding_by_top20_sponsors.csv", index=False)

    # 8) Outliers
    outliers = outlier_analysis(pairs_cat, lower=0.5, upper=15)
    outliers.to_csv(out_tables / "holding_outliers.csv", index=False)

    print("\nSaved tables to:", out_tables)
    print("Saved plots to:", out_plots)
    print("\nFiles created:")
    for p in sorted(out_tables.glob("*.csv")):
        print(" -", p.name)
    for p in sorted(out_plots.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
