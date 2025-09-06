# scripts/diagnostics_overview.py
from pathlib import Path
import pandas as pd
import tomllib

from tools.datasets import load_pairs_and_unmatched


def _pick_exit_category(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Use original PitchBook exit labels. You can choose which field via config:
      [analysis]
      exit_category = "Exit_DealType" | "Exit_DealType2" | "Exit_DealType3" | "combined"
    """
    mode = (cfg.get("analysis", {}).get("exit_category") or "Exit_DealType")
    mode_norm = str(mode).strip().lower()
    if mode_norm == "exit_dealtype2" and "Exit_DealType2" in df.columns:
        return df["Exit_DealType2"].astype(str)
    if mode_norm == "exit_dealtype3" and "Exit_DealType3" in df.columns:
        return df["Exit_DealType3"].astype(str)
    if mode_norm == "combined" and {"Exit_DealType","Exit_DealType2","Exit_DealType3"}.issubset(df.columns):
        return (
            df[["Exit_DealType","Exit_DealType2","Exit_DealType3"]]
            .astype(str)
            .apply(lambda r: " | ".join([x for x in dict.fromkeys(r) if x and x.lower() != "nan"]), axis=1)
        )
    # default
    return df.get("Exit_DealType", pd.Series(index=df.index, dtype="object")).astype(str)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())

    out_tables = root / cfg["outputs"]["tables_dir"]
    out_tables.mkdir(parents=True, exist_ok=True)

    # Source-of-truth (already schema-normalized by our loader)
    pairs, unmatched, entries_all = load_pairs_and_unmatched(root / "config.toml")

    # Basic sizes
    paired_n   = len(pairs)
    unmatched_n = len(unmatched)
    total_seed = paired_n + unmatched_n

    # Company counts (schema uses Entry_* for entries/unmatched)
    uniq_comp_paired    = pairs.get("Entry_Company", pd.Series(dtype=object)).nunique()
    uniq_comp_unmatched = unmatched.get("Entry_Company", pd.Series(dtype=object)).nunique()

    # Exit distribution (paired only)
    exit_cat = _pick_exit_category(pairs, cfg).fillna("Unknown").str.strip().str.lower()
    exit_dist = (
        exit_cat.replace({"": "unknown"})
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "ExitCategory", 0: "count"})
        .sort_values("count", ascending=False)
    )

    # Quick IRR & Holding summaries (only if present and valid)
    irr_series = pairs.get("IRR_XIRR", pd.Series(dtype=float))
    irr_stats = irr_series.dropna()
    hold_series = pairs.get("HoldingYears", pd.Series(dtype=float))
    hold_stats = hold_series.dropna()

    summary = {
        "Total_seed_P2P": total_seed,
        "Paired": paired_n,
        "Unmatched": unmatched_n,
        "Pairing_rate_%": round(100 * paired_n / total_seed, 2) if total_seed else 0.0,
        "Unique_companies_paired": int(uniq_comp_paired),
        "Unique_companies_unmatched": int(uniq_comp_unmatched),
        # optional perf snapshots
        "IRR_mean": float(irr_stats.mean()) if len(irr_stats) else None,
        "IRR_median": float(irr_stats.median()) if len(irr_stats) else None,
        "Holding_mean_years": float(hold_stats.mean()) if len(hold_stats) else None,
        "Holding_median_years": float(hold_stats.median()) if len(hold_stats) else None,
    }
    summary_df = pd.DataFrame([summary])

    # Save
    summary_df.to_csv(out_tables / "overview_summary.csv", index=False)
    exit_dist.to_csv(out_tables / "overview_exit_distribution.csv", index=False)

    # Console
    print("\n📊 Overview Diagnostics")
    print("-" * 40)
    for k, v in summary.items():
        print(f"{k:25}: {v}")
    print("\nTop exit categories (paired):")
    print(exit_dist.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
