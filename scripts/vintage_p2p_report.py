# scripts/vintage_p2p_report.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import tomllib

from tools.datasets import load_pairs_and_unmatched
from tools.safety import add_completeness_flags
from tools.irr_utils import (
    load_universe_if_any,
    mark_distributions,
    deal_cashflows_from_pairs,
    xirr,
)


def _to_numeric_money(series: pd.Series) -> pd.Series:
    """Coerce money-like strings to numeric (strip $, commas)."""
    if series is None:
        return series
    s = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def main():
    # ----------------------------
    # Paths & config
    # ----------------------------
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())

    out_tables = root / cfg["outputs"]["tables_dir"]
    out_tables.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load source-of-truth tables
    # ----------------------------
    pairs, unmatched, entries_all = load_pairs_and_unmatched(root / "config.toml")

    # Normalize deal sizes in PAIRS (to numeric)
    for col in ["Entry_DealSize", "Exit_DealSize"]:
        if col in pairs.columns:
            pairs[col] = _to_numeric_money(pairs[col])

    # Add completeness flags and HoldingYears (dates-only rule)
    pairs = add_completeness_flags(pairs)

    # ----------------------------
    # Optional distributions & IRR
    # ----------------------------
    # If a universe Excel is configured, use it to pick up mid-hold distributions.
    universe = load_universe_if_any(root, cfg)
    if universe is not None:
        universe_marked = mark_distributions(universe, cfg["irr"])
        cashflows, deal_metrics = deal_cashflows_from_pairs(pairs, universe_marked, cfg["irr"])
    else:
        # Minimal IRR path: compute IRR only when both dates & sizes exist.
        recs = []
        for i, r in pairs.reset_index(drop=True).iterrows():
            if bool(r.get("EligibleForIRR")):
                irr_val = xirr(
                    [r["Entry_DealDate"], r["Exit_DealDate"]],
                    [-(r["Entry_DealSize"]), (r["Exit_DealSize"])],
                )
                total_gains = r["Exit_DealSize"]  # no distributions in this path
                recs.append({
                    "PairKey": i,
                    "Company": r.get("Entry_Company"),
                    "EntryDate": r.get("Entry_DealDate"),
                    "ExitDate": r.get("Exit_DealDate"),
                    "Entry_DealSize": r.get("Entry_DealSize"),
                    "Exit_DealSize":  r.get("Exit_DealSize"),
                    "TotalDistributions": 0.0,
                    "TotalGains": total_gains,
                    "IRR_XIRR": irr_val,
                    "MOIC": (r["Exit_DealSize"] / r["Entry_DealSize"]) if r["Entry_DealSize"] else None,
                    "HoldingYears": r.get("HoldingYears"),
                })
            else:
                recs.append({
                    "PairKey": i,
                    "Company": r.get("Entry_Company"),
                    "EntryDate": r.get("Entry_DealDate"),
                    "ExitDate": r.get("Exit_DealDate"),
                    "Entry_DealSize": r.get("Entry_DealSize"),
                    "Exit_DealSize":  r.get("Exit_DealSize"),
                    "TotalDistributions": None,
                    "TotalGains": None,
                    "IRR_XIRR": None,
                    "MOIC": None,
                    "HoldingYears": r.get("HoldingYears"),
                })
        cashflows = pd.DataFrame(columns=["PairKey", "Company", "Date", "Amount", "Kind"])
        deal_metrics = pd.DataFrame(recs)

    # ----------------------------
    # Merge metrics back to pairs
    # ----------------------------
    pairs = pairs.reset_index(drop=True).copy()
    pairs["PairKey"] = pairs.index
    deal_metrics = deal_metrics.sort_values("PairKey").reset_index(drop=True)

    # keep `pairs` names intact; metrics get `_m` if overlapping
    pairs_m = pairs.merge(deal_metrics, on="PairKey", how="left", suffixes=("", "_m"))

    # normalize key columns that might have collided
    holding_candidates = [c for c in ["HoldingYears", "HoldingYears_m", "HoldingYears_x", "HoldingYears_y"] if
                          c in pairs_m.columns]
    if not holding_candidates:
        pairs_m["HoldingYears"] = (pd.to_datetime(pairs_m["Exit_DealDate"]) - pd.to_datetime(
            pairs_m["Entry_DealDate"])).dt.days / 365.25
    else:
        if "HoldingYears" not in pairs_m.columns:
            pairs_m["HoldingYears"] = pairs_m[holding_candidates[0]]

    if "IRR_XIRR" not in pairs_m.columns and "IRR_XIRR_m" in pairs_m.columns:
        pairs_m["IRR_XIRR"] = pairs_m["IRR_XIRR_m"]

    # ----------------------------
    # Diagnostics (completeness)
    # ----------------------------
    diagnostics = {
        "total_pairs": int(len(pairs_m)),
        "has_both_dates": int(pairs_m["HasBothDates"].sum()),
        "has_both_sizes": int(pairs_m["HasBothSizes"].sum()),
        "eligible_for_irr": int(pairs_m["EligibleForIRR"].sum()),
        "holdingyears_available": int(pairs_m["HoldingYears"].notna().sum()),
    }
    pd.DataFrame([diagnostics]).to_csv(out_tables / "diagnostics_pairs_completeness.csv", index=False)

    # ----------------------------
    # Per-deal realized metrics (respecting completeness)
    # ----------------------------
    per_deal_cols = [
        "Entry_DealID", "Entry_Company", "Entry_DealDate", "Entry_DealSize",
        "Exit_DealID",  "Exit_Company",  "Exit_DealDate",  "Exit_DealSize",
        "HasBothDates", "HasBothSizes", "EligibleForIRR",
        "HoldingYears", "IRR_XIRR", "TotalDistributions", "TotalGains",
        "Exit_DealType", "Exit_DealType2", "Exit_DealType3", "EntryYear",
    ]
    per_deal = pairs_m.reindex(columns=per_deal_cols)
    per_deal.to_csv(out_tables / "per_deal_realized_metrics.csv", index=False)

    # ----------------------------
    # (1) Vintage overview: count & going-in spend (paired + unmatched)
    # ----------------------------
    vintage_overview = (
        entries_all
        .groupby("EntryYear", as_index=False)
        .agg(
            total_entries_n=("Entry_DealID", "count"),
            total_entry_spend=("Entry_DealSize", "sum"),
        )
        .sort_values("EntryYear")
    )
    vintage_overview.to_csv(out_tables / "vintage_overview_entries_spend.csv", index=False)

    # ----------------------------
    # (2a) Exit breakdown by original PitchBook labels
    # ----------------------------
    # Choose exit category column via config (default: Exit_DealType)
    exit_col = (cfg.get("analysis", {}).get("exit_category") or "Exit_DealType")
    if exit_col not in pairs_m.columns:
        exit_col = "Exit_DealType"

    pairs_m["ExitCategory"] = pairs_m[exit_col].astype(str).str.strip().str.lower()

    alias = (cfg.get("labels", {}).get("exit_alias") or {})
    if alias:
        pairs_m["ExitCategory"] = (
            pairs_m["ExitCategory"]
            .astype("string").str.strip().str.lower()
            .map(lambda x: alias.get(x, x))
        )

    def avg_irr_safe(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.mean()) if len(s) else float("nan")

    exits_group = (
        pairs_m
        .groupby(["EntryYear", "ExitCategory"], as_index=False)
        .agg(
            exits_n=("Entry_DealID", "count"),
            entry_spend_for_exits=("Entry_DealSize", "sum"),
            avg_irr=("IRR_XIRR", avg_irr_safe),
        )
    )

    denom = vintage_overview.rename(columns={
        "total_entries_n": "denom_entries_n",
        "total_entry_spend": "denom_entry_spend",
    })

    breakdown = exits_group.merge(denom, on="EntryYear", how="left")
    breakdown["frac_by_number"] = (
        breakdown["exits_n"] / breakdown["denom_entries_n"].replace(0, pd.NA)
    )
    breakdown["frac_by_entry_spend"] = (
        breakdown["entry_spend_for_exits"] / breakdown["denom_entry_spend"].replace(0, pd.NA)
    )
    breakdown = breakdown.sort_values(["EntryYear", "ExitCategory"])
    breakdown.to_csv(out_tables / "vintage_exit_breakdown.csv", index=False)

    # ----------------------------
    # (2b) Unmatched fractions (by number & spend)
    # ----------------------------
    unmatched_v = (
        unmatched
        .groupby("EntryYear", as_index=False)
        .agg(
            unmatched_n=("Entry_DealID", "count"),
            unmatched_entry_spend=("Entry_DealSize", "sum"),
        )
        .merge(denom, on="EntryYear", how="right")  # keep all vintages
    )
    unmatched_v["unmatched_n"] = unmatched_v["unmatched_n"].fillna(0).astype(int)
    unmatched_v["unmatched_entry_spend"] = unmatched_v["unmatched_entry_spend"].fillna(0.0)
    unmatched_v["fraction_by_number"] = (
        unmatched_v["unmatched_n"] / unmatched_v["denom_entries_n"].replace(0, pd.NA)
    )
    unmatched_v["fraction_by_spend"] = (
        unmatched_v["unmatched_entry_spend"] / unmatched_v["denom_entry_spend"].replace(0, pd.NA)
    )
    unmatched_v = unmatched_v[
        ["EntryYear", "unmatched_n", "fraction_by_number", "unmatched_entry_spend", "fraction_by_spend"]
    ]
    unmatched_v.to_csv(out_tables / "vintage_unmatched_fractions.csv", index=False)

    # ----------------------------
    # Console summary
    # ----------------------------
    print("Saved tables:")
    for f in [
        "diagnostics_pairs_completeness.csv",
        "per_deal_realized_metrics.csv",
        "vintage_overview_entries_spend.csv",
        "vintage_exit_breakdown.csv",
        "vintage_unmatched_fractions.csv",
    ]:
        print(" -", out_tables / f)


if __name__ == "__main__":
    main()
