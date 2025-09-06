# scripts/report_all_universe.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from tools.io import load_and_clean
from tools.transforms import add_p2p_flag, add_entry_exit_flags
from tools.pairing import pair_entry_exit
from tools.p2p_vintage import (
    map_exit_type,                # we'll label exit types WITHOUT filtering to P2P
    vintage_overview_from_entries # we'll reuse this on all entries
)
from tools.exit_quality import build_and_plot_quality_panels
from tools.export import export_table


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config.toml"
    out_tables = root / "artifacts" / "tables"
    out_plots  = root / "artifacts" / "plots"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    # 1) Load events & basic flags (no P2P filter here)
    events, _ = load_and_clean(cfg)
    events = add_p2p_flag(events)         # still useful, but we won't filter by it
    events = add_entry_exit_flags(events)

    # 2) Universe overview of entries (for counts & going-in spend)
    entries_all = events[events["IsEntry"]].copy()
    overview_all = vintage_overview_from_entries(entries_all)
    export_table(
        overview_all,
        out_tables / "universe_vintage_overview",
        title="Universe Vintage Overview (Firms & Going-In Spend)",
        formats=("csv", "html", "png"),
        money_cols=["total_going_in_spend"],
    )

    # 3) Pairing across the full universe
    pairs_all, audit_all = pair_entry_exit(
        events,
        carry_cols_entry=["DealType","DealType2","DealType3","DealSize","Sponsor","PublicToPrivate","DealSynopsis"],
        carry_cols_exit =["DealType","DealType2","DealType3","DealSize","Sponsor","DealSynopsis"],
    )

    if pairs_all.empty:
        print("No pairs produced in the full universe.")
        return

    # Add Vintage (entry cohort) and ExitYear
    pairs_all["Vintage"] = pd.to_datetime(pairs_all["EntryDate"]).dt.year
    pairs_all["ExitYear"] = pd.to_datetime(pairs_all["ExitDate"]).dt.year

    # Label baseline exit types (without filtering)
    pairs_all["ExitTypeLabel"] = pairs_all.apply(map_exit_type, axis=1)

    # Save basic diagnostics
    basic_diag = pd.DataFrame({
        "total_pairs":[len(pairs_all)],
        "unique_companies_paired":[pairs_all["Company"].nunique()],
        "median_holding_years":[pairs_all["HoldingYears"].median()],
        "mean_holding_years":[pairs_all["HoldingYears"].mean()]
    })
    export_table(
        basic_diag,
        out_tables / "universe_pairing_summary",
        title="Universe Pairing Summary",
        formats=("csv","html","png"),
    )
    audit_all.to_csv(out_tables / "universe_pairing_audit.csv", index=False)

    # 4) Exit-quality & baseline-type panels (counts + $; fraction + absolute; full vs nosize)
    # Full sample
    _ = build_and_plot_quality_panels(
        pairs_all,
        out_tables=out_tables,
        out_plots=out_plots,
        year_col="Vintage",
        exit_type_col="ExitTypeLabel",
        size_col="Entry_DealSize",
        exclude_missing_size=False,
        label_prefix="universe_exit_quality_full"
    )
    # Excluding missing sizes
    _ = build_and_plot_quality_panels(
        pairs_all,
        out_tables=out_tables,
        out_plots=out_plots,
        year_col="Vintage",
        exit_type_col="ExitTypeLabel",
        size_col="Entry_DealSize",
        exclude_missing_size=True,
        label_prefix="universe_exit_quality_nosize"
    )

    print("\n[ALL-UNIVERSE] Done. Tables/plots written to artifacts/.")


if __name__ == "__main__":
    main()
