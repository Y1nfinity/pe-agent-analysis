# scripts/report_p2p_only.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from tools.io import load_and_clean
from tools.transforms import add_p2p_flag, add_entry_exit_flags
from tools.pairing import pair_entry_exit
from tools.p2p_vintage import (
    compute_irr_proxy,
    filter_p2p_and_vintage,             # filters to P2P and adds ExitTypeLabel + Vintage
    vintage_overview_from_entries,
    build_exit_breakdown_and_nonexit,
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

    # 1) Load & tag
    events, _ = load_and_clean(cfg)
    events = add_p2p_flag(events)
    events = add_entry_exit_flags(events)

    # 2) P2P entries overview (base for (i) and denominators)
    entries_p2p = events[events["IsEntry"] & (events["PublicToPrivate"] == True)].copy()
    overview_p2p = vintage_overview_from_entries(entries_p2p)
    export_table(
        overview_p2p,
        out_tables / "p2p_vintage_overview",
        title="P2P Vintage Overview (Firms & Going-In Spend)",
        formats=("csv","html","png"),
        money_cols=["total_going_in_spend"],
    )

    # 3) Pairing (carry fields we need), IRR proxy, P2P filter, exit labels
    pairs, audit = pair_entry_exit(
        events,
        carry_cols_entry=["DealType","DealType2","DealType3","DealSize","Sponsor","PublicToPrivate","DealSynopsis"],
        carry_cols_exit =["DealType","DealType2","DealType3","DealSize","Sponsor","DealSynopsis"],
    )
    if pairs.empty:
        print("No pairs produced (P2P subset cannot be built).")
        return

    pairs = compute_irr_proxy(pairs)
    pairs["EntryYear"] = pd.to_datetime(pairs["EntryDate"]).dt.year
    pairs_p2p = filter_p2p_and_vintage(pairs)  # adds Vintage + ExitTypeLabel

    # 4) Professor’s breakdown tables (exits & non-exits within vintage)
    breakdown, nonexit = build_exit_breakdown_and_nonexit(pairs_p2p, entries_p2p)
    export_table(
        breakdown,
        out_tables / "p2p_vintage_exit_breakdown",
        title="P2P Exit Breakdown by Vintage (Fractions & Avg IRR)",
        formats=("csv","html","png"),
        percent_cols=["fraction_by_number","fraction_by_entry_spend"],
        money_cols=["entry_spend"],
    )
    export_table(
        nonexit,
        out_tables / "p2p_vintage_nonexit",
        title="P2P Non-Exited Fractions by Vintage",
        formats=("csv","html","png"),
        percent_cols=["fraction_by_number_no_exit","fraction_by_entry_spend_no_exit"],
        money_cols=["entry_spend_no_exit"],
    )
    audit.to_csv(out_tables / "p2p_pairing_audit.csv", index=False)

    # 5) Exit-quality & baseline-type panels (counts + $; fraction + absolute; full vs nosize)
    _ = build_and_plot_quality_panels(
        pairs_p2p,
        out_tables=out_tables,
        out_plots=out_plots,
        year_col="Vintage",
        exit_type_col="ExitTypeLabel",
        size_col="Entry_DealSize",
        exclude_missing_size=False,
        label_prefix="p2p_exit_quality_full"
    )
    _ = build_and_plot_quality_panels(
        pairs_p2p,
        out_tables=out_tables,
        out_plots=out_plots,
        year_col="Vintage",
        exit_type_col="ExitTypeLabel",
        size_col="Entry_DealSize",
        exclude_missing_size=True,
        label_prefix="p2p_exit_quality_nosize"
    )

    print("\n[P2P-ONLY] Done. Tables/plots written to artifacts/.")


if __name__ == "__main__":
    main()
