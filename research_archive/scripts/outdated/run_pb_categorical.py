# scripts/run_pb_categorical.py
import tomllib


def read_pitchbook_tables(root: Path, cfg: dict):
    tables_dir = root / cfg["outputs"]["tables_dir"]
    per_deal = pd.read_csv(tables_dir / "per_deal_realized_metrics.csv")
    per_deal = assign_exit_category(per_deal)
    breakdown = pd.read_csv(tables_dir / "vintage_exit_breakdown.csv")
    unmatched_v = pd.read_csv(tables_dir / "vintage_unmatched_fractions.csv")
    return per_deal, breakdown, unmatched_v




def main():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())
    out_dir = root / "artifacts" / "plots_categorical"
    ensure_dir(out_dir)

    per_deal, breakdown, unmatched_v = read_pitchbook_tables(root, cfg)

    # ✅ Assign unified exit category
    per_deal = assign_exit_category(per_deal)

    # Core exit-type composition visuals
    plot_exit_counts_by_vintage(breakdown, out_dir)
    plot_exit_spend_by_vintage(breakdown, out_dir)
    plot_exit_mix_fraction(breakdown, out_dir)

    # Paired/unmatched pairing analytics
    plot_paired_unmatched_by_vintage(breakdown, unmatched_v, out_dir)

    # Performance visuals
    plot_box_metric_by_exit(per_deal, metric="IRR_XIRR", ylabel="IRR (Annualized)", title="IRR by Exit Type", out=out_dir)
    plot_box_metric_by_exit(per_deal, metric="HoldingYears", ylabel="Holding Period (Years)", title="Holding Period by Exit Type", out=out_dir)
    plot_heatmap_avg_metric(breakdown, metric_col="avg_irr", out=out_dir)

    print(f"✅ Saved all PitchBook categorical plots to: {out_dir}")


if __name__ == "__main__":
    main()
