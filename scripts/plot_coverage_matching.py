from __future__ import annotations
from pathlib import Path
import argparse
import tomllib

# import loaders & charts directly from your existing script
from scripts.plot_p2p_overview import (
    read_tables,
    bar_paired_unmatched_by_vintage,
    bar_paired_unmatched_spend_by_vintage,
    stacked_paired_unmatched_fraction_by_vintage,
    ensure_dir,
)

def main():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "config.toml").read_text())

    ap = argparse.ArgumentParser(description="Generate matched vs. unmatched coverage charts only.")
    ap.add_argument("--out-dir", default=str(root / "artifacts" / "plots"))
    ap.add_argument("--which", choices=["all","counts","spend","fractions"], default="all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    per_deal, breakdown, vint_over, unmatched_v = read_tables(root, cfg)

    if args.which in ("all", "counts"):
        bar_paired_unmatched_by_vintage(vint_over, breakdown, unmatched_v,
                                        out_dir / "paired_vs_unmatched_by_vintage.png")

    if args.which in ("all", "spend"):
        bar_paired_unmatched_spend_by_vintage(vint_over, breakdown, unmatched_v,
                                              out_dir / "paired_vs_unmatched_by_spend.png")

    if args.which in ("all", "fractions"):
        stacked_paired_unmatched_fraction_by_vintage(vint_over, breakdown, unmatched_v,
                                                     out_dir / "paired_unmatched_fractions_by_vintage.png")

    print("Saved coverage plots to:", out_dir)

if __name__ == "__main__":
    main()
