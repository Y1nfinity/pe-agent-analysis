from pathlib import Path
import pandas as pd
from tools.io import load_and_clean
from tools.transforms import add_p2p_flag, add_entry_exit_flags
from tools.pairing import pair_entry_exit   # <-- new import
from tools.holding import holding_stats

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config.toml"

    # Load & flags
    df, _ = load_and_clean(cfg)
    df = add_p2p_flag(df)
    df = add_entry_exit_flags(df)

    # Pair with basic filters (tune as needed)
    pairs, audit = pair_entry_exit(
        df,
        restrict_same_sponsor=False,   # set True if you want sponsor-consistent pairing
        min_hold_days=0,               # e.g., try 30 to remove same-month flips
        max_gap_years=None             # e.g., set 25 to drop extreme long chains
    )

    # Quick stats
    print("Rows (events):", len(df))
    print("Entries:", int(df["IsEntry"].sum()), "| Exits:", int(df["IsExit"].sum()))
    print("Paired exits:", len(pairs))
    print("\nHoldingYears (describe):")
    print(pairs["HoldingYears"].describe())

    # Save outputs
    outdir = root / "artifacts" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(outdir / "paired_holding_periods.csv", index=False)
    audit.to_csv(outdir / "pairing_audit.csv", index=False)
    print(f"\nSaved:\n- {outdir/'paired_holding_periods.csv'}\n- {outdir/'pairing_audit.csv'}")

    print(holding_stats(pairs))
