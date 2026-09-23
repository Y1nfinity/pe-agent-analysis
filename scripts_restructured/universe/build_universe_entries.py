"""
build_universe_entries.py
=========================

Extracts PURE entry deals from the PitchBook master file.
These form the denominator for entry-year exit completion analysis.

Input:
    data/clean/pitchbook_master_clean.parquet

Output:
    data/clean/universe_entries.parquet
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path


ENTRY_TYPES_PRIMARY = {
    "buyout/lbo",
    "management buyout",
    "management buy-in",
    "lbo",
    "buyout",
}

ENTRY_TYPES_SECONDARY = {
    "public to private",
    "take-private",
    "secondary buyout",
    "add-on",
    "asset acquisition",
    "corporate divestiture",
    "spin-off",
    "distressed acquisition",
    "recapitalization",   # minor, can be flagged
}

def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_entries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize deal types lower-case
    def norm(x):
        return str(x).strip().lower() if pd.notnull(x) else ""

    df["deal_type_norm"] = df["deal_type"].map(norm)
    df["deal_type2_norm"] = df.get("Deal Type 2", "").map(norm)
    df["deal_type3_norm"] = df.get("Deal Type 3", "").map(norm)

    # Identify entry events
    df["is_entry"] = (
        df["deal_type_norm"].isin(ENTRY_TYPES_PRIMARY)
        | df["deal_type2_norm"].isin(ENTRY_TYPES_PRIMARY.union(ENTRY_TYPES_SECONDARY))
        | df["deal_type3_norm"].isin(ENTRY_TYPES_PRIMARY.union(ENTRY_TYPES_SECONDARY))
    )

    entries = df[df["is_entry"]].copy()

    # Use PitchBook deal date as entry date
    entries = entries.rename(columns={"deal_date": "entry_date"})
    entries["entry_year"] = entries["entry_date"].dt.year

    # Normalize deal size for entry value
    entries["deal_size_usd_m_entry"] = entries["deal_size_usd_m"]

    # Keep only usable columns
    keep = [
        "deal_id",
        "company_name_clean",
        "entry_date",
        "entry_year",
        "deal_type",
        "deal_size_usd_m_entry",
        "exit_category",  # may exist, but not used here
        "HQ Location"
    ]

    entries = entries[keep].copy()

    return entries


def run_build_entries(master_path: Path, out_path: Path):
    print(f"🔹 Loading master universe file:\n   {master_path}")
    df = load_master(master_path)

    print("🔹 Filtering to entry-only deals...")
    entries = build_entries(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries.to_parquet(out_path, index=False)

    print(f"✅ Saved universe entries → {out_path.resolve()}")
    print(f"   Rows: {len(entries)}")
    print("   (All unmatched exits will be handled in the next script.)")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    MASTER = ROOT / "data" / "clean" / "pitchbook_master_clean.parquet"
    OUT    = ROOT / "data" / "clean" / "universe_entries.parquet"
    run_build_entries(MASTER, OUT)
