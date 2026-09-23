"""
link_entry_exit.py
==================
Driver script for linking take-private entries (public_2_private_all.xlsx)
to subsequent private equity exits (pitchbook_export.xlsx).

This script:
  1. Loads both datasets (raw and cleaned).
  2. Prepares and standardizes the data.
  3. Links entries to exits using company names and chronology.
  4. Saves a clean dataset of entry–exit pairs to:
        data/clean/entry_exit_pairs.parquet
"""

from pathlib import Path
from scripts_restructured import data_cleaning as dc
from scripts_restructured import entry_exit_linking as link


# ---------------------------------------------------------------------
# 1. Define paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

print("🔹 Project root:", ROOT)
print("🔹 Data directory:", DATA_DIR)


# ---------------------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------------------
print("\n🔹 Loading input data...")

entries = dc.load_raw_data(RAW_DIR / "public_2_private_all.xlsx")
exits = dc.load_raw_data(RAW_DIR / "pitchbook_export.xlsx")

# If needed for reference or consistency (not required for linking)
try:
    merged = dc.load_raw_data(CLEAN_DIR / "exits_merged.parquet")
    print(f"   Loaded exits_merged.parquet ({merged.shape[0]:,} rows)")
except FileNotFoundError:
    print("⚠️ exits_merged.parquet not found — continuing without it.")


# ---------------------------------------------------------------------
# 3. Prepare and standardize data
# ---------------------------------------------------------------------
print("\n🔹 Preparing datasets...")
entries = link.prepare_entry_data(entries)
exits = link.prepare_exit_data(exits)

print(f"   Entries dataset: {entries.shape}")
print(f"   Exits dataset:   {exits.shape}")


# ---------------------------------------------------------------------
# 4. Link entries to exits
# ---------------------------------------------------------------------
print("\n🔹 Linking entry–exit pairs...")
pairs = link.link_entries_exits(entries, exits, fuzzy=False)

print(f"✅ Successfully linked {pairs.shape[0]:,} entry–exit pairs.")
print("   Example rows:")
print(pairs.head(10))


# ---------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------
out_path = CLEAN_DIR / "entry_exit_pairs.parquet"
pairs.to_parquet(out_path, index=False)

print(f"\n💾 Saved entry–exit pairs to: {out_path.resolve()}")
print("✅ Done.")

# ---------------------------------------------------------------------
# 6. Diagnostics: linkage success rates
# ---------------------------------------------------------------------
n_entries = entries["company_name_clean"].nunique()
n_exits = exits["company_name_clean"].nunique()
n_pairs = pairs["company_name_clean"].nunique()


print("\n📊 LINKAGE DIAGNOSTICS")
print(f"Unique entry companies: {n_entries:,}")
print(f"Unique exit companies:  {n_exits:,}")
print(f"Matched (linked) companies: {n_pairs:,}")
print(f"Match rate (entries matched to exits): {n_pairs / n_entries:.2%}")


print("\n📊 ENTRY–EXIT SANITY CHECK SUMMARY")
print("--------------------------------------------")
print(f"Total linked pairs: {len(pairs):,}")
print("\nHolding period stats:")
print(pairs["holding_period_years"].describe(percentiles=[0.25, 0.5, 0.75]).round(2))

print("\nExit type distribution:")
print(pairs["exit_type"].value_counts())

print("\nShortest 3 holding periods:")
print(pairs.nsmallest(3, "holding_period_years")[["company_name_clean", "entry_date", "exit_date", "holding_period_years"]])

print("\nLongest 3 holding periods:")
print(pairs.nlargest(3, "holding_period_years")[["company_name_clean", "entry_date", "exit_date", "holding_period_years"]])



