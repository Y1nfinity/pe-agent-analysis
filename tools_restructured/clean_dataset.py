from pathlib import Path
from scripts_restructured import data_cleaning as dc
from scripts_restructured import exit_categorization as ec

# --- Automatically find the project root (2 directories up from this file) ---
root = Path(__file__).resolve().parents[1]  # <- might be parents[1] or [2] depending on your structure
raw_dir = root / "data" / "raw"

print(f"Resolved project root: {root}")
print(f"Looking for files in: {raw_dir}")

# --- Load data using absolute paths ---
pb_main = dc.load_raw_data(raw_dir / "pitchbook_export.xlsx")
pb_tpp  = dc.load_raw_data(raw_dir / "public_2_private_all.xlsx")

print("Files loaded successfully ✅")

merged = dc.merge_pitchbook_datasets(pb_main, pb_tpp)
qc = dc.basic_quality_checks(merged)

dc.save_clean(merged, "data/clean/exits_merged.parquet")

print("✅ Cleaned dataset saved:", merged.shape)
print(qc)

print("PitchBook Export Columns:")
print(pb_main.columns.tolist())

print("\nPublic-to-Private Columns:")
print(pb_tpp.columns.tolist())