from scripts_restructured import data_cleaning as dc
from scripts_restructured import exit_categorization as ec
from scripts_restructured import analysis as an
from scripts_restructured import report_utils as ru

# --- Load and rebuild analysis ---
merged = dc.load_raw_data("data/clean/exits_merged.parquet")
categorized = ec.assign_exit_category(merged, source="pitchbook")
categorized = an.ensure_year_column(categorized)
summaries = an.build_summary_package(categorized)

# --- Export summary tables ---
ru.export_table(
    summaries["counts_by_year"],
    "outputs/tables/Exit_Counts_By_Year.csv",
    "Annual number of private equity exits (PitchBook sample, 1977–2025)."
)

ru.export_table(
    summaries["counts_by_year_and_category"],
    "outputs/tables/Exit_Mix_By_Category.csv",
    "Exit composition by category per year (counts and shares)."
)

# --- Attach figure captions & organize folder ---
ru.build_graphs_folder({
    "Figure_01_Exits_By_Year.png": "Annual PE exit volume, 1977–2025.",
    "Figure_02_ExitMix_Counts.png": "Exit mix by category (counts).",
    "Figure_03_ExitMix_Shares.png": "Exit mix by category (share of total exits).",
    "Figure_04_TP_vs_SBO.png": "Take-private vs secondary buyout trends."
})

print("✅ Tables and figure captions exported to outputs/ folder.")
