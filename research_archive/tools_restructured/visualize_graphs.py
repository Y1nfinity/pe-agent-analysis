# visualize_graphs.py
from scripts_restructured import data_cleaning as dc
from scripts_restructured import exit_categorization as ec
from scripts_restructured import analysis as an
from scripts_restructured import visualization as vis

# --- Step 1. Load cleaned data ---
merged = dc.load_raw_data("data/clean/exits_merged.parquet")

# --- Step 2. Categorize ---
categorized = ec.assign_exit_category(merged, source="pitchbook")

# --- Step 3. Add year column & build summaries ---
categorized = an.ensure_year_column(categorized)
summaries = an.build_summary_package(categorized)

# --- Step 4. Plot figures ---
vis.plot_exits_by_year(
    summaries["counts_by_year"],
    title="Private Equity Exits by Year",
    outfile="outputs/figures/Figure_01_Exits_By_Year.png"
)

vis.plot_exit_mix_by_category(
    summaries["counts_by_year_and_category"],
    title="Exit Mix by Year (Counts)",
    outfile="outputs/figures/Figure_02_ExitMix_Counts.png"
)

vis.plot_exit_share_by_category(
    summaries["counts_by_year_and_category"],
    title="Exit Mix by Year (Shares)",
    outfile="outputs/figures/Figure_03_ExitMix_Shares.png"
)

vis.plot_take_private_vs_private_to_private(
    summaries["counts_by_year_and_category"],
    title="Take-Private vs Secondary Buyout Exits",
    outfile="outputs/figures/Figure_04_TP_vs_SBO.png"
)

print("✅ Visualization complete — all figures saved to outputs/figures/")
