from scripts_restructured import analysis as an
from scripts_restructured import data_cleaning as dc
from scripts_restructured import exit_categorization as ec

# 1. Load cleaned + categorized data
merged = dc.load_raw_data("data/clean/exits_merged.parquet")
categorized = ec.assign_exit_category(merged, source="pitchbook")

# 2. Add year column
categorized = an.ensure_year_column(categorized)

# 3. Run analytical summaries
summaries = an.build_summary_package(categorized)

# 4. Inspect key tables
for name, df in summaries.items():
    print(f"\n{name} — {df.shape}")
    print(df.head(100))

print(categorized["exit_category"].value_counts(dropna=False))
# 1. How many exits per category?
print(categorized["exit_category"].value_counts())

# 2. What’s your year coverage?
print(categorized["year"].isna().mean())

# 3. Which categories are missing valid years?
print(categorized.groupby("exit_category")["year"].apply(lambda x: x.notna().mean()))

