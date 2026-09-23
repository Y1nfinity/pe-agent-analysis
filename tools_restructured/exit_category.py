from scripts_restructured import exit_categorization as ec
from scripts_restructured import data_cleaning as dc

merged = dc.load_raw_data("data/clean/exits_merged.parquet")

categorized = ec.assign_exit_category(merged, source="pitchbook")
categorized = ec.flag_key_segments(categorized)

print("✅ Categorized successfully:", categorized.shape)
print(categorized["exit_category"].value_counts())


# Show top unique values in exit_type and secondary columns
#for col in ["exit_type", "exit_type_2", "exit_type_3"]:
    #if col in categorized.columns:
        #print(f"\n{col} unique values:")
        #print(categorized[col].dropna().unique())
