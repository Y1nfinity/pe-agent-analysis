from pathlib import Path
import pandas as pd

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = Path(r"C:\Users\azhao\PycharmProjects\PEAgent\data\raw\pitchbook_exit_completed.xlsx")

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_excel(file_path)

# ============================================================
# 2. VALIDATE REQUIRED COLUMN
# ============================================================

if "Deal Synopsis" not in df.columns:
    raise ValueError("Column 'Deal Synopsis' not found in dataset.")

# Preserve raw text column for export / review
df["Deal Synopsis"] = df["Deal Synopsis"].astype(str)
synopsis = df["Deal Synopsis"].str.lower()

# ============================================================
# 3. DEFINE TARGET PHRASES
# ============================================================

phrases = [
    "undisclosed sum",
    "undisclosed",
    "unreported",
]

# ============================================================
# 4. PHRASE FREQUENCY COUNTS
# ============================================================

results = []

for phrase in phrases:
    mask = synopsis.str.contains(phrase, na=False, regex=False)
    count = int(mask.sum())
    pct = (count / len(df)) * 100 if len(df) > 0 else 0.0

    results.append({
        "phrase": phrase,
        "count": count,
        "percent_of_rows": pct
    })

results_df = pd.DataFrame(results)

print(f"\nDataset rows: {len(df):,}\n")
print("Phrase frequency in 'Deal Synopsis':\n")
print(results_df.to_string(index=False))

# ============================================================
# 5. EXPORT ALL MATCHING ROWS TO CSV
# ============================================================

def identify_phrase(text: str) -> str | None:
    text = str(text).lower()
    for p in phrases:
        if p in text:
            return p
    return None

pattern = "|".join(phrases)
match_mask = synopsis.str.contains(pattern, na=False, regex=True)

matching_rows = df.loc[match_mask].copy()
matching_rows["matched_phrase"] = matching_rows["Deal Synopsis"].apply(identify_phrase)

output_csv = file_path.parent / "deal_synopsis_phrase_matches.csv"
matching_rows.to_csv(output_csv, index=False)

print(f"\nExported {len(matching_rows):,} matching rows.")
print(f"CSV saved to: {output_csv}")

# ============================================================
# 6. ADDITIONAL DIAGNOSTIC:
#    YEARLY FREQUENCY OF MATCHES
# ============================================================

# Try to identify a usable deal-date column
possible_date_cols = [
    "Deal Date",
    "deal_date",
    "Entry Date",
    "entry_date",
    "Exit Date",
    "exit_date",
]

date_col = next((c for c in possible_date_cols if c in df.columns), None)

if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["deal_year"] = df[date_col].dt.year
    df["has_target_phrase"] = match_mask

    yearly_diag = (
        df.dropna(subset=["deal_year"])
          .groupby("deal_year")
          .agg(
              total_rows=("has_target_phrase", "size"),
              matched_rows=("has_target_phrase", "sum"),
          )
          .reset_index()
    )

    yearly_diag["matched_percent"] = (
        yearly_diag["matched_rows"] / yearly_diag["total_rows"] * 100
    )

    yearly_output_csv = file_path.parent / "deal_synopsis_phrase_matches_by_year.csv"
    yearly_diag.to_csv(yearly_output_csv, index=False)

    print("\nYearly diagnostic:")
    print(yearly_diag.to_string(index=False))
    print(f"\nYearly diagnostic CSV saved to: {yearly_output_csv}")

else:
    print("\nNo recognized date column found, so yearly diagnostic was skipped.")

# ============================================================
# 7. ADDITIONAL DIAGNOSTIC:
#    FREQUENCY BY DEAL TYPE
# ============================================================

possible_type_cols = [
    "Deal Type",
    "deal_type",
    "Deal Type 2",
    "Deal Type 3",
]

type_col = next((c for c in possible_type_cols if c in df.columns), None)

if type_col is not None:
    df[type_col] = df[type_col].astype(str)
    df["has_target_phrase"] = match_mask

    type_diag = (
        df.groupby(type_col)
          .agg(
              total_rows=("has_target_phrase", "size"),
              matched_rows=("has_target_phrase", "sum"),
          )
          .reset_index()
          .sort_values(["matched_rows", "total_rows"], ascending=[False, False])
    )

    type_diag["matched_percent"] = (
        type_diag["matched_rows"] / type_diag["total_rows"] * 100
    )

    type_output_csv = file_path.parent / "deal_synopsis_phrase_matches_by_type.csv"
    type_diag.to_csv(type_output_csv, index=False)

    print("\nDeal-type diagnostic:")
    print(type_diag.head(25).to_string(index=False))
    print(f"\nDeal-type diagnostic CSV saved to: {type_output_csv}")

else:
    print("\nNo recognized deal type column found, so deal-type diagnostic was skipped.")

# ============================================================
# 8. OPTIONAL SUMMARY FILE
# ============================================================

summary_output_csv = file_path.parent / "deal_synopsis_phrase_summary.csv"
results_df.to_csv(summary_output_csv, index=False)

print(f"\nPhrase summary CSV saved to: {summary_output_csv}")
print("\nDone.")