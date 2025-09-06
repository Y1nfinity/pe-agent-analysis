from pathlib import Path
import pandas as pd

# --- locate the cleaned parquet ---
project_root = Path(__file__).resolve().parents[1]
clean_path = project_root / "data" / "interim" / "clean.parquet"

print(f"Reading: {clean_path}")
df = pd.read_parquet(clean_path)

# --- basic shape & columns ---
print("\n=== SHAPE & COLUMNS ===")
print(df.shape)
print(df.columns.tolist())

# --- dtypes (so we see dates/numbers are correct) ---
print("\n=== DTYPES ===")
print(df.dtypes)

# --- quick head ---
print("\n=== HEAD(10) ===")
print(df.head(10).to_string())

# --- date range (sanity check) ---
if "DealDate" in df.columns:
    print("\n=== DEAL DATE RANGE ===")
    print(df["DealDate"].min(), "→", df["DealDate"].max())

# --- missingness in key columns ---
key_cols = [c for c in ["Company","DealDate","DealType","DealType2","DealType3","DealSize","Sponsor"] if c in df.columns]
print("\n=== NULL COUNTS (key cols) ===")
print(df[key_cols].isna().sum())

# --- top values in deal type columns (helps us define entry/exit sets) ---
def top_vals(col, n=20):
    if col in df.columns:
        print(f"\n=== TOP {n} '{col}' VALUES ===")
        print(df[col].value_counts(dropna=True).head(n).to_string())

for col in ["DealType","DealType2","DealType3"]:
    top_vals(col, n=20)
