"""
diagnose_entry_exit_parquet.py
==============================

Diagnostics for:
    data/clean/entry_exit_pairs.parquet

Goal:
    - Verify shape and schema
    - Inspect matching quality
    - Check exit-type composition
    - Validate dollarization
    - Sanity-check holding periods
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path


def main():
    ROOT = Path(__file__).resolve().parents[1]
    path = ROOT / "data" / "clean" / "entry_exit_pairs.parquet"

    print(f"🔹 Loading parquet:\n   {path}")
    df = pd.read_parquet(path)

    print(f"\n✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    # --- 1. Schema --------------------------------------------------------
    print("📌 Columns:")
    print(df.columns.tolist())

    print("\n📌 Data types:")
    print(df.dtypes)

    print("\n📌 Sample rows:")
    print(df.head(10))

    # --- 2. Coverage by entry year ---------------------------------------
    df["entry_year"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.year

    print("\n📊 Entries per year:")
    print(df["entry_year"].value_counts().sort_index())

    # --- 3. Match quality ------------------------------------------------
    print("\n📊 Match type breakdown:")
    if "match_type" in df.columns:
        print(df["match_type"].value_counts(dropna=False))
    else:
        print("⚠️ No 'match_type' column found.")

    print("\n📊 Fuzzy match score distribution:")
    if "match_score" in df.columns:
        print(df["match_score"].describe())
    else:
        print("⚠️ No 'match_score' column found.")

    # --- 4. Exit composition ---------------------------------------------
    print("\n📊 Exit category breakdown:")
    if "exit_category" in df.columns:
        print(df["exit_category"].value_counts(dropna=False))
    else:
        print("⚠️ No 'exit_category' column found.")

    # --- 5. Holding periods ----------------------------------------------
    if "holding_period_years" in df.columns:
        print("\n📊 Holding period summary (years):")
        print(df["holding_period_years"].describe())

        print("\n📊 Holding-period distribution (buckets):")
        bins = [0, 1, 3, 5, 10, 20, 50]
        print(pd.cut(df["holding_period_years"], bins=bins).value_counts().sort_index())
    else:
        print("⚠️ No 'holding_period_years' column found.")

    # --- 6. Dollarization -------------------------------------------------
    print("\n💰 Entry deal size coverage:")
    if "deal_size_usd_m_entry" in df.columns:
        print(df["deal_size_usd_m_entry"].notna().mean().round(3))
        print(df["deal_size_usd_m_entry"].describe())
    else:
        print("⚠️ No 'deal_size_usd_m_entry' column found.")

    print("\n💰 Exit deal size coverage:")
    if "deal_size_usd_m_exit" in df.columns:
        print(df["deal_size_usd_m_exit"].notna().mean().round(3))
        print(df["deal_size_usd_m_exit"].describe())
    else:
        print("⚠️ No 'deal_size_usd_m_exit' column found.")

    # --- 7. Uniqueness & duplication -------------------------------------
    print("\n🔎 Unique firms:")
    print(df["company_name_clean"].nunique())

    print("\n🔎 Duplicate firm/date pairs:")
    dup = df.duplicated(subset=["company_name_clean", "entry_date"])
    print(dup.sum())

    # --- 8. Rows with obvious issues -------------------------------------
    print("\n🚨 Potential issues:")

    print("• Negative holding periods:",
          (df["holding_period_years"] < 0).sum() if "holding_period_years" in df else "N/A")

    print("• Missing entry dates:",
          df["entry_date"].isna().sum() if "entry_date" in df else "N/A")

    print("• Missing exit dates:",
          df["exit_date"].isna().sum() if "exit_date" in df else "N/A")

    print("• Entries with no dollar value:",
          df["deal_size_usd_m_entry"].isna().sum() if "deal_size_usd_m_entry" in df else "N/A")

    print("• Exits with no dollar value:",
          df["deal_size_usd_m_exit"].isna().sum() if "deal_size_usd_m_exit" in df else "N/A")

    print("\n✅ Diagnosis complete.")


if __name__ == "__main__":
    main()
