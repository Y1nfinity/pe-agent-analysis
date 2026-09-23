"""
diagnose_pitchbook_raw.py
=========================

Quick structural and content diagnostics for:
    data/raw/pitchbook_export.xlsx

Goal:
    - Understand what this file actually contains
    - Check if it is:
        * All PE buyouts?
        * Only take-privates?
        * A broader PE universe?
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path


def main():
    ROOT = Path(__file__).resolve().parents[1]
    raw_path = ROOT / "data" / "raw" / "pitchbook_export.xlsx"

    print(f"🔹 Loading raw PitchBook file:\n   {raw_path}")
    df = pd.read_excel(raw_path, sheet_name=0)
    print(f"\n✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    # --- 1. Columns & basic info ------------------------------------------
    print("📌 Columns:")
    print(df.columns.tolist())

    print("\n📌 Sample rows:")
    print(df.head(10))

    # --- 2. Deal date / year coverage -------------------------------------
    df["deal_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    df["year"] = df["deal_date"].dt.year

    year_counts = df["year"].value_counts().sort_index()
    print("\n📊 Deals per year:")
    print(year_counts)

    # --- 3. Deal type structure -------------------------------------------
    print("\n📊 Top 20 Deal Type values:")
    print(df["Deal Type"].value_counts(dropna=False).head(20))

    print("\n📊 Top 20 Deal Type 2 values:")
    if "Deal Type 2" in df.columns:
        print(df["Deal Type 2"].value_counts(dropna=False).head(20))
    else:
        print("  (No 'Deal Type 2' column found)")

    print("\n📊 Top 20 Deal Type 3 values:")
    if "Deal Type 3" in df.columns:
        print(df["Deal Type 3"].value_counts(dropna=False).head(20))
    else:
        print("  (No 'Deal Type 3' column found)")

    # --- 4. Financing / status fields -------------------------------------
    for col in ["Deal Status", "Financing Status", "Business Status"]:
        if col in df.columns:
            print(f"\n📊 {col} value counts:")
            print(df[col].value_counts(dropna=False).head(20))

    # --- 5. Take-private intensity ----------------------------------------
    # Look for "Public to Private", "Take-Private", etc. in deal-type fields
    def is_take_private(row) -> bool:
        text = " ".join(
            str(row.get(c, "")) for c in ["Deal Type", "Deal Type 2", "Deal Type 3"]
        ).lower()
        return ("public to private" in text) or ("take-private" in text)

    df["is_tp_flag"] = df.apply(is_take_private, axis=1)

    tp_share = df["is_tp_flag"].mean()
    print(f"\n📊 Approximate take-private share of this file: {tp_share:.2%}")

    print("\n📊 Take-private deals per year:")
    tp_year_counts = df.loc[df["is_tp_flag"], "year"].value_counts().sort_index()
    print(tp_year_counts)

    # --- 6. Deal size coverage --------------------------------------------
    if "Deal Size" in df.columns:
        df["deal_size"] = pd.to_numeric(df["Deal Size"], errors="coerce")
        coverage = df["deal_size"].notna().mean()
        print(f"\n📊 Deal size non-missing coverage: {coverage:.2%}")
        print("\n🔎 Deal size summary (non-missing):")
        print(df["deal_size"].describe())
    else:
        print("\n⚠️ No 'Deal Size' column found — cannot dollarize.")


if __name__ == "__main__":
    main()
