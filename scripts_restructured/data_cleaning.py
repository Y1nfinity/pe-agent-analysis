"""
data_cleaning.py — CANONICAL VERSION
===================================
Produces schema-clean, categorization-safe tables
for entry-exit linking.

Exports:

    load_raw_data()
    prepare_entry_data()
    prepare_exit_data()

Guarantees:
    - all_deal_types exists (for categorize_exits)
    - all_deal_types_entry
    - all_deal_types_exit
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scripts_restructured.utilities.name_cleaning import clean_company_name


# ============================================================================
# Utilities
# ============================================================================

def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path.suffix}")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
                  .str.replace(" ", "_")
                  .str.replace("/", "_")
                  .str.replace("-", "_")
    )
    return df


def _to_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def _to_usd_m(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value / 1e6 if value > 1e6 else float(value)

    s = str(value).lower().replace(",", "").replace("$", "")
    try:
        if "bn" in s:
            return float(s.replace("bn", "")) * 1000
        if "m" in s:
            return float(s.replace("m", ""))
        if "k" in s:
            return float(s.replace("k", "")) / 1000
        x = float(s)
        return x / 1e6 if x > 1e6 else x
    except:
        return np.nan


def _combine_types(df: pd.DataFrame, pattern: str) -> pd.Series:
    cols = [c for c in df.columns if pattern in c]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].astype(str).agg("; ".join, axis=1)


# ============================================================================
# Entry Cleaning
# ============================================================================

def prepare_entry_data(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = _clean_columns(df_raw)

    name_cols = ["companies", "company", "portfolio_company", "company_name"]
    date_cols = ["deal_date", "announce_date", "entry_date"]
    size_cols = ["deal_size", "entry_size", "value"]
    type_cols = [c for c in df.columns if "deal_type" in c]

    for c in name_cols:
        if c in df:
            df["company_name_clean"] = df[c].astype(str).map(clean_company_name)
            break
    else:
        raise ValueError("No company name column in entries data")

    for c in date_cols:
        if c in df:
            df["entry_date"] = _to_datetime(df[c])
            break
    else:
        raise ValueError("No deal date in entries data")

    df["entry_type"] = df[type_cols].iloc[:, 0] if type_cols else pd.NA
    df["all_deal_types"] = _combine_types(df, "deal_type")
    df["all_deal_types_entry"] = df["all_deal_types"]

    df["deal_size_usd_m_entry"] = np.nan
    for c in size_cols:
        if c in df:
            df["deal_size_usd_m_entry"] = df[c].apply(_to_usd_m)
            break

    keep = [
        "company_name_clean",
        "entry_date",
        "entry_type",
        "deal_size_usd_m_entry",
        "all_deal_types",
        "all_deal_types_entry",
    ]
    if "deal_id" in df:
        df["deal_id"] = df["deal_id"].astype(str)
        keep.append("deal_id")

    return df[keep]


# ============================================================================
# Exit Cleaning
# ============================================================================

def prepare_exit_data(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = _clean_columns(df_raw)

    name_cols = ["companies", "company", "portfolio_company", "company_name"]
    date_cols = ["deal_date", "exit_date", "announcement_date"]
    size_cols = ["deal_size", "post_valuation", "value"]
    type_cols = [c for c in df.columns if "deal_type" in c]

    for c in name_cols:
        if c in df:
            df["company_name_clean"] = df[c].astype(str).map(clean_company_name)
            break
    else:
        raise ValueError("No company name column in exits data")

    for c in date_cols:
        if c in df:
            df["exit_date"] = _to_datetime(df[c])
            break
    else:
        raise ValueError("No exit date in exits data")

    df["exit_type"] = df[type_cols].iloc[:, 0] if type_cols else pd.NA

    df["all_deal_types"] = _combine_types(df, "deal_type")
    df["all_deal_types_exit"] = df["all_deal_types"]

    df["deal_size_usd_m_exit"] = np.nan
    for c in size_cols:
        if c in df:
            df["deal_size_usd_m_exit"] = df[c].apply(_to_usd_m)
            break

    keep = [
        "company_name_clean",
        "exit_date",
        "exit_type",
        "deal_size_usd_m_exit",
        "all_deal_types",
        "all_deal_types_exit",
    ]
    if "deal_id" in df:
        df["deal_id"] = df["deal_id"].astype(str)
        keep.append("deal_id")

    return df[keep]
