# tools/matching.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import os
import zipfile
import pandas as pd

# Map source columns -> normalized names (only those present will be renamed)
COLMAP = {
    "Deal ID": "DealID",
    "Companies": "Company",
    "Deal Date": "DealDate",
    "Deal Type": "DealType",
    "Deal Type 2": "DealType2",
    "Deal Type 3": "DealType3",
    "Deal Size": "DealSize",
    "Deal Status": "DealStatus",
    "Investor Funds": "InvestorFund",
    "Deal Synopsis": "DealSynopsis",
    "Sponsor": "Sponsor",
    "Primary Industry Code": "PrimaryIndustry",
}

ENTRY_TYPES = {"buyout/lbo", "pe growth/expansion"}
EXIT_TYPES  = {"merger/acquisition", "ipo", "reverse merger", "merger of equals"}

def _assert_valid_xlsx(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")
    try:
        with zipfile.ZipFile(path) as z:
            if "[Content_Types].xml" not in z.namelist():
                raise ValueError(f"Not a valid .xlsx (missing [Content_Types].xml): {path}")
    except zipfile.BadZipFile:
        # Some Excel files (older .xls) won’t be zip; let pandas handle .xls/.xlsm too.
        # We only strictly validate .xlsx files.
        if path.suffix.lower() == ".xlsx":
            raise ValueError(f"Not a readable .xlsx ZIP: {path}")

def _first_or_named_sheet(xls: pd.ExcelFile, sheet_name: Optional[str]) -> str:
    if sheet_name and sheet_name in xls.sheet_names:
        return sheet_name
    return xls.sheet_names[0]

def load_excel(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel sheet (first sheet if sheet_name is None/empty), rename known columns, parse dates, normalize."""
    _assert_valid_xlsx(path)
    xls = pd.ExcelFile(path, engine="openpyxl")
    use_sheet = _first_or_named_sheet(xls, sheet_name if sheet_name else None)
    df = pd.read_excel(xls, sheet_name=use_sheet, engine="openpyxl")

    # rename known columns if present
    rename = {c: COLMAP[c] for c in df.columns if c in COLMAP}
    df = df.rename(columns=rename)

    if "DealDate" in df.columns:
        df["DealDate"] = pd.to_datetime(df["DealDate"], errors="coerce")

    if "Company" in df.columns:
        df["Company_norm"] = (
            df["Company"].astype(str)
            .str.lower().str.strip()
            .str.replace(r"[\,\.\-]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
        )

    if "DealID" in df.columns:
        df["DealID"] = df["DealID"].astype(str)

    return df

def add_entry_exit_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out.get("DealType", pd.Series(index=out.index, dtype="object")).astype(str).str.lower()
    out["IsEntry"] = dt.isin(ENTRY_TYPES)
    out["IsExit"]  = dt.isin(EXIT_TYPES)
    return out

def _holding_years(entry_date: pd.Timestamp, exit_date: pd.Timestamp):
    if pd.isna(entry_date) or pd.isna(exit_date):
        return pd.NA
    return (exit_date - entry_date).days / 365.25

def build_pairs_from_seeds(
    p2p_seed: pd.DataFrame,
    universe: pd.DataFrame,
    topk: int = 3,
    min_hold_days: int = 0,
    max_hold_years: Optional[float] = None,
):
    """Match each seed entry to the nearest exit in the universe (same normalized company, date after entry, different DealID)."""
    def within_bounds(hy):
        if pd.isna(hy):
            return False
        if min_hold_days and hy < (min_hold_days / 365.25):
            return False
        if max_hold_years is not None and hy > max_hold_years:
            return False
        return True

    # Ensure flags on universe; treat seed rows as entries
    universe_f = add_entry_exit_flags(universe)
    seed_f = add_entry_exit_flags(p2p_seed)
    if "IsEntry" in seed_f.columns:
        seed_f = seed_f[seed_f["IsEntry"]].copy()

    pairs, unmatched, candidates = [], [], []

    for _, seed in seed_f.iterrows():
        comp = seed.get("Company_norm", "")
        entry_date = seed.get("DealDate", pd.NaT)
        seed_dealid = str(seed.get("DealID", ""))

        if not comp or pd.isna(entry_date):
            unmatched.append(seed)
            continue

        cand = universe_f[
            (universe_f["Company_norm"] == comp) &
            (universe_f["IsExit"]) &
            (universe_f["DealDate"] > entry_date)
        ].copy()

        if "DealID" in cand.columns:
            cand = cand[cand["DealID"].astype(str) != seed_dealid]  # prevent self-match

        if cand.empty:
            unmatched.append(seed)
            continue

        cand["EntryDate"] = entry_date
        cand["HoldingYears"] = cand["DealDate"].apply(lambda d: _holding_years(entry_date, d))
        cand["AbsDaysAfter"] = (cand["DealDate"] - entry_date).dt.days
        cand = cand[cand["HoldingYears"].apply(within_bounds)]
        cand = cand.sort_values(["AbsDaysAfter", "DealDate"])

        if cand.empty:
            unmatched.append(seed)
            continue

        # Keep top-k candidates for QA
        keep_cols = ["DealID","Company","Company_norm","DealType","DealType2","DealType3","DealDate","DealSize","DealStatus","EntryDate","HoldingYears","AbsDaysAfter"]
        cand_out = cand[keep_cols].head(topk).copy()
        cand_out["Seed_DealID"]   = seed.get("DealID")
        cand_out["Seed_Company"]  = seed.get("Company")
        cand_out["Seed_DealDate"] = seed.get("DealDate")
        cand_out["Seed_DealSize"] = seed.get("DealSize")
        candidates.append(cand_out)

        # Choose best (nearest exit)
        best = cand.iloc[0:1].copy()
        best = best.rename(columns={
            "DealID":"Exit_DealID",
            "Company":"Exit_Company",
            "DealDate":"Exit_DealDate",
            "DealSize":"Exit_DealSize",
            "DealType":"Exit_DealType",
            "DealType2":"Exit_DealType2",
            "DealType3":"Exit_DealType3",
            "DealStatus":"Exit_DealStatus",
        })
        best["Entry_DealID"]    = seed.get("DealID")
        best["Entry_Company"]   = seed.get("Company")
        best["Entry_DealDate"]  = seed.get("DealDate")
        best["Entry_DealSize"]  = seed.get("DealSize")
        best["Entry_DealType"]  = seed.get("DealType")
        best["Entry_DealType2"] = seed.get("DealType2")
        best["Entry_DealType3"] = seed.get("DealType3")

        pairs.append(best[[
            "Entry_DealID","Entry_Company","Entry_DealDate","Entry_DealSize","Entry_DealType","Entry_DealType2","Entry_DealType3",
            "Exit_DealID","Exit_Company","Exit_DealDate","Exit_DealSize","Exit_DealType","Exit_DealType2","Exit_DealType3",
            "HoldingYears","AbsDaysAfter"
        ]])

    pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    unmatched_df = pd.DataFrame(unmatched) if unmatched else pd.DataFrame()
    candidates_df = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return pairs_df, unmatched_df, candidates_df
