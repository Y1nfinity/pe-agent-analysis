# tools/datasets.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

REQUIRED_PAIR_COLS = [
    "Entry_DealID","Entry_Company","Entry_DealDate","Entry_DealSize","Entry_DealType","Entry_DealType2","Entry_DealType3",
    "Exit_DealID","Exit_Company","Exit_DealDate","Exit_DealSize","Exit_DealType","Exit_DealType2","Exit_DealType3",
]

MIN_UNMATCHED_REQ = ["Entry_DealID","Entry_Company","Entry_DealDate"]
OPT_UNMATCHED     = ["Entry_DealSize","Entry_DealType","Entry_DealType2","Entry_DealType3"]

def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _norm_header(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-_]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

# 1) Exact map for YOUR unmatched headers → Entry_* schema
EXACT_UNMATCHED_TO_ENTRY = {
    "DealID":        "Entry_DealID",
    "Company":       "Entry_Company",
    "DealDate":      "Entry_DealDate",
    "DealSize":      "Entry_DealSize",
    "DealType":      "Entry_DealType",
    "DealType2":     "Entry_DealType2",
    "DealType3":     "Entry_DealType3",
    # others in your file are not required for denominators; we keep them as-is
}

# 2) Fallback alias map (normalized header → Entry_*)
UNMATCHED_ALIASES = {
    "dealid": "Entry_DealID",
    "companies": "Entry_Company",
    "company": "Entry_Company",
    "dealdate": "Entry_DealDate",
    "dealsize": "Entry_DealSize",
    "dealtype": "Entry_DealType",
    "dealtype2": "Entry_DealType2",
    "dealtype3": "Entry_DealType3",
    "entrydate": "Entry_DealDate",
    "entrysize": "Entry_DealSize",
    "purchaseprice": "Entry_DealSize",
    "transactionvalue": "Entry_DealSize",
}

def _auto_map_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce unmatched CSV into Entry_* schema.
    Priority:
      (a) exact rename using your provided headers,
      (b) alias-based rename using normalized names.
    Only Entry_DealID, Entry_Company, Entry_DealDate are strictly required.
    """
    # (a) exact rename if keys exist
    rename_exact = {c: EXACT_UNMATCHED_TO_ENTRY[c] for c in df.columns if c in EXACT_UNMATCHED_TO_ENTRY}
    df2 = df.rename(columns=rename_exact).copy()

    # (b) fill any remaining via alias mapping on normalized headers
    norm2orig = {_norm_header(c): c for c in df2.columns}
    for norm_name, canonical in UNMATCHED_ALIASES.items():
        if canonical not in df2.columns and norm_name in norm2orig:
            df2 = df2.rename(columns={norm2orig[norm_name]: canonical})

    # Ensure minimal required
    missing_min = [c for c in MIN_UNMATCHED_REQ if c not in df2.columns]
    if missing_min:
        raise ValueError(
            "Unmatched CSV missing minimal required columns even after mapping.\n"
            f"Required: {MIN_UNMATCHED_REQ}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Create optional if missing
    for c in OPT_UNMATCHED:
        if c not in df2.columns:
            df2[c] = np.nan

    # Dates
    df2 = _parse_dates(df2, ["Entry_DealDate"])
    # Convenience
    df2["EntryYear"] = df2["Entry_DealDate"].dt.year
    df2["IsCensored"] = True
    return df2

def load_pairs_and_unmatched(cfg_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg_path = Path(cfg_path)
    import tomllib
    cfg = tomllib.loads(cfg_path.read_text())

    root = cfg_path.parent
    pairs_path     = root / cfg["datasets"]["pairs_csv"]
    unmatched_path = root / cfg["datasets"]["unmatched_csv"]

    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")
    if not unmatched_path.exists():
        raise FileNotFoundError(f"Unmatched CSV not found: {unmatched_path}")

    pairs = pd.read_csv(pairs_path)
    unmatched_raw = pd.read_csv(unmatched_path)

    # Pairs must be standard
    miss_p = [c for c in REQUIRED_PAIR_COLS if c not in pairs.columns]
    if miss_p:
        raise ValueError(f"pairs CSV missing columns: {miss_p}")

    # Normalize unmatched using your header mapping
    unmatched = _auto_map_unmatched(unmatched_raw)

    # Pairs extra convenience
    pairs = _parse_dates(pairs, ["Entry_DealDate","Exit_DealDate"])
    pairs["EntryYear"] = pairs["Entry_DealDate"].dt.year
    pairs["ExitYear"]  = pairs["Exit_DealDate"].dt.year
    pairs["HoldingYears"] = (pairs["Exit_DealDate"] - pairs["Entry_DealDate"]).dt.days / 365.25

    # Unified entries (for denominators)
    entries_cols = [
        "Entry_DealID","Entry_Company","Entry_DealDate","Entry_DealSize",
        "Entry_DealType","Entry_DealType2","Entry_DealType3","EntryYear"
    ]
    entries_pairs = pairs[entries_cols].copy()
    entries_unmatched = unmatched[entries_cols].copy()
    entries_all = pd.concat(
        [entries_pairs.assign(Source="paired"), entries_unmatched.assign(Source="unmatched")],
        ignore_index=True
    )

    return pairs, unmatched, entries_all


