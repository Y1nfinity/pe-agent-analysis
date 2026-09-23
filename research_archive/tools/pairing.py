# tools/transforms.py

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np

def pair_entry_exit(
    df: pd.DataFrame,
    company_col: str = "Company",
    date_col: str = "DealDate",
    entry_col: str = "IsEntry",
    exit_col: str = "IsExit",
    sponsor_col: str = "Sponsor",
    # Strategy & filters
    strategy: str = "last_entry_before_exit",  # future: "first_entry_in_window"
    restrict_same_sponsor: bool = False,       # only pair if sponsor matches (entry vs exit)
    min_hold_days: int = 0,                    # drop if exit-entry < this many days
    max_gap_years: Optional[float] = None,     # drop if exit-entry > this (e.g., 25)
    # Metadata to carry along
    carry_cols_entry: Optional[List[str]] = None,
    carry_cols_exit: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pair each exit with the most recent prior entry per company, with optional sponsor restriction,
    minimum holding filter, optional max gap, and a full audit trail of drops.

    Returns:
        pairs_df: one row per paired exit with EntryDate, ExitDate, HoldingYears, notes
        audit_df: log of dropped/ambiguous/unpaired cases with reasons
    """
    required = {company_col, date_col, entry_col, exit_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for pairing: {missing}")

    # Defaults for columns to carry over from the original rows
    if carry_cols_entry is None:
        carry_cols_entry = ["DealType", "DealType2", "DealType3", "DealSize"]
    if carry_cols_exit is None:
        carry_cols_exit = ["DealType", "DealType2", "DealType3", "DealSize"]

    # Defensive copies & normalization
    d = df.copy()
    d = d.sort_values([company_col, date_col], kind="mergesort")  # stable sort
    # Ensure datetime
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    # Prepare containers
    rows: List[Dict] = []
    audit: List[Dict] = []

    # Iterate company by company
    for comp, g in d.groupby(company_col, sort=False):
        g = g.reset_index(drop=True)
        # stack of entry candidates (tuples of index, date, sponsor)
        entries: List[Dict] = []

        # optional: track counts for notes
        entry_count = int(g[entry_col].sum())
        exit_count  = int(g[exit_col].sum())

        for idx, r in g.iterrows():
            r_date = r[date_col]

            # Push entries on the stack
            if r[entry_col] is True:
                entries.append({
                    "idx": idx,
                    "date": r_date,
                    "sponsor": (r.get(sponsor_col) if sponsor_col in g.columns else None)
                })
                continue

            # On exit, try to match the latest prior entry
            if r[exit_col] is True:
                if not entries:
                    audit.append({
                        "Company": comp,
                        "EventIndex": idx,
                        "EventDate": r_date,
                        "Reason": "ExitWithoutPriorEntry",
                        "EntryCountCompany": entry_count,
                        "ExitCountCompany": exit_count
                    })
                    continue

                # find most-recent prior entry (strictly before exit date)
                candidate = None
                for e in reversed(entries):
                    if pd.isna(e["date"]) or pd.isna(r_date):
                        continue
                    if e["date"] < r_date:
                        candidate = e
                        # sponsor restriction if requested
                        if restrict_same_sponsor:
                            s_entry = e.get("sponsor")
                            s_exit  = r.get(sponsor_col) if sponsor_col in g.columns else None
                            if (s_entry != s_exit):
                                # keep looking for an earlier entry that matches sponsor
                                candidate = None
                                continue
                        break

                if candidate is None:
                    audit.append({
                        "Company": comp,
                        "EventIndex": idx,
                        "EventDate": r_date,
                        "Reason": "NoEligibleEntryPrior",
                        "EntryCountCompany": entry_count,
                        "ExitCountCompany": exit_count
                    })
                    continue

                # Build the paired row with metadata
                e_idx = candidate["idx"]
                e_row = g.loc[e_idx]
                x_row = r

                entry_date = candidate["date"]
                exit_date  = r_date
                if pd.isna(entry_date) or pd.isna(exit_date) or not (entry_date < exit_date):
                    audit.append({
                        "Company": comp,
                        "EventIndex": idx,
                        "EventDate": r_date,
                        "Reason": "InvalidDatesOrNonIncreasing",
                        "EntryDate": entry_date,
                        "ExitDate": exit_date
                    })
                    continue

                hold_days = (exit_date - entry_date).days
                hold_years = hold_days / 365.25

                # Filters: min hold & max gap
                if min_hold_days and hold_days < min_hold_days:
                    audit.append({
                        "Company": comp,
                        "EventIndex": idx,
                        "EventDate": r_date,
                        "Reason": "BelowMinHoldDays",
                        "HoldDays": hold_days,
                        "MinHoldDays": min_hold_days
                    })
                    continue

                if (max_gap_years is not None) and (hold_years > max_gap_years):
                    audit.append({
                        "Company": comp,
                        "EventIndex": idx,
                        "EventDate": r_date,
                        "Reason": "AboveMaxGapYears",
                        "HoldYears": hold_years,
                        "MaxGapYears": max_gap_years
                    })
                    continue

                # Carry selected metadata from entry & exit rows (with suffixes)
                entry_meta = {f"Entry_{c}": e_row.get(c) for c in carry_cols_entry if c in g.columns}
                exit_meta  = {f"Exit_{c}":  x_row.get(c) for c in carry_cols_exit  if c in g.columns}

                rows.append({
                    company_col: comp,
                    "EntryDate": entry_date,
                    "ExitDate":  exit_date,
                    "HoldingYears": hold_years,
                    "PairingNotes": _pair_notes(entry_count, exit_count),
                    **entry_meta,
                    **exit_meta
                })

    # Build outputs
    pairs_df = pd.DataFrame(rows)
    if not pairs_df.empty:
        pairs_df["EntryYear"] = pd.to_datetime(pairs_df["EntryDate"]).dt.year
        pairs_df["ExitYear"]  = pd.to_datetime(pairs_df["ExitDate"]).dt.year

    audit_df = pd.DataFrame(audit)
    return pairs_df, audit_df


def _pair_notes(entry_count: int, exit_count: int) -> str:
    """
    Lightweight label for context on company multiplicity.
    """
    if entry_count <= 1 and exit_count <= 1:
        return "SingleEntrySingleExit"
    if entry_count > 1 and exit_count <= 1:
        return "MultipleEntriesSingleExit"
    if entry_count <= 1 and exit_count > 1:
        return "SingleEntryMultipleExits"
    return "MultipleEntriesMultipleExits"
