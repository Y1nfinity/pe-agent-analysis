# tools/matching_resolver.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _normalize_money(s):
    return pd.to_numeric(
        pd.Series(s, dtype="string").str.replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce",
    )

def _holding_days(entry_date, exit_date):
    if pd.isna(entry_date) or pd.isna(exit_date):
        return np.nan
    return (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days

def _apply_exit_alias(s: pd.Series, alias: dict) -> pd.Series:
    if not alias:
        return s
    s_norm = s.astype("string").str.strip().str.lower()
    mapped = s_norm.map(lambda x: alias.get(x, x))
    return mapped

def auto_select_best(cands: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    From a candidates table (top-k per seed), pick one Exit per Entry_DealID using rules.
    Expected columns (at minimum):
      Entry_DealID, Entry_DealDate, Entry_Company, Entry_Sponsor,
      Exit_DealID_cand, Exit_DealDate_cand, Exit_Sponsor_cand,
      score, Exit_DealType_cand, Exit_DealType2_cand, Exit_DealType3_cand,
      Entry_DealSize, Exit_DealSize_cand
    """
    if cands.empty:
        return cands.head(0)

    c = cands.copy()

    # numeric sizes
    for col in ["Entry_DealSize", "Exit_DealSize_cand"]:
        if col in c.columns:
            c[col] = _normalize_money(c[col])

    # compute hold days for filtering
    c["hold_days"] = [
        _holding_days(ed, xd) for ed, xd in zip(c["Entry_DealDate"], c["Exit_DealDate_cand"])
    ]

    # rule filters
    min_score     = float(cfg.get("min_score", 0.6))
    min_hold_days = int(cfg.get("min_hold_days", 0))
    max_hold_years = cfg.get("max_hold_years", None)
    max_hold_days = int(max_hold_years * 365.25) if max_hold_years else None
    prefer_same_sponsor = bool(cfg.get("prefer_same_sponsor", True))
    dedupe_exit_ids = bool(cfg.get("dedupe_exit_ids", True))

    c = c[c["score"] >= min_score]
    c = c[c["hold_days"].fillna(0) >= min_hold_days]
    if max_hold_days:
        c = c[c["hold_days"].fillna(10**9) <= max_hold_days]

    # rank candidates per seed:
    # 1) prefer same sponsor
    # 2) closest exit date after entry (min hold_days)
    # 3) higher score
    # 4) larger Exit_DealSize (tie-break)
    c["_same_sponsor"] = (
        c.get("Entry_Sponsor", "").astype("string").str.lower().str.strip()
        == c.get("Exit_Sponsor_cand", "").astype("string").str.lower().str.strip()
    )

    c = c.sort_values(
        by=["Entry_DealID", "_same_sponsor", "hold_days", "score", "Exit_DealSize_cand"],
        ascending=[True, False, True, False, False],
        kind="mergesort",
    )

    best = c.groupby("Entry_DealID", as_index=False).head(1).copy()

    # de-duplicate Exit_DealID across different seeds, if requested
    if dedupe_exit_ids and "Exit_DealID_cand" in best.columns:
        # If the same exit is chosen for multiple entries, keep the one with the highest score
        best = (
            best.sort_values(by=["Exit_DealID_cand", "score"], ascending=[True, False])
                .drop_duplicates(subset=["Exit_DealID_cand"], keep="first")
        )

    # reshape to the pairs schema used downstream
    colmap = {
        "Exit_DealID_cand":"Exit_DealID",
        "Exit_DealDate_cand":"Exit_DealDate",
        "Exit_Sponsor_cand":"Exit_Sponsor",
        "Exit_DealType_cand":"Exit_DealType",
        "Exit_DealType2_cand":"Exit_DealType2",
        "Exit_DealType3_cand":"Exit_DealType3",
        "Exit_DealSize_cand":"Exit_DealSize",
    }
    best = best.rename(columns=colmap)

    # add flags
    best["SelectedBy"] = "auto"
    return best

def apply_overrides(pairs_auto: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """
    overrides CSV expected columns:
      Entry_DealID, Exit_DealID, (optional) reason
    """
    if overrides is None or overrides.empty:
        return pairs_auto

    pairs = pairs_auto.set_index("Entry_DealID").copy()
    ov = overrides.dropna(subset=["Entry_DealID","Exit_DealID"]).copy()
    ov = ov.drop_duplicates(subset=["Entry_DealID"], keep="last").set_index("Entry_DealID")

    # Apply: set Exit_DealID for those entries; other Exit_* fields will be filled later by joining to universe if you desire
    for eid, row in ov.iterrows():
        if eid in pairs.index:
            pairs.loc[eid, "Exit_DealID"] = row["Exit_DealID"]
            pairs.loc[eid, "SelectedBy"] = "override"

    pairs = pairs.reset_index()
    return pairs
