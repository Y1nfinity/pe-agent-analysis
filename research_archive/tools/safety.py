# tools/safety.py
from __future__ import annotations
import pandas as pd
import numpy as np

def add_completeness_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # presence flags
    out["HasEntryDate"] = out["Entry_DealDate"].notna()
    out["HasExitDate"]  = out["Exit_DealDate"].notna()
    out["HasEntrySize"] = out["Entry_DealSize"].notna()
    out["HasExitSize"]  = out["Exit_DealSize"].notna()

    out["HasBothDates"] = out["HasEntryDate"] & out["HasExitDate"]
    out["HasBothSizes"] = out["HasEntrySize"] & out["HasExitSize"]

    # HoldingYears only if both dates available
    out["HoldingYears"] = np.where(
        out["HasBothDates"],
        (pd.to_datetime(out["Exit_DealDate"]) - pd.to_datetime(out["Entry_DealDate"])).dt.days / 365.25,
        np.nan,
    )

    # IRR eligibility: both dates + both sizes
    out["EligibleForIRR"] = out["HasBothDates"] & out["HasBothSizes"]

    return out
