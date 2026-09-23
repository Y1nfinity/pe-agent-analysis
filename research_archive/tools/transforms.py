from __future__ import annotations
import pandas as pd
import re

# P2P detection (seen in DealType2/3)
P2P_PHRASES = ("public to private", "take private", "take-private", "going private", "public-to-private")
# P2P_REGEX = re.compile(r"\bpublic[-\s]?to[-\s]?private\b", flags=re.IGNORECASE)
# Primary DealType buckets based on your peek
ENTRY_TYPES_PRIMARY = {
    "buyout/lbo",
    "pe growth/expansion",
}
EXIT_TYPES_PRIMARY = {
    "merger/acquisition",
    "ipo",
    "reverse merger",
    "merger of equals",
}
FINANCING_TYPES_PRIMARY = {  # ignore for holding periods
    "public investment 2nd offering",
    "pipe",
}

# Secondary/tertiary tags that signal entry/exit flavor
ENTRY_TYPES_SECONDARY = {
    "management buyout",  # usually accompanies buyout/lbo
    "public to private",  # take-private entry
    "add-on",             # acquisition by portco; doesn't start a new hold for the parent
}
EXIT_TYPES_SECONDARY = {
    "secondary buyout",
    "asset divestiture (corporate)",
    "corporate divestiture",
}

def compute_holding_periods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HoldingYears"] = (df["ExitDate"] - df["EntryDate"]).dt.days / 365.25
    return df

def add_p2p_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def is_p2p_row(row):
        vals = []
        for c in ("DealType","DealType2","DealType3","DealSynopsis"):
            if c in df.columns:
                vals.append((row.get(c) or ""))
        blob = " | ".join(v.strip().lower() for v in vals)
        return any(ph in blob for ph in P2P_PHRASES)
    df["PublicToPrivate"] = df.apply(is_p2p_row, axis=1)
    return df

def add_entry_exit_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d1 = df["DealType"].fillna("").str.strip().str.lower()

    # STRICT: decide entry/exit from primary only
    df["IsEntry"] = d1.isin(ENTRY_TYPES_PRIMARY)
    df["IsExit"]  = d1.isin(EXIT_TYPES_PRIMARY)

    # convenience dates
    df["EntryDate"] = df["DealDate"].where(df["IsEntry"])
    df["ExitDate"]  = df["DealDate"].where(df["IsExit"])
    return df

def pair_entry_exit(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each company, match each exit to the most recent prior entry (strictly earlier date).
    Returns one row per (Company, ExitDate) with EntryDate, ExitDate, HoldingYears.
    """
    required = {"Company","DealDate","IsEntry","IsExit"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns for pairing: {missing}")

    d = df[["Company","DealDate","IsEntry","IsExit"]].sort_values(["Company","DealDate"]).copy()

    # track last entry per company
    last_entry_dates = []
    last_seen = pd.NaT
    last_company = None
    for comp, date, is_entry in d[["Company","DealDate","IsEntry"]].itertuples(index=False):
        if comp != last_company:
            last_seen = pd.NaT
            last_company = comp
        if is_entry:
            last_seen = date
        last_entry_dates.append(last_seen)
    d["LastEntryDate"] = last_entry_dates

    # keep only exits with a prior entry strictly before the exit date
    exits = d[d["IsExit"]].copy()
    exits = exits[exits["LastEntryDate"].notna()]
    exits = exits[exits["LastEntryDate"] < exits["DealDate"]].copy()

    exits = exits.rename(columns={"DealDate":"ExitDate"})
    exits["EntryDate"] = exits["LastEntryDate"]
    exits = exits.drop(columns=["IsEntry","IsExit","LastEntryDate"])

    exits["HoldingYears"] = (exits["ExitDate"] - exits["EntryDate"]).dt.days / 365.25

    # (Optional) If you want to remove micro-holds (same-week bookkeeping), uncomment:
    # exits = exits[exits["HoldingYears"] >= (30/365.25)]

    return exits


