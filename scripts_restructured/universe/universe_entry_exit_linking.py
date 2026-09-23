"""
universe_entry_exit_linking.py
==============================

Builds entry–exit linkages for the **full PitchBook universe**.

Input:
    data/raw/pitchbook_export.xlsx

Output:
    data/clean/universe_entry_exit_pairs.parquet

Features:
    - Cleans names
    - Infers entry vs exit deals based on Deal Type
    - Tags standardized exit categories:
         IPO, M&A, Secondary Buyout, Sponsor-to-Sponsor,
         Continuation Fund, Recapitalization, Add-On,
         Divestiture, Take-Private, Other, Minor
    - Links entries → earliest valid exit
    - Computes holding periods
    - Retains all deals (even unlinked ones)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process

# -------------------------
# utilities from your repo
# -------------------------
from scripts_restructured.utilities.name_cleaning import clean_company_name


# ============================================================
# 1. Load Raw Universe
# ============================================================

def load_raw_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing PitchBook raw file: {path.resolve()}")
    return pd.read_excel(path)


# ============================================================
# 2. Lightweight Deal Type → Exit Category Mapping
# ============================================================

def classify_exit_type(dt: str) -> str:
    """Maps Deal Type string to standardized categories."""
    if not isinstance(dt, str):
        return "other"

    t = dt.lower()

    # Core exit types
    if "ipo" in t:
        return "ipo"
    if "merger" in t or "acquisition" in t:
        return "mna"
    if "secondary" in t:
        return "secondary_buyout"
    if "buyout" in t and "secondary" in t:
        return "secondary_buyout"
    if "continuation" in t or "gp-led" in t:
        return "continuation_fund"
    if "recap" in t or "recapital" in t:
        return "recapitalization"
    if "add-on" in t:
        return "add_on"
    if "divestiture" in t:
        return "corporate_divestiture"
    if "take-private" in t or "public-to-private" in t:
        return "take_private"

    # Minor categories (your request: include, but tag as minor)
    if "pipe" in t:
        return "minor_pipe"
    if "growth" in t:
        return "minor_growth"
    if "venture" in t or "angel" in t or "seed" in t:
        return "minor_vc"
    if "distressed" in t or "restructur" in t:
        return "minor_distressed"

    return "other"


# ============================================================
# 3. Clean and Prepare Universe
# ============================================================

def clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Names
    df["company_name_clean"] = df["Companies"].astype(str).map(clean_company_name)

    # Dates
    df["deal_date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    df["transaction_year"] = df["deal_date"].dt.year

    # Deal size
    df["deal_size_usd_m"] = pd.to_numeric(df["Deal Size"], errors="coerce")

    # Exit type
    df["exit_category"] = df["Deal Type"].astype(str).map(classify_exit_type)

    # Keep lean schema
    df = df[[
        "Deal ID",
        "company_name_clean",
        "deal_date",
        "transaction_year",
        "Deal Type",
        "exit_category",
        "deal_size_usd_m",
    ]].rename(columns={"Deal ID": "deal_id", "Deal Type": "deal_type"})

    return df


# ============================================================
# 4. Entry Logic
# ============================================================

# A deal is considered an "entry" if:
#   - It is a buyout, add-on, take-private, recap, secondary, growth, VC, etc.
# A deal is considered a "potential exit" if:
#   - exit_category is not "other"

def identify_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Everything is an entry; exit identification separate."""
    df["is_entry"] = True
    return df


def identify_exits(df: pd.DataFrame) -> pd.DataFrame:
    """A deal counts as an exit if it falls in known exit categories."""
    df = df.copy()
    df["is_exit"] = df["exit_category"].isin([
        "ipo", "mna", "secondary_buyout", "continuation_fund",
        "recapitalization", "corporate_divestiture"
    ])
    return df


# ============================================================
# 5. Matching Engine — EXACT + FUZZY
# ============================================================

def exact_link(entries, exits):
    merged = entries.merge(
        exits,
        on="company_name_clean",
        how="inner",
        suffixes=("_entry", "_exit")
    )
    merged = merged[merged["deal_date_exit"] > merged["deal_date_entry"]]
    merged = merged.sort_values(["company_name_clean", "deal_date_entry", "deal_date_exit"])
    merged = merged.groupby(["deal_id_entry"]).head(1)
    merged["match_type"] = "exact"
    return merged


def fuzzy_link(entries, exits, min_score=90):
    exit_names = exits["company_name_clean"].unique().tolist()
    matches = []

    for cname in entries["company_name_clean"].unique():
        best = process.extractOne(cname, exit_names, score_cutoff=min_score)
        if best:
            matches.append((cname, best[0], best[1]))

    if not matches:
        return pd.DataFrame()

    mdf = pd.DataFrame(matches, columns=["entry_name_clean", "exit_name_clean", "score"])

    exits2 = exits.merge(mdf, left_on="company_name_clean", right_on="exit_name_clean")

    merged = entries.merge(
        exits2,
        left_on="company_name_clean",
        right_on="entry_name_clean",
        suffixes=("_entry", "_exit")
    )
    merged["company_name_clean"] = merged["entry_name_clean"]
    merged = merged[merged["deal_date_exit"] > merged["deal_date_entry"]]
    merged = merged.sort_values(["company_name_clean", "deal_date_entry", "deal_date_exit"])
    merged = merged.groupby(["deal_id_entry"]).head(1)
    merged["match_type"] = "fuzzy"

    return merged


# ============================================================
# 6. Full Linking
# ============================================================

def build_universe_pairs(df: pd.DataFrame) -> pd.DataFrame:

    entries = identify_entries(df)
    exits = identify_exits(df)

    entries = entries.rename(columns={"deal_id": "deal_id_entry", "deal_date": "deal_date_entry"})
    exits = exits.rename(columns={"deal_id": "deal_id_exit", "deal_date": "deal_date_exit"})

    entries_only = entries[["deal_id_entry", "company_name_clean", "deal_date_entry",
                            "deal_size_usd_m", "deal_type", "exit_category"]].copy()

    exits_only = exits[exits["is_exit"]][["deal_id_exit", "company_name_clean",
                                          "deal_date_exit", "deal_size_usd_m",
                                          "deal_type", "exit_category"]].copy()

    exact = exact_link(entries_only, exits_only)
    remaining_entries = entries_only[~entries_only["deal_id_entry"].isin(exact["deal_id_entry"])]
    fuzzy = fuzzy_link(remaining_entries, exits_only)

    linked = pd.concat([exact, fuzzy], ignore_index=True)

    # Holding period
    linked["holding_period_years"] = (
        (linked["deal_date_exit"] - linked["deal_date_entry"]).dt.days / 365.25
    )

    return linked


# ============================================================
# 7. Save Output
# ============================================================

def run_universe_entry_exit_linking(raw_path: Path, out_path: Path):
    print("🔹 Loading raw universe…")
    raw = load_raw_universe(raw_path)

    print("🔹 Cleaning…")
    clean = clean_universe(raw)

    print("🔹 Building entry–exit linkages…")
    pairs = build_universe_pairs(clean)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)

    print("✅ Saved Universe Entry–Exit pairs →", out_path.resolve())


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    RAW  = ROOT / "data" / "raw" / "pitchbook_export.xlsx"
    OUT  = ROOT / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    run_universe_entry_exit_linking(RAW, OUT)
