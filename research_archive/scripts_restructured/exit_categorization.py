"""
exit_categorization.py (REWRITTEN)
----------------------------------
Purpose:
    Assign high-quality PE exit categories using all deal-type fields:
        deal_type_1, deal_type_2, deal_type_3  → combined in all_deal_types.

Fixes the original data-loss problem by:
    - Incorporating ALL deal-type columns
    - Providing deterministic classification
    - Producing BOTH exit_type (fine) and exit_category (canonical)

Inputs:
    df with at least:
        company_name_clean
        all_deal_types (list[str])
        close_date or exit_date

Outputs:
    df with:
        exit_type       (fine-grained)
        exit_category   (coarse canonical taxonomy)

"""

from __future__ import annotations
from typing import List
import pandas as pd

# =============================================================================
# CANONICAL TAXONOMY (Coarse)
# =============================================================================

CANONICAL_CATEGORIES = [
    "ipo",
    "mna_exit",
    "secondary_buyout",
    "lbo_buyout",
    "add_on",
    "pipe",
    "recapitalization",
    "take_private",
    "private_to_private",
    "corporate_divestiture",
    "growth_equity",
    "distressed",
    "other",
]


# =============================================================================
# PATTERN MAPS (Fine-grained exit_type)
# =============================================================================
# These are matched against ANY string in all_deal_types.
# All lowercase.

PATTERNS = {
    "ipo": [
        "ipo", "initial public offering", "reverse merger"
    ],

    "mna_exit": [
        "merger", "acquisition", "trade sale",
        "strategic acquisition", "asset acquisition",
        "merger of equals"
    ],

    "secondary_buyout": [
        "secondary buyout", "sponsor-to-sponsor",
        "sponsor to sponsor", "sponsor buyout"
    ],

    "lbo_buyout": [
        "buyout", "lbo", "mbo", "management buyout", "management buy-in",
        "buyout (lbo, mbo)"
    ],

    "add_on": [
        "add-on", "addon", "bolt-on", "add on"
    ],

    "pipe": [
        "pipe", "public investment", "2nd offering", "follow-on"
    ],

    "recapitalization": [
        "recap", "recapitalization", "dividend recapitalization",
        "leveraged recapitalization"
    ],

    "take_private": [
        "public to private", "take private", "take-private",
        "buyout/lbo public", "public buyout"
    ],

    "private_to_private": [
        "private to private"
    ],

    "corporate_divestiture": [
        "divestiture", "spin-off", "spin off", "corporate divestiture"
    ],

    "growth_equity": [
        "growth", "expansion", "minority investment"
    ],

    "distressed": [
        "distressed", "debt repayment", "debt conversion"
    ],
}


# =============================================================================
# HELPER: Match any pattern against all_deal_types
# =============================================================================

def _match_pattern(deal_types: List[str], patterns: List[str]) -> bool:
    """Return True if ANY pattern matches ANY deal_type substring."""
    for dt in deal_types:
        for p in patterns:
            if p in dt:
                return True
    return False


# =============================================================================
# CORE CLASSIFICATION LOGIC
# =============================================================================

def assign_exit_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign exit_type (fine-grained) using all_deal_types.
    Guarantees every row gets exactly one category.
    """

    if "all_deal_types" not in df.columns:
        raise KeyError(
            "Expected 'all_deal_types' column. "
            "Run data_cleaning.normalize_values() first."
        )

    def classify(row):
        dtypes = row["all_deal_types"]

        # evaluate in order of decreasing specificity
        for label, patterns in PATTERNS.items():
            if _match_pattern(dtypes, patterns):
                return label

        return "other"

    df["exit_type"] = df.apply(classify, axis=1)
    return df


# =============================================================================
# CANONICAL (COARSE) EXIT CATEGORY
# =============================================================================

CATEGORY_MAPPING = {
    # canonical → fine-grained variants
    "ipo": ["ipo"],
    "mna_exit": ["mna_exit"],
    "secondary_buyout": ["secondary_buyout"],
    "lbo_buyout": ["lbo_buyout"],
    "add_on": ["add_on"],
    "pipe": ["pipe"],
    "recapitalization": ["recapitalization"],
    "take_private": ["take_private"],
    "private_to_private": ["private_to_private"],
    "corporate_divestiture": ["corporate_divestiture"],
    "growth_equity": ["growth_equity"],
    "distressed": ["distressed"],
}


def assign_exit_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map fine-grained exit types -> canonical exit categories.
    """
    if "exit_type" not in df.columns:
        df = assign_exit_type(df)

    def canonical(x: str) -> str:
        for cat, types in CATEGORY_MAPPING.items():
            if x in types:
                return cat
        return "other"

    df["exit_category"] = df["exit_type"].apply(canonical)
    return df


# =============================================================================
# PUBLIC API
# =============================================================================

def categorize_exits(df: pd.DataFrame) -> pd.DataFrame:
    """
    FULL pipeline:
        1. exit_type       (fine-grained)
        2. exit_category   (canonical)

    Ensures NO missing values and deterministic results.
    """
    df = assign_exit_type(df)
    df = assign_exit_category(df)
    return df
