"""
name_cleaning.py
================
Centralized, robust company-name cleaning utilities
used by all parts of the PEAgent project.

This ensures consistent matching between entry and exit datasets.
"""

import re
import unicodedata
from functools import lru_cache

# Common corporate suffixes that should be removed
CORPORATE_SUFFIXES = [
    r"\binc\b", r"\binc.\b", r"\bcorp\b", r"\bcorp.\b", r"\bcorporation\b",
    r"\bltd\b", r"\bltd.\b", r"\bco\b", r"\bco.\b", r"\bcompany\b",
    r"\bplc\b", r"\bholdings\b", r"\bholding\b", r"\bgroup\b",
    r"\bsa\b", r"\bag\b", r"\bse\b", r"\bllc\b", r"\bgmbh\b",
    r"\bpartners\b", r"\btechnologies\b", r"\btechnology\b",
]

SUFFIX_PATTERN = re.compile("|".join(CORPORATE_SUFFIXES), flags=re.IGNORECASE)


@lru_cache(maxsize=50000)
def clean_company_name(raw: str) -> str:
    """
    Clean company names into a standardized canonical format.
    This function MUST be used wherever we rely on company-name matching.

    Steps:
    - Convert to lowercase
    - Unicode normalize (fix accented chars)
    - Remove URLs, tickers, parentheticals
    - Remove punctuation
    - Remove corporate suffixes ('inc', 'corp', 'ltd', ...)
    - Collapse whitespace
    """

    if raw is None:
        return ""

    # Convert to string and lowercase
    name = str(raw).strip().lower()

    # Normalize unicode (e.g., accented characters)
    name = unicodedata.normalize("NFKD", name)

    # Remove URLs or web-style identifiers
    name = re.sub(r"http\S+|www\.\S+", " ", name)

    # Remove ticker symbols in parentheses, e.g. "Apple (AAPL)"
    name = re.sub(r"\([^)]*\)", " ", name)

    # Remove punctuation entirely
    name = re.sub(r"[^\w\s]", " ", name)

    # Remove corporate suffixes
    name = SUFFIX_PATTERN.sub(" ", name)

    # Remove digits (rarely useful for name matching)
    name = re.sub(r"\d+", " ", name)

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def clean_company_name_series(series):
    """
    Vectorized version for Pandas Series.
    """
    return series.astype(str).apply(clean_company_name)
