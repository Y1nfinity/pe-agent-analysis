# tools/io.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

try:
    import toml
except ImportError:  # friendly message if package missing
    raise SystemExit("Please install 'toml' first:  pip install toml")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Internal schema (final column names we use downstream)
# Left side: keys in [columns] section of config.toml
# Right side: target column names inside the codebase
INTERNAL_SCHEMA_MAP = {
    "company":          "Company",
    "deal_id":          "DealID",
    "deal_date":        "DealDate",
    "dealtype1":        "DealType",
    "dealtype2":        "DealType2",
    "dealtype3":        "DealType3",
    "deal_size":        "DealSize",
    "investor_fund":    "InvestorFund",
    "sponsor":          "Sponsor",
    "deal_synopsis":    "DealSynopsis",
    "deal_size_status": "DealSizeStatus",
    "post_valuation":   "PostValuation",
    "deal_status":      "DealStatus",
    "financing_status": "FinancingStatus",
    "business_status":  "BusinessStatus",
    "primary_industry": "PrimaryIndustry",
    "verticals":        "Verticals",
    "description":      "Description",
    "hq_location":      "HQLocation",
    "company_website":  "CompanyWebsite",
    "company_url":      "CompanyURL",
}

TEXT_COLS = [
    "Company", "DealType", "DealType2", "DealType3",
    "Sponsor", "InvestorFund", "DealSynopsis", "DealStatus",
    "FinancingStatus", "BusinessStatus", "PrimaryIndustry",
    "Verticals", "Description", "HQLocation",
    "CompanyWebsite", "CompanyURL", "DealSizeStatus"
]

NUMERIC_COLS = ["DealSize", "PostValuation"]

DATE_COLS = ["DealDate"]  # Entry/Exit will be derived later in transforms


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return toml.load(config_path)


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV / Excel / Parquet by extension."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suf}")


def _build_rename_map(cfg_columns: Dict[str, str]) -> Dict[str, str]:
    """
    Build a pandas-rename map from the raw PitchBook labels → internal names.
    cfg_columns maps our config keys (company, deal_date, ...) → PitchBook labels.
    We invert it and map to INTERNAL_SCHEMA_MAP targets.
    """
    rename_map = {}
    for cfg_key, pb_label in cfg_columns.items():
        target = INTERNAL_SCHEMA_MAP.get(cfg_key)
        if not target:
            # unrecognized config key; skip silently
            continue
        rename_map[pb_label] = target
    return rename_map


def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in TEXT_COLS:
        if c in df.columns:
            # Preserve a raw copy for Company (optional)
            if c == "Company" and "Company_raw" not in df.columns:
                df["Company_raw"] = df["Company"]
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .replace({"nan": None, "none": None, "": None})
            )
    return df


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _coerce_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_clean(config_path: str | Path = "config.toml") -> Tuple[pd.DataFrame, Path]:
    """
    Reads config.toml, loads the raw PitchBook export, renames columns to our internal schema,
    normalizes types, and writes a clean parquet to data/interim.

    Returns:
        (clean_df, output_parquet_path)
    """
    cfg = _load_config(config_path)

    # Resolve paths relative to the config file’s directory
    cfg_dir = Path(config_path).resolve().parent
    in_path = (cfg_dir / cfg["data"]["input_path"]).resolve()
    out_path = (cfg_dir / cfg["data"]["clean_path"]).resolve()

    logger.info(f"Loading raw data: {in_path}")
    raw = _read_any(in_path)

    # Build rename map and apply
    rename_map = _build_rename_map(cfg["columns"])
    missing = [pb for pb in rename_map.keys() if pb not in raw.columns]
    if missing:
        logger.warning(f"These columns from config were not found in the input: {missing}")

    df = raw.rename(columns=rename_map).copy()

    # Keep only columns we know (optional but helps keep things tidy)
    keep_cols = list(set(rename_map.values()))
    # also keep the original columns if you want; here we keep only standardized ones:
    df = df[[c for c in keep_cols if c in df.columns]]

    # Type normalization
    df = _normalize_strings(df)
    df = _coerce_dates(df)
    df = _coerce_numbers(df)

    # Add helpful deriveds you’ll want later (non-destructive)
    if "DealDate" in df.columns and "DealDateYear" not in df.columns:
        df["DealDateYear"] = df["DealDate"].dt.year

    # Write parquet
    _ensure_dirs(out_path)
    logger.info(f"Writing clean parquet → {out_path}")
    df.to_parquet(out_path, index=False)

    return df, out_path
