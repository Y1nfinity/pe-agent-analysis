from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
import sys
import os
import zipfile
import tempfile
import re


# =============================================================================
# Path setup
# =============================================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.name_cleaning import clean_company_name


# =============================================================================
# Constants
# =============================================================================

STRICT_DATE_WINDOW_DAYS = 30
RELAXED_DATE_WINDOW_DAYS = 365
FUZZY_MIN_SCORE = 90
FUZZY_REVIEW_SCORE = 85
MIN_EXIT_GAP_DAYS = 180


# =============================================================================
# Loaders
# =============================================================================

def _sanitize_xlsx_autofilters(path: Path) -> Path:
    """
    Creates a temporary copy of an .xlsx file with worksheet/table autofilters removed.

    This fixes openpyxl errors like:
        ValueError: Value must be either numerical or a string containing a wildcard

    The original Excel file is not modified.
    """
    temp_dir = Path(tempfile.mkdtemp())
    sanitized_path = temp_dir / f"{path.stem}_sanitized.xlsx"

    with zipfile.ZipFile(path, "r") as zin:
        with zipfile.ZipFile(sanitized_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                if item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                    text = data.decode("utf-8", errors="ignore")
                    text = re.sub(r"<autoFilter[\s\S]*?</autoFilter>", "", text)
                    text = re.sub(r"<autoFilter[^>]*/>", "", text)
                    data = text.encode("utf-8")

                if item.filename.startswith("xl/tables/") and item.filename.endswith(".xml"):
                    text = data.decode("utf-8", errors="ignore")
                    text = re.sub(r"<autoFilter[\s\S]*?</autoFilter>", "", text)
                    text = re.sub(r"<autoFilter[^>]*/>", "", text)
                    data = text.encode("utf-8")

                zout.writestr(item, data)

    return sanitized_path


def load_pitchbook_raw(path: Path) -> pd.DataFrame:
    """
    Loads a PitchBook Excel export.

    First tries normal pandas/openpyxl loading.
    If the workbook has broken Excel autofilter metadata, creates a sanitized
    temporary copy with autofilters removed and reads that instead.
    """
    if not path.exists():
        raise FileNotFoundError(f"PitchBook file not found: {path}")

    try:
        return pd.read_excel(path)
    except ValueError as e:
        msg = str(e)

        if "wildcard" in msg or "Value must be either numerical" in msg:
            print("⚠️ openpyxl could not read the workbook because of Excel autofilter metadata.")
            print("   Creating sanitized temporary copy without autofilters...")

            sanitized_path = _sanitize_xlsx_autofilters(path)
            return pd.read_excel(sanitized_path)

        raise


def load_bloomberg_raw(path: Path) -> pd.DataFrame:
    """
    Loads the Bloomberg M&A CSV export.
    """
    if not path.exists():
        raise FileNotFoundError(f"Bloomberg file not found: {path}")

    return pd.read_csv(path)


def resolve_data_file(folder: Path, stem: str) -> Path:
    """
    Finds a raw/clean data file by stem, allowing common extensions.

    Example:
        resolve_data_file(RAW, "pitchbook_public_2_private_all")

    Will search for:
        pitchbook_public_2_private_all.xlsx
        pitchbook_public_2_private_all.xls
        pitchbook_public_2_private_all.csv
        pitchbook_public_2_private_all.parquet
    """
    candidates = [
        folder / f"{stem}.xlsx",
        folder / f"{stem}.xls",
        folder / f"{stem}.csv",
        folder / f"{stem}.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find file with stem '{stem}' in {folder}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def load_table(path: Path) -> pd.DataFrame:
    """
    General loader for Excel, CSV, and Parquet files.
    Uses the Excel sanitizer for .xlsx/.xls files.
    """
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return load_pitchbook_raw(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


# =============================================================================
# Column helpers
# =============================================================================

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
        .replace({"nan": None, "None": None, "": None})
        .pipe(pd.to_numeric, errors="coerce")
    )


# =============================================================================
# Standardizers
# =============================================================================

def prepare_pitchbook_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw PitchBook export into a standardized deal table.

    Expected PitchBook columns may include:
        Deal ID
        Companies
        Deal Date
        Deal Type
        Deal Type 2
        Deal Type 3
        Deal Size
        Deal Status
        Investors
        Deal Synopsis
    """
    df = df.copy()

    company_col = _first_existing_col(df, ["Companies", "Company", "Target", "Target Name"])
    date_col = _first_existing_col(df, ["Deal Date", "Announce Date", "Announcement Date", "Close Date"])
    value_col = _first_existing_col(df, ["Deal Size", "Deal Value", "Transaction Value", "Announced Total Value (mil.)"])
    id_col = _first_existing_col(df, ["Deal ID", "DealID", "Transaction ID"])

    if company_col is None:
        raise ValueError(f"Could not find a PitchBook company column. Available columns: {list(df.columns)}")

    if date_col is None:
        raise ValueError(f"Could not find a PitchBook deal date column. Available columns: {list(df.columns)}")

    out = pd.DataFrame()

    out["pb_row_id"] = range(len(df))
    out["pb_deal_id"] = df[id_col] if id_col else pd.NA
    out["pb_company_name"] = df[company_col].astype(str)
    out["company_name_clean"] = out["pb_company_name"].map(clean_company_name)
    out["pb_deal_date"] = pd.to_datetime(df[date_col], errors="coerce")

    out["pb_deal_type"] = df["Deal Type"] if "Deal Type" in df.columns else pd.NA
    out["pb_deal_type_2"] = df["Deal Type 2"] if "Deal Type 2" in df.columns else pd.NA
    out["pb_deal_type_3"] = df["Deal Type 3"] if "Deal Type 3" in df.columns else pd.NA
    out["pb_deal_status"] = df["Deal Status"] if "Deal Status" in df.columns else pd.NA
    out["pb_investors"] = df["Investors"] if "Investors" in df.columns else pd.NA
    out["pb_deal_synopsis"] = df["Deal Synopsis"] if "Deal Synopsis" in df.columns else pd.NA

    if value_col:
        out["pb_deal_value"] = _to_numeric_series(df[value_col])
    else:
        out["pb_deal_value"] = pd.NA

    out = out.dropna(subset=["company_name_clean", "pb_deal_date"])
    out = out[out["company_name_clean"].astype(str).str.len() > 0]

    return out


def prepare_pitchbook_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts public-to-private entry file into standardized entry format.
    """
    pb = prepare_pitchbook_deals(df)

    entries = pb.rename(
        columns={
            "pb_row_id": "entry_row_id",
            "pb_deal_id": "entry_deal_id",
            "pb_company_name": "entry_company_name",
            "pb_deal_date": "entry_date",
            "pb_deal_type": "entry_deal_type",
            "pb_deal_type_2": "entry_deal_type_2",
            "pb_deal_type_3": "entry_deal_type_3",
            "pb_deal_status": "entry_deal_status",
            "pb_investors": "entry_investors",
            "pb_deal_synopsis": "entry_deal_synopsis",
            "pb_deal_value": "entry_deal_value",
        }
    )

    return entries


def prepare_bloomberg_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Bloomberg M&A export into standardized deal table.

    Expected Bloomberg columns:
        Deal Type
        Announce Date
        Target Name
        Acquirer Name
        Seller Name
        Announced Total Value (mil.)
        Payment Type
        TV/EBITDA
        Deal Status
    """
    df = df.copy()

    required = ["Target Name", "Announce Date"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Bloomberg export missing required columns: {missing}. Available columns: {list(df.columns)}")

    out = pd.DataFrame()

    out["bb_row_id"] = range(len(df))
    out["bb_company_name"] = df["Target Name"].astype(str)
    out["bb_company_name_clean"] = out["bb_company_name"].map(clean_company_name)
    out["bb_deal_date"] = pd.to_datetime(df["Announce Date"], errors="coerce")

    out["bb_deal_type"] = df["Deal Type"] if "Deal Type" in df.columns else pd.NA
    out["bb_deal_status"] = df["Deal Status"] if "Deal Status" in df.columns else pd.NA
    out["bb_acquirer_name"] = df["Acquirer Name"] if "Acquirer Name" in df.columns else pd.NA
    out["bb_seller_name"] = df["Seller Name"] if "Seller Name" in df.columns else pd.NA
    out["bb_payment_type"] = df["Payment Type"] if "Payment Type" in df.columns else pd.NA
    out["bb_tv_ebitda"] = df["TV/EBITDA"] if "TV/EBITDA" in df.columns else pd.NA

    if "Announced Total Value (mil.)" in df.columns:
        out["bb_deal_value"] = _to_numeric_series(df["Announced Total Value (mil.)"])
    else:
        out["bb_deal_value"] = pd.NA

    out = out.dropna(subset=["bb_company_name_clean", "bb_deal_date"])
    out = out[out["bb_company_name_clean"].astype(str).str.len() > 0]

    return out


# =============================================================================
# Match scoring helpers
# =============================================================================

def _date_diff_days(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left - right).abs().dt.days


def _value_diff_abs(pb_value: pd.Series, bb_value: pd.Series) -> pd.Series:
    return (pb_value - bb_value).abs()


def _value_diff_pct(pb_value: pd.Series, bb_value: pd.Series) -> pd.Series:
    denom = pb_value.abs()
    return ((pb_value - bb_value).abs() / denom).where(denom > 0)


def _assign_review_flag(df: pd.DataFrame) -> pd.Series:
    """
    Conservative review flag. Value difference is informative but not required.
    """
    high = (
        (df["match_score"] >= 95)
        & (df["date_diff_days"] <= 90)
    )

    medium = (
        (df["match_score"] >= 90)
        & (df["date_diff_days"] <= 365)
    )

    return pd.Series(
        [
            "high_confidence" if h else "possible_match" if m else "manual_review"
            for h, m in zip(high, medium)
        ],
        index=df.index,
    )


def _assign_research_grade_flag(df: pd.DataFrame) -> pd.Series:
    """
    Conservative research-grade match definition.

    This is stricter than review_flag and is intended for headline validation.
    """
    exact_good = (
        df["match_type"].isin(["strict_exact", "relaxed_exact", "exact"])
        & (df["date_diff_days"] <= 180)
    )

    fuzzy_good = (
        df["match_type"].isin(["fuzzy"])
        & (df["match_score"] >= 95)
        & (df["date_diff_days"] <= 180)
    )

    fuzzy_with_close_date = (
        df["match_type"].isin(["fuzzy"])
        & (df["match_score"] >= 90)
        & (df["date_diff_days"] <= 90)
    )

    return exact_good | fuzzy_good | fuzzy_with_close_date


def _select_best_by_pitchbook_deal(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches

    matches = matches.copy()

    tier_rank = {
        "strict_exact": 1,
        "relaxed_exact": 2,
        "fuzzy": 3,
        "fuzzy_review": 4,
        "exact": 1,
    }

    matches["match_tier_rank"] = matches["match_type"].map(tier_rank).fillna(99)

    matches = matches.sort_values(
        by=[
            "pb_row_id",
            "match_tier_rank",
            "match_score",
            "date_diff_days",
            "value_diff_abs",
        ],
        ascending=[True, True, False, True, True],
    )

    return matches.groupby("pb_row_id", as_index=False).head(1)


def _select_best_by_entry(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches

    matches = matches.copy()

    tier_rank = {
        "strict_exact": 1,
        "relaxed_exact": 2,
        "fuzzy": 3,
        "fuzzy_review": 4,
        "exact": 1,
    }

    matches["match_tier_rank"] = matches["match_type"].map(tier_rank).fillna(99)

    matches = matches.sort_values(
        by=[
            "entry_row_id",
            "match_tier_rank",
            "match_score",
            "date_diff_days",
            "value_diff_abs",
        ],
        ascending=[True, True, False, True, True],
    )

    return matches.groupby("entry_row_id", as_index=False).head(1)


# =============================================================================
# Overlap matching: PitchBook raw universe vs Bloomberg raw universe
# =============================================================================

def exact_overlap_match(
    pb: pd.DataFrame,
    bb: pd.DataFrame,
    date_window_days: int,
    match_type: str,
) -> pd.DataFrame:
    merged = pb.merge(
        bb,
        left_on="company_name_clean",
        right_on="bb_company_name_clean",
        how="inner",
    )

    if merged.empty:
        return merged

    merged["date_diff_days"] = _date_diff_days(
        merged["pb_deal_date"],
        merged["bb_deal_date"],
    )

    merged = merged[merged["date_diff_days"] <= date_window_days]

    if merged.empty:
        return merged

    merged["match_type"] = match_type
    merged["match_score"] = 100

    merged["value_diff_abs"] = _value_diff_abs(
        merged["pb_deal_value"],
        merged["bb_deal_value"],
    )

    merged["value_diff_pct"] = _value_diff_pct(
        merged["pb_deal_value"],
        merged["bb_deal_value"],
    )

    return merged


def fuzzy_overlap_match(
    pb: pd.DataFrame,
    bb: pd.DataFrame,
    min_score: int = FUZZY_MIN_SCORE,
    date_window_days: int = RELAXED_DATE_WINDOW_DAYS,
    include_review_band: bool = True,
) -> pd.DataFrame:
    bb_names = bb["bb_company_name_clean"].dropna().unique()

    if len(bb_names) == 0:
        return pd.DataFrame()

    matches = []

    for pb_name in pb["company_name_clean"].dropna().unique():
        best = process.extractOne(pb_name, bb_names, scorer=fuzz.WRatio)

        if not best:
            continue

        bb_name, score, _ = best

        if score >= min_score:
            match_type = "fuzzy"
        elif include_review_band and score >= FUZZY_REVIEW_SCORE:
            match_type = "fuzzy_review"
        else:
            continue

        matches.append(
            {
                "company_name_clean": pb_name,
                "bb_company_name_clean": bb_name,
                "match_score": score,
                "match_type": match_type,
            }
        )

    if not matches:
        return pd.DataFrame()

    match_df = pd.DataFrame(matches)

    merged = (
        pb.merge(match_df, on="company_name_clean", how="inner")
        .merge(bb, on="bb_company_name_clean", how="inner")
    )

    merged["date_diff_days"] = _date_diff_days(
        merged["pb_deal_date"],
        merged["bb_deal_date"],
    )

    merged = merged[merged["date_diff_days"] <= date_window_days]

    if merged.empty:
        return merged

    merged["value_diff_abs"] = _value_diff_abs(
        merged["pb_deal_value"],
        merged["bb_deal_value"],
    )

    merged["value_diff_pct"] = _value_diff_pct(
        merged["pb_deal_value"],
        merged["bb_deal_value"],
    )

    return merged


def build_pitchbook_bloomberg_overlap(pb: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    strict = exact_overlap_match(
        pb,
        bb,
        date_window_days=STRICT_DATE_WINDOW_DAYS,
        match_type="strict_exact",
    )

    used_pb = strict["pb_row_id"].unique() if not strict.empty else []

    relaxed = exact_overlap_match(
        pb[~pb["pb_row_id"].isin(used_pb)],
        bb,
        date_window_days=RELAXED_DATE_WINDOW_DAYS,
        match_type="relaxed_exact",
    )

    used_pb = (
        pd.concat([strict, relaxed], ignore_index=True, sort=False)["pb_row_id"].unique()
        if not relaxed.empty or not strict.empty
        else []
    )

    fuzzy = fuzzy_overlap_match(
        pb[~pb["pb_row_id"].isin(used_pb)],
        bb,
        min_score=FUZZY_MIN_SCORE,
        date_window_days=RELAXED_DATE_WINDOW_DAYS,
        include_review_band=True,
    )

    all_matches = pd.concat([strict, relaxed, fuzzy], ignore_index=True, sort=False)

    if all_matches.empty:
        return all_matches

    best = _select_best_by_pitchbook_deal(all_matches)

    best["review_flag"] = _assign_review_flag(best)
    best["research_grade_match"] = _assign_research_grade_flag(best)

    return best


# =============================================================================
# Entry validation: TP entries vs Bloomberg
# =============================================================================

def validate_entries_against_bloomberg(entries: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    entry_as_pb = entries.rename(
        columns={
            "entry_row_id": "pb_row_id",
            "entry_deal_id": "pb_deal_id",
            "entry_company_name": "pb_company_name",
            "entry_date": "pb_deal_date",
            "entry_deal_type": "pb_deal_type",
            "entry_deal_type_2": "pb_deal_type_2",
            "entry_deal_type_3": "pb_deal_type_3",
            "entry_deal_status": "pb_deal_status",
            "entry_investors": "pb_investors",
            "entry_deal_synopsis": "pb_deal_synopsis",
            "entry_deal_value": "pb_deal_value",
        }
    )

    overlap = build_pitchbook_bloomberg_overlap(entry_as_pb, bb)

    if overlap.empty:
        return overlap

    out = overlap.rename(
        columns={
            "pb_row_id": "entry_row_id",
            "pb_deal_id": "entry_deal_id",
            "pb_company_name": "entry_company_name",
            "pb_deal_date": "entry_date",
            "pb_deal_type": "entry_deal_type",
            "pb_deal_type_2": "entry_deal_type_2",
            "pb_deal_type_3": "entry_deal_type_3",
            "pb_deal_status": "entry_deal_status",
            "pb_investors": "entry_investors",
            "pb_deal_synopsis": "entry_deal_synopsis",
            "pb_deal_value": "entry_deal_value",
        }
    )

    out["entry_validation_status"] = out["review_flag"].map(
        {
            "high_confidence": "validated_by_bloomberg",
            "possible_match": "possible_bloomberg_match",
            "manual_review": "manual_review",
        }
    )

    return out


# =============================================================================
# Bloomberg exit pairing: PitchBook entries vs later Bloomberg deals
# =============================================================================

def exact_bloomberg_exit_match(entries: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    merged = entries.merge(
        bb,
        left_on="company_name_clean",
        right_on="bb_company_name_clean",
        how="inner",
    )

    if merged.empty:
        return merged

    merged["days_after_entry"] = (merged["bb_deal_date"] - merged["entry_date"]).dt.days
    merged = merged[merged["days_after_entry"] >= MIN_EXIT_GAP_DAYS]

    if merged.empty:
        return merged

    merged["match_type"] = "exact"
    merged["match_score"] = 100
    merged["holding_period_years"] = merged["days_after_entry"] / 365.25
    merged["exit_validation_status"] = "found_bloomberg_exit"

    return merged


def fuzzy_bloomberg_exit_match(
    entries: pd.DataFrame,
    bb: pd.DataFrame,
    min_score: int = FUZZY_MIN_SCORE,
) -> pd.DataFrame:
    bb_names = bb["bb_company_name_clean"].dropna().unique()

    if len(bb_names) == 0:
        return pd.DataFrame()

    matches = []

    for entry_name in entries["company_name_clean"].dropna().unique():
        best = process.extractOne(entry_name, bb_names, scorer=fuzz.WRatio)

        if not best:
            continue

        bb_name, score, _ = best

        if score >= min_score:
            matches.append(
                {
                    "company_name_clean": entry_name,
                    "bb_company_name_clean": bb_name,
                    "match_score": score,
                }
            )

    if not matches:
        return pd.DataFrame()

    match_df = pd.DataFrame(matches)

    merged = (
        entries.merge(match_df, on="company_name_clean", how="inner")
        .merge(bb, on="bb_company_name_clean", how="inner")
    )

    merged["days_after_entry"] = (merged["bb_deal_date"] - merged["entry_date"]).dt.days
    merged = merged[merged["days_after_entry"] >= MIN_EXIT_GAP_DAYS]

    if merged.empty:
        return merged

    merged["match_type"] = "fuzzy"
    merged["holding_period_years"] = merged["days_after_entry"] / 365.25
    merged["exit_validation_status"] = "found_bloomberg_exit"

    return merged


def find_bloomberg_exits_for_entries(entries: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    exact = exact_bloomberg_exit_match(entries, bb)

    used_entries = exact["entry_row_id"].unique() if not exact.empty else []

    fuzzy = fuzzy_bloomberg_exit_match(
        entries[~entries["entry_row_id"].isin(used_entries)],
        bb,
        min_score=FUZZY_MIN_SCORE,
    )

    all_matches = pd.concat([exact, fuzzy], ignore_index=True, sort=False)

    if all_matches.empty:
        return all_matches

    all_matches = all_matches.sort_values(
        by=["entry_row_id", "bb_deal_date", "match_score"],
        ascending=[True, True, False],
    )

    earliest = all_matches.groupby("entry_row_id", as_index=False).head(1)

    return earliest


# =============================================================================
# PitchBook pair-level validation against Bloomberg
# =============================================================================

def prepare_pitchbook_entry_exit_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes an existing PitchBook entry-exit pair file.

    Designed to handle common outputs from the TP pairing script:
        entry_deal_id / exit_deal_id
        entry_company_name / exit_company_name
        entry_date / exit_date

    Also handles pandas merge suffixes:
        deal_id_x / deal_id_y
        Companies_x / Companies_y
        Deal Date_x / Deal Date_y
    """
    df = pair_df.copy()

    entry_id_col = _pick_col(df, ["entry_deal_id", "deal_id_x", "Deal ID_x", "pb_entry_deal_id"])
    exit_id_col = _pick_col(df, ["exit_deal_id", "deal_id_y", "Deal ID_y", "pb_exit_deal_id"])

    entry_company_col = _pick_col(
        df,
        [
            "entry_company_name",
            "pb_entry_company_name",
            "Companies_x",
            "Company_x",
            "pb_company_name_x",
            "Companies",
            "Company",
        ],
    )

    exit_company_col = _pick_col(
        df,
        [
            "exit_company_name",
            "pb_exit_company_name",
            "Companies_y",
            "Company_y",
            "pb_company_name_y",
            "bb_company_name",
            "Companies",
            "Company",
        ],
    )

    entry_date_col = _pick_col(
        df,
        [
            "entry_date",
            "pb_entry_date",
            "Deal Date_x",
            "pb_deal_date_x",
        ],
    )

    exit_date_col = _pick_col(
        df,
        [
            "exit_date",
            "pb_exit_date",
            "Deal Date_y",
            "pb_deal_date_y",
            "bb_deal_date",
        ],
    )

    entry_value_col = _pick_col(
        df,
        [
            "entry_deal_value",
            "pb_entry_deal_value",
            "Deal Size_x",
            "pb_deal_value_x",
        ],
    )

    exit_value_col = _pick_col(
        df,
        [
            "exit_deal_value",
            "pb_exit_deal_value",
            "Deal Size_y",
            "pb_deal_value_y",
            "bb_deal_value",
        ],
    )

    exit_type_col = _pick_col(
        df,
        [
            "exit_category",
            "exit_deal_type",
            "Deal Type_y",
            "pb_deal_type_y",
            "bb_deal_type",
        ],
    )

    required = {
        "entry_company": entry_company_col,
        "entry_date": entry_date_col,
        "exit_company": exit_company_col,
        "exit_date": exit_date_col,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        raise ValueError(
            f"Could not standardize PitchBook entry-exit pair file. Missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()

    out["pair_row_id"] = range(len(df))
    out["entry_deal_id"] = df[entry_id_col] if entry_id_col else pd.NA
    out["exit_deal_id"] = df[exit_id_col] if exit_id_col else pd.NA

    out["entry_company_name"] = df[entry_company_col].astype(str)
    out["entry_company_name_clean"] = out["entry_company_name"].map(clean_company_name)
    out["entry_date"] = pd.to_datetime(df[entry_date_col], errors="coerce")

    out["exit_company_name"] = df[exit_company_col].astype(str)
    out["exit_company_name_clean"] = out["exit_company_name"].map(clean_company_name)
    out["exit_date"] = pd.to_datetime(df[exit_date_col], errors="coerce")

    out["entry_deal_value"] = _to_numeric_series(df[entry_value_col]) if entry_value_col else pd.NA
    out["exit_deal_value"] = _to_numeric_series(df[exit_value_col]) if exit_value_col else pd.NA
    out["pitchbook_exit_type"] = df[exit_type_col] if exit_type_col else pd.NA

    out["pitchbook_holding_period_years"] = (
        out["exit_date"] - out["entry_date"]
    ).dt.days / 365.25

    out = out.dropna(
        subset=[
            "entry_company_name_clean",
            "entry_date",
            "exit_company_name_clean",
            "exit_date",
        ]
    )

    out = out[out["exit_date"] > out["entry_date"]]

    return out


def _match_pair_side_to_bloomberg(
    pair_side: pd.DataFrame,
    bb: pd.DataFrame,
    side: str,
    date_window_days: int = RELAXED_DATE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Matches either the entry side or exit side of a PitchBook pair to Bloomberg.

    side must be either:
        "entry"
        "exit"
    """
    if side not in ["entry", "exit"]:
        raise ValueError("side must be either 'entry' or 'exit'")

    name_col = f"{side}_company_name_clean"
    date_col = f"{side}_date"
    value_col = f"{side}_deal_value"

    side_df = pair_side[
        [
            "pair_row_id",
            name_col,
            date_col,
            value_col,
        ]
    ].copy()

    side_df = side_df.rename(
        columns={
            name_col: "company_name_clean",
            date_col: "pb_deal_date",
            value_col: "pb_deal_value",
        }
    )

    side_df["pb_row_id"] = side_df["pair_row_id"]
    side_df["pb_company_name"] = side_df["company_name_clean"]
    side_df["pb_deal_id"] = side_df["pair_row_id"]
    side_df["pb_deal_type"] = pd.NA
    side_df["pb_deal_type_2"] = pd.NA
    side_df["pb_deal_type_3"] = pd.NA
    side_df["pb_deal_status"] = pd.NA
    side_df["pb_investors"] = pd.NA
    side_df["pb_deal_synopsis"] = pd.NA

    strict = exact_overlap_match(
        side_df,
        bb,
        date_window_days=STRICT_DATE_WINDOW_DAYS,
        match_type="strict_exact",
    )

    used = strict["pb_row_id"].unique() if not strict.empty else []

    relaxed = exact_overlap_match(
        side_df[~side_df["pb_row_id"].isin(used)],
        bb,
        date_window_days=date_window_days,
        match_type="relaxed_exact",
    )

    used = (
        pd.concat([strict, relaxed], ignore_index=True, sort=False)["pb_row_id"].unique()
        if not strict.empty or not relaxed.empty
        else []
    )

    fuzzy = fuzzy_overlap_match(
        side_df[~side_df["pb_row_id"].isin(used)],
        bb,
        min_score=FUZZY_MIN_SCORE,
        date_window_days=date_window_days,
        include_review_band=True,
    )

    matches = pd.concat([strict, relaxed, fuzzy], ignore_index=True, sort=False)

    if matches.empty:
        return matches

    matches = _select_best_by_pitchbook_deal(matches)

    matches["review_flag"] = _assign_review_flag(matches)
    matches["research_grade_match"] = _assign_research_grade_flag(matches)

    prefix = f"bb_{side}_"

    keep_cols = [
        "pb_row_id",
        "bb_row_id",
        "bb_company_name",
        "bb_deal_date",
        "bb_deal_type",
        "bb_deal_status",
        "bb_deal_value",
        "bb_acquirer_name",
        "bb_seller_name",
        "match_type",
        "match_score",
        "date_diff_days",
        "value_diff_abs",
        "value_diff_pct",
        "review_flag",
        "research_grade_match",
    ]

    matches = matches[keep_cols].copy()

    matches = matches.rename(
        columns={
            "pb_row_id": "pair_row_id",
            "bb_row_id": f"{prefix}row_id",
            "bb_company_name": f"{prefix}company_name",
            "bb_deal_date": f"{prefix}date",
            "bb_deal_type": f"{prefix}deal_type",
            "bb_deal_status": f"{prefix}deal_status",
            "bb_deal_value": f"{prefix}deal_value",
            "bb_acquirer_name": f"{prefix}acquirer_name",
            "bb_seller_name": f"{prefix}seller_name",
            "match_type": f"{side}_match_type",
            "match_score": f"{side}_match_score",
            "date_diff_days": f"{side}_date_diff_days",
            "value_diff_abs": f"{side}_value_diff_abs",
            "value_diff_pct": f"{side}_value_diff_pct",
            "review_flag": f"{side}_review_flag",
            "research_grade_match": f"{side}_research_grade_match",
        }
    )

    return matches


def validate_pitchbook_pairs_against_bloomberg(
    pitchbook_pairs: pd.DataFrame,
    bb: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validates existing PitchBook entry-exit pairs against Bloomberg.

    For each PitchBook pair:
        1. Try to validate the entry in Bloomberg.
        2. Try to validate the exit in Bloomberg.
        3. Classify the pair into a validation category.
    """
    pairs = prepare_pitchbook_entry_exit_pairs(pitchbook_pairs)

    entry_matches = _match_pair_side_to_bloomberg(
        pair_side=pairs,
        bb=bb,
        side="entry",
        date_window_days=RELAXED_DATE_WINDOW_DAYS,
    )

    exit_matches = _match_pair_side_to_bloomberg(
        pair_side=pairs,
        bb=bb,
        side="exit",
        date_window_days=RELAXED_DATE_WINDOW_DAYS,
    )

    out = pairs.copy()

    if not entry_matches.empty:
        out = out.merge(entry_matches, on="pair_row_id", how="left")

    if not exit_matches.empty:
        out = out.merge(exit_matches, on="pair_row_id", how="left")

    if "entry_research_grade_match" not in out.columns:
        out["entry_research_grade_match"] = False

    if "exit_research_grade_match" not in out.columns:
        out["exit_research_grade_match"] = False

    out["entry_research_grade_match"] = out["entry_research_grade_match"].fillna(False)
    out["exit_research_grade_match"] = out["exit_research_grade_match"].fillna(False)

    if "bb_entry_row_id" in out.columns:
        out["entry_found_in_bloomberg"] = out["bb_entry_row_id"].notna()
    else:
        out["entry_found_in_bloomberg"] = False

    if "bb_exit_row_id" in out.columns:
        out["exit_found_in_bloomberg"] = out["bb_exit_row_id"].notna()
    else:
        out["exit_found_in_bloomberg"] = False

    out["pair_validation_category"] = "manual_review_or_possible_match"

    out.loc[
        out["entry_research_grade_match"] & out["exit_research_grade_match"],
        "pair_validation_category",
    ] = "both_entry_and_exit_validated"

    out.loc[
        out["entry_research_grade_match"] & ~out["exit_research_grade_match"],
        "pair_validation_category",
    ] = "entry_validated_exit_not_validated"

    out.loc[
        ~out["entry_research_grade_match"] & out["exit_research_grade_match"],
        "pair_validation_category",
    ] = "entry_not_validated_exit_validated"

    out.loc[
        ~out["entry_found_in_bloomberg"] & ~out["exit_found_in_bloomberg"],
        "pair_validation_category",
    ] = "neither_entry_nor_exit_found"

    out["pair_research_grade_validated"] = (
        out["pair_validation_category"] == "both_entry_and_exit_validated"
    )

    if "entry_review_flag" in out.columns:
        entry_manual = out["entry_review_flag"].eq("manual_review")
    else:
        entry_manual = False

    if "exit_review_flag" in out.columns:
        exit_manual = out["exit_review_flag"].eq("manual_review")
    else:
        exit_manual = False

    out["manual_review_needed"] = (
        out["pair_validation_category"].eq("manual_review_or_possible_match")
        | entry_manual
        | exit_manual
    )

    return out


# =============================================================================
# Summaries
# =============================================================================

def compute_overlap_summary(pb: pd.DataFrame, bb: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    total_pb = len(pb)
    total_bb = len(bb)

    matched_pb = overlap["pb_row_id"].nunique() if not overlap.empty else 0
    matched_bb = overlap["bb_row_id"].nunique() if not overlap.empty else 0

    rows = [
        ("total_pitchbook_deals", total_pb),
        ("total_bloomberg_deals", total_bb),
        ("matched_pitchbook_deals", matched_pb),
        ("matched_bloomberg_deals", matched_bb),
        ("pitchbook_overlap_pct", matched_pb / total_pb if total_pb else 0),
        ("bloomberg_overlap_pct", matched_bb / total_bb if total_bb else 0),
    ]

    if not overlap.empty:
        for match_type, count in overlap["match_type"].value_counts().items():
            rows.append((f"matches_{match_type}", count))

        for flag, count in overlap["review_flag"].value_counts().items():
            rows.append((f"review_flag_{flag}", count))

        if "research_grade_match" in overlap.columns:
            rows.append(("research_grade_matches", int(overlap["research_grade_match"].sum())))
            rows.append(
                (
                    "research_grade_match_pct_of_pitchbook",
                    overlap["research_grade_match"].sum() / total_pb if total_pb else 0,
                )
            )

    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_entry_validation_summary(entries: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    total_entries = len(entries)
    matched_entries = validation["entry_row_id"].nunique() if not validation.empty else 0

    rows = [
        ("total_pitchbook_entries", total_entries),
        ("entries_found_in_bloomberg", matched_entries),
        ("entry_bloomberg_validation_pct", matched_entries / total_entries if total_entries else 0),
    ]

    if not validation.empty:
        for status, count in validation["entry_validation_status"].value_counts().items():
            rows.append((f"entry_status_{status}", count))

        if "research_grade_match" in validation.columns:
            rows.append(("entry_research_grade_matches", int(validation["research_grade_match"].sum())))
            rows.append(
                (
                    "entry_research_grade_match_pct",
                    validation["research_grade_match"].sum() / total_entries if total_entries else 0,
                )
            )

    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_entry_exit_summary(entries: pd.DataFrame, entry_exit: pd.DataFrame) -> pd.DataFrame:
    total_entries = len(entries)
    entries_with_exit = entry_exit["entry_row_id"].nunique() if not entry_exit.empty else 0

    rows = [
        ("total_pitchbook_entries", total_entries),
        ("entries_with_bloomberg_exit", entries_with_exit),
        ("entry_exit_pairing_pct", entries_with_exit / total_entries if total_entries else 0),
    ]

    if not entry_exit.empty:
        rows.append(("avg_holding_period_years", entry_exit["holding_period_years"].mean()))
        rows.append(("median_holding_period_years", entry_exit["holding_period_years"].median()))

        for match_type, count in entry_exit["match_type"].value_counts().items():
            rows.append((f"exit_match_type_{match_type}", count))

    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_pair_validation_summary(pair_validation: pd.DataFrame) -> pd.DataFrame:
    total_pairs = len(pair_validation)

    rows = [
        ("total_pitchbook_entry_exit_pairs", total_pairs),
        (
            "pairs_both_entry_and_exit_research_grade_validated",
            int(pair_validation["pair_research_grade_validated"].sum()) if total_pairs else 0,
        ),
        (
            "pair_research_grade_validation_pct",
            pair_validation["pair_research_grade_validated"].mean() if total_pairs else 0,
        ),
        (
            "pairs_entry_found_in_bloomberg",
            int(pair_validation["entry_found_in_bloomberg"].sum()) if total_pairs else 0,
        ),
        (
            "pairs_exit_found_in_bloomberg",
            int(pair_validation["exit_found_in_bloomberg"].sum()) if total_pairs else 0,
        ),
        (
            "pairs_manual_review_needed",
            int(pair_validation["manual_review_needed"].sum()) if total_pairs else 0,
        ),
    ]

    if total_pairs:
        for category, count in pair_validation["pair_validation_category"].value_counts().items():
            rows.append((f"pair_category_{category}", count))

    return pd.DataFrame(rows, columns=["metric", "value"])


# =============================================================================
# Excel export
# =============================================================================

def write_excel_report(
    out_path: Path,
    overlap_summary: pd.DataFrame,
    entry_validation_summary: pd.DataFrame | None,
    entry_exit_summary: pd.DataFrame | None,
    pair_validation_summary: pd.DataFrame | None,
    overlap: pd.DataFrame,
    pb: pd.DataFrame,
    bb: pd.DataFrame,
    entry_validation: pd.DataFrame | None,
    entry_exit: pd.DataFrame | None,
    pair_validation: pd.DataFrame | None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched_pb_ids = set(overlap["pb_row_id"]) if not overlap.empty else set()
    matched_bb_ids = set(overlap["bb_row_id"]) if not overlap.empty else set()

    unmatched_pb = pb[~pb["pb_row_id"].isin(matched_pb_ids)].copy()
    unmatched_bb = bb[~bb["bb_row_id"].isin(matched_bb_ids)].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        overlap_summary.to_excel(writer, sheet_name="overlap_summary", index=False)

        if entry_validation_summary is not None:
            entry_validation_summary.to_excel(writer, sheet_name="entry_validation_summary", index=False)

        if entry_exit_summary is not None:
            entry_exit_summary.to_excel(writer, sheet_name="entry_exit_summary", index=False)

        if pair_validation_summary is not None:
            pair_validation_summary.to_excel(writer, sheet_name="pair_validation_summary", index=False)

        if not overlap.empty:
            overlap.to_excel(writer, sheet_name="overlap_matches", index=False)

            overlap[overlap["review_flag"] == "high_confidence"].to_excel(
                writer,
                sheet_name="high_confidence",
                index=False,
            )

            overlap[overlap["review_flag"] != "high_confidence"].to_excel(
                writer,
                sheet_name="needs_review",
                index=False,
            )

            if "research_grade_match" in overlap.columns:
                overlap[overlap["research_grade_match"]].to_excel(
                    writer,
                    sheet_name="research_grade_overlap",
                    index=False,
                )

        unmatched_pb.to_excel(writer, sheet_name="unmatched_pitchbook", index=False)
        unmatched_bb.to_excel(writer, sheet_name="unmatched_bloomberg", index=False)

        if entry_validation is not None and not entry_validation.empty:
            entry_validation.to_excel(writer, sheet_name="entry_validation", index=False)

        if entry_exit is not None and not entry_exit.empty:
            entry_exit.to_excel(writer, sheet_name="bb_independent_exits", index=False)

        if pair_validation is not None and not pair_validation.empty:
            pair_validation.to_excel(writer, sheet_name="pair_level_validation", index=False)

            pair_validation[
                pair_validation["pair_validation_category"] == "both_entry_and_exit_validated"
            ].to_excel(
                writer,
                sheet_name="pairs_both_validated",
                index=False,
            )

            pair_validation[
                pair_validation["pair_validation_category"] == "entry_validated_exit_not_validated"
            ].to_excel(
                writer,
                sheet_name="entry_only_validated",
                index=False,
            )

            pair_validation[
                pair_validation["pair_validation_category"] == "entry_not_validated_exit_validated"
            ].to_excel(
                writer,
                sheet_name="exit_only_validated",
                index=False,
            )

            pair_validation[
                pair_validation["pair_validation_category"] == "neither_entry_nor_exit_found"
            ].to_excel(
                writer,
                sheet_name="neither_validated",
                index=False,
            )

            pair_validation[
                pair_validation["manual_review_needed"]
            ].to_excel(
                writer,
                sheet_name="pair_manual_review",
                index=False,
            )


# =============================================================================
# Public runner
# =============================================================================

def run_bloomberg_validation(
    pitchbook_raw: pd.DataFrame,
    bloomberg_raw: pd.DataFrame,
    tp_entries_raw: pd.DataFrame | None = None,
    pitchbook_pair_raw: pd.DataFrame | None = None,
    clean_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame | None]:
    pb = prepare_pitchbook_deals(pitchbook_raw)
    bb = prepare_bloomberg_deals(bloomberg_raw)

    overlap = build_pitchbook_bloomberg_overlap(pb, bb)
    overlap_summary = compute_overlap_summary(pb, bb, overlap)

    entry_validation = None
    entry_exit = None
    entry_validation_summary = None
    entry_exit_summary = None

    pair_validation = None
    pair_validation_summary = None

    if tp_entries_raw is not None:
        entries = prepare_pitchbook_entries(tp_entries_raw)

        entry_validation = validate_entries_against_bloomberg(entries, bb)
        entry_exit = find_bloomberg_exits_for_entries(entries, bb)

        entry_validation_summary = compute_entry_validation_summary(entries, entry_validation)
        entry_exit_summary = compute_entry_exit_summary(entries, entry_exit)

    if pitchbook_pair_raw is not None:
        pair_validation = validate_pitchbook_pairs_against_bloomberg(
            pitchbook_pairs=pitchbook_pair_raw,
            bb=bb,
        )
        pair_validation_summary = compute_pair_validation_summary(pair_validation)

    if clean_dir is not None:
        clean_dir.mkdir(parents=True, exist_ok=True)

        pb.to_parquet(clean_dir / "pitchbook_standardized.parquet", index=False)
        bb.to_parquet(clean_dir / "bloomberg_standardized.parquet", index=False)
        overlap.to_parquet(clean_dir / "pitchbook_bloomberg_overlap.parquet", index=False)

        if entry_validation is not None:
            entry_validation.to_parquet(clean_dir / "bloomberg_entry_validation.parquet", index=False)

        if entry_exit is not None:
            entry_exit.to_parquet(clean_dir / "bloomberg_entry_exit_pairs.parquet", index=False)

        if pair_validation is not None:
            pair_validation.to_parquet(
                clean_dir / "pitchbook_pair_validation_against_bloomberg.parquet",
                index=False,
            )

    if output_dir is not None:
        write_excel_report(
            out_path=output_dir / "bloomberg_validation_summary.xlsx",
            overlap_summary=overlap_summary,
            entry_validation_summary=entry_validation_summary,
            entry_exit_summary=entry_exit_summary,
            pair_validation_summary=pair_validation_summary,
            overlap=overlap,
            pb=pb,
            bb=bb,
            entry_validation=entry_validation,
            entry_exit=entry_exit,
            pair_validation=pair_validation,
        )

    return {
        "pitchbook_standardized": pb,
        "bloomberg_standardized": bb,
        "overlap": overlap,
        "overlap_summary": overlap_summary,
        "entry_validation": entry_validation,
        "entry_validation_summary": entry_validation_summary,
        "entry_exit": entry_exit,
        "entry_exit_summary": entry_exit_summary,
        "pair_validation": pair_validation,
        "pair_validation_summary": pair_validation_summary,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    RAW = ROOT / "data" / "raw"
    CLEAN = ROOT / "data" / "clean"
    OUTPUT = ROOT / "data" / "output"

    pitchbook_path = RAW / "pitchbook_export.xlsx"

    # Updated Bloomberg M&A export filename.
    bloomberg_path = RAW / "ma_export_compelted deals filter 1.csv"

    tp_entries_path = None
    try:
        tp_entries_path = resolve_data_file(RAW, "pitchbook_public_2_private_all")
    except FileNotFoundError:
        pass

    pitchbook_pair_path = None
    candidate_pair_paths = [
        CLEAN / "tp_entry_exit_pairs.parquet",
        CLEAN / "tp_entry_exit_pairs.xlsx",
        RAW / "tp_entry_exit_pairs.parquet",
        RAW / "tp_entry_exit_pairs.xlsx",
    ]

    for path in candidate_pair_paths:
        if path.exists():
            pitchbook_pair_path = path
            break

    pitchbook_raw = load_pitchbook_raw(pitchbook_path)
    bloomberg_raw = load_bloomberg_raw(bloomberg_path)

    if tp_entries_path is not None:
        tp_entries_raw = load_table(tp_entries_path)
        print(f"✅ Found P2P entry file: {tp_entries_path}")
    else:
        tp_entries_raw = None
        print("⚠️ No P2P entry file found with stem: pitchbook_public_2_private_all")
        print("   Running raw PitchBook ↔ Bloomberg overlap only unless pair file exists.")

    if pitchbook_pair_path is not None:
        pitchbook_pair_raw = load_table(pitchbook_pair_path)
        print(f"✅ Found PitchBook entry-exit pair file: {pitchbook_pair_path}")
    else:
        pitchbook_pair_raw = None
        print("⚠️ No existing PitchBook entry-exit pair file found.")
        print("   Pair-level validation will be skipped.")
        print("   Expected one of:")

        for path in candidate_pair_paths:
            print(f"   - {path}")

    outputs = run_bloomberg_validation(
        pitchbook_raw=pitchbook_raw,
        bloomberg_raw=bloomberg_raw,
        tp_entries_raw=tp_entries_raw,
        pitchbook_pair_raw=pitchbook_pair_raw,
        clean_dir=CLEAN,
        output_dir=OUTPUT,
    )

    print("\n==============================")
    print("Bloomberg validation complete")
    print("==============================")

    print("\nRaw overlap summary:")
    print(outputs["overlap_summary"])

    if outputs["entry_validation_summary"] is not None:
        print("\nP2P entry validation summary:")
        print(outputs["entry_validation_summary"])

    if outputs["entry_exit_summary"] is not None:
        print("\nBloomberg independent entry-exit summary:")
        print(outputs["entry_exit_summary"])

    if outputs["pair_validation_summary"] is not None:
        print("\nPitchBook pair-level validation summary:")
        print(outputs["pair_validation_summary"])

    print("\nSaved outputs:")
    print(f"- {CLEAN / 'pitchbook_standardized.parquet'}")
    print(f"- {CLEAN / 'bloomberg_standardized.parquet'}")
    print(f"- {CLEAN / 'pitchbook_bloomberg_overlap.parquet'}")
    print(f"- {OUTPUT / 'bloomberg_validation_summary.xlsx'}")

    if tp_entries_raw is not None:
        print(f"- {CLEAN / 'bloomberg_entry_validation.parquet'}")
        print(f"- {CLEAN / 'bloomberg_entry_exit_pairs.parquet'}")

    if pitchbook_pair_raw is not  None:
        print(f"- {CLEAN / 'pitchbook_pair_validation_against_bloomberg.parquet'}")