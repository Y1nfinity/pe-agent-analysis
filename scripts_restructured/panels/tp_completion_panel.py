"""
tp_completion_panel.py
======================

Builds "completion" panels for Take-Private deals using
universe_entry_exit_pairs.parquet but filtering only TP entries.

TP filter = entry_type contains "public" (same logic used in your TP pipeline)

Outputs:
    data/clean/tp_completion_panel.parquet
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# Load pairs
# ------------------------------------------------------------
def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairs file: {path}")
    return pd.read_parquet(path)


# ------------------------------------------------------------
# TP filter
# ------------------------------------------------------------
def filter_take_private(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your repo’s TP logic:
        entry_type contains "public"
    but pairs do not have entry_type.

    Instead we infer from exit_category_exit or deal_type_entry.

    Most robust: deal_type_entry contains "public" or "take-private".
    """
    d = df.copy()

    if "deal_type_entry" not in d.columns:
        # Use fallback heuristic
        d["is_tp"] = d["exit_category_exit"].eq("take_private")
    else:
        d["is_tp"] = d["deal_type_entry"].str.contains("public", case=False, na=False)

    return d[d["is_tp"]].copy()


# ------------------------------------------------------------
# Build completion panel
# ------------------------------------------------------------
def build_completion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_exit"] = df["deal_id_exit"].notna()
    df["entry_year"] = df["deal_date_entry"].dt.year

    out = df[[
        "deal_id_entry",
        "company_name_clean",
        "deal_date_entry",
        "deal_size_usd_m_entry",
        "deal_id_exit",
        "deal_date_exit",
        "deal_size_usd_m_exit",
        "exit_category_exit",
        "holding_period_years",
        "has_exit",
        "entry_year",
    ]].copy()

    return out


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def run_tp_completion(raw_pairs: Path, out_path: Path):
    print("🔹 Loading linked pairs…")
    df = load_pairs(raw_pairs)

    print("🔹 Filtering take-private entries…")
    df_tp = filter_take_private(df)

    print("🔹 Building completion panel…")
    panel = build_completion(df_tp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    print("✅ Saved TP completion panel →", out_path.resolve())


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    PAIRS = ROOT / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    OUT = ROOT / "data" / "clean" / "tp_completion_panel.parquet"

    run_tp_completion(PAIRS, OUT)
