"""
universe_completion_panel.py
============================

Builds "completion" panels for the Full Universe using
universe_entry_exit_pairs.parquet, which already contains:

- deal_id_entry
- deal_date_entry
- deal_size_usd_m_entry
- exit fields (where matched)
- holding_period_years
- exit_category_exit
- match_type (exact / fuzzy)
- possibly null exit fields for unexited deals

Outputs:
    data/clean/universe_completion_panel.parquet
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# Load the universe_entry_exit_pairs file
# ------------------------------------------------------------
def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairs file: {path}")
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Build completion panel
# ------------------------------------------------------------
def build_completion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces ONE ROW PER ENTRY DEAL, regardless of whether
    it matched to an exit.
    """

    df = df.copy()

    # Identify which entries exited
    df["has_exit"] = df["deal_id_exit"].notna()

    # Extract clean columns
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
    ]].copy()

    # Entry year
    out["entry_year"] = out["deal_date_entry"].dt.year

    return out


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def run_universe_completion(raw_pairs: Path, out_path: Path):
    print("🔹 Loading universe entry–exit pairs…")
    df = load_pairs(raw_pairs)

    print("🔹 Building completion panel…")
    panel = build_completion(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    print("✅ Saved universe completion panel →", out_path.resolve())


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    PAIRS = ROOT / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    OUT   = ROOT / "data" / "clean" / "universe_completion_panel.parquet"

    run_universe_completion(PAIRS, OUT)
