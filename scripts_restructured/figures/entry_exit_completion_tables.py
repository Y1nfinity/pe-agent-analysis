"""
entry_exit_completion_tables.py
===============================

Exports memo-ready tables for:
    (1) Full Universe completion statistics
    (2) Take-Private completion statistics

Inputs:
    data/clean/universe_completion_panel.parquet
    data/clean/tp_completion_panel.parquet

Outputs:
    outputs/tables/completion/universe_completion_table.csv
    outputs/tables/completion/tp_completion_table.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Load completion panels
# ------------------------------------------------------------
def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing completion panel: {path}")
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Summarize per entry year
# ------------------------------------------------------------
def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m_entry"], errors="coerce")

    grp = df.groupby("entry_year")

    summary = pd.DataFrame({
        "entry_year": grp.size().index,
        "N_total": grp.size().values,
        "N_exited": grp["has_exit"].sum().values,
        "V_total": grp["deal_size_usd_m_entry"].sum().values,
        "V_exited": grp.loc[df["has_exit"], "deal_size_usd_m_entry"].sum().values,
    })

    # Exit percentages
    summary["pct_N_exited"] = summary["N_exited"] / summary["N_total"]
    summary["pct_V_exited"] = summary["V_exited"] / summary["V_total"]
    summary["pct_V_exited"] = summary["pct_V_exited"].fillna(0)

    return summary.sort_values("entry_year")


# ------------------------------------------------------------
# Export CSV
# ------------------------------------------------------------
def export_table(df: pd.DataFrame, outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"✅ Saved table → {outfile}")


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def run_all():
    ROOT = Path(__file__).resolve().parents[2]

    # Input panels
    UNIV = ROOT / "data" / "clean" / "universe_completion_panel.parquet"
    TP   = ROOT / "data" / "clean" / "tp_completion_panel.parquet"

    # Output directory
    OUTDIR = ROOT / "outputs" / "tables" / "completion"

    df_univ = summarize(load_panel(UNIV))
    df_tp   = summarize(load_panel(TP))

    export_table(df_univ, OUTDIR / "universe_completion_table.csv")
    export_table(df_tp,   OUTDIR / "tp_completion_table.csv")


if __name__ == "__main__":
    run_all()
