"""
entry_exit_completion_bars.py
=============================

Two figures:

1) Full universe – entry-year completion:
   - For each entry year X, fraction of deals that have *any* matched exit
     between year X and 2025 (by count and by entry $).
2) Take-privates – same, but restricted to p2p entries.

Visualization logic (per advisor feedback):

For each entry year (x-axis):
  • Blue bar (N):
      - Full bar outline at height = 1.0 (100% of entries by count)
      - Solid fill up to % of entries that have a matched exit
      - Text label with "% exited (N)" inside the solid region
  • Red bar ($):
      - Same idea but for entry deal value (sum of entry deal sizes)

Inputs:
  - data/raw/public_2_private_all.xlsx      (TP entry universe)
  - data/clean/pitchbook_master_clean.parquet (Universe entry universe)
  - data/clean/entry_exit_pairs.parquet        (TP linked pairs)
  - data/clean/universe_entry_exit_pairs.parquet (Universe linked pairs)

Outputs:
  - outputs/figures/TP_EntryCompletion.png
  - outputs/figures/Universe_EntryCompletion.png
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- existing utilities ---
from scripts_restructured.data_cleaning import load_raw_data, prepare_entry_data


# ============================================================
# 1. HELPERS: LOAD & MERGE
# ============================================================

def load_tp_entries(root: Path) -> pd.DataFrame:
    """Load and clean take-private entry universe."""
    raw_path = root / "data" / "raw" / "public_2_private_all.xlsx"
    entries_raw = load_raw_data(raw_path)
    entries = prepare_entry_data(entries_raw).copy()
    # ensure canonical names
    if "deal_size_usd_m" in entries.columns and "deal_size_usd_m_entry" not in entries.columns:
        entries["deal_size_usd_m_entry"] = entries["deal_size_usd_m"]
    entries["entry_year"] = entries["entry_date"].dt.year
    return entries


def load_universe_entries(root: Path) -> pd.DataFrame:
    """Load cleaned PitchBook master as universe entries."""
    uni_path = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe master not found: {uni_path}")
    df = pd.read_parquet(uni_path).copy()
    df["entry_date"] = df["deal_date"]
    df["entry_year"] = df["transaction_year"]
    df["deal_size_usd_m_entry"] = df["deal_size_usd_m"]
    return df


def load_tp_pairs(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"TP pairs not found: {p}")
    df = pd.read_parquet(p).copy()
    # keep only linking keys we need
    df = df[["company_name_clean", "entry_date", "deal_size_usd_m_entry", "exit_date"]]
    df["has_exit"] = df["exit_date"].notna()
    return df


def load_universe_pairs(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Universe pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    # --- Normalize column names across universe linking versions ---
    rename_map = {}

    # Entry date normalization
    if "deal_date_entry" in df.columns:
        rename_map["deal_date_entry"] = "entry_date"

    # Exit date normalization
    if "deal_date_exit" in df.columns:
        rename_map["deal_date_exit"] = "exit_date"

    # Deal size normalization
    if "deal_size_usd_m_entry" not in df.columns and "deal_size_usd_m" in df.columns:
        rename_map["deal_size_usd_m"] = "deal_size_usd_m_entry"

    # Entry deal ID normalization
    if "deal_id_entry" not in df.columns and "deal_id" in df.columns:
        rename_map["deal_id"] = "deal_id_entry"

    df = df.rename(columns=rename_map)

    # --- Keep essential columns only ---
    keep_cols = [
        "deal_id_entry",
        "company_name_clean",
        "entry_date",
        "deal_size_usd_m_entry",
        "exit_date",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]  # safe filtering
    df = df[keep_cols].copy()

    # --- Add exit flag ---
    df["has_exit"] = df["exit_date"].notna()

    return df



def build_completion_frame(entries: pd.DataFrame,
                           pairs: pd.DataFrame,
                           year_min: int = 1995,
                           year_max: int = 2025) -> pd.DataFrame:
    """
    entries: full entry universe (one row per deal)
    pairs:   linked subset with 'entry_date', 'company_name_clean', 'has_exit'

    Returns per-entry-year completion summary with:
        - N_entries
        - N_exited
        - frac_exited_N
        - V_entry_total
        - V_entry_exited
        - frac_exited_value
    """

    # --- join entries to pairs to flag exits ---
    key_cols = ["company_name_clean", "entry_date"]
    pairs_small = pairs[key_cols + ["has_exit"]].drop_duplicates()

    merged = entries.merge(
        pairs_small,
        on=key_cols,
        how="left",
        suffixes=("", "_pair"),
    )
    merged["has_exit"] = merged["has_exit"].fillna(False)

    # restrict to range
    merged = merged[(merged["entry_year"] >= year_min) & (merged["entry_year"] <= year_max)]

    # ensure numeric sizes
    merged["deal_size_usd_m_entry"] = pd.to_numeric(
        merged["deal_size_usd_m_entry"], errors="coerce"
    )

    grp = merged.groupby("entry_year")

    N_entries = grp.size()
    N_exited = grp["has_exit"].sum()

    V_entry_total = grp["deal_size_usd_m_entry"].sum(min_count=1)
    V_entry_exited = (
        merged[merged["has_exit"]]
        .groupby("entry_year")["deal_size_usd_m_entry"]
        .sum(min_count=1)
    )

    df_out = pd.DataFrame({
        "entry_year": N_entries.index,
        "N_entries": N_entries.values,
        "N_exited": N_exited.values,
        "frac_exited_N": (N_exited / N_entries).values,
        "V_entry_total": V_entry_total.values,
        "V_entry_exited": V_entry_exited.reindex(N_entries.index).fillna(0.0).values,
    })

    df_out["frac_exited_value"] = df_out["V_entry_exited"] / df_out["V_entry_total"].replace(0, pd.NA)

    return df_out.sort_values("entry_year").reset_index(drop=True)


# ============================================================
# 2. PLOTTING
# ============================================================

def plot_completion_bars(summary: pd.DataFrame,
                         title: str,
                         outfile: Path,
                         year_min: int = 1995,
                         year_max: int = 2025):
    """
    Draws, for each entry_year:
      - Blue outlined bar (100% N) with solid fill up to frac_exited_N
      - Red outlined bar (100% $) with solid fill up to frac_exited_value
    """

    df = summary[(summary["entry_year"] >= year_min) &
                 (summary["entry_year"] <= year_max)].copy()
    if df.empty:
        print(f"⚠️ No data in range {year_min}-{year_max} for {title}")
        return

    years = df["entry_year"].tolist()
    x = range(len(years))

    frac_N = df["frac_exited_N"].astype(float).fillna(0.0).values
    frac_V = df["frac_exited_value"].astype(float).fillna(0.0).values

    fig, ax = plt.subplots(figsize=(14, 6))

    width = 0.35
    x_blue = [i - width / 2 for i in x]
    x_red = [i + width / 2 for i in x]

    # BLUE (count)
    # Outline bar at 100%
    ax.bar(x_blue, 1.0, width=width, fill=False, edgecolor="blue", linewidth=1.5, label="% exited (N)")
    # Solid fill for exited fraction
    ax.bar(x_blue, frac_N, width=width, color="blue", alpha=0.5)

    # RED (value)
    ax.bar(x_red, 1.0, width=width, fill=False, edgecolor="red", linewidth=1.5, label="% exited ($)")
    ax.bar(x_red, frac_V, width=width, color="red", alpha=0.5)

    # Labels
    for i, (xn, xv, fn, fv) in enumerate(zip(x_blue, x_red, frac_N, frac_V)):
        if fn > 0:
            ax.text(xn, fn - 0.02, f"{fn*100:.0f}%", ha="center", va="top", fontsize=7, color="white")
        if fv > 0:
            ax.text(xv, fv - 0.02, f"{fv*100:.0f}%", ha="center", va="top", fontsize=7, color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(years, rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of entries exited")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile}")


# ============================================================
# 3. MASTER DRIVER
# ============================================================

def run_all(year_min: int = 1995, year_max: int = 2025):
    ROOT = Path(__file__).resolve().parents[2]
    figs = ROOT / "outputs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- TAKE-PRIVATE ---
    tp_entries = load_tp_entries(ROOT)
    tp_pairs = load_tp_pairs(ROOT)
    tp_summary = build_completion_frame(tp_entries, tp_pairs, year_min, year_max)
    plot_completion_bars(
        tp_summary,
        title="Take-Private Deals — Entry-Year Completion (Exits by Count and Value)",
        outfile=figs / "TP_EntryCompletion.png",
        year_min=year_min,
        year_max=year_max,
    )

    # --- UNIVERSE ---
    univ_entries = load_universe_entries(ROOT)
    univ_pairs = load_universe_pairs(ROOT)
    univ_summary = build_completion_frame(univ_entries, univ_pairs, year_min, year_max)
    plot_completion_bars(
        univ_summary,
        title="Full Universe — Entry-Year Completion (Exits by Count and Value)",
        outfile=figs / "Universe_EntryCompletion.png",
        year_min=year_min,
        year_max=year_max,
    )

    # (Optional) save summary tables if you want:
    panels = ROOT / "outputs" / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    tp_summary.to_csv(panels / "tp_entry_completion_panel.csv", index=False)
    univ_summary.to_csv(panels / "universe_entry_completion_panel.csv", index=False)
    print(f"✅ Completion panels saved under {panels.resolve()}")


if __name__ == "__main__":
    run_all(year_min=2000, year_max=2025)
