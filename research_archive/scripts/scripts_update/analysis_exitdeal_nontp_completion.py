from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


# ============================================================
# 1. PATH & CONFIGURATION
# ============================================================

def find_project_root() -> Path:
    """
    Script: PEAgent/scripts/scripts_update/analysis_exitdeal_nontp_completion.py
    ROOT should be: PEAgent/
    """
    return Path(__file__).resolve().parents[2]


# ============================================================
# 2. DATA LOADING & PREP
# ============================================================

def load_universe_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Universe master not found: {p}")
    df = pd.read_parquet(p).copy()

    # Standardize column names
    df["entry_date"] = pd.to_datetime(df.get("deal_date"), errors='coerce')
    # Use deal_size_usd_m as the canonical size column
    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = pd.to_numeric(df["deal_size_usd_m"], errors="coerce")

    df["entry_year"] = df["entry_date"].dt.year
    return df


def load_tp_entries(root: Path) -> pd.DataFrame:
    p = root / "data" / "raw" / "pitchbook_public_2_private_all.xlsx"
    if not p.exists():
        raise FileNotFoundError(f"TP raw data not found: {p}")

    df = pd.read_excel(p)
    df['entry_date'] = pd.to_datetime(df.get('Deal Date'), errors='coerce')
    df['deal_size_usd_m_entry'] = pd.to_numeric(df.get('Deal Size'), errors='coerce')
    # Match the canonical cleaning logic for names
    df['company_name_clean'] = df['Companies'].astype(str).str.lower().str.replace('[^a-z0-9]', '', regex=True)
    return df


def load_non_tp_pairs(root: Path) -> pd.DataFrame:
    p = root / "data" / "clean" / "non_tp_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Non-TP pairs not found: {p}")

    df = pd.read_parquet(p).copy()
    # Normalize exit columns
    rename_map = {
        "Exit_Date": "exit_date",
        "Entry_Date": "entry_date",
        "V_entry": "deal_size_usd_m_entry"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
    df["has_exit"] = df["exit_date"].notna()
    return df


def build_completion_frame(entries: pd.DataFrame,
                           pairs: pd.DataFrame,
                           y_min: int,
                           y_max: int) -> pd.DataFrame:
    """Builds the aggregation frame: Exits as % of Total Entries."""
    key_cols = ["company_name_clean", "entry_date"]

    # Deduplicate exit pairs
    pairs_small = pairs.sort_values("exit_date", na_position="last").drop_duplicates(subset=key_cols, keep="first")

    merged = entries.merge(pairs_small[key_cols + ["has_exit"]], on=key_cols, how="left")
    merged["has_exit"] = merged["has_exit"].fillna(False).astype(bool)

    # Filter restricted range
    merged = merged[(merged["entry_year"] >= y_min) & (merged["entry_year"] <= y_max)]

    grp = merged.groupby("entry_year")

    # Aggregate
    df_out = pd.DataFrame({
        "entry_year": grp.groups.keys(),
        "N_entries": grp.size().values,
        "N_exited": grp["has_exit"].sum().values,
        "V_total": grp["deal_size_usd_m_entry"].sum(min_count=1).values,
        "V_exited": merged[merged["has_exit"]].groupby("entry_year")["deal_size_usd_m_entry"].sum(min_count=1).reindex(
            grp.groups.keys()).fillna(0).values
    })

    # Calculate percentages for the chart
    df_out["pct_N"] = (df_out["N_exited"] / df_out["N_entries"]) * 100
    df_out["pct_V"] = (df_out["V_exited"] / df_out["V_total"].replace(0, np.nan)).fillna(0) * 100

    return df_out.sort_values("entry_year")


# ============================================================
# 3. VISUALIZATION
# ============================================================

def plot_non_tp_completion(summary: pd.DataFrame, title: str, outfile: Path):
    years = summary["entry_year"].tolist()
    x = np.arange(len(years))

    # Style configuration
    width, custom_blue, custom_red = 0.42, "#6666b2", "#b96666"
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    # Bars (Zero-Gap)
    rects1 = ax1.bar(x - width / 2, summary["pct_N"], width, color=custom_blue, label='% Exited (Count)', zorder=3)
    rects2 = ax2.bar(x + width / 2, summary["pct_V"], width, color=custom_red, label='% Exited (Value)', zorder=3)

    # Multi-line Constrained Labels
    def add_labels(ax, rects):
        for rect in rects:
            h = rect.get_height()
            if h < 0.5: continue

            label_text = f"{int(h)}\n%"
            x_pos = rect.get_x() + rect.get_width() / 2.0
            y_pos = h - 1.5 if h > 10 else h + 0.5
            va = 'top' if h > 10 else 'bottom'

            txt = ax.text(x_pos, y_pos, label_text, ha='center', va=va,
                          fontsize=9, fontweight='bold', color='black', zorder=10)
            txt.set_clip_path(rect)

    add_labels(ax1, rects1)
    add_labels(ax2, rects2)

    # Formatting
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 100)
        ax.margins(y=0)

    ax1.set_xlabel('Entry Year', fontweight='bold')
    ax1.set_ylabel('% Exited (by Count)', color=custom_blue, fontweight='bold')
    ax2.set_ylabel('% Exited (by Entry Value)', color=custom_red, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(y)}" for y in years], rotation=90)

    ax1.grid(axis='y', linestyle='-', alpha=0.3, zorder=0)
    ax1.axhline(0, color='black', linewidth=1, zorder=4)
    ax2.grid(False)

    ax1.set_title(title, fontsize=16, fontweight='bold', pad=30)

    # --- FIX: COLLECT HANDLES FROM BOTH AXES ---
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()

    # Merge and draw in the top right
    ax1.legend(handler1 + handler2, label1 + label2,
               loc='upper right', frameon=True, shadow=True, fancybox=True)

    plt.tight_layout()

    # Ensure folder exists and save
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure → {outfile.name}")
    plt.show()


# ============================================================
# 4. MAIN DRIVER
# ============================================================

def main():
    ROOT = find_project_root()
    print(f"Project ROOT: {ROOT}")

    Y_MIN, Y_MAX = 2000, 2022

    try:
        # 1. Load Denominators
        print("Loading entry universes...")
        univ = load_universe_entries(ROOT)
        tp = load_tp_entries(ROOT)

        # 2. Strict Anti-Join for Non-TP Universe
        print("Performing anti-join...")
        key_cols = ['company_name_clean', 'entry_date']
        univ['__key'] = univ[key_cols].astype(str).agg('|'.join, axis=1)
        tp['__key'] = tp[key_cols].astype(str).agg('|'.join, axis=1)

        non_tp_entries = univ[~univ['__key'].isin(set(tp['__key']))].copy()
        print(f"Non-TP denominator derived: {len(non_tp_entries)} entries.")

        # 3. Load Exits (Numerators)
        print("Loading exit pairs...")
        non_tp_pairs = load_non_tp_pairs(ROOT)

        # 4. Process Summary
        summary = build_completion_frame(non_tp_entries, non_tp_pairs, Y_MIN, Y_MAX)

        # 5. Render
        output_path = ROOT / "reports" / "figures" / "NonTP_Exit_Rates.png"
        plot_non_tp_completion(
            summary,
            "Non-Take-Private Entry Cohorts (2000-2022): Realization Rates",
            output_path
        )

        print(f"Successfully saved chart to {output_path}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()