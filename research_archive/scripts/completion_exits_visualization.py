from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


# ============================================================
# 1. HELPERS: LOAD & MERGE
# ============================================================

# --- Placeholder for scripts_restructured.data_cleaning ---
def load_raw_data(path):
    """Placeholder: Load data from raw path."""
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    # Add other formats if needed
    raise ValueError(f"Unsupported file format: {path}")


def prepare_entry_data(df):
    """
    Placeholder: Clean and prep entry data.
    The resulting deal size column 'deal_size_usd_m' is in Millions of USD.
    """
    df = df.copy()
    if 'Deal Date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['Deal Date'], errors='coerce')

    if 'Deal Size' in df.columns and 'deal_size_usd_m' not in df.columns:
        df['deal_size_usd_m'] = pd.to_numeric(df['Deal Size'], errors='coerce')

    # Ensure company_name_clean is created by the cleaning pipeline
    if 'company_name_clean' not in df.columns and 'Companies' in df.columns:
        # Simplified cleaning for demonstration
        df['company_name_clean'] = df['Companies'].astype(str).str.lower().str.replace('[^a-z0-9]', '', regex=True)

    return df


# --- End Placeholder ---

def load_tp_entries(root: Path) -> pd.DataFrame:
    """Load and clean take-private entry universe."""
    raw_path = root / "data" / "raw" / "public_2_private_all.xlsx"
    if not raw_path.exists():
        raise FileNotFoundError(f"TP raw data not found: {raw_path}")

    # Using the placeholder helper functions
    entries_raw = load_raw_data(raw_path)
    entries = prepare_entry_data(entries_raw).copy()

    # ensure canonical names
    if "deal_size_usd_m" in entries.columns and "deal_size_usd_m_entry" not in entries.columns:
        entries["deal_size_usd_m_entry"] = entries["deal_size_usd_m"]

    # Ensure entry_date is a datetime object
    if pd.api.types.is_datetime64_any_dtype(entries['entry_date']):
        entries["entry_year"] = entries["entry_date"].dt.year
    else:
        entries["entry_year"] = None

    # Required key columns for matching
    required_key_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry"]
    for col in required_key_cols:
        if col not in entries.columns:
            raise ValueError(f"TP entries missing required key column: {col}")

    return entries


def load_universe_entries(root: Path) -> pd.DataFrame:
    """Load cleaned PitchBook master as universe entries."""
    uni_path = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe master not found: {uni_path}")
    df = pd.read_parquet(uni_path).copy()

    # Standardizing names from clean master data
    if "deal_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["deal_date"], errors='coerce')

    if "transaction_year" in df.columns:
        df["entry_year"] = df["transaction_year"].astype(int)

    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = df["deal_size_usd_m"]

    # Required key columns for matching
    required_key_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry"]
    for col in required_key_cols:
        if col not in df.columns:
            # We are less strict here as this is the raw universe, but good practice
            print(f"Warning: Universe entries missing key column {col}")

    return df


def load_tp_pairs(root: Path) -> pd.DataFrame:
    """Load linked TP pairs."""
    p = root / "data" / "clean" / "tp_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"TP pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    if "Deal Size_x" in df.columns and "deal_size_usd_m_entry" not in df.columns:
        df.rename(columns={"Deal Size_x": "deal_size_usd_m_entry"}, inplace=True)

    # Convert dates
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')

    keep_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry", "exit_date"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["has_exit"] = df["exit_date"].notna()
    return df


def load_universe_pairs(root: Path) -> pd.DataFrame:
    """Load linked Full Universe pairs."""
    p = root / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Universe pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    # --- Normalize column names across universe linking versions ---
    rename_map = {}
    if "deal_date_entry" in df.columns:
        rename_map["deal_date_entry"] = "entry_date"
    if "deal_date_exit" in df.columns:
        rename_map["deal_date_exit"] = "exit_date"
    if "deal_size_usd_m_entry" not in df.columns and "deal_size_usd_m" in df.columns:
        rename_map["deal_size_usd_m"] = "deal_size_usd_m_entry"
    df = df.rename(columns=rename_map)

    # Convert dates
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')

    # --- Keep essential columns only ---
    keep_cols = [
        "company_name_clean",
        "entry_date",
        "deal_size_usd_m_entry",
        "exit_date",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # --- Add exit flag ---
    df["has_exit"] = df["exit_date"].notna()

    return df


def load_non_tp_pairs(root: Path) -> pd.DataFrame:
    """
    Load linked Non-Take-Private pairs (Universe - TP).
    This file contains the *exited* subset of non-TP deals.
    """
    p = root / "data" / "clean" / "non_tp_entry_exit_pairs.parquet"
    if not p.exists():
        print(f"ERROR: Non-TP pairs file NOT FOUND at {p}")
        print("Please ensure you run 'seperate_tp_universe_parquet.py' successfully first.")
        raise FileNotFoundError(f"Non-TP pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    # --- Robust Column Renaming/Checking ---
    rename_map = {}
    for original, canonical in [
        ("Exit_Date", "exit_date"),
        ("Entry_Date", "entry_date"),
        ("V_entry", "deal_size_usd_m_entry"),
    ]:
        if original in df.columns and canonical not in df.columns:
            rename_map[original] = canonical

    df = df.rename(columns=rename_map)

    # Convert dates
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')

    # Ensure required entry_year and has_exit columns exist
    if "entry_date" in df.columns and "entry_year" not in df.columns:
        df["entry_year"] = df["entry_date"].dt.year
    df["has_exit"] = df["exit_date"].notna()

    return df


def build_completion_frame(entries: pd.DataFrame,
                           pairs: pd.DataFrame,
                           year_min: int = 1995,
                           year_max: int = 2025) -> pd.DataFrame:
    """
    entries: full entry universe for this set (the denominator)
    pairs:   linked subset with 'entry_date', 'company_name_clean', 'has_exit' (the numerator flag)

    Returns per-entry-year completion summary.
    """
    if entries.empty:
        return pd.DataFrame()

    # --- join entries to pairs to flag exits ---
    key_cols = ["company_name_clean", "entry_date"]

    # Filter pairs to unique exit status (in case of multiple exits/rows)
    pairs_small = pairs.sort_values("exit_date", na_position="last").drop_duplicates(subset=key_cols, keep="first")
    pairs_small = pairs_small[key_cols + ["has_exit"]]

    # Merge the full set of entries (denominator) with the exit flags (numerator)
    merged = entries.merge(
        pairs_small,
        on=key_cols,
        how="left",
        suffixes=("", "_pair"),
    )
    # Deals that didn't merge (no exit pair found) must be flagged as no exit.
    merged["has_exit"] = merged["has_exit"].fillna(False).astype(bool)

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

    # Calculate exited value using only rows where has_exit is True
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

    df_out["frac_exited_value"] = df_out["V_entry_exited"] / df_out["V_entry_total"].replace(0, np.nan)
    df_out["frac_exited_value"] = df_out["frac_exited_value"].fillna(0)

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
    Dual-axis implementation. Sets scale based on plot title:
    - Take-Private: $ Million
    - Full Universe/Non-Take-Private: $ Billion
    """

    df = summary[(summary["entry_year"] >= year_min) &
                 (summary["entry_year"] <= year_max)].copy()

    if df.empty:
        print(f"⚠️ No data for {title}")
        return

    years = df["entry_year"].tolist()
    x = np.arange(len(years))
    bar_width = 0.4
    bar_sep = 0.05

    # --- Determine Value Scale based on title ---
    is_tp_plot = "Take-Private Deals" in title
    if is_tp_plot:
        # Input is in Millions, Scale Factor = 1 for plot label to be in Millions
        scale_factor = 1
        value_unit = "$ Million"
    else:
        # Input is in Millions, Scale Factor = 1000 for plot label to be in Billions
        scale_factor = 1000
        value_unit = "$ Billion"

    # --- extract series and apply scaling ---
    N_total = df["N_entries"].fillna(0).values
    N_exit = df["N_exited"].fillna(0).values

    V_total = df["V_entry_total"].fillna(0).values / scale_factor
    V_exit = df["V_entry_exited"].fillna(0).values / scale_factor

    frac_N = df["frac_exited_N"].fillna(0).values
    frac_V = df["frac_exited_value"].fillna(0).values

    fig, axN = plt.subplots(figsize=(14, 8))
    axV = axN.twinx()

    x_blue = x - (bar_width / 2) - (bar_sep / 2)
    x_red = x + (bar_width / 2) + (bar_sep / 2)

    # =======================
    # LEFT AXIS — COUNT (BLUE)
    # =======================
    axN.bar(
        x_blue, N_total, width=bar_width,
        fill=False, edgecolor="blue", linewidth=1.5,
        label="Total Entries (N)"
    )

    axN.bar(
        x_blue, N_exit, width=bar_width,
        color="blue", alpha=0.6,
        label="Exited Entries (N)"
    )

    # =======================
    # RIGHT AXIS — VALUE (RED)
    # =======================
    axV.bar(
        x_red, V_total, width=bar_width,
        fill=False, edgecolor="red", linewidth=1.5,
        label=f"Total Entry Value ({value_unit})"
    )

    axV.bar(
        x_red, V_exit, width=bar_width,
        color="red", alpha=0.6,
        label=f"Exited Entry Value ({value_unit})"
    )

    # =======================
    # PERCENT LABELS (INSIDE, BLACK)
    # =======================
    # Adjusting label placement to be less reliant on dynamic y-limits
    min_label_height_N = N_total.max() * 0.05
    min_label_height_V = V_total.max() * 0.05
    TEXT_COLOR = "black"

    for xn, xv, fn, fv, nE, vE in zip(x_blue, x_red, frac_N, frac_V, N_exit, V_exit):
        # N Labels
        if nE > 0:
            if nE < N_total.max() * 0.2:  # For very small bars, put label slightly above
                label_y = nE + min_label_height_N * 0.1
                label_va = "bottom"
            else:
                label_y = nE * 0.9
                label_va = "top"

            axN.text(xn, label_y, f"{fn * 100:.0f}\n%",
                     ha="center", va=label_va, fontsize=8, color=TEXT_COLOR, fontweight="bold")

        # Value Labels
        if vE > 0:
            if vE < V_total.max() * 0.2:
                label_y = vE + min_label_height_V * 0.1
                label_va = "bottom"
            else:
                label_y = vE * 0.9
                label_va = "top"

            axV.text(xv, label_y, f"{fv * 100:.0f}\n%",
                     ha="center", va=label_va, fontsize=8, color=TEXT_COLOR, fontweight="bold")

    # =======================
    # AXES & LABELS
    # =======================
    axN.set_ylabel("Number of Deals (Count)", color="blue")
    axV.set_ylabel(f"Entry Value ({value_unit})", color="red")

    axN.tick_params(axis='y', colors='blue')
    axV.tick_params(axis='y', colors='red')

    axN.set_xticks(x)
    axN.set_xticklabels([f"{int(y)}" for y in years], rotation=90)
    axN.set_title(title, fontsize=14, pad=15)
    axN.set_xlabel("Entry Year")

    # =======================
    # LEGEND (MERGED)
    # =======================
    h1, l1 = axN.get_legend_handles_labels()
    h2, l2 = axV.get_legend_handles_labels()

    # Reordering the labels for clarity in the legend
    l_out = ["Total Entries (N)", "Exited Entries (N)", f"Total Entry Value ({value_unit})",
             f"Exited Entry Value ({value_unit})"]
    h_out = [h1[0], h1[1], h2[0], h2[1]]

    axN.legend(h_out, l_out, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=4)

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile.name}")


# ============================================================
# 3. MASTER DRIVER
# ============================================================

def run_all(year_min: int = 1995, year_max: int = 2025):
    # Determine the root directory. This assumes the script is in ROOT/scripts/
    ROOT = Path(__file__).resolve().parents[1]

    figs = ROOT / "outputs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    panels = ROOT / "outputs" / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    print(f"Starting analysis, saving outputs to: {figs.resolve().parent}")
    key_cols_for_entry_match = ['company_name_clean', 'entry_date', 'deal_size_usd_m_entry']

    # --- Load all necessary base data (Denominator) ---
    try:
        univ_entries = load_universe_entries(ROOT)
        tp_entries = load_tp_entries(ROOT)
    except FileNotFoundError as e:
        print(f"Skipping all analysis due to missing master entry data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading entry data: {e}")
        return

    # --- DERIVE CORRECT NON-TP ENTRY UNIVERSE (The Denominator Fix) ---
    print("\n--- Deriving Non-Take-Private Entry Universe (The Denominator) ---")

    # Create keys on the full entry universe
    if all(col in univ_entries.columns for col in key_cols_for_entry_match):
        univ_entries['__key'] = univ_entries[key_cols_for_entry_match].astype(str).agg('|'.join, axis=1)
    else:
        print(f"ERROR: Full Universe Entries missing key columns for anti-join: {key_cols_for_entry_match}")
        return

    # Create keys on the TP entries (to be excluded)
    if all(col in tp_entries.columns for col in key_cols_for_entry_match):
        tp_entries['__key'] = tp_entries[key_cols_for_entry_match].astype(str).agg('|'.join, axis=1)
        tp_keys = set(tp_entries['__key'])
    else:
        print(f"ERROR: TP Entries missing key columns for anti-join: {key_cols_for_entry_match}")
        return

    # Perform anti-join: keep rows from the full universe whose key is NOT in the TP set
    non_tp_entries = univ_entries[~univ_entries['__key'].isin(tp_keys)].copy()
    non_tp_entries = non_tp_entries.drop(columns=['__key'])
    print(f"Non-TP Entry Universe derived: {len(non_tp_entries)} total entries.")

    # --- 1. TAKE-PRIVATE ---
    print("\n--- Processing Take-Private Completion ---")
    try:
        tp_pairs = load_tp_pairs(ROOT)
        tp_summary = build_completion_frame(tp_entries.drop(columns='__key'), tp_pairs, year_min, year_max)
        plot_completion_bars(
            tp_summary,
            title="Take-Private Deals — Entry-Year Completion (Exits by Count and Value)",
            outfile=figs / "TP_EntryCompletion.png",
            year_min=year_min,
            year_max=year_max,
        )
        tp_summary.to_csv(panels / "tp_entry_completion_panel.csv", index=False)
    except FileNotFoundError as e:
        print(f"Skipping TP Completion: {e}")
    except ValueError as e:
        print(f"Skipping TP Completion due to data issue: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during TP Completion: {e}")

    # --- 2. NON-TAKE-PRIVATE (New Analysis - FIXED) ---
    print("\n--- Processing Non-Take-Private Completion (Using Correct Denominator) ---")
    try:
        non_tp_pairs = load_non_tp_pairs(ROOT)
        # FIX: Use the full non_tp_entries as the denominator (entries)
        # and the linked non_tp_pairs as the numerator flags (pairs)
        non_tp_summary = build_completion_frame(non_tp_entries, non_tp_pairs, year_min, year_max)

        plot_completion_bars(
            non_tp_summary,
            title="Non-Take-Private Deals — Entry-Year Completion (Exits by Count and Value)",
            outfile=figs / "NonTP_EntryCompletion.png",
            year_min=year_min,
            year_max=year_max,
        )
        non_tp_summary.to_csv(panels / "non_tp_entry_completion_panel.csv", index=False)

    except FileNotFoundError as e:
        print(f"Skipping Non-TP Completion: {e}")
    except ValueError as e:
        print(f"Skipping Non-TP Completion due to data issue: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Non-TP Completion: {e}")

    # --- 3. FULL UNIVERSE ---
    print("\n--- Processing Full Universe Completion ---")
    try:
        univ_pairs = load_universe_pairs(ROOT)
        # Ensure the temporary key is removed from the full universe entries if it's still there
        univ_entries_clean = univ_entries.drop(columns='__key', errors='ignore')
        univ_summary = build_completion_frame(univ_entries_clean, univ_pairs, year_min, year_max)
        plot_completion_bars(
            univ_summary,
            title="Full Universe — Entry-Year Completion (Exits by Count and Value)",
            outfile=figs / "Universe_EntryCompletion.png",
            year_min=year_min,
            year_max=year_max,
        )
        univ_summary.to_csv(panels / "universe_entry_completion_panel.csv", index=False)
    except FileNotFoundError as e:
        print(f"Skipping Universe Completion: {e}")
    except ValueError as e:
        print(f"Skipping Universe Completion due to data issue: {e}")

    print(f"\n✅ All completion panels saved under {panels.resolve()}")


if __name__ == "__main__":
    run_all(year_min=2000, year_max=2025)