from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# NOTE: Placeholder for external dependency - assuming they are defined elsewhere
# from scripts_restructured.data_cleaning import load_raw_data, prepare_entry_data
# Since I cannot see those definitions, I will assume simple loading/preprocessing functions
# are available or replace them with simple placeholders where needed.

# ============================================================
# 1. HELPERS: LOAD & MERGE
# ============================================================

# --- Placeholder for scripts_restructured.data_cleaning ---
def load_raw_data(path):
    """Placeholder: Load data from raw path."""
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    # Add other formats if needed


def prepare_entry_data(df):
    """Placeholder: Clean and prep entry data."""
    # This is a simplification; your actual prepare_entry_data is more complex
    # but we need to ensure required columns exist.
    df = df.copy()
    if 'Deal Date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['Deal Date'])
    if 'Deal Size' in df.columns and 'deal_size_usd_m' not in df.columns:
        # Assuming Deal Size needs translation to float USD M
        df['deal_size_usd_m'] = pd.to_numeric(df['Deal Size'], errors='coerce')
        df['deal_size_usd_m'] = df['deal_size_usd_m'] / 1000000

        # Ensure company_name_clean is created by the cleaning pipeline
    if 'company_name_clean' not in df.columns and 'Companies' in df.columns:
        # Simplified cleaning for demonstration; actual cleaning should use the dedicated function
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
    entries["entry_year"] = entries["entry_date"].dt.year
    return entries


def load_universe_entries(root: Path) -> pd.DataFrame:
    """Load cleaned PitchBook master as universe entries."""
    uni_path = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe master not found: {uni_path}")
    df = pd.read_parquet(uni_path).copy()

    # Standardizing names from clean master data
    if "deal_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["deal_date"])

    # Assuming 'transaction_year' is the year of the entry deal
    if "transaction_year" in df.columns:
        df["entry_year"] = df["transaction_year"].astype(int)

    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = df["deal_size_usd_m"]

    return df


def load_tp_pairs(root: Path) -> pd.DataFrame:
    """
    Load linked TP pairs. NOTE: Path corrected to use 'tp_entry_exit_pairs.parquet'
    as discussed previously to avoid conflicts.
    """
    # *** CRITICAL PATH CORRECTION ***
    p = root / "data" / "clean" / "tp_entry_exit_pairs.parquet"
    if not p.exists():
        # Fallback check for the file name used in the user's provided code
        fallback_path = root / "data" / "clean" / "entry_exit_pairs.parquet"
        if fallback_path.exists():
            print(f"Warning: Using fallback path {fallback_path.name}. Rename your output file.")
            p = fallback_path
        else:
            raise FileNotFoundError(f"TP pairs not found (expected {p.name} or {fallback_path.name})")

    df = pd.read_parquet(p).copy()

    # Handle the inconsistent column name 'Deal Size_x' if it exists in the linked data
    if "Deal Size_x" in df.columns and "deal_size_usd_m_entry" not in df.columns:
        df.rename(columns={"Deal Size_x": "deal_size_usd_m_entry"}, inplace=True)

    # keep only linking keys we need
    keep_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry", "exit_date"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["has_exit"] = df["exit_date"].notna()
    return df


def load_universe_pairs(root: Path) -> pd.DataFrame:
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
    # Need to normalize company_name_clean and entry_date to ensure the merge works
    key_cols = ["company_name_clean", "entry_date"]

    # Handle potential duplicate entries in the pairs file (e.g., multiple exits for one entry)
    pairs_small = pairs.sort_values("exit_date", na_position="last").drop_duplicates(subset=key_cols, keep="first")
    pairs_small = pairs_small[key_cols + ["has_exit"]]

    merged = entries.merge(
        pairs_small,
        on=key_cols,
        how="left",
        suffixes=("", "_pair"),
    )
    # Deals that didn't merge must have no exit.
    # FIX: Explicitly cast to boolean after fillna to avoid Pandas FutureWarning.
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
        # Reindex to ensure all entry years are covered, even if V_entry_exited is 0
        "V_entry_exited": V_entry_exited.reindex(N_entries.index).fillna(0.0).values,
    })

    # Calculate fraction, replace 0 V_entry_total with NA to prevent division by zero warning
    df_out["frac_exited_value"] = df_out["V_entry_exited"] / df_out["V_entry_total"].replace(0, np.nan)

    # Fill NaN fractions (where V_entry_total was 0) with 0
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
    Dual-axis implementation:

    LEFT AXIS (BLUE):
        - Outline bar = total entries (N)
        - Fill bar    = exited entries (N)

    RIGHT AXIS (RED):
        - Outline bar = total entry value ($)
        - Fill bar    = exited entry value ($)

    Percent exited is written inside bars.
    """

    df = summary[(summary["entry_year"] >= year_min) &
                 (summary["entry_year"] <= year_max)].copy()

    if df.empty:
        print(f"⚠️ No data for {title}")
        return

    years = df["entry_year"].tolist()
    x = np.arange(len(years))  # Use numpy for x array
    width = 0.5

    # --- extract series ---
    N_total = df["N_entries"].fillna(0).values
    N_exit = df["N_exited"].fillna(0).values

    # Convert Value to Billions for clearer right-axis labels
    V_total = df["V_entry_total"].fillna(0).values / 1000
    V_exit = df["V_entry_exited"].fillna(0).values / 1000

    frac_N = df["frac_exited_N"].fillna(0).values
    frac_V = df["frac_exited_value"].fillna(0).values

    # FIX: Increased figure height from 6 to 8 for better vertical spacing
    fig, axN = plt.subplots(figsize=(14, 8))
    axV = axN.twinx()

    x_blue = x - width / 2
    x_red = x + width / 2

    # =======================
    # LEFT AXIS — COUNT (BLUE)
    # =======================
    # Outline bar (Total N)
    axN.bar(
        x_blue, N_total, width=width,
        fill=False, edgecolor="blue", linewidth=1.5,
        label="Total Entries (N)"
    )

    # Fill bar (Exited N)
    axN.bar(
        x_blue, N_exit, width=width,
        color="blue", alpha=0.6,
        label="Exited Entries (N)"
    )

    # =======================
    # RIGHT AXIS — VALUE (RED)
    # =======================
    # Outline bar (Total Value, now in Billions)
    axV.bar(
        x_red, V_total, width=width,
        fill=False, edgecolor="red", linewidth=1.5,
        label="Total Entry Value ($B)"
    )

    # Fill bar (Exited Value, now in Billions)
    axV.bar(
        x_red, V_exit, width=width,
        color="red", alpha=0.6,
        label="Exited Entry Value ($B)"
    )

    # =======================
    # PERCENT LABELS (INSIDE)
    # =======================
    for xn, xv, fn, fv, nE, vE in zip(x_blue, x_red, frac_N, frac_V, N_exit, V_exit):
        # N Labels
        if nE > 0:
            axN.text(xn, nE * 0.9, f"{fn * 100:.0f}%",
                     ha="center", va="top", fontsize=8, color="white", fontweight="bold")

        # Value Labels (note V_exit is in Billions here)
        if vE > 0:
            axV.text(xv, vE * 0.9, f"{fv * 100:.0f}%",
                     ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    # =======================
    # AXES & LABELS
    # =======================
    axN.set_ylabel("Number of Deals (Count)", color="blue")
    axV.set_ylabel("Entry Value ($ Billion)", color="red")

    axN.tick_params(axis='y', colors='blue')
    axV.tick_params(axis='y', colors='red')

    axN.set_xticks(x)
    axN.set_xticklabels(years, rotation=90)
    axN.set_title(title, fontsize=14, pad=15)
    axN.set_xlabel("Entry Year")

    # =======================
    # LEGEND (MERGED)
    # =======================
    h1, l1 = axN.get_legend_handles_labels()
    h2, l2 = axV.get_legend_handles_labels()

    # We combine and remove duplicates from the fill/outline labels
    unique_handles = []
    unique_labels = []

    # Manually select the handles for clarity (one filled bar, one outline bar from each set)
    # Order: N Exited (Filled), N Total (Outline), V Exited (Filled), V Total (Outline)
    h_out = [h1[1], h1[0], h2[1], h2[0]]
    l_out = ["% Exited (N)", "Total Entries (N)", "% Exited (Value $B)", "Total Entry Value ($B)"]

    axN.legend(h_out, l_out, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=4)

    plt.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust for legend
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile}")


# ============================================================
# 3. MASTER DRIVER
# ============================================================

def run_all(year_min: int = 1995, year_max: int = 2025):
    # Determine the root directory. This assumes the script is inside a 'scripts' folder,
    # and the project root is one level above 'scripts'.
    # Original: Path(__file__).resolve().parents[2]
    # Fixed: parents[1] points to the PEAgent directory.
    ROOT = Path(__file__).resolve().parents[1]

    figs = ROOT / "outputs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- TAKE-PRIVATE ---
    print("--- Processing Take-Private Completion ---")
    try:
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
    except FileNotFoundError as e:
        print(f"Skipping TP Completion: {e}")

    # --- UNIVERSE ---
    print("--- Processing Universe Completion ---")
    try:
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
    except FileNotFoundError as e:
        print(f"Skipping Universe Completion: {e}")

    # (Optional) save summary tables if you want:
    panels = ROOT / "outputs" / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    if 'tp_summary' in locals():
        tp_summary.to_csv(panels / "tp_entry_completion_panel.csv", index=False)
    if 'univ_summary' in locals():
        univ_summary.to_csv(panels / "universe_entry_completion_panel.csv", index=False)

    print(f"✅ Completion panels saved under {panels.resolve()}")


if __name__ == "__main__":
    run_all(year_min=2000, year_max=2025)