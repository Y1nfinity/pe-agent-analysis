from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re  # Added for case-insensitive string filtering


# NOTE: Placeholder for external dependency - assuming they are defined elsewhere

# ============================================================
# 1. HELPERS: LOAD & MERGE
# ============================================================

# --- Placeholder for scripts_restructured.data_cleaning ---
# Note: These are placeholders and assume your actual project structure
# provides robust implementations for these functions.
def load_raw_data(path):
    """Placeholder: Load data from raw path."""
    print(f"Loading raw data (Placeholder): {path}")
    # Simplified placeholder for data loading
    try:
        if path.suffix in [".xlsx", ".xls"]:
            # Ensure the correct engine is used for complex excel files if needed
            return pd.read_excel(path, engine='openpyxl')
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            print(f"Warning: Unknown file type {path.suffix}. Attempting to read as CSV.")
            return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading raw data from {path}: {e}")
        return pd.DataFrame()  # Return empty frame on failure


def prepare_entry_data(df):
    """Placeholder: Clean and prep entry data."""
    # This is a simplification; your actual prepare_entry_data is more complex
    # but we need to ensure required columns exist.
    df = df.copy()

    # 1. Entry Date
    if 'Deal Date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['Deal Date'], errors='coerce')

    # 2. Deal Size
    if 'Deal Size' in df.columns and 'deal_size_usd_m' not in df.columns:
        # Assuming Deal Size needs translation to float USD M
        df['deal_size_usd_m'] = pd.to_numeric(df['Deal Size'], errors='coerce')
        # Assuming the value is in USD (not millions) and needs conversion to millions
        # If your raw data is already in millions, remove the division by 1M
        df['deal_size_usd_m'] = df['deal_size_usd_m'] / 1000000

        # 3. Company Name Cleaning (Crucial for linking)
    if 'company_name_clean' not in df.columns and 'Companies' in df.columns:
        # Placeholder for robust cleaning using your expected function logic
        df['company_name_clean'] = df['Companies'].astype(str).str.lower().str.replace('[^a-z0-9]', '', regex=True)

    return df


# --- End Placeholder ---

def load_tp_entries(root: Path) -> pd.DataFrame:
    """Load and clean take-private entry universe from the specialized export."""
    raw_path = root / "data" / "raw" / "public_2_private_all.xlsx"
    if not raw_path.exists():
        raise FileNotFoundError(f"TP raw data not found: {raw_path}")

    entries_raw = load_raw_data(raw_path)
    entries = prepare_entry_data(entries_raw).copy()

    if "deal_size_usd_m" in entries.columns and "deal_size_usd_m_entry" not in entries.columns:
        entries["deal_size_usd_m_entry"] = entries["deal_size_usd_m"]

    if "entry_date" in entries.columns:
        entries["entry_year"] = entries["entry_date"].dt.year
    else:
        raise ValueError("entry_date column missing after preparation.")

    return entries


def load_universe_entries(root: Path) -> pd.DataFrame:
    """Load cleaned PitchBook master as the full universe of entries."""
    uni_path = root / "data" / "clean" / "pitchbook_master_clean.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe master not found: {uni_path}")
    df = pd.read_parquet(uni_path).copy()

    if "deal_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["deal_date"], errors='coerce')

    if "transaction_year" in df.columns:
        df["entry_year"] = df["transaction_year"].astype('Int64')
    elif "entry_date" in df.columns:
        df["entry_year"] = df["entry_date"].dt.year.astype('Int64')
    else:
        raise ValueError("Cannot determine 'entry_year'. Check master file columns.")

    if "deal_size_usd_m" in df.columns:
        df["deal_size_usd_m_entry"] = df["deal_size_usd_m"]

    return df


def load_entry_subset(root: Path, subset_type: str = "universe") -> pd.DataFrame:
    """
    Loads the full PitchBook universe and filters it for specific subsets.

    subset_type:
        - 'universe': Full PitchBook master
        - 'non_tp': Excludes deals flagged as 'Public to Private'
    """
    df = load_universe_entries(root)

    if subset_type == "universe":
        print("Loading full PitchBook universe.")
        return df

    elif subset_type == "non_tp":
        print("Loading Non-Take-Private subset...")

        # --- CRITICAL FILTER FIX for Non-TP ---
        # Look for common column names that might contain the TP flag.
        potential_tp_cols = [
            'deal_type_2', 'deal_type_3', 'transaction_type_2', 'transaction_type_3',
            'deal_type_secondary', 'deal_type_tertiary'
        ]

        # Filter columns that actually exist in the dataframe (case-insensitive search)
        df_cols_lower = [col.lower() for col in df.columns]
        tp_cols = []
        for p_col in potential_tp_cols:
            if p_col in df_cols_lower:
                # Find the actual case-sensitive column name
                tp_cols.append(df.columns[df_cols_lower.index(p_col)])

        if not tp_cols:
            print(
                "Warning: Could not find deal type columns (like deal_type_2/3) for Non-TP filter. Returning full universe.")
            return df

        # Create a boolean series where True means 'is_tp'
        is_tp = pd.Series(False, index=df.index)
        tp_pattern = re.compile(r'public to private', re.IGNORECASE)

        for col in tp_cols:
            # Check the column for the pattern, handling NaNs
            is_tp = is_tp | df[col].astype(str).str.contains(tp_pattern, na=False)

        non_tp_df = df[~is_tp].copy()

        print(f"Original entries: {len(df)}. Non-TP entries: {len(non_tp_df)}. Filtered out: {is_tp.sum()}")
        return non_tp_df

    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")


def load_tp_pairs(root: Path) -> pd.DataFrame:
    """Load linked TP pairs. Includes entry and exit details."""
    p = root / "data" / "clean" / "tp_entry_exit_pairs.parquet"
    # Fallback check for the file name used in the user's provided code
    fallback_path = root / "data" / "clean" / "entry_exit_pairs.parquet"
    if not p.exists() and fallback_path.exists():
        print(f"Warning: Using fallback path {fallback_path.name}.")
        p = fallback_path
    elif not p.exists():
        raise FileNotFoundError(f"TP pairs not found (expected {p.name} or {fallback_path.name})")

    df = pd.read_parquet(p).copy()

    # Normalize column names
    rename_map = {}
    if "Deal Size_x" in df.columns: rename_map["Deal Size_x"] = "deal_size_usd_m_entry"
    if "Deal Date_x" in df.columns: rename_map["Deal Date_x"] = "entry_date"
    if "Deal Date_y" in df.columns: rename_map["Deal Date_y"] = "exit_date"
    if "Deal Type_y" in df.columns: rename_map["Deal Type_y"] = "exit_type"
    if "Deal Size_y" in df.columns: rename_map["Deal Size_y"] = "deal_size_usd_m_exit"

    df = df.rename(columns=rename_map)

    keep_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry", "exit_date", "exit_type",
                 "deal_size_usd_m_exit"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["has_exit"] = df["exit_date"].notna()
    return df


def load_universe_pairs(root: Path) -> pd.DataFrame:
    """Load linked universe pairs, which includes TP and Non-TP exits."""
    p = root / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Universe pairs not found: {p}")

    df = pd.read_parquet(p).copy()

    # --- Normalize column names ---
    rename_map = {}
    if "deal_date_entry" in df.columns: rename_map["deal_date_entry"] = "entry_date"
    if "deal_date_exit" in df.columns: rename_map["deal_date_exit"] = "exit_date"
    if "deal_size_usd_m_entry" not in df.columns and "deal_size_usd_m" in df.columns:
        rename_map["deal_size_usd_m"] = "deal_size_usd_m_entry"
    if "deal_type_exit" in df.columns: rename_map["deal_type_exit"] = "exit_type"
    if "deal_size_usd_m_exit" not in df.columns and "deal_size_usd_m_y" in df.columns:
        rename_map["deal_size_usd_m_y"] = "deal_size_usd_m_exit"

    df = df.rename(columns=rename_map)

    # --- Keep essential columns only ---
    keep_cols = ["company_name_clean", "entry_date", "deal_size_usd_m_entry", "exit_date", "exit_type",
                 "deal_size_usd_m_exit"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["has_exit"] = df["exit_date"].notna()

    return df


def build_completion_frame(entries: pd.DataFrame,
                           pairs: pd.DataFrame,
                           year_min: int = 1995,
                           year_max: int = 2025) -> pd.DataFrame:
    """
    Builds per-entry-year completion summary (Exits vs. Total Entries).
    """

    key_cols = ["company_name_clean", "entry_date"]

    pairs_small = pairs.sort_values("exit_date", na_position="last").drop_duplicates(subset=key_cols, keep="first")
    pairs_small = pairs_small[key_cols + ["has_exit"]]

    merged = entries.merge(
        pairs_small,
        on=key_cols,
        how="left",
        suffixes=("", "_pair"),
    )

    merged["has_exit"] = merged["has_exit"].fillna(False).astype(bool)

    merged = merged[(merged["entry_year"] >= year_min) & (merged["entry_year"] <= year_max)]

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

    df_out["frac_exited_value"] = df_out["V_entry_exited"] / df_out["V_entry_total"].replace(0, np.nan)
    df_out["frac_exited_value"] = df_out["frac_exited_value"].fillna(0)

    return df_out.sort_values("entry_year").reset_index(drop=True)


# Define standard colors for consistent exit type plotting
EXIT_COLORS = {
    "Acquisition": "#1f77b4",  # Blue
    "IPO": "#2ca02c",  # Green
    "Secondary Buyout": "#ff7f0e",  # Orange
    "Liquidation": "#d62728",  # Red
    "Other": "#9467bd",  # Purple
    "Missing/Unknown": "#7f7f7f"  # Gray
}


def analyze_exit_type_breakdown(entries: pd.DataFrame,
                                pairs: pd.DataFrame,
                                year_min: int = 1995,
                                year_max: int = 2025) -> pd.DataFrame:
    """
    Analyzes the breakdown of exits by exit type and entry year.
    We are interested in the ENTRY value and the COUNT of the deals that exited.
    """
    key_cols = ["company_name_clean", "entry_date"]

    pairs_small = pairs.sort_values("exit_date", na_position="last").drop_duplicates(
        subset=key_cols, keep="first"
    )

    exited_pairs = pairs_small[pairs_small["has_exit"]].copy()

    def map_exit_type(raw_type):
        if pd.isna(raw_type):
            return "Missing/Unknown"
        raw_type = str(raw_type).strip().lower()
        if "acquisition" in raw_type or "merger" in raw_type or "trade sale" in raw_type:
            return "Acquisition"
        elif "initial public offering" in raw_type or "ipo" in raw_type:
            return "IPO"
        elif "secondary buyout" in raw_type or "sbo" in raw_type:
            return "Secondary Buyout"
        elif "liquidation" in raw_type or "wind-down" in raw_type:
            return "Liquidation"
        else:
            return "Other"

    exited_pairs["exit_type_clean"] = exited_pairs["exit_type"].apply(map_exit_type)

    merged = entries.merge(
        exited_pairs[["company_name_clean", "entry_date", "deal_size_usd_m_exit", "exit_type_clean"]],
        on=key_cols,
        how="inner",
    )

    merged = merged[(merged["entry_year"] >= year_min) & (merged["entry_year"] <= year_max)]

    grp = merged.groupby(["entry_year", "exit_type_clean"])

    df_n = grp.size().reset_index(name="N")
    df_v = grp["deal_size_usd_m_entry"].sum().reset_index(name="V_entry_usd_m")

    df_out = df_n.merge(df_v, on=["entry_year", "exit_type_clean"])

    return df_out


# ============================================================
# 2. ANALYSIS: HOLDING PERIOD BY EXIT YEAR (New Task E)
# ============================================================

def analyze_holding_period(pairs: pd.DataFrame,
                           year_min: int = 1995,
                           year_max: int = 2025) -> pd.DataFrame:
    """
    Calculates the mean holding period for deals grouped by their EXIT YEAR.
    Holding period is calculated as (Exit Date - Entry Date) in years.
    """

    # 1. Filter for successfully exited deals
    exited_deals = pairs[pairs["has_exit"]].copy()

    if exited_deals.empty:
        print("Warning: No exited deals found in the pairs dataset.")
        return pd.DataFrame()

    # 2. Calculate Exit Year
    exited_deals["exit_year"] = exited_deals["exit_date"].dt.year

    # 3. Calculate Holding Period (in years)
    # Convert timedelta to total days, then divide by 365.25 (for leap years)
    holding_period_days = (exited_deals["exit_date"] - exited_deals["entry_date"]).dt.days
    exited_deals["holding_period_years"] = holding_period_days / 365.25

    # 4. Restrict to exit year range
    df_filtered = exited_deals[
        (exited_deals["exit_year"] >= year_min) & (exited_deals["exit_year"] <= year_max)
        ].copy()

    # 5. Group by Exit Year and calculate summary statistics
    grp = df_filtered.groupby("exit_year")

    hp_summary = grp.agg(
        N_exits=('holding_period_years', 'size'),
        mean_holding_period_years=('holding_period_years', 'mean'),
        median_holding_period_years=('holding_period_years', 'median'),
        std_holding_period_years=('holding_period_years', 'std'),
        Q25_holding_period_years=('holding_period_years', lambda x: x.quantile(0.25)),
        Q75_holding_period_years=('holding_period_years', lambda x: x.quantile(0.75)),
    ).reset_index()

    return hp_summary.sort_values("exit_year")


# ============================================================
# 3. PLOTTING
# ============================================================

def plot_completion_bars(summary: pd.DataFrame,
                         title: str,
                         outfile: Path,
                         year_min: int = 1995,
                         year_max: int = 2025):
    """
    Dual-axis implementation for Entry-Year Completion.
    """

    df = summary[(summary["entry_year"] >= year_min) &
                 (summary["entry_year"] <= year_max)].copy()

    if df.empty:
        print(f"⚠️ No data for {title}")
        return

    years = df["entry_year"].tolist()
    x = np.arange(len(years))
    width = 0.45

    N_total = df["N_entries"].fillna(0).values
    N_exit = df["N_exited"].fillna(0).values

    V_total = df["V_entry_total"].fillna(0).values / 1000
    V_exit = df["V_entry_exited"].fillna(0).values / 1000

    frac_N = df["frac_exited_N"].fillna(0).values
    frac_V = df["frac_exited_value"].fillna(0).values

    fig, axN = plt.subplots(figsize=(14, 8))
    axV = axN.twinx()

    x_blue = x - width / 2
    x_red = x + width / 2

    # LEFT AXIS — COUNT (BLUE)
    axN.bar(x_blue, N_total, width=width, fill=False, edgecolor="blue", linewidth=1.5, label="Total Entries (N)")
    axN.bar(x_blue, N_exit, width=width, color="blue", alpha=0.6, label="Exited Entries (N)")

    # RIGHT AXIS — VALUE (RED)
    axV.bar(x_red, V_total, width=width, fill=False, edgecolor="red", linewidth=1.5, label="Total Entry Value ($B)")
    axV.bar(x_red, V_exit, width=width, color="red", alpha=0.6, label="Exited Entry Value ($B)")

    # PERCENT LABELS (INSIDE)
    for xn, xv, fn, fv, nE, vE in zip(x_blue, x_red, frac_N, frac_V, N_exit, V_exit):
        if nE > 0:
            axN.text(xn, nE * 0.9, f"{fn * 100:.0f}%", ha="center", va="top", fontsize=8, color="white",
                     fontweight="bold")
        if vE > 0:
            axV.text(xv, vE * 0.9, f"{fv * 100:.0f}%", ha="center", va="top", fontsize=8, color="white",
                     fontweight="bold")

    # AXES & LABELS
    axN.set_ylabel("Number of Deals (Count)", color="blue")
    axV.set_ylabel("Entry Value ($ Billion)", color="red")

    axN.tick_params(axis='y', colors='blue')
    axV.tick_params(axis='y', colors='red')

    axN.set_xticks(x)
    axN.set_xticklabels(years, rotation=90)
    axN.set_title(title, fontsize=14, pad=15)
    axN.set_xlabel("Entry Year")

    # LEGEND (MERGED)
    h1, l1 = axN.get_legend_handles_labels()
    h2, l2 = axV.get_legend_handles_labels()

    h_out = [h1[1], h1[0], h2[1], h2[0]]
    l_out = ["% Exited (N)", "Total Entries (N)", "% Exited (Value $B)", "Total Entry Value ($B)"]

    axN.legend(h_out, l_out, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=4)

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile}")


def plot_exit_type_stacked_bars(summary: pd.DataFrame,
                                type_col: str,
                                title: str,
                                outfile: Path,
                                year_min: int = 1995,
                                year_max: int = 2025):
    """
    Plots stacked bars of exit type breakdown by entry year.
    """

    df = summary[(summary["entry_year"] >= year_min) &
                 (summary["entry_year"] <= year_max)].copy()

    if df.empty:
        print(f"⚠️ No data for {title}")
        return

    value_col = "N" if type_col == "N" else "V_entry_usd_m"
    y_label = "Number of Exits (Count)" if type_col == "N" else "Entry Value of Exited Deals ($ Billion)"

    pivot_df = df.pivot(index="entry_year", columns="exit_type_clean", values=value_col).fillna(0)

    for category in EXIT_COLORS.keys():
        if category not in pivot_df.columns:
            pivot_df[category] = 0

    ordered_categories = [c for c in EXIT_COLORS.keys() if c in pivot_df.columns]
    plot_df = pivot_df[ordered_categories]

    if type_col == "V":
        plot_df = plot_df / 1000

    years = plot_df.index.tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(len(years))

    for category in ordered_categories:
        values = plot_df[category].values
        color = EXIT_COLORS.get(category, '#cccccc')
        ax.bar(years, values, bottom=bottom, label=category, color=color)
        bottom += values

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Entry Year")
    ax.set_ylabel(y_label)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)

    totals = bottom
    for i, total in enumerate(totals):
        if total > 0.5 or (type_col == "N" and total > 5):
            label = f"{int(total)}" if type_col == "N" else f"{total:.1f}"
            ax.text(years[i], total + (max(totals) * 0.01), label,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=len(ordered_categories))
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile}")


def plot_holding_period_line(summary: pd.DataFrame,
                             title: str,
                             outfile: Path,
                             year_min: int = 1995,
                             year_max: int = 2025):
    """
    Plots mean holding period by exit year with 25th and 75th percentile bounds.
    """
    df = summary[(summary["exit_year"] >= year_min) &
                 (summary["exit_year"] <= year_max)].copy()

    if df.empty:
        print(f"⚠️ No data for {title}")
        return

    years = df["exit_year"].tolist()

    fig, ax = plt.subplots(figsize=(14, 8))

    # 1. Plot Mean Holding Period (Line)
    ax.plot(years, df["mean_holding_period_years"],
            marker='o', linestyle='-', color='blue', linewidth=2,
            label="Mean Holding Period (Years)")

    # 2. Plot Median Holding Period (Dashed Line)
    ax.plot(years, df["median_holding_period_years"],
            marker='s', linestyle='--', color='darkcyan', linewidth=1.5,
            label="Median Holding Period (Years)")

    # 3. Fill the interquartile range (IQR)
    ax.fill_between(years,
                    df["Q25_holding_period_years"],
                    df["Q75_holding_period_years"],
                    color='blue', alpha=0.1,
                    label="Interquartile Range (25th to 75th Pctl)")

    # 4. Add labels for N_exits (Right axis)
    ax2 = ax.twinx()
    ax2.bar(years, df["N_exits"], color='gray', alpha=0.2, label='Number of Exits (N)', width=0.8)
    ax2.set_ylabel("Number of Exits (N)", color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.grid(False)  # Turn off grid for the count axis

    # 5. Axes and Labels
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Exit Year")
    ax.set_ylabel("Holding Period (Years)", color='blue')
    ax.tick_params(axis='y', colors='blue')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)
    ax.grid(True, axis='y', linestyle='--')

    # 6. Merge Legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=4)

    plt.tight_layout(rect=[0, 0.25, 1, 1])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved figure → {outfile}")


# ============================================================
# 4. MASTER DRIVER
# ============================================================

def run_all(year_min: int = 1995, year_max: int = 2025):
    ROOT = Path(__file__).resolve().parents[1]

    figs = ROOT / "outputs" / "figures"
    panels = ROOT / "outputs" / "panels"
    figs.mkdir(parents=True, exist_ok=True)
    panels.mkdir(parents=True, exist_ok=True)

    # --- Load all pairs first (used for all Universe analysis) ---
    try:
        univ_pairs = load_universe_pairs(ROOT)
        univ_entries = load_entry_subset(ROOT, subset_type="universe")
    except FileNotFoundError as e:
        print(f"Skipping all Universe-based analysis: {e}")
        univ_pairs = None
        univ_entries = None

    # ====================================================================
    # COMPLETION ANALYSIS (A, B, C)
    # ====================================================================

    # --- 1. TAKE-PRIVATE (TP) COMPLETION ---
    print("\n--- 1. Processing Take-Private Completion ---")
    tp_entries = None
    tp_pairs = None
    try:
        tp_entries = load_tp_entries(ROOT)
        tp_pairs = load_tp_pairs(ROOT)
        tp_summary = build_completion_frame(tp_entries, tp_pairs, year_min, year_max)
        tp_summary.to_csv(panels / "tp_entry_completion_panel.csv", index=False)

        plot_completion_bars(
            tp_summary,
            title="Take-Private Deals — Entry-Year Completion (Exits by Count and Value)",
            outfile=figs / "TP_EntryCompletion.png",
            year_min=year_min,
            year_max=year_max,
        )
    except FileNotFoundError as e:
        print(f"Skipping TP Completion: {e}")

    # --- 2. NON-TAKE-PRIVATE (NON-TP) COMPLETION ---
    if univ_pairs is not None and univ_entries is not None:
        print("\n--- 2. Processing Non-Take-Private Completion (Fix implemented for column check) ---")
        try:
            # FIX: load_entry_subset is updated to robustly check for TP columns
            non_tp_entries = load_entry_subset(ROOT, subset_type="non_tp")

            non_tp_summary = build_completion_frame(non_tp_entries, univ_pairs, year_min, year_max)
            non_tp_summary.to_csv(panels / "non_tp_entry_completion_panel.csv", index=False)

            plot_completion_bars(
                non_tp_summary,
                title="Non-Take-Private Deals — Entry-Year Completion (Exits by Count and Value)",
                outfile=figs / "NonTP_EntryCompletion.png",
                year_min=year_min,
                year_max=year_max,
            )
        except Exception as e:
            print(f"Skipping Non-TP Completion due to error: {e}")

    # --- 3. FULL UNIVERSE COMPLETION ---
    if univ_pairs is not None and univ_entries is not None:
        print("\n--- 3. Processing Full Universe Completion ---")
        try:
            univ_summary = build_completion_frame(univ_entries, univ_pairs, year_min, year_max)
            univ_summary.to_csv(panels / "universe_entry_completion_panel.csv", index=False)

            plot_completion_bars(
                univ_summary,
                title="Full Universe — Entry-Year Completion (Exits by Count and Value)",
                outfile=figs / "Universe_EntryCompletion.png",
                year_min=year_min,
                year_max=year_max,
            )
        except Exception as e:
            print(f"Skipping Full Universe Completion due to error: {e}")

    # ====================================================================
    # EXIT TYPE BREAKDOWN (D)
    # ====================================================================

    print("\n--- 4. Processing Exit Type Breakdown (Stacked Bars) ---")

    # 4a. TP Exit Type Breakdown
    if tp_entries is not None and tp_pairs is not None:
        print("--- 4a. Take-Private Exit Type Breakdown ---")
        try:
            tp_exit_panel = analyze_exit_type_breakdown(tp_entries, tp_pairs, year_min, year_max)
            tp_exit_panel.to_csv(panels / "tp_exit_type_panel.csv", index=False)

            plot_exit_type_stacked_bars(
                tp_exit_panel, 'N',
                title="Take-Private Deals — Exit Type by Entry Year (Count)",
                outfile=figs / "TP_ExitType_N.png",
                year_min=year_min, year_max=year_max
            )

            plot_exit_type_stacked_bars(
                tp_exit_panel, 'V',
                title="Take-Private Deals — Exit Type by Entry Year (Entry Value $B)",
                outfile=figs / "TP_ExitType_V.png",
                year_min=year_min, year_max=year_max
            )
        except Exception as e:
            print(f"Skipping TP Exit Type Analysis due to error: {e}")

    # 4b. Full Universe Exit Type Breakdown
    if univ_entries is not None and univ_pairs is not None:
        print("--- 4b. Full Universe Exit Type Breakdown ---")
        try:
            univ_exit_panel = analyze_exit_type_breakdown(univ_entries, univ_pairs, year_min, year_max)
            univ_exit_panel.to_csv(panels / "universe_exit_type_panel.csv", index=False)

            plot_exit_type_stacked_bars(
                univ_exit_panel, 'N',
                title="Full Universe — Exit Type by Entry Year (Count)",
                outfile=figs / "Universe_ExitType_N.png",
                year_min=year_min, year_max=year_max
            )

            plot_exit_type_stacked_bars(
                univ_exit_panel, 'V',
                title="Full Universe — Exit Type by Entry Year (Entry Value $B)",
                outfile=figs / "Universe_ExitType_V.png",
                year_min=year_min, year_max=year_max
            )
        except Exception as e:
            print(f"Skipping Universe Exit Type Analysis due to error: {e}")

    # ====================================================================
    # HOLDING PERIOD ANALYSIS (E)
    # ====================================================================

    print("\n--- 5. Processing Holding Period by Exit Year ---")

    # 5a. TP Holding Period
    if tp_pairs is not None:
        print("--- 5a. Take-Private Holding Period ---")
        try:
            tp_hp_panel = analyze_holding_period(tp_pairs, year_min, year_max)
            tp_hp_panel.to_csv(panels / "tp_holding_period_panel.csv", index=False)

            plot_holding_period_line(
                tp_hp_panel,
                title="Take-Private Deals — Average Holding Period by Exit Year",
                outfile=figs / "TP_HoldingPeriod.png",
                year_min=year_min, year_max=year_max
            )
        except Exception as e:
            print(f"Skipping TP Holding Period Analysis due to error: {e}")

    # 5b. Full Universe Holding Period
    if univ_pairs is not None:
        print("--- 5b. Full Universe Holding Period ---")
        try:
            univ_hp_panel = analyze_holding_period(univ_pairs, year_min, year_max)
            univ_hp_panel.to_csv(panels / "universe_holding_period_panel.csv", index=False)

            plot_holding_period_line(
                univ_hp_panel,
                title="Full Universe Deals — Average Holding Period by Exit Year",
                outfile=figs / "Universe_HoldingPeriod.png",
                year_min=year_min, year_max=year_max
            )
        except Exception as e:
            print(f"Skipping Universe Holding Period Analysis due to error: {e}")

    print(f"\n✅ All analyses panels and figures saved.")


if __name__ == "__main__":
    run_all(year_min=2000, year_max=2025)