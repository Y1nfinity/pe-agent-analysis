from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# LOAD
# =============================================================================

def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find clean PitchBook master: {path}")
    # Note: This script expects the OUTPUT of the universe linking script
    return pd.read_parquet(path)


# =============================================================================
# PANEL 1 — UNIVERSE ENTRY-YEAR VIEW
# =============================================================================

def build_universe_entry_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For all entries (every deal):
        - N deals
        - N deals categorized as "exits" (based on exit_category)
        - Breakdown by exit_category
        - Fraction exits
    """

    df = df.copy()
    # --- FIX: Use deal_date_entry to derive entry_year ---
    if "deal_date_entry" in df.columns:
        df["entry_year"] = df["deal_date_entry"].dt.year
    else:
        # Fallback if the data frame passed is the pre-linked universe (which has 'transaction_year')
        # But for post-linked data, we rely on the specific entry/exit columns.
        df["entry_year"] = df["transaction_year"]

        # What counts as an "exit"?
    # Anything that has an exit_category that is NOT "other"
    # NOTE: In the linked file, exit_category_entry is the category of the ENTRY deal.
    # We are tracking entry deals here, so we use the *entry* deal's category.
    if 'exit_category_entry' in df.columns:
        df["is_exit"] = df["exit_category_entry"].ne("other")
    elif 'exit_category' in df.columns:
        df["is_exit"] = df["exit_category"].ne("other")
    else:
        df["is_exit"] = False  # Should not happen with clean data

    grouped = df.groupby("entry_year")

    out = pd.DataFrame({
        "N_deals": grouped.size(),
        "N_exits": grouped["is_exit"].sum(),
    })

    out["frac_exits"] = out["N_exits"] / out["N_deals"]

    # Breakdown by exit_category
    breakdown = (
        df[df["is_exit"]]
        .groupby(["entry_year", "exit_category_entry" if 'exit_category_entry' in df.columns else 'exit_category'])
        .size()
        .unstack(fill_value=0)
    )

    # Clean up breakdown column names for consistency
    breakdown.columns = [col.replace('_entry', '') for col in breakdown.columns]

    return out.join(breakdown, how="left").fillna(0).reset_index()


# =============================================================================
# PANEL 2 — UNIVERSE EXIT-YEAR VIEW
# =============================================================================

def build_universe_exit_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For all *linked* deals that have an exit:
        - N exits
        - Breakdown by exit_category (of the exit deal)
        - exit_value_usd_m
    """

    # Filter for rows that actually represent an *exit* deal being linked
    df = df[df["deal_id_exit"].notna()].copy()

    # --- FIX: Use deal_date_exit to derive exit_year ---
    df["exit_year"] = df["deal_date_exit"].dt.year

    # Use the exit deal's category and size
    df.rename(columns={
        "exit_category_exit": "exit_category",
        "deal_size_usd_m_exit": "exit_value_usd_m"
    }, inplace=True)

    # Remove "other" (unclassified) exits if they somehow slipped through linking
    df = df[df["exit_category"].ne("other")].copy()

    grouped = df.groupby("exit_year")

    out = pd.DataFrame({
        "N_exits": grouped.size(),
        "exit_value_usd_m": grouped["exit_value_usd_m"].sum(),
    })

    breakdown = (
        df.groupby(["exit_year", "exit_category"])
        .size()
        .unstack(fill_value=0)
    )

    return out.join(breakdown, how="left").reset_index()


# =============================================================================
# PANEL 3 — DEAL-YEAR ACTIVITY (UNIVERSE)
# =============================================================================

def build_universe_deal_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    This panel should use the full original universe (pre-linked data)
    to show total activity, but since we are running this script
    with the *linked* data, we need to adapt.

    We will combine entry and exit data to proxy for overall deal activity.

    Note: If you want a panel of ALL raw deals, you should run this function
    on the raw cleaned data from the first half of universe_linking.py.

    For now, we'll use the entry year count from Panel 1 as a proxy
    for the deal-year activity of entries.
    """
    # For simplicity and to avoid reloading the raw file,
    # we'll just re-run the entry year panel logic.

    df = df.copy()

    if "deal_date_entry" in df.columns:
        df["transaction_year"] = df["deal_date_entry"].dt.year
    elif "transaction_year_entry" in df.columns:
        df["transaction_year"] = df["transaction_year_entry"]
    else:
        # Fallback to a core year column if available
        df["transaction_year"] = df["deal_date_entry"].dt.year

    grouped = df.groupby("transaction_year")

    # Use deal_size_usd_m_entry as the value for the transaction
    deal_value_col = "deal_size_usd_m_entry" if "deal_size_usd_m_entry" in df.columns else "deal_size_usd_m"

    out = pd.DataFrame({
        "N_deals": grouped.size(),
        "total_value_usd_m": grouped[deal_value_col].sum(),
    })

    # deal_type breakdown (using the entry deal type)
    deal_type_col = "deal_type_entry" if "deal_type_entry" in df.columns else "deal_type"
    breakdown = (
        df.groupby(["transaction_year", deal_type_col])
        .size()
        .unstack(fill_value=0)
    )

    # Clean up breakdown column names for consistency
    breakdown.columns = [col.replace('_entry', '') for col in breakdown.columns]

    return out.join(breakdown, how="left").reset_index()


# =============================================================================
# FIGURES
# =============================================================================

def plot_basic_line(df, x, y, title, outpath):
    plt.figure(figsize=(12, 6))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_all_figures(entry, exit_, deal, outdir: Path):
    plot_basic_line(
        entry, "entry_year", "N_deals",
        "Universe — Deal Counts by Entry Year",
        outdir / "Universe_EntryYear_NDeals.png"
    )

    plot_basic_line(
        entry, "entry_year", "frac_exits",
        "Universe — Fraction of Deals that are Exits (Entry-Year View)",
        outdir / "Universe_EntryYear_FracExits.png"
    )

    plot_basic_line(
        exit_, "exit_year", "N_exits",
        "Universe — Exit Counts by Exit Year",
        outdir / "Universe_ExitYear_NExits.png"
    )

    plot_basic_line(
        exit_, "exit_year", "exit_value_usd_m",
        "Universe — Exit Value by Exit Year",
        outdir / "Universe_ExitYear_ExitValue.png"
    )

    plot_basic_line(
        deal, "transaction_year", "N_deals",
        "Universe — Total Deals per Year",
        outdir / "Universe_DealYear_NDeals.png"
    )


# =============================================================================
# DRIVER
# =============================================================================

def run_universe_panels(master_path: Path, panel_dir: Path, fig_dir: Path):
    # Note: master_path typically refers to the CLEANED universe dataset from step 1
    df = load_master(master_path)

    entry_panel = build_universe_entry_year_panel(df)
    exit_panel = build_universe_exit_year_panel(df)
    deal_panel = build_universe_deal_year_panel(df)

    panel_dir.mkdir(parents=True, exist_ok=True)

    entry_panel.to_csv(panel_dir / "universe_entry_year_panel.csv", index=False)
    exit_panel.to_csv(panel_dir / "universe_exit_year_panel.csv", index=False)
    deal_panel.to_csv(panel_dir / "universe_deal_year_panel.csv", index=False)

    save_all_figures(entry_panel, exit_panel, deal_panel, fig_dir)

    print("✅ Universe Panels & Figures Saved")
    print("   Panels:", panel_dir.resolve())
    print("   Figures:", fig_dir.resolve())


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    # This should likely point to the clean universe parquet file
    MASTER = ROOT / "data" / "clean" / "universe_entry_exit_pairs.parquet"
    PANELS = ROOT / "outputs" / "panels_universe"
    FIGS = ROOT / "outputs" / "figures_universe"

    run_universe_panels(MASTER, PANELS, FIGS)