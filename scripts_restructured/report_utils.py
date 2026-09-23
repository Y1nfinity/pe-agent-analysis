"""
report_utils.py
---------------
Purpose:
    Export cleaned analytical outputs (tables and figures) alongside
    1–2 sentence captions for academic or policy reports.

Supports:
    - Saving tables to CSV with a paired _description.txt file
    - Saving captions for figures
    - Consistent labeling (Figure_##_ExitType_Description.png)
    - Building a single organized "Graphs & Visualization Folder"

Usage Example:
    from scripts_restructured import report_utils as ru

    ru.export_table(df_summary, "outputs/tables/Exit_Summary.csv",
                    "Summary table of exit counts and proportions by year.")

    ru.save_figure_with_caption("outputs/figures/Figure_01_TakePrivate_Trend.png",
                                "Figure 1. Annual trend in take-private exits.")

    ru.build_graphs_folder({
        "Figure_01_TakePrivate_Trend.png": "Annual trend in take-private exits.",
        "Figure_02_ExitMix_Shares.png": "Exit mix by type, share of annual total."
    })
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# -----------------------------------------------------------------------------
# Export tables
# -----------------------------------------------------------------------------
def export_table(df: pd.DataFrame, out_csv: str, description: Optional[str] = None) -> None:
    """
    Export a DataFrame as CSV and, if provided, a sidecar description text file.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save
    out_csv : str
        Path to output .csv file
    description : Optional[str]
        Optional 1–2 sentence description of the table
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if description:
        desc_path = out_path.parent / f"{out_path.stem}_description.txt"
        desc_path.write_text(description.strip())

# -----------------------------------------------------------------------------
# Export figure captions
# -----------------------------------------------------------------------------
def save_figure_with_caption(fig_path: str, caption: str) -> None:
    """
    Save a short caption alongside a figure as <figure>_caption.txt.

    Parameters
    ----------
    fig_path : str
        Path to the saved figure
    caption : str
        Caption text (e.g., "Figure 3. Exit mix by category.")
    """
    path = Path(fig_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    caption_file = path.parent / f"{path.stem}_caption.txt"
    caption_file.write_text(caption.strip())

# -----------------------------------------------------------------------------
# Label helper
# -----------------------------------------------------------------------------
def label_figure(number: int, exit_type: str, desc: str) -> str:
    """
    Generate a standardized filename for figures.

    Returns:
        e.g., 'Figure_03_TakePrivate_Trend.png'
    """
    safe_type = exit_type.replace(" ", "").replace("-", "")
    safe_desc = desc.replace(" ", "").replace("-", "")
    return f"Figure_{number:02d}_{safe_type}_{safe_desc}.png"

# -----------------------------------------------------------------------------
# Folder builder
# -----------------------------------------------------------------------------
def build_graphs_folder(figures: Dict[str, str],
                        base_dir: str = "outputs/figures") -> Path:
    """
    Ensure a complete folder of graphs with captions exists.

    Parameters
    ----------
    figures : dict
        Mapping of {figure_filename -> caption_text}
    base_dir : str
        Output base directory for figures and captions

    Returns:
        Path to created folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    for fname, caption in figures.items():
        caption_file = base_path / f"{Path(fname).stem}_caption.txt"
        caption_file.write_text(caption.strip())
    return base_path

# -----------------------------------------------------------------------------
# Batch export utility
# -----------------------------------------------------------------------------
def export_summary_package(
    tables: Dict[str, pd.DataFrame],
    figures: Dict[str, str],
    out_root: str = "outputs"
) -> None:
    """
    Save a complete package of tables and figures for review.

    Parameters
    ----------
    tables : Dict[str, pd.DataFrame]
        Mapping of table_name -> DataFrame
    figures : Dict[str, str]
        Mapping of figure_filename -> caption_text
    out_root : str
        Root folder (default = 'outputs')
    """
    root = Path(out_root)
    tables_dir = root / "tables"
    figs_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    for name, df in tables.items():
        export_table(df, tables_dir / f"{name}.csv", f"Data table: {name}")

    # Save figure captions
    for fname, cap in figures.items():
        save_figure_with_caption(figs_dir / fname, cap)

    print(f"✅ Exported tables and captions to '{root}'")
