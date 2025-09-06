# tools/export.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import math

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _guess_percent_cols(df: pd.DataFrame) -> list[str]:
    """
    Heuristic: columns whose names suggest fractions/IRR/percent.
    """
    keys = ("fraction", "irr", "_share", "pct", "percent")
    out = [c for c in df.columns if any(k in str(c).lower() for k in keys)]
    # keep only numeric
    out = [c for c in out if pd.api.types.is_numeric_dtype(df[c])]
    return out


def _guess_money_cols(df: pd.DataFrame) -> list[str]:
    """
    Heuristic: columns whose names suggest money/spend/deal size/valuation.
    """
    keys = ("deal", "spend", "valuation", "price", "expenditure", "entry_")
    out = [c for c in df.columns if any(k in str(c).lower() for k in keys)]
    # keep only numeric
    out = [c for c in out if pd.api.types.is_numeric_dtype(df[c])]
    return out


def _format_df_for_display(
    df: pd.DataFrame,
    percent_cols: Optional[Iterable[str]] = None,
    money_cols: Optional[Iterable[str]] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Returns a *string-formatted* copy of df for HTML/PNG export.
    """
    disp = df.copy()

    if percent_cols is None:
        percent_cols = _guess_percent_cols(disp)
    if money_cols is None:
        money_cols = _guess_money_cols(disp)

    # Format percents (assume values like 0.42 -> 42.00%)
    for c in percent_cols:
        try:
            disp[c] = (disp[c] * 100).map(lambda x: f"{x:.{decimals}f}%")
        except Exception:
            pass

    # Format money with thousands separators
    for c in money_cols:
        try:
            disp[c] = disp[c].map(lambda x: f"${x:,.{decimals}f}")
        except Exception:
            pass

    # For other floats, round for legibility
    for c in disp.columns:
        if c in percent_cols or c in money_cols:
            continue
        if pd.api.types.is_float_dtype(disp[c]):
            disp[c] = disp[c].map(lambda x: f"{x:.{decimals}f}")

    return disp


def _dataframe_to_png(
    df_display: pd.DataFrame,
    path: Path,
    max_rows: int = 40,
    col_width: float = 2.2,   # inches per column (heuristic)
    row_height: float = 0.35, # inches per row
    header_height: float = 0.45,
    font_size: int = 9,
    title: Optional[str] = None,
) -> None:
    """
    Render a DataFrame (already string-formatted) to a PNG via matplotlib.
    No external dependencies beyond matplotlib.
    """
    _ensure_dir(path)

    df_show = df_display.copy()
    truncated = False
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows).copy()
        truncated = True

    nrows, ncols = df_show.shape
    fig_w = min(22, max(6, ncols * col_width))
    fig_h = max(2.0, header_height + nrows * row_height + (0.4 if title else 0.2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Title
    if title:
        ax.set_title(title, loc="left", fontsize=font_size + 2, pad=10)

    # Build table
    table = ax.table(
        cellText=df_show.values,
        colLabels=[str(c) for c in df_show.columns],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0, 1, 1]  # fill axes
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Column widths heuristic
    # Scale by the longest string length in each column
    for i, col in enumerate(df_show.columns):
        max_len = max(len(str(col)), *(len(str(v)) for v in df_show[col].values))
        # Normalize length ~ characters to relative width
        rel = min(1.6, 0.7 + max_len / 18.0)
        table.auto_set_column_width(i)
        for (row, col_idx), cell in table.get_celld().items():
            if col_idx == i:
                # pad a bit
                cell.PAD = 0.01
        # Matplotlib table lacks per-column width setter; bbox handles total width.
        # We keep a consistent font and rely on fig_w to prevent overflow.

    # Zebra stripes for readability
    for row in range(1, nrows + 1):
        color = (0.96, 0.96, 0.96) if row % 2 == 0 else (1, 1, 1)
        for col in range(0, ncols):
            table[(row, col)].set_facecolor(color)

    # Header styling
    for col in range(0, ncols):
        table[(0, col)].set_facecolor((0.9, 0.9, 0.9))
        table[(0, col)].set_text_props(weight="bold")

    # Truncation note
    if truncated:
        ax.text(0, -0.02, f"Showing first {max_rows} rows (truncated).",
                fontsize=font_size, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def export_table(
    df: pd.DataFrame,
    base_path: Path,
    *,
    title: Optional[str] = None,
    formats: Iterable[str] = ("csv", "html", "png"),
    percent_cols: Optional[Iterable[str]] = None,
    money_cols: Optional[Iterable[str]] = None,
    decimals: int = 2,
) -> dict[str, Path]:
    """
    Export a DataFrame to CSV/HTML/PNG with smart formatting for percents and currency.

    Args:
        df: DataFrame to export.
        base_path: Path WITHOUT extension, e.g., Path("artifacts/tables/my_table")
        title: Optional title for PNG.
        formats: subset of {"csv","html","png"}.
        percent_cols, money_cols: override auto-detection if needed.
        decimals: numeric precision for display.

    Returns:
        dict of {'csv': path, 'html': path, 'png': path} for files created.
    """
    out_paths: dict[str, Path] = {}

    # CSV (raw numeric values)
    if "csv" in formats:
        csv_path = base_path.with_suffix(".csv")
        _ensure_dir(csv_path)
        df.to_csv(csv_path, index=False)
        out_paths["csv"] = csv_path

    # HTML (formatted)
    if "html" in formats:
        html_path = base_path.with_suffix(".html")
        _ensure_dir(html_path)
        disp = _format_df_for_display(df, percent_cols, money_cols, decimals)
        html = disp.to_html(index=False, border=0)
        html_path.write_text(html, encoding="utf-8")
        out_paths["html"] = html_path

    # PNG (formatted)
    if "png" in formats:
        png_path = base_path.with_suffix(".png")
        disp = _format_df_for_display(df, percent_cols, money_cols, decimals)
        _dataframe_to_png(disp, png_path, title=title)
        out_paths["png"] = png_path

    return out_paths
