from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PATHING
# ============================================================

def find_project_root() -> Path:
    """Finds the PEAgent root directory."""
    current = Path(__file__).resolve().parent

    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent

    fallback = Path(r"C:\Users\azhao\PycharmProjects\PEAgent")
    return fallback if fallback.exists() else Path(__file__).resolve().parents[2]


# ============================================================
# 2. GARAMOND / EXCEL-STYLE CONSTANTS
# ============================================================

FONT_FAMILY = "Garamond, EB Garamond, Cormorant Garamond, Times New Roman, serif"

EXCEL_BLUE = "#4472C4"
EXCEL_ORANGE = "#ED7D31"
EXCEL_RED = "#C00000"
EXCEL_YELLOW = "#FFC000"
EXCEL_GREY = "#A6A6A6"
EXCEL_DARK_GREY = "#636363"
GRID_COLOR = "#EFEFEF"


def apply_garamond_layout(fig: go.Figure, title: str) -> go.Figure:
    """
    Applies Garamond / Excel-style formatting to a Plotly figure.
    """
    fig.update_layout(
        template="plotly_white",

        font=dict(
            family=FONT_FAMILY,
            size=16,
            color="black",
        ),

        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="top",
            font=dict(
                family=FONT_FAMILY,
                size=24,
                color="black",
            ),
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",

        xaxis=dict(
            title=dict(
                text="Vintage (Entry Year)",
                standoff=12,
                font=dict(
                    family=FONT_FAMILY,
                    size=18,
                    color="black",
                ),
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14,
                color="black",
            ),
            tickmode="linear",
            dtick=2,
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            ticklen=5,
            tickcolor="black",
            zeroline=False,
        ),

        yaxis=dict(
            title=dict(
                text="Share of Entry Deal Value (%)",
                standoff=12,
                font=dict(
                    family=FONT_FAMILY,
                    size=18,
                    color="black",
                ),
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14,
                color="black",
            ),
            tickformat=".0f",
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            ticklen=5,
            tickcolor="black",
            rangemode="tozero",
            zeroline=False,
        ),

        legend=dict(
            title=dict(
                text="Exit Strategy",
                font=dict(
                    family=FONT_FAMILY,
                    size=15,
                    color="black",
                ),
            ),
            font=dict(
                family=FONT_FAMILY,
                size=14,
                color="black",
            ),
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="white",
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),

        margin=dict(
            t=130,
            b=75,
            l=80,
            r=45,
        ),

        bargap=0.15,
    )

    return fig


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_p2p_linked_master(root: Path) -> pd.DataFrame:
    xlsx_file = root / "data" / "clean" / "p2p_linked_master_updated.xlsx"
    csv_file = root / "data" / "clean" / "p2p_linked_master_updated.csv"
    parquet_file = root / "data" / "clean" / "p2p_linked_master_updated.parquet"

    if xlsx_file.exists():
        input_file = xlsx_file
    elif csv_file.exists():
        input_file = csv_file
    elif parquet_file.exists():
        input_file = parquet_file
    else:
        print(f"❌ ERROR: Could not find data file in {root / 'data' / 'clean'}")
        return pd.DataFrame()

    try:
        if input_file.suffix.lower() == ".xlsx":
            df = pd.read_excel(input_file, engine="openpyxl")
        elif input_file.suffix.lower() == ".csv":
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_file)
        else:
            print(f"❌ ERROR: Unsupported file type: {input_file}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return pd.DataFrame()

    print(f"✅ Loaded: {input_file}")
    return df


# ============================================================
# 4. DATA ENGINEERING
# ============================================================

def normalize_exit_type(x) -> str:
    """
    Normalizes exit strategy labels for consistent chart stacking.
    """
    if pd.isna(x):
        return "Unmatched"

    x_lower = str(x).strip().lower()

    if "ipo" in x_lower:
        return "IPO"

    if "strategic" in x_lower or "trade" in x_lower:
        return "Strategic"

    if "continuation" in x_lower:
        return "Continuation Fund"

    if "club" in x_lower or "secondary" in x_lower:
        return "Secondaries"

    if "active" in x_lower or "unreal" in x_lower:
        return "Unmatched"

    return str(x).strip()


def prepare_exit_cohort_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "deal_date_entry",
        "deal_type_exit_final",
        "deal_size_entry",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df.copy()

    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["entry_year"] = df["deal_date_entry"].dt.year

    df["deal_type_exit_final"] = (
        df["deal_type_exit_final"]
        .apply(normalize_exit_type)
        .fillna("Unmatched")
    )

    df["deal_size_entry"] = pd.to_numeric(
        df["deal_size_entry"],
        errors="coerce",
    )

    current_year = pd.Timestamp.now().year

    df = df[
        (df["entry_year"] >= current_year - 25)
        & (df["entry_year"] <= current_year)
    ].copy()

    df = df.dropna(subset=["entry_year"])
    df["entry_year"] = df["entry_year"].astype(int)

    df_value = df.dropna(subset=["deal_size_entry"]).copy()

    if df_value.empty:
        print("❌ No data available after filtering for non-missing entry deal size.")
        return pd.DataFrame()

    return df_value


def build_value_weighted_cohort_summary(df_value: pd.DataFrame) -> pd.DataFrame:
    cohort_value = (
        df_value
        .groupby(["entry_year", "deal_type_exit_final"], as_index=False)["deal_size_entry"]
        .sum()
        .rename(columns={"deal_size_entry": "entry_value"})
    )

    if cohort_value.empty:
        print("❌ No grouped value data available.")
        return pd.DataFrame()

    return cohort_value


# ============================================================
# 5. CHART
# ============================================================

def build_exit_composition_chart(cohort_value: pd.DataFrame) -> go.Figure:
    exit_types = [
        "Strategic",
        "IPO",
        "Secondaries",
        "Continuation Fund",
        "Unmatched",
    ]

    extras = [
        t for t in cohort_value["deal_type_exit_final"].dropna().unique()
        if t not in exit_types
    ]

    exit_types_extended = exit_types + sorted(extras)

    color_map = {
        "Strategic": EXCEL_BLUE,
        "IPO": EXCEL_ORANGE,
        "Secondaries": EXCEL_RED,
        "Continuation Fund": EXCEL_YELLOW,
        "Unmatched": EXCEL_GREY,
    }

    for t in extras:
        color_map.setdefault(t, EXCEL_DARK_GREY)

    title = "Exit Composition by Vintage Weighted by Entry Deal Value"

    fig = px.bar(
        cohort_value,
        x="entry_year",
        y="entry_value",
        color="deal_type_exit_final",
        title=title,
        category_orders={
            "deal_type_exit_final": exit_types_extended,
        },
        color_discrete_map=color_map,
        barmode="relative",
    )

    fig.update_layout(
        barnorm="percent",
    )

    fig = apply_garamond_layout(
        fig,
        title=title,
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate=(
            "Entry Year: %{x}"
            "<br>Share of Entry Value: %{y:.1f}%"
            "<br>Exit Strategy: %{fullData.name}"
            "<extra></extra>"
        ),
    )

    return fig


# ============================================================
# 6. MAIN
# ============================================================

def analyze_exit_cohorts_by_entry_value():
    ROOT = find_project_root()

    FIGURE_DIR = ROOT / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_p2p_linked_master(ROOT)

    if df.empty:
        print("❌ No data loaded. Stopping.")
        return

    df_value = prepare_exit_cohort_data(df)

    if df_value.empty:
        print("❌ No usable cohort data. Stopping.")
        return

    cohort_value = build_value_weighted_cohort_summary(df_value)

    if cohort_value.empty:
        print("❌ No cohort summary available. Stopping.")
        return

    print("\n========== EXIT COHORT VALUE CHECK ==========")
    print(f"Rows used:              {len(df_value):,}")
    print(f"Entry years:            {df_value['entry_year'].nunique():,}")
    print(f"Exit strategies:        {df_value['deal_type_exit_final'].nunique():,}")
    print(f"Total entry value:      {df_value['deal_size_entry'].sum():,.1f}")
    print("=============================================\n")

    fig_value = build_exit_composition_chart(cohort_value)

    png_output_path = FIGURE_DIR / "exit_composition_entry_value_100pct.png"
    html_output_path = FIGURE_DIR / "exit_composition_entry_value_100pct.html"

    print(f"Exporting graphic to {png_output_path}...")

    try:
        fig_value.write_image(
            str(png_output_path),
            scale=2,
        )
        print(f"✅ Success! Static PNG saved to: {png_output_path}")

    except Exception as e:
        print(f"⚠️ Static export failed. Check if kaleido is installed. Error: {e}")

    try:
        fig_value.write_html(
            str(html_output_path),
            include_plotlyjs="cdn",
            full_html=True,
        )
        print(f"✅ Interactive HTML chart saved to: {html_output_path}")

    except Exception as e:
        print(f"⚠️ HTML export failed. Error: {e}")


if __name__ == "__main__":
    analyze_exit_cohorts_by_entry_value()