import pandas as pd
import plotly.express as px
from pathlib import Path


def find_project_root() -> Path:
    """Finds the 'PEAgent' root directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent
    return Path(r"C:\Users\azhao\PycharmProjects\PEAgent")


def analyze_exit_cohorts():
    # 1. Setup Paths
    ROOT = find_project_root()
    GRAPHICS_DIR = ROOT / "graphics"
    GRAPHICS_DIR.mkdir(exist_ok=True)

    xlsx_file = ROOT / "data" / "clean" / "p2p_linked_master_updated.xlsx"
    csv_file = ROOT / "data" / "clean" / "p2p_linked_master_updated.csv"
    INPUT_FILE = xlsx_file if xlsx_file.exists() else csv_file

    if not INPUT_FILE.exists():
        print(f"ERROR: Could not find data file in {ROOT / 'data' / 'clean'}")
        return

    # 2. Load Data
    try:
        df = (
            pd.read_excel(INPUT_FILE, engine="openpyxl")
            if INPUT_FILE.suffix.lower() == ".xlsx"
            else pd.read_csv(INPUT_FILE)
        )
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 3. Data Engineering
    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["entry_year"] = df["deal_date_entry"].dt.year

    # Normalize exit categories to match your desired labeling + colors
    def normalize_exit_type(x) -> str:
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

        # fallback: keep original label (but trimmed)
        return str(x).strip()

    df["deal_type_exit_final"] = df["deal_type_exit_final"].apply(normalize_exit_type)

    # Handle missing exits as "Active / Unrealized"
    df["deal_type_exit_final"] = df["deal_type_exit_final"].fillna("Unmatched")

    # Filter for the last 25 years
    current_year = pd.Timestamp.now().year
    df = df[(df["entry_year"] >= current_year - 25) & (df["entry_year"] <= current_year)].copy()
    df = df.dropna(subset=["entry_year"])
    df["entry_year"] = df["entry_year"].astype(int)

    # Group and calculate counts
    cohort_counts = (
        df.groupby(["entry_year", "deal_type_exit_final"])
        .size()
        .reset_index(name="count")
    )

    if cohort_counts.empty:
        print("No data available after filtering/grouping.")
        return

    # Sort exit types for consistent stacking (bottom -> top)
    exit_types = [
        "Strategic",            # blue
        "IPO",                  # orange
        "Secondaries",          # red
        "Continuation Fund",    # yellow
        "Unmatched",  # grey
    ]

    # If you have unexpected categories, keep them visible (but after the core set)
    extras = [t for t in cohort_counts["deal_type_exit_final"].unique() if t not in exit_types]
    exit_types_extended = exit_types + sorted(extras)

    # Excel-ish palette with your semantic mapping
    color_map = {
        "Strategic": "#4472C4",            # Excel blue
        "IPO": "#ED7D31",                  # Excel orange
        "Secondaries": "#C00000",          # Excel red
        "Continuation Fund": "#FFC000",    # Excel yellow
        "Unmatched": "#A6A6A6",  # Excel grey
    }
    for t in extras:
        color_map.setdefault(t, "#636363")  # fallback dark gray

    # --- SHARED VISUAL CONFIGURATION (centered, Times New Roman, Excel-like) ---
    shared_layout = dict(
        template="plotly_white",
        font=dict(family="Times New Roman", size=12, color="black"),
        title=dict(
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="top",
            font=dict(family="Times New Roman", size=16, color="black"),
        ),
        xaxis=dict(
            title=dict(text="Vintage (Entry Year)", standoff=10),
            tickmode="linear",
            dtick=2,
            showgrid=False,
            showline=True,
            linecolor="black",
            ticks="outside",
            ticklen=5,
            tickcolor="black",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Share of Deals (%)", standoff=10),
            tickformat=".0f",
            showgrid=True,
            gridcolor="#EFEFEF",
            showline=True,
            linecolor="black",
            ticks="outside",
            ticklen=5,
            tickcolor="black",
            rangemode="tozero",
            zeroline=False,
        ),
        legend=dict(
            title=dict(text="Exit Strategy"),
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(family="Times New Roman", size=11),
        ),
        margin=dict(t=120, b=70, l=70, r=40),
        bargap=0.15,
    )

    # --- CHART: 100% STACKED BAR ---
    fig_cohort = px.bar(
        cohort_counts,
        x="entry_year",
        y="count",
        color="deal_type_exit_final",
        title="Exit Composition by Vintage (100% Stacked)",
        category_orders={"deal_type_exit_final": exit_types_extended},
        color_discrete_map=color_map,
        barmode="relative",
    )
    fig_cohort.update_layout(**shared_layout)
    fig_cohort.update_layout(barnorm="percent")

    # Clean hover
    fig_cohort.update_traces(
        hovertemplate="Entry Year: %{x}<br>Share: %{y:.1f}%<extra></extra>"
    )

    # 4. Export PNG
    print(f"Exporting graphics to {GRAPHICS_DIR}...")
    try:
        fig_cohort.write_image(str(GRAPHICS_DIR / "exit_composition_100pct.png"), scale=2)
        print("Success! 100% stacked chart saved.")
    except Exception as e:
        print(f"Static export failed (check if 'kaleido' is installed): {e}")


if __name__ == "__main__":
    analyze_exit_cohorts()