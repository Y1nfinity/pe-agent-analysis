from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


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

    return Path(r"C:\Users\azhao\PycharmProjects\PEAgent")


# ============================================================
# 2. GARAMOND / EXCEL-STYLE CONSTANTS
# ============================================================

FONT_FAMILY = "Garamond, EB Garamond, Cormorant Garamond, Times New Roman, serif"

EXCEL_BLUE = "#4472C4"
EXCEL_RED = "#C00000"
EXCEL_GREY = "#7F7F7F"
GRID_COLOR = "#EFEFEF"

PLOTLY_TEMPLATE = "plotly_white"


def apply_garamond_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """
    Applies Garamond-first formatting to a Plotly figure.
    """
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=title,
            font=dict(
                family=FONT_FAMILY,
                size=24,
                color="black",
            ),
            x=0.5,
            xanchor="center",
        ) if title else None,
        font=dict(
            family=FONT_FAMILY,
            size=16,
            color="black",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=40, t=90, b=70),
        legend=dict(
            font=dict(
                family=FONT_FAMILY,
                size=15,
                color="black",
            ),
            bgcolor="white",
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        xaxis=dict(
            title_font=dict(
                family=FONT_FAMILY,
                size=18,
                color="black",
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14,
                color="black",
            ),
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=False,
        ),
        yaxis=dict(
            title_font=dict(
                family=FONT_FAMILY,
                size=18,
                color="black",
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14,
                color="black",
            ),
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=False,
        ),
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

def prepare_holding_period_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["deal_date_entry", "deal_date_exit"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df.copy()

    df["deal_date_entry"] = pd.to_datetime(df["deal_date_entry"], errors="coerce")
    df["deal_date_exit"] = pd.to_datetime(df["deal_date_exit"], errors="coerce")

    exited_df = df.dropna(subset=["deal_date_entry", "deal_date_exit"]).copy()

    exited_df["hold_period_years"] = (
        exited_df["deal_date_exit"] - exited_df["deal_date_entry"]
    ).dt.days / 365.25

    exited_df = exited_df[
        (exited_df["hold_period_years"] > 0)
        & (exited_df["hold_period_years"] < 15)
    ].copy()

    exited_df["exit_year"] = exited_df["deal_date_exit"].dt.year
    exited_df["hold_period_years"] = exited_df["hold_period_years"].round(2)

    return exited_df


# ============================================================
# 5. CHART BUILDERS
# ============================================================

def build_histogram(exited_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        exited_df,
        x="hold_period_years",
        nbins=20,
        marginal="box",
        labels={
            "hold_period_years": "Years Held",
        },
        color_discrete_sequence=[EXCEL_BLUE],
    )

    fig = apply_garamond_layout(
        fig,
        title="Distribution of Holding Periods",
    )

    fig.update_traces(
        marker=dict(
            color=EXCEL_BLUE,
            line=dict(
                color="white",
                width=1,
            ),
        ),
        opacity=0.85,
    )

    fig.update_layout(
        bargap=0.1,
        showlegend=False,
    )

    fig.update_xaxes(
        title_text="Years Held",
    )

    fig.update_yaxes(
        title_text="Frequency",
    )

    return fig


def build_trend_chart(exited_df: pd.DataFrame) -> go.Figure:
    yearly_avg = (
        exited_df
        .groupby("exit_year", as_index=False)["hold_period_years"]
        .mean()
    )

    fig = px.line(
        yearly_avg,
        x="exit_year",
        y="hold_period_years",
        markers=True,
        line_shape="spline",
        labels={
            "exit_year": "Exit Year",
            "hold_period_years": "Average Years Held",
        },
    )

    fig.update_traces(
        line=dict(
            color=EXCEL_RED,
            width=3,
        ),
        marker=dict(
            color=EXCEL_RED,
            size=9,
        ),
        mode="lines+markers+text",
        text=[f"{v:.1f}" for v in yearly_avg["hold_period_years"]],
        textposition="top center",
        textfont=dict(
            family=FONT_FAMILY,
            size=13,
            color=EXCEL_RED,
        ),
    )

    fig = apply_garamond_layout(
        fig,
        title="Average Holding Period Trend by Exit Year",
    )

    fig.update_xaxes(
        title_text="Exit Year",
        dtick=2,
    )

    fig.update_yaxes(
        title_text="Average Years Held",
    )

    return fig


def build_exit_type_boxplot(exited_df: pd.DataFrame) -> go.Figure:
    if "deal_type_exit" not in exited_df.columns:
        fig = go.Figure()

        fig.add_annotation(
            text="deal_type_exit column not found",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                family=FONT_FAMILY,
                size=18,
                color="black",
            ),
        )

        fig = apply_garamond_layout(
            fig,
            title="Holding Period by Exit Strategy",
        )

        return fig

    fig = px.box(
        exited_df,
        x="deal_type_exit",
        y="hold_period_years",
        color="deal_type_exit",
        points="all",
        labels={
            "deal_type_exit": "Exit Strategy",
            "hold_period_years": "Years Held",
        },
    )

    fig = apply_garamond_layout(
        fig,
        title="Holding Period by Exit Strategy",
    )

    fig.update_layout(
        showlegend=False,
    )

    fig.update_traces(
        marker=dict(
            size=6,
            opacity=0.55,
        ),
        line=dict(
            width=1.5,
        ),
    )

    fig.update_xaxes(
        title_text="Exit Strategy",
        tickangle=-30,
    )

    fig.update_yaxes(
        title_text="Years Held",
    )

    return fig


# ============================================================
# 6. EXPORT
# ============================================================

def write_html_report(output_path: Path, figs: list[tuple[str, go.Figure]], graphics_dir: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head>
            <title>Holding Period Analytics</title>
            <style>
                body {{
                    font-family: {FONT_FAMILY};
                    margin: 3%;
                    background: #f4f7f6;
                    color: #333;
                }}
                .card {{
                    background: white;
                    padding: 24px;
                    margin-bottom: 28px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e1e4e8;
                }}
                h1 {{
                    color: #111;
                    font-size: 34px;
                    margin-bottom: 8px;
                }}
                h2 {{
                    color: #111;
                    font-size: 24px;
                    margin-bottom: 14px;
                }}
                .meta {{
                    color: #666;
                    font-size: 16px;
                    margin-bottom: 24px;
                }}
            </style>
        </head>
        <body>
            <h1>Holding Period Analysis Dashboard</h1>
            <div class="meta">Report Generated: Same Folder | Graphics: {graphics_dir}</div>
        """)

        for title, fig in figs:
            f.write(f"<div class='card'><h2>{title}</h2>")
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("</div>")

        f.write("</body></html>")


# ============================================================
# 7. MAIN
# ============================================================

def analyze_holding_periods():
    ROOT = find_project_root()

    GRAPHICS_DIR = ROOT / "reports" / "figures"
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_p2p_linked_master(ROOT)

    if df.empty:
        print("❌ No data loaded. Stopping.")
        return

    exited_df = prepare_holding_period_data(df)

    if exited_df.empty:
        print("❌ No valid holding-period data. Stopping.")
        return

    print("\n========== HOLDING PERIOD CHECK ==========")
    print(f"Exited deals used:       {len(exited_df):,}")
    print(f"Median holding period:   {exited_df['hold_period_years'].median():.2f} years")
    print(f"Mean holding period:     {exited_df['hold_period_years'].mean():.2f} years")
    print(f"Min holding period:      {exited_df['hold_period_years'].min():.2f} years")
    print(f"Max holding period:      {exited_df['hold_period_years'].max():.2f} years")
    print("==========================================\n")

    fig_hist = build_histogram(exited_df)
    fig_trend = build_trend_chart(exited_df)
    fig_box = build_exit_type_boxplot(exited_df)

    print(f"Exporting PNGs to {GRAPHICS_DIR}...")

    try:
        fig_hist.write_image(str(GRAPHICS_DIR / "hold_period_dist.png"), scale=2)
        fig_trend.write_image(str(GRAPHICS_DIR / "hold_period_trend.png"), scale=2)
        fig_box.write_image(str(GRAPHICS_DIR / "hold_period_by_exit_type.png"), scale=2)
    except Exception as e:
        print(f"⚠️ Static export failed. Check whether kaleido is installed. Error: {e}")

    output_path = Path(__file__).parent / "holding_period_report.html"

    write_html_report(
        output_path=output_path,
        figs=[
            ("1. Period Distribution & Outliers", fig_hist),
            ("2. Temporal Trends", fig_trend),
            ("3. Strategy Comparison", fig_box),
        ],
        graphics_dir=GRAPHICS_DIR,
    )

    print(f"✅ Success! Report saved to: {output_path}")


if __name__ == "__main__":
    analyze_holding_periods()