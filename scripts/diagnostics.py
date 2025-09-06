# scripts/diagnostics.py
from pathlib import Path
import pandas as pd


def dataset_diagnostics(file_path: str, sheet: str | None = None) -> None:
    path = Path(file_path)

    # --- Load depending on file type ---
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name= "Sheet1")
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"\n📊 Diagnostics for {path.name}")
    print("-" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nSample rows:")
    print(df.head(3))
    print("-" * 50)


if __name__ == "__main__":
    # 👉 change this to your dataset path
    dataset_diagnostics("C:/Users/azhao/PycharmProjects/PEAgent/data/raw/pitchbook_export.xlsx")

    # dataset_diagnostics("Pitchbook_MandatoryDisclosure_Analysis.parquet")
