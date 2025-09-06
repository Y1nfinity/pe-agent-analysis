from pathlib import Path
from tools.io import load_and_clean

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]   # goes up one level from /scripts
    config_path = project_root / "config.toml"

    df, out_path = load_and_clean(config_path)
    print(f"Wrote cleaned parquet: {out_path}")
    print(df.head(3))
