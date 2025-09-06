# scripts/peek_pairs_sizes.py
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
pairs_path = root / "artifacts" / "tables" / "p2p_seed_pairs.csv"
pairs = pd.read_csv(pairs_path)

print("Columns:", list(pairs.columns))
for c in ["Entry_DealSize","Exit_DealSize"]:
    print(f"\n=== {c} ===")
    if c in pairs:
        print("dtype:", pairs[c].dtype)
        print("non-null:", pairs[c].notna().sum(), " / ", len(pairs))
        print("non-zero:", (pd.to_numeric(pairs[c], errors="coerce").fillna(0)!=0).sum())
        print(pairs[[c]].head(5))
    else:
        print("MISSING in CSV")
