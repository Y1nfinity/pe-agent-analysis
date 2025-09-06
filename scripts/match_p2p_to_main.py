# scripts/match_p2p_to_main.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import tomllib  # Python 3.11+; if older, install 'tomli' and use that instead.

from tools.matching import load_excel, build_pairs_from_seeds

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "config.toml"

    # Read config
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    pm = cfg["p2p_matching"]
    seed_file = project_root / pm["seed_file"]
    univ_file = project_root / pm["universe_file"]
    seed_sheet = pm.get("seed_sheet") or None
    univ_sheet = pm.get("universe_sheet") or None
    outdir = project_root / pm["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading P2P seed:      {seed_file} :: {seed_sheet or '[first sheet]'}")
    p2p_seed = load_excel(seed_file, seed_sheet)

    print(f"📥 Loading universe file: {univ_file} :: {univ_sheet or '[first sheet]'}")
    universe = load_excel(univ_file, univ_sheet)

    print("🔗 Matching P2P entries to universe exits…")
    pairs, unmatched, candidates = build_pairs_from_seeds(
        p2p_seed, universe, topk=3, min_hold_days=0, max_hold_years=None
    )

    # Save outputs
    pairs.to_csv(outdir / "p2p_seed_pairs.csv", index=False)
    unmatched.to_csv(outdir / "p2p_unmatched_seed.csv", index=False)
    candidates.to_csv(outdir / "p2p_candidates_topk.csv", index=False)

    print(f"\n✅ Done.")
    print(f"Pairs created: {len(pairs):,}")
    print(f"Unmatched seeds: {len(unmatched):,}")
    print(f"Top-3 candidates written: {len(candidates):,}")
    print(f"📂 Outputs in: {outdir}")

if __name__ == "__main__":
    main()
