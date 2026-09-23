# research_archive

Earlier and exploratory work from this project, kept for reference. **The polished, reproducible deliverable is [`handoff_package/`](../handoff_package/README.md)** — start there. Nothing here is guaranteed to run standalone (paths, inputs, and package versions weren't maintained after each script was superseded), and several folders overlap in purpose because they represent different passes at the same analysis rather than a single pipeline.

## Layout

- **`scripts/`** — the main working folder, including `scripts/scripts_update/`, the most recent iteration of the analysis before it was distilled into `handoff_package/`, and `scripts/outdated/`, explicitly-retired earlier versions.
- **`scripts_restructured/`** — an attempt to reorganize the analysis into a cleaner package structure (`universe/`, `panels/`, `figures/`, `yearly_analysis/`).
- **`tools/`** and **`tools_restructured/`** — shared helper modules (matching, IRR calculations, plotting utilities) used by the scripts above, again in two organizational passes.

## Why some analyses exist in 2–3 versions

A few files are genuinely worth reading together rather than dismissing as clutter — they show the debugging process, not just noise:

- **`scripts/scripts_update/entry_exit_linker.py` → `p2p_entry_exit_linker.py` → `p2p_entry_exit_linker_with_chrono.py`**: three attempts at linking take-private entries to their exits. The first over-paired one entry to multiple exits; the second fixed that but initially looked like it dropped still-active (unrealized) deals; the third made that handling explicit. `p2p_entry_exit_linker.py` is the version that actually produced the data used downstream — confirmed against the tracked `data/clean/p2p_linked_master.csv` output during the `handoff_package/` build.
- **`analysis_exitcohort.py` vs. `analysis_exitcohort_weightedbydollar.py`**: the same exit-composition analysis, first by deal count, then by dollar value — a deliberate second cut at the question, not a duplicate.
- **`analysis_holdingperiod.py` vs. `analysis_holdingperiods_avgonly.py`**: the latter adds standard-error and confidence-interval bands around the same average holding-period trend.
