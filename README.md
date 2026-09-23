# PEAgent

A Python project analyzing private equity exits, take-private (public-to-private) deals, and continuation funds using PitchBook, Bloomberg, and Preqin data.

## `handoff_package/` is the final version of the pipeline

This the polished, reproducible analysis. It lives in [`handoff_package/`](handoff_package/README.md). Everything else in this repo represents earlier work; `handoff_package/` is the final, verified version of the pipeline, it has been simplified down to: only what's needed to regenerate five specific charts from raw data, with pinned dependencies, numbered pipeline scripts, and a README documenting setup, data lineage, and what was verified against the original outputs. If you're reviewing this project, start there.

## `research_archive/`

Everything else — [`research_archive/`](research_archive/README.md) — is earlier exploratory and iterative work: alternative versions of the same analyses, data-cleaning experiments, and scripts superseded by later ones. It's kept for reference and isn't meant to be read line-by-line; see its README for a short guide to what's there and why some of it exists in multiple versions.

## Setup

```bash
git clone https://github.com/Y1nfinity/pe-agent-analysis.git
cd pe-agent-analysis/handoff_package
pip install -r requirements.txt
```

## Addendum

This code was written with the help of artificial intelligence. While not my first foray into coding, it's my first time compiling a project in this way. I used ChatGPT to help organize the code and to write some of the search/matching algorithms in `research_archive/`, and used Claude (Anthropic) to isolate, verify, and clean up the reproducible pipeline in `handoff_package/`.
