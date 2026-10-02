# Archive

Files kept for reference that the current (COLET-based) thesis no longer uses. They were
moved here on 2026-10-01, when the team switched datasets from GAZELOAD to COLET; see
`docs/rrl-decision-log.md`.

- `gazeload/docs/`: the GAZELOAD-era analysis.
  - EDA, data-quality and feature-engineering notes, with their figures and scripts.
  - The old `methodology.md` and `results.md`.
  - `methodology-replan.md`, superseded by `docs/methodology-colet.md`.
  - The within-participant concept note and the thesis notes.
- `gazeload/outputs/`: all tables, figures, predictions and SHAP outputs from the GAZELOAD
  runs (Aug 4 and Sep 6 2026).
- `old-drafts/`: the thesis drafts (Chapters 1–5, markdown and humanized versions) and the
  original Chapter 1–3 PDF and its copy. These carry the rejected COLET + GAZELOAD
  cross-dataset scope.

- `gazeload/src/` and `gazeload/tests/`: a complete snapshot of the GAZELOAD pipeline code
  (`run_experiment.py`, `figures.py` and the modules they import), archived on 2026-10-02 so that
  `src/` can be rewritten for COLET. It runs as a set: from the repo root, `python
  archive/gazeload/src/run_experiment.py` (needs the GAZELOAD data path in its `config.py`).
