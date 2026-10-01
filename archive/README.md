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

The code in `src/` was left in place. It is GAZELOAD-specific and will be adapted for COLET.
Running it as it is would recreate `outputs/` at the repository root.
