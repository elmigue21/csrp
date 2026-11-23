# Changes: v4-6 → v4-7 (post-defense rewrite, 2026-11-11)

**Paper:** `CSRP-Chapter-1-to-5-draft-COLET-v4-7.docx`. Source text: the `.md` of the same name.
**Length:** 16,473 → 9,765 words; 19 → 13 tables; 8 figures. Chapters 1–5 and the references; no appendices.

## What changed
| Part | Change |
|---|---|
| Method | 10 fixed features → 17 statistics per signal + 3 rates (173, 140 kept); top-N feature selection inside each fold; no per-person standardization; P2, permutation importance, Kendall's τ, LR coefficients, gain importance and all post-hoc analyses removed |
| Objectives | 5 rewritten: flatten, LOPO with in-fold selection, every metric, performance by N, SHAP + selection count |
| Ch. 2 | Standardization paragraph removed; feature-selection paragraph added |
| Ch. 3 | Feature extraction, model development, metrics and importance rewritten; Table 7 (analysis plan) and the standardization section removed |
| Ch. 4 | All model numbers replaced from `outputs/final/results/`; new tables: every metric (11), by N (12), blink alone / without blink (13) |
| Ch. 5 | Rewritten; limitations 15 → 8 bullets; recommendations 5 → 4 |
| Figures | 2, 4, 6, 7, 8 redrawn by `src/paper_figures.py` (`outputs/final/paper_figures/`); 1, 3, 5 unchanged |
| Appendices A, B | Removed (gain importance; 10 s windows) |

## References (75, was 86)
- **Removed (12):** Breiman, Fisher (permutation importance); Oppelt, Gado, Appel 2023, Albuquerque
  (standardization); Gelman (coefficients); Strobl (gain appendix); Appel 2018, Stolte (P2);
  Pluchino; Chakraborty. Full audit: `rrl-recheck-post-defense.md`.
- **Added (1):** Guyon & Elisseeff 2003 (filter feature selection). Details checked by web search
  only: **read it before final submission.**
- Renumbered in order of first citation.
- **Still to check:** that Fawcett [71] defines precision, recall and F1 (from memory), and the
  metadata-only entries flagged in the audit (Hart 2006, Fawcett).

## Not changed
Figure 1, the cleaning steps (C1–C10), data quality, sanity check and manipulation check numbers
(same data and cleaning).

## Rebuild
`python src/paper_figures.py`, then `python docs/build_v4_7_docx.py`, which
keeps the v4-6 title page and styles and copies the old reference entries.
