# Vendored ML / Data Science Agent Skills

26 skills covering model building, dataset cleaning, feature engineering, tuning,
and general data-science workflow. Vendored from two upstream repos, both MIT.

## Coverage map

| Need | Skills |
|---|---|
| **Creating an ML model** | `sklearn-pipelines`, `scikit-learn`, `pytorch-training-loop`, `llm-finetuning`, `rag-pipeline` |
| **Cleaning datasets** | `data-cleaning`, `data-quality-frameworks`, `polars`, `pandas-patterns` |
| **Data scientist workflow** | `exploratory-data-analysis`, `statistical-analysis`, `statsmodels`, `matplotlib`, `shap`, `model-card`, `aeon` (time series) |
| **Tuning models** | `hyperparameter-tuning`, `model-evaluation`, `split-strategy`, `imbalanced-data`, `ml-debugging` |
| **Creating feature columns** | `feature-engineering`, `target-leakage-detection` |
| **Ops / reproducibility** | `experiment-tracking`, `reproducible-ml`, `model-serving` |

## Provenance

| Source | License | Skills taken |
|---|---|---|
| [param087/agent-ml-skills](https://github.com/param087/agent-ml-skills) | MIT | `data-cleaning`, `feature-engineering`, `hyperparameter-tuning`, `sklearn-pipelines`, `pytorch-training-loop`, `exploratory-data-analysis`, `model-evaluation`, `pandas-patterns`, `experiment-tracking`, `reproducible-ml`, `ml-debugging`, `imbalanced-data`, `model-serving`, `llm-finetuning`, `rag-pipeline` |
| [andikarachman/data-science-plugin](https://github.com/andikarachman/data-science-plugin) | MIT (declared in README; no LICENSE file upstream) | `target-leakage-detection`, `split-strategy`, `shap`, `statistical-analysis`, `statsmodels`, `polars`, `data-quality-frameworks`, `model-card`, `matplotlib`, `aeon`, `scikit-learn` |

License texts are preserved in [`vendor/licenses/`](../../vendor/licenses/).

Six skills carry their own upstream attribution to **K-Dense Inc.** in their frontmatter
(`aeon`, `scikit-learn`, `statsmodels` — BSD-3-Clause; `shap`, `statistical-analysis` — MIT;
`matplotlib` — matplotlib's own license). That attribution has been left intact.

## Local modifications

The `andikarachman` skills were written for that repo's `/ds:*` slash-command plugin. Since
only the skills were vendored (not the commands or agents), they were decoupled:

- Removed `/ds:eda`, `/ds:experiment`, `/ds:validate`, … references from `description` fields,
  rewriting each trigger in plain terms so model-driven invocation works.
- Relabelled the `**Role in the ds plugin:**` paragraphs to `**Scope and boundaries:**`,
  dropping sentences that described plugin workflow steps and keeping the skill-boundary guidance.
- Remapped cross-references to non-vendored skills onto the vendored equivalents:
  `pandas-pro`→`pandas-patterns`, `data-preprocessing`→`data-cleaning`,
  `tuning-hyperparameters`→`hyperparameter-tuning`, and the `data-profiler`/`feature-engineer`/
  `model-evaluator` *agents*→the `exploratory-data-analysis`/`feature-engineering`/`model-evaluation` skills.
- Dropped `disable-model-invocation: true` from `data-quality-frameworks`, which would
  otherwise have blocked the model from ever invoking it.
- Fixed two dangling links to `references/data_validation_schemas.md`, a file that shipped
  only with the un-vendored `data-preprocessing` skill.

`param087` provides the concise spine (~70–90 lines each). `andikarachman` supplies
deeper references (300–650 lines, some with bundled `references/` and `scripts/`).
Overlapping concepts were deduplicated to one skill each — duplicate descriptions
degrade skill selection, since the agent matches on the `description` field.

### Deliberately excluded

- **`pymc-labs/python-analytics-skills`**, **`Emily2040/data-science-agent-skills`** — no
  license declared (defaults to all-rights-reserved). Worth revisiting if they add one;
  Emily2040's `leakage-adversary` and `drift-monitor-designer` are strong.
- **`probabl-ai/skills`** (BSD-3, from the company behind scikit-learn) — high pedigree, but
  its `build-ml-pipeline` mandates [skrub](https://skrub-data.org) DataOps graphs over plain
  `sklearn.Pipeline`. Adopt only if you want that dependency; it would also collide with
  `sklearn-pipelines`.
- **`addyosmani/agent-skills`**, **`lyndonkl/claude`** — general SWE and
  writing/reasoning packs with little genuine ML content. See the note below.

### Dropped as duplicates

From `andikarachman`, these collided with `param087` equivalents and were not taken:
`data-preprocessing`, `eda-checklist`, `exploratory-data-analysis`, `experiment-tracking`,
`pandas-pro`, `reproducibility-checklist`, `tuning-hyperparameters`, `setup`.

## Updating

Upstreams were cloned at `--depth 1`. To refresh, re-clone and re-copy the skill
directories listed in the provenance table above, then re-check that no two
`description:` fields overlap.
