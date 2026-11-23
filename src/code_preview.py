"""One-page code tour for the defense panel: the real code of each pipeline step, in order.

The code shown is read from src/ every time, so the page always matches the code that produced
the results. Open outputs/final/code_preview.html in a browser; no internet is needed.
run_extract.py writes it on every run.

Usage: python src/code_preview.py
"""
from __future__ import annotations

import ast
import html
import sys
from pathlib import Path

import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from config import FINAL_DIR

# (step title, file, function or class, what it does, what to say to the panel)
STEPS = [
    ("1. Read one recording", "data.py", "load_recording",
     "Loads the gaze, pupil and blink tables of one participant and activity.",
     "Each recording is read on its own; nothing from other people is mixed in."),
    ("2. Clean the signals", "features.py", "recording_signals",
     "Trims stray samples, drops low-confidence data, cleans blinks, pupil and gaze, and returns "
     "every cleaned signal as a list of values plus the blink, fixation and saccade rates.",
     "Cleaning follows published guidelines for Pupil Core data (Chapter 3, Table 4)."),
    ("3. Detect fixations and saccades", "events.py", "detect_events",
     "Labels each gaze sample as saccade (faster than 45 deg/s) or fixation, and drops fixations "
     "shorter than 55 ms (I-VT, the COLET authors' settings).",
     "Same velocity threshold and minimum fixation as the COLET paper."),
    ("4. Summarize each signal (flattening)", "flatten.py", "summarize",
     "Turns one signal into 17 numbers: min, max, range, mean, median, SD, variance, CV, IQR, "
     "6 percentiles, skewness and kurtosis.",
     "We did not pick statistics by hand: every signal gets all 17."),
    ("5. Build the feature row of a recording", "flatten.py", "flat_features",
     "Applies step 4 to every signal and adds the three rates: 173 features, 140 after removing "
     "copies and unusable statistics.",
     "One row per recording: 88 rows (44 people x A1 and A4)."),
    ("6. Pick the best N features (inside each fold)", "topn.py", "TopNByAUC",
     "Scores every feature alone by how well it separates low from high load (|AUC - 0.5|) on the "
     "training people only, and keeps the best N.",
     "Selection is refit in every fold, so the held-out person never influences which features "
     "are chosen."),
    ("7a. Logistic Regression pipeline", "topn.py", "make_logistic_regression_topn",
     "Fill blanks -> pick top N -> scale -> elastic-net Logistic Regression. Every step is fitted "
     "on the training people only.",
     "Scaling and imputation are part of the model, so they never see the test person."),
    ("7b. XGBoost pipeline", "topn.py", "make_xgboost_topn",
     "Pick top N -> XGBoost with shallow trees.",
     "Same features and the same selection step as Logistic Regression."),
    ("8. Leave-one-participant-out with inner tuning", "evaluate.py", "run_lopo",
     "44 rounds: hide one person, tune N and the model settings with 5 inner folds on the other 43 "
     "(30 random candidates), then predict the hidden person.",
     "Every prediction is for a person the model never saw: no leakage across people."),
    ("9. Every metric", "evaluate.py", "score",
     "ROC-AUC, accuracy, balanced accuracy, precision, recall and F1 (per class and macro), and "
     "the Brier score.",
     "We report every metric, not only ROC-AUC."),
    ("10. Chance test (leakage check)", "evaluate.py", "permutation_test",
     "Shuffles each person's two labels 200 times and reruns the whole pipeline, feature selection "
     "included. If something leaked, shuffled runs would also score high.",
     "None of 200 shuffled runs reached the real score (p = 0.005)."),
    ("11. Feature importance (SHAP)", "run_model.py", "fold_shap",
     "Explains each fold's model on the person it never saw; a feature not picked in that fold "
     "counts as 0.",
     "Both models are explained with the same method, on held-out people."),
]


def _source(file: str, name: str) -> tuple[str, int]:
    path = SRC / file
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            lines = text.splitlines()[start - 1:node.end_lineno]
            return "\n".join(lines), start
    raise KeyError(f"{name} not found in {file}")


def _results_box(results_dir: Path) -> str:
    path = results_dir / "results.csv"
    if not path.exists():
        return ""
    r = pd.read_csv(path).set_index("model")
    needed = {"roc_auc", "auc_ci_low", "auc_ci_high", "accuracy", "precision_macro", "recall_low",
              "recall_high", "macro_f1", "brier", "permutation_p"}
    if not needed <= set(r.columns):
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(m)}</td><td>{x.roc_auc:.3f} ({x.auc_ci_low:.3f}–{x.auc_ci_high:.3f})</td>"
        f"<td>{x.accuracy:.3f}</td><td>{x.precision_macro:.3f}</td>"
        f"<td>{x.recall_low:.3f} / {x.recall_high:.3f}</td><td>{x.macro_f1:.3f}</td>"
        f"<td>{x.brier:.3f}</td><td>{x.permutation_p:.3f}</td></tr>"
        for m, x in r.iterrows())
    return ("<h2>Result these steps produce</h2><table class='res'><tr><th>Model</th>"
            "<th>ROC-AUC (95% CI)</th><th>Accuracy</th><th>Precision</th><th>Recall low / high</th>"
            f"<th>F1</th><th>Brier</th><th>Chance test p</th></tr>{rows}</table>"
            "<p class='note'>From outputs/final/results/results.csv (leave-one-participant-out, "
            "44 people, 88 recordings).</p>")


def code_preview(results_dir: Path | None = None) -> str:
    fmt = HtmlFormatter(style="default", linenos="table", cssclass="code")
    nav, parts = [], []
    for i, (title, file, name, what, say) in enumerate(STEPS):
        code, start = _source(file, name)
        fmt.linenostart = start
        nav.append(f"<li><a href='#s{i}'>{html.escape(title)}</a></li>")
        parts.append(
            f"<section id='s{i}'><h2>{html.escape(title)}</h2>"
            f"<p class='loc'><code>src/{file}</code>, <code>{name}</code>, line {start}</p>"
            f"<p><b>What it does:</b> {html.escape(what)}</p>"
            f"<p class='say'><b>Say to the panel:</b> {html.escape(say)}</p>"
            f"{highlight(code, PythonLexer(), fmt)}</section>")
    css = fmt.get_style_defs(".code")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Code tour</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:16px;color:#111;background:#fff;line-height:1.5}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:32px;border-top:1px solid #ddd;padding-top:16px}}
.loc{{color:#555;margin:0}} .say{{background:#eef4fc;padding:8px 12px;border-left:4px solid #2a78d6}}
.code{{overflow-x:auto;font-size:13px;border:1px solid #ddd}} .code pre{{margin:0}}
.code td.linenos{{color:#999;padding-right:8px;user-select:none}}
table.res{{border-collapse:collapse;font-size:14px}} table.res td,table.res th{{border:1px solid #ccc;padding:4px 8px}}
.note{{color:#555;font-size:13px}} ol li{{margin:2px 0}}
{css}
</style></head><body>
<h1>Code tour: from raw eye data to results</h1>
<p>The real code of each step, read from <code>src/</code> when this page was made, in the order the
pipeline runs. Run everything with <code>python src/run_extract.py</code> (steps 1–5) and
<code>python src/run_model.py</code> (steps 6–11).</p>
<ol>{''.join(nav)}</ol>
{_results_box(results_dir) if results_dir else ''}
{''.join(parts)}
</body></html>
"""


def main(final_dir=FINAL_DIR) -> Path:
    path = Path(final_dir) / "code_preview.html"
    path.write_text(code_preview(Path(final_dir) / "results"), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
