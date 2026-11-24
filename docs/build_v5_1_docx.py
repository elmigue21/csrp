"""Build CSRP-...-Pina-v5-1.docx from the v5-1 markdown, using v4-6.docx as the template.

Keeps the v4-6 title page, styles and page setup; rebuilds the body. Old reference
entries are copied as-is (italics kept) and renumbered by order of first citation.
"""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Emu, Pt

ROOT = Path(r"C:\dev\csrp")
SRC = ROOT / "docs" / "old" / "CSRP-Chapter-1-to-5-draft-COLET-v4-6.docx"
MD = ROOT / "docs" / "CSRP-Chapter-1-to-5-draft-Pina-v5-1.md"
OUT = ROOT / "docs" / "CSRP-Chapter-1-to-5-draft-Pina-v5-1.docx"

NEW_REFS = {
    "guyon": "I. Guyon and A. Elisseeff, \u201cAn introduction to variable and feature selection,\u201d "
             "*J. Mach. Learn. Res.*, vol. 3, pp. 1157\u20131182, 2003.",
    "pina": "J. Pina et al., \u201cA comprehensive analysis of the influence of cognitive load on physiological "
            "signals in virtual reality,\u201d in *Proc. IEEE Int. Symp. Mixed Augmented Reality (ISMAR)*, 2025, "
            "pp. 433\u2013443, doi: 10.1109/ISMAR67309.2025.00054. Dataset: Zenodo, record 18669059.",
    "bielsa": "J. Bielsa, J. Pina, A. Serrano, and D. Martin, \u201cAn analysis of gaze behavior under multisensory "
              "cognitive load in immersive environments,\u201d *Comput. Graph.*, vol. 137, Art. no. 104613, 2026, "
              "doi: 10.1016/j.cag.2026.104613.",
}
OLD_FIG = {"figure1": 44, "figure3": 112, "figure5": 169}      # paragraph index in v4-6
WIDTH_IN = {"fig2_method.png": 4.3, "fig4_preprocessing.png": 4.4, "fig6_roc.png": 3.6,
            "fig7_by_n.png": 5.6, "fig8_shap.png": 6.0}
CITE = re.compile(r"\{(\d+|[a-z]+)\}")

doc = docx.Document(SRC)
body = doc.element.body
paras = doc.paragraphs
old_fig = {k: copy.deepcopy(paras[i]._p) for k, i in OLD_FIG.items()}
old_refs = {}
for p in paras:
    m = re.match(r"\[(\d+)\]\t", p.text)
    if m:
        old_refs[m.group(1)] = copy.deepcopy(p._p)
title_end = paras[9]._p                                   # page break after the title page
keep = True
for el in list(body):
    if el.tag == qn("w:sectPr"):
        continue
    if not keep:
        body.remove(el)
    if el is title_end:
        keep = False

# ---------------------------------------------------------------- citation numbers
md = MD.read_text(encoding="utf-8")
order = []
for key in CITE.findall(md):
    if key not in order:
        order.append(key)
num = {k: i + 1 for i, k in enumerate(order)}
missing = [k for k in order if k not in old_refs and k not in NEW_REFS]
assert not missing, missing


def cites(text):
    return CITE.sub(lambda m: f"[{num[m.group(1)]}]", text)


def runs(p, text, size=None, bold_all=False):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", cites(text)):
        if not part:
            continue
        b = part.startswith("**")
        i = not b and part.startswith("*") and part.endswith("*") and len(part) > 1
        r = p.add_run(part.strip("*") if (b or i) else part)
        r.bold = True if (b or bold_all) else None
        r.italic = True if i else None
        if size:
            r.font.size = Pt(size)


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    runs(p, text)
    return p


def page_break():
    doc.add_paragraph().add_run().add_break(docx.enum.text.WD_BREAK.PAGE)


def heading1(text):
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parts = text.split("|")
    for j, t in enumerate(parts):
        if j:
            p.add_run().add_break()
        p.add_run(t)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    ncol = len(cells[0])
    t = doc.add_table(rows=len(cells), cols=ncol)
    t.style = doc.styles["Table Grid"]
    t.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
    weight = [max(min(max(len(cites(r[j])) for r in cells), 40),
                  max(len(wd) for r in cells for wd in cites(r[j]).split() or [""]) + 2)
              for j in range(ncol)]
    widths = [Emu(int(9360 * wt / sum(weight) * 635)) for wt in weight]
    t.autofit = False
    for j, col in enumerate(t.columns):
        col.width = widths[j]
    for i, r in enumerate(cells):
        for j, c in enumerate(r):
            cell = t.cell(i, j)
            cell.width = widths[j]
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.keep_with_next = i < len(cells) - 1
            runs(p, c, size=9, bold_all=(i == 0))
    doc.add_paragraph()                                    # space after the table, as in v4-6


fig_no = 0


def figure(caption, path):
    global fig_no
    fig_no += 1
    if path.startswith("@docx:"):
        el = copy.deepcopy(old_fig[path[6:]])
        body.insert(len(body) - 1, el)
        docx.text.paragraph.Paragraph(el, doc._body).paragraph_format.keep_with_next = True
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name = Path(path).name
        p.add_run().add_picture(str(ROOT / path), width=Emu(int(WIDTH_IN[name] * 914400)))
        p.paragraph_format.keep_with_next = True
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run(f"Figure {fig_no}. ").bold = True
    cap.add_run(caption)


# ---------------------------------------------------------------- body
lines = md.splitlines()
i = 0
first_chapter = True
while i < len(lines):
    line = lines[i]
    if not line.strip():
        i += 1
        continue
    if line.startswith("# "):
        if not first_chapter:
            page_break()
        first_chapter = False
        heading1(line[2:])
    elif line.startswith("## "):
        doc.add_paragraph(line[3:], style="Heading 2").alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif line.startswith("### "):
        doc.add_paragraph(line[4:], style="Heading 3").alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif line.startswith("|"):
        block = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i])
            i += 1
        table(block)
        continue
    elif line.startswith("!["):
        m = re.match(r"!\[(.*)\]\((.*)\)", line)
        figure(m.group(1), m.group(2))
    elif re.match(r"\*\*Table \d+\.\*\*", line):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.keep_with_next = True
        runs(p, line)
    else:
        para(line)
    i += 1

# ---------------------------------------------------------------- references
page_break()
heading1("REFERENCES")
for k in order:
    n = num[k]
    if k in NEW_REFS:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Emu(457200)
        p.paragraph_format.first_line_indent = Emu(-457200)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.add_run(f"[{n}]\t")
        runs(p, NEW_REFS[k])
        continue
    el = copy.deepcopy(old_refs[k])
    t = el.find(".//" + qn("w:t"))
    t.text = f"[{n}]"
    body.insert(len(body) - 1, el)

# drop images no longer referenced
used = set(re.findall(r'r:embed="([^"]+)"', body.xml))
for rid, rel in list(doc.part.rels.items()):
    if "image" in rel.reltype and rid not in used:
        doc.part.drop_rel(rid)

doc.save(OUT)
print("refs:", len(order), "figures:", fig_no)
print("old->new:", {k: num[k] for k in order})
dropped = sorted(set(old_refs) - set(order), key=int)
print("old refs not cited:", dropped)
