import ast
import importlib
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colet_pipeline.ipynb"


def test_notebook_imports_resolve():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))]
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                    "run_colet", "checks", "data", "dataset", "config"}:
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"{node.module}.{alias.name}"
