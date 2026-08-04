"""Every name this codebase imports from itself has to exist.

This guards a failure mode the rest of the suite cannot see. Most first-party imports here
are written inside function bodies — deliberately, to keep torch, kornia and SimpleITK off
the startup path — and a function-local import raises only when that function is called. So
deleting or renaming a public function leaves the whole test suite green and breaks the app
at run time, at whatever depth the caller happens to sit.

That is not hypothetical. Rewriting the scale-bar detector kept only the private
`_detect_scale_bar` and dropped the public `extract_pixel_size_from_scale_bar` wrapper. Both
`run_pipeline.resolve_pixel_size` and the scale matcher in `webui.api` import that name
inside a function. Every spatial and quant run then died on its first step with

    Pipeline failed: import failed: cannot import name
    'extract_pixel_size_from_scale_bar' from 'oasis.common.pixel_size_util'

while every test passed, because the scale-bar tests all called the private function
directly and no test drives a pipeline. It was found by running the real app.

Cheap to run: the modules that pull heavy dependencies do so lazily inside their own
functions, so importing all of them takes well under a second.
"""
import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIRST_PARTY = ("oasis", "run_pipeline")
# legacy/ is retired code kept for the record and is not imported by the app.
SKIP_DIRS = {".venv", "legacy", "build", "dist", "__pycache__", ".git"}


def _sources():
    return [p for p in ROOT.rglob("*.py") if not SKIP_DIRS & set(p.parts)]


def _first_party_imports():
    """{module: {name: [(file, line, enclosing_function)]}} for `from <first-party> import`."""
    found = {}
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        enclosing = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    enclosing.setdefault(id(sub), node.name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
                continue
            if node.module.split(".")[0] not in FIRST_PARTY:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.setdefault(node.module, {}).setdefault(alias.name, []).append(
                    (str(path.relative_to(ROOT)), node.lineno, enclosing.get(id(node), "")))
    return found


def _resolves(module, name):
    """True if `from module import name` would work — as an attribute or as a submodule."""
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return True
    try:                                    # `from oasis.quant import segment` is a submodule,
        return importlib.util.find_spec(f"{module}.{name}") is not None   # not an attribute
    except (ImportError, AttributeError, ValueError):
        return False


@pytest.fixture(scope="module")
def imports():
    found = _first_party_imports()
    assert found, "no first-party imports found — the scan has drifted from the tree"
    return found


def test_every_first_party_module_imports(imports):
    broken = []
    for module in sorted(imports):
        try:
            importlib.import_module(module)
        except Exception as exc:            # noqa: BLE001 — reporting, not handling
            broken.append(f"{module}: {type(exc).__name__}: {exc}")
    assert not broken, "first-party modules that will not import:\n  " + "\n  ".join(broken)


def test_every_imported_name_exists(imports):
    missing = []
    for module in sorted(imports):
        try:
            importlib.import_module(module)
        except Exception:                   # noqa: BLE001 — the test above reports this
            continue
        for name, sites in sorted(imports[module].items()):
            if _resolves(module, name):
                continue
            for path, line, func in sites:
                where = f"inside {func}(), so it raises only when called" if func \
                        else "at module level"
                missing.append(f"{path}:{line}  from {module} import {name}  ({where})")
    assert not missing, ("names imported from first-party modules that do not exist:\n  "
                         + "\n  ".join(missing))
