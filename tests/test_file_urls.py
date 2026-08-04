"""A filesystem path is not a URL, and concatenating "file://" onto one is wrong.

This shipped in v0.1.0 and broke the Windows bundle outright. `app.py` opened its window
with `url=f"file://{html_path}"`. On Windows that path is `C:\\ProgramData\\...\\index.html`, so
the result is `file://C:\\ProgramData\\...`, where everything after the two slashes parses as the
URL's HOST and the path is empty. The window had nothing to load and the app died at
startup with PyInstaller's "Failed to execute script" dialog.

It passed CI on all three platforms because the frozen smoke test ran
`--oasis-worker run_pipeline`, which segments an image headlessly and shares almost no code
with startup. Nothing in the pipeline ever opened a window.

POSIX was never correct either, only lucky: a POSIX path already starts with "/", so two
slashes plus that one happen to make three. Spaces were broken everywhere, on every
platform, since neither form escapes them.
"""
import re
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ["app.py", "tools/capture_help.py", "oasis"]


def test_no_source_concatenates_a_path_onto_the_file_scheme():
    """Checked on the AST, not on the text.

    A regex over lines also matches the comments and docstrings that quote the broken
    pattern in order to explain it, which would make the fix's own explanation fail the
    test. The AST sees an f-string or a concatenation and never sees a comment.
    """
    import ast

    bad = []
    for rel in SEARCH:
        target = ROOT / rel
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                hit = False
                # f"file://{path}"
                if isinstance(node, ast.JoinedStr):
                    first = node.values[0] if node.values else None
                    hit = (isinstance(first, ast.Constant)
                           and str(first.value).startswith("file://")
                           and len(node.values) > 1)
                # "file://" + path
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    hit = (isinstance(node.left, ast.Constant)
                           and str(node.left.value).startswith("file://"))
                if hit:
                    bad.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    assert not bad, ("a path is concatenated onto the file scheme; use Path(...).as_uri():"
                     "\n  " + "\n  ".join(bad))


def test_as_uri_is_correct_where_concatenation_is_not():
    """The property that makes as_uri() the right answer, stated on a Windows path."""
    win = PureWindowsPath(r"C:\ProgramData\OASIS\_MEI1\oasis\webui\index.html")
    assert win.as_uri() == (
        "file:///C:/ProgramData/OASIS/_MEI1/oasis/webui/index.html")
    # what the old code produced: the drive and path swallowed into the host
    from urllib.parse import urlparse
    broken = urlparse(f"file://{win}")
    assert broken.path == "" and broken.netloc.startswith("C:")


def test_a_space_in_the_path_is_escaped():
    """OASIS is checked out here under a directory with a space in it, and a bundle lands
    in one on macOS too. Neither concatenated form escapes it."""
    assert " " not in Path("/tmp/My Slides/index.html").as_uri()
    assert "%20" in Path("/tmp/My Slides/index.html").as_uri()


def test_the_app_can_check_its_ui_without_opening_a_window():
    """The gap that let this ship. `--oasis-check-ui` is what CI now runs on every bundle."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "--oasis-check-ui" in src
    assert "def check_ui" in src
    for hook in ("packaging/build.sh", ".github/workflows/release.yml"):
        assert "--oasis-check-ui" in (ROOT / hook).read_text(encoding="utf-8"), (
            f"{hook} does not run the UI check, so a bundle that cannot start can ship")


def test_the_ui_helper_in_the_page_handles_a_windows_path():
    """The same bug had a second home: overlay images in the results view."""
    js = (ROOT / "oasis" / "webui" / "index.html").read_text(encoding="utf-8")
    assert "function fileUrl(" in js
    assert not re.search(r'src="file://\$\{', js), (
        "an image source still concatenates a raw path onto file://")
