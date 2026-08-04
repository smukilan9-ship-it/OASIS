"""Text files must be opened with an explicit encoding.

`open()` and `Path.read_text()` fall back to `locale.getencoding()`, which is UTF-8 on
macOS and Linux and **cp1252 on Windows**. Two things break, and both had already
happened here:

1. The Windows CI job had been red for weeks behind `continue-on-error`, with 34 errors
   and one failure, every one of them `UnicodeDecodeError: 'charmap' codec can't decode
   byte 0x90` from reading `oasis/webui/index.html`. That file is UTF-8 and holds µ, →
   and ✓, so on Windows it simply cannot be read at all.

2. Worse than a red build, because it is silent: `write_detections_csv` emits the header
   `Centroid X µm`, and `load_detection_centroids_csv` matches on that exact string. Write
   the file on macOS as UTF-8, read it on Windows as cp1252, and the header arrives as
   `Centroid X Âµm`. No exception. The column just never matches, the all-cell support
   loads zero points, the dense null's support gate fails, and the pair is withheld with a
   reason that describes the tissue rather than the encoding.

So this is not tidiness. An implicit encoding is a per-platform behaviour change in code
whose whole purpose is to give the same answer everywhere.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "legacy", "build", "dist", "__pycache__", ".git", "node_modules",
             "validation_reports", "validation_datasets"}
# Openers that are not the builtin and do their own decoding (or none).
NOT_BUILTIN = {"Image", "np", "gzip", "tarfile", "zipfile", "io", "h5py", "nib", "tifffile"}


def _sources():
    for p in sorted(ROOT.rglob("*.py")):
        if SKIP_DIRS & set(p.relative_to(ROOT).parts):
            continue
        try:
            yield p, ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue        # legacy/ holds embedded Groovy; it is excluded above anyway


def _is_binary_mode(call):
    for i, a in enumerate(call.args):
        if i == 1 and isinstance(a, ast.Constant) and isinstance(a.value, str):
            return "b" in a.value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return "b" in str(kw.value.value)
    return False


def _offenders():
    bad = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if any(k.arg == "encoding" for k in node.keywords):
                continue
            # `open` is not the only opener. tempfile's handles take the same default, and
            # one of them was writing the page script for `node --check`: fine on a UTF-8
            # box, cp1252 bytes handed to a UTF-8 reader on Windows.
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            owner = (f.value.id if isinstance(f, ast.Attribute)
                     and isinstance(f.value, ast.Name) else None)
            openers = ("open", "NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile")
            if name in openers and owner not in NOT_BUILTIN:
                if not _is_binary_mode(node):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}()")
            elif isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text"):
                owner = f.value.id if isinstance(f.value, ast.Name) else None
                if owner not in NOT_BUILTIN:
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} .{f.attr}()")
    return bad


def test_no_text_file_is_opened_without_an_encoding():
    bad = _offenders()
    assert not bad, (
        "these read or write text with the platform default encoding, which is cp1252 on "
        "Windows:\n  " + "\n  ".join(bad))


def test_the_detections_header_survives_a_windows_round_trip():
    """The concrete silent failure: the µ in the centroid column.

    Not hypothetical. `spatial.load_detection_centroids_csv` looks up the literal
    "Centroid X µm", and that lookup returning nothing is indistinguishable, downstream,
    from a section that genuinely has no cells.
    """
    from oasis.quant.segment import _CSV_COLUMNS

    header = [c for c in _CSV_COLUMNS if "Centroid X" in c]
    assert header, "the centroid column was renamed; update load_detection_centroids_csv too"
    assert "µ" in header[0], header[0]
    # UTF-8 out, cp1252 in is exactly what a cross-platform hand-off does. It must not
    # silently produce a different string.
    assert header[0].encode("utf-8").decode("cp1252", errors="replace") != header[0], (
        "this assertion documents WHY the encoding must be explicit; if it ever fails the "
        "header became ASCII and the hazard is gone")


def test_index_html_needs_utf8_to_read_at_all():
    """The file every UI test reads. Proves the CI failure rather than assuming it."""
    index = ROOT / "oasis" / "webui" / "index.html"
    raw = index.read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")
    assert raw.decode("utf-8")
