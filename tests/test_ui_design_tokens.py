"""The design-token layer has to be internally consistent, or the app silently loses colour.

These exist because a find/replace that routed hard-coded colours through tokens also
rewrote the two lines that DEFINE them, producing `--surface: var(--surface)`. A
self-referential custom property is invalid at computed-value time, so every consumer
resolves to `unset` — table headers lost their background, two gradients lost their entire
`background` shorthand, and the primary and Run buttons rendered white-on-white on hover.
Nothing failed; it just looked wrong, in a way no test could see.
"""
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "oasis" / "webui" / "index.html"


def _stylesheet():
    html = INDEX.read_text(encoding="utf-8")
    a = html.index("<style>")
    return html[a:html.index("</style>", a)], html


def test_no_custom_property_references_itself():
    css, _ = _stylesheet()
    cycles = [m.group(0) for m in re.finditer(r"(--[a-z0-9-]+):\s*([^;]+);", css)
              if f"var({m.group(1)})" in m.group(2)]
    assert not cycles, f"custom properties that reference themselves: {cycles}"


def test_every_var_used_is_defined():
    css, html = _stylesheet()
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", html))
    assert not (used - defined), f"var() names with no definition: {sorted(used - defined)}"


def test_the_status_triads_are_complete():
    """Each status idea needs text, surface and border, or call sites invent their own."""
    css, _ = _stylesheet()
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    for base in ("ok", "warn", "bad", "info"):
        for suffix in ("", "-weak", "-line"):
            assert f"--{base}{suffix}" in defined, f"--{base}{suffix} is missing"


def test_cell_call_colours_are_not_the_status_tokens():
    """Red means "called positive" on the overlay — the inverse of --bad. Keeping these
    separate is what stops a future sweep from inverting the legend's meaning."""
    css, _ = _stylesheet()
    assert "--cell-pos" in css and "--cell-neg" in css
    legend = [ln for ln in css.splitlines() if ".qr-key i.pos" in ln]
    assert legend, "the overlay legend rule is missing"
    assert "var(--cell-pos)" in legend[0] and "var(--cell-neg)" in legend[0], (
        "the legend must key the overlay's own colours, not the status tokens")
