"""Nothing may hand `.scroll` an inline `display`, because that silently un-grids it.

`.scroll` is `display: grid` in the stylesheet, and two behaviours ride on that:

  * `justify-items: center` is what centres the column on every tab;
  * `grid-auto-rows: max-content` is what stops a card with `overflow: hidden` from being
    squeezed to nothing (see test_ui_stylesheet.py for the run that rendered three and a
    half rows of a four-image table).

An inline style beats both. `calibShow()` hid the Classifier form with `display:none` while
the labelling canvas was up and restored it with `display:block` — which is a perfectly
reasonable-looking line that quietly turns the grid into a block box. The symptom is not an
error: the cards keep their width and simply sit against the left edge with half the tab
empty beside them, on that tab only, and only after someone has used the labeller once.

The fix is to clear the property (`= ''`) so the stylesheet wins again. This test pins the
general rule, because the next person to hide a panel will reach for `display = 'block'`
just as readily.
"""
import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "oasis/webui/index.html"


@pytest.fixture(scope="module")
def html():
    return INDEX.read_text(encoding="utf-8")


def _scroll_display_assignments(html):
    """Every `<something>.style.display = ...` whose target was resolved from `.scroll`."""
    hits = []
    for m in re.finditer(r"(\w+)\.style\.display\s*=\s*([^;\n]+)", html):
        var, value = m.group(1), m.group(2).strip()
        # Look back for the declaration of this variable; only flag ones bound to .scroll.
        decl = re.search(r"(?:const|let|var)\s+" + re.escape(var) + r"\s*=\s*[^;\n]*\.scroll[^;\n]*",
                         html[:m.start()])
        if decl:
            hits.append((var, value, html[:m.start()].count("\n") + 1))
    return hits


def test_scroll_is_never_given_a_non_empty_inline_display(html):
    offenders = []
    for var, value, line in _scroll_display_assignments(html):
        # 'none' is fine — hiding is the point, and it restores cleanly. Anything that
        # assigns a *layout* mode is not: it outranks `display: grid` from the sheet.
        for bad in ("'block'", '"block"', "'flex'", '"flex"', "'inline-block'"):
            if bad in value:
                offenders.append(f"line {line}: {var}.style.display = {value}")
    assert not offenders, (
        "an inline display was assigned to a .scroll element, which overrides "
        "`display: grid` and silently drops both the centred column and the "
        "row-squeeze guard:\n  " + "\n  ".join(offenders))


def test_the_classifier_form_is_restored_by_clearing_the_property(html):
    """The specific line this was written for, so a revert is caught rather than inferred."""
    m = re.search(r"#page-classifier > \.scroll.*?style\.display\s*=\s*([^;\n]+)", html, re.S)
    assert m, "the Classifier form show/hide no longer resolves .scroll the expected way"
    expr = m.group(1)
    assert "'block'" not in expr and '"block"' not in expr, (
        "the Classifier form is restored with display:block again — the column will render "
        "flush left after anyone opens the labelling canvas")
    assert "''" in expr or '""' in expr, (
        "expected the inline display to be cleared (= '') so the stylesheet's grid wins")
