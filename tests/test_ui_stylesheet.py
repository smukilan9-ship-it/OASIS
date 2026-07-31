"""The stylesheet has to survive parsing, because CSS fails silently and takes rules with it.

There is no build step and no linter between this file and the shipped app, and a broken
CSS rule produces no error anywhere — not in the console, not in a test, not in the page.
It just doesn't apply, and the layout is subtly wrong in a way that reads as a design
choice rather than a defect.

The failure this was written for, twice over:

    /* Centre with GRID ... explanation ...
       `/* The step rail stays put ... */
    .wizard-stepper-card { position: sticky; top: 0; }
    .wizard-nav { margin: 6px 0 4px }` did exactly that ... */
    .scroll { display: grid; justify-items: center; }     <-- DISCARDED

A rule was written inside the comment that documents a neighbouring rule. CSS comments do
not nest: the first close-delimiter ends the comment, the prose after it parses as a
selector, and error recovery throws away tokens up to and through the next block — which
was the `.scroll` centring rule. The sticky stepper worked, so the change looked correct;
`.scroll > *` still set the width, so the column was the right SIZE. It was simply no
longer centred, and every tab in the app rendered a 760 px column against the left edge
with 750 px of empty space beside it.

That is the exact complaint this rule was added to fix, silently reintroduced by an edit
to its own comment.
"""
import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "oasis/webui/index.html"

OPEN, CLOSE = "/" + "*", "*" + "/"   # spelled out so this file is not its own tripwire


@pytest.fixture(scope="module")
def css():
    html = INDEX.read_text()
    blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.S)
    assert blocks, "no <style> block found — the regex has drifted from the markup"
    return "\n".join(blocks)


def _strip_comments(text):
    """Strip comments the way a CSS parser does: the FIRST close-delimiter ends one."""
    return re.sub(re.escape(OPEN) + r".*?" + re.escape(CLOSE), "", text, flags=re.S)


def test_no_comment_delimiter_survives_stripping(css):
    """A leftover delimiter means a comment was opened or closed where it wasn't meant to be.

    Catches both directions: a rule buried inside a comment (leaves a trailing close), and
    a comment left unterminated (leaves a dangling open that eats the rest of the sheet).
    """
    rest = _strip_comments(css)
    leftovers = []
    for delim, label in ((CLOSE, "unopened comment close"), (OPEN, "unterminated comment")):
        for m in re.finditer(re.escape(delim), rest):
            line = rest[:m.start()].count("\n") + 1
            context = " ".join(rest[max(0, m.start() - 90):m.start() + 20].split())[-100:]
            leftovers.append(f"{label} near stripped-line {line}: ...{context}")
    assert not leftovers, "comment delimiters left in the stylesheet:\n  " + "\n  ".join(leftovers)


def test_braces_balance(css):
    body = _strip_comments(css)
    depth = 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, "a closing brace with no opener — a rule was cut in half"
    assert depth == 0, f"{depth} unclosed rule block(s) — everything after is discarded"


def test_no_selector_reads_like_prose(css):
    """A discarded rule is invisible; a selector full of English is how it gets discarded.

    After the comment above broke, the sheet contained a "selector" that was two sentences
    of explanation. Real selectors have no sentence punctuation and no backticks.
    """
    body = _strip_comments(css)
    body = re.sub(r"@(media|supports|keyframes)[^{]*\{", "{", body)   # at-rule preludes are prose-free
    offenders = []
    for m in re.finditer(r"(^|[};])\s*([^{};@]{1,400}?)\s*\{", body, re.S):
        sel = " ".join(m.group(2).split())
        if not sel or sel.endswith(":"):        # nested block inside an at-rule
            continue
        if "`" in sel or re.search(r"[a-z]{2}\. [A-Z]", sel) or sel.count(" ") > 14:
            offenders.append(sel[:110])
    assert not offenders, f"selectors that are actually prose (rule was discarded): {offenders}"


def test_the_column_stays_centred(css):
    """The load-bearing rule itself, guarded against outright deletion.

    Both halves matter and they live in different rules: `.scroll` centres the column and
    `.scroll > *` sizes it. Losing only the centring still renders a correctly sized column
    — against the left edge — which is why this needs asserting rather than eyeballing.

    Note this is a text search, so it cannot tell a live rule from one the parser discarded;
    the two tests above are what catch the discard case.
    """
    body = _strip_comments(css)
    centring = re.search(r"\.scroll\s*\{[^}]*justify-items:\s*center[^}]*\}", body, re.S)
    assert centring, ".scroll no longer centres its children — every tab will crowd left"
    sizing = re.search(r"\.scroll\s*>\s*\*\s*\{[^}]*width:\s*min\(var\(--col\)", body, re.S)
    assert sizing, ".scroll > * no longer sets the column width"
