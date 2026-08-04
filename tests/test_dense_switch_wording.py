"""When the run switches to the dense null, the chip says so instead of warning.

The per-marker chip reports the architecture verdict inside the certified window. On tissue
whose structures sit just above the 75 um bandwidth the verdict is "caution", and the run
responds by switching to the dense morphology-conditioned null — which is the correct null
for that architecture, not a compromise.

Rendered together, the two lines contradicted each other:

    CD8: caution - structures ~96.6 um across        (amber)
    Dense-tissue morphological null - morphology-conditioned cross-K   (violet)

An amber "caution" beside a decision that has already been taken reads as doubt about the
result, and invites a reader to discount a finding that is in fact being measured against
the null its tissue calls for. The chip now names the action and takes the colour of the
null it switched to, so the summary and the expanded parameter table agree.

The wording only changes when the dense null was actually selected. A "caution" verdict that
did NOT lead to a switch is still a caution, and must keep saying so.
"""
import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "oasis/webui/index.html"


@pytest.fixture(scope="module")
def js():
    return INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chip(js):
    m = re.search(r"function spatialBwChip\(([^)]*)\)\s*\{(.*?)\n\}", js, re.S)
    assert m, "spatialBwChip is gone"
    return m.group(1), m.group(2)


def test_the_chip_is_told_whether_the_dense_null_was_selected(chip):
    args, _ = chip
    assert len(args.split(",")) >= 3, (
        "spatialBwChip no longer receives the dense-null flag, so it cannot tell a caution "
        "that was acted on from one that was not")


def test_a_caution_that_led_to_a_switch_says_switching(chip):
    _, body = chip
    assert "'switching'" in body, "the switched-to-dense wording is gone; it will read 'caution'"
    # The flag has to be READ before the wording is assigned. Checking order rather than a
    # literal `if (...)` keeps this from breaking on how the condition happens to be wrapped.
    assert "denseSelected" in body.split("'switching'")[0], (
        "the wording is no longer gated on the dense null having been selected — every "
        "caution would be relabelled a switch")


def test_a_caution_without_a_switch_still_says_caution(chip):
    """The plain verdict has to survive, or a real warning is renamed into an action."""
    _, body = chip
    assert "caution:['#f59e0b','caution']" in body.replace(" ", ""), (
        "the unswitched caution label was removed — every caution would now read as a switch")


def test_both_chip_call_sites_pass_the_flag(js):
    calls = re.findall(r"spatialBwChip\(m,\s*e([^)]*)\)", js)
    assert calls, "no spatialBwChip call sites found — the regex has drifted from the markup"
    bare = [c for c in calls if not c.strip()]
    assert not bare, (
        f"{len(bare)} spatialBwChip call site(s) still omit the dense-null flag, so that "
        "screen keeps showing 'caution' next to a dense-null decision")


def test_the_parameter_table_uses_the_same_word(js):
    """The expanded table sits directly under the chip; disagreeing there is worse."""
    m = re.search(r"architecture status`\]\s*=(.*?);", js, re.S)
    assert m, "the architecture-status parameter row is gone"
    assert "'switching'" in m.group(1), (
        "the parameter table still reports 'caution' where the chip above it says "
        "'switching'")


def test_caution_is_never_labelled_a_valid_bandwidth(js):
    """"75 µm bandwidth valid · caution" survived the routing change that made it false.

    `caution` counted as valid until §15.3 measured the reweighted null's false engagement
    rate at 0.17-0.25 in that margin against a nominal 0.05. It has routed to the dense null
    since. A screen that prints "valid" beside a run which rejected the 75 um null is telling
    the operator the opposite of what the run did.
    """
    src = js
    assert "75 µm bandwidth valid · caution" not in src
    i = src.index("function spatialBandwidthStatusLabel")
    body = src[i:i + 1800]
    assert "caution: 'Architecture marginally coarser than 75 µm'" in body


def test_every_status_label_call_site_knows_whether_the_run_switched(js):
    """The label reads "switching" only when it can see that the run switched.

    Both call sites — the review card and the results banner — describe the same decision,
    so a call that omits the flag silently reverts one of them to the old wording.
    """
    src = js
    calls = [ln for ln in src.splitlines()
             if "spatialBandwidthStatusLabel(" in ln and "function " not in ln
             and "typeof" not in ln]
    assert calls, "no call sites found — the label was renamed or removed"
    for ln in calls:
        assert "," in ln.split("spatialBandwidthStatusLabel(", 1)[1].split(")")[0], (
            f"call site passes no dense-selected flag: {ln.strip()}")
