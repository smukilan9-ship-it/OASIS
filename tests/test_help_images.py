"""Every picture the help card names must exist, and ship.

A missing help image fails the way missing images always do: a broken frame, or with the
`onerror` handler that hides it, nothing at all. No console error a user would see, no test
failure, just a step that quietly lost its illustration. The two ways it happens are a
renamed file and a bundle that forgot to carry the folder, so both are checked here.

One step deliberately has no image. `cls-4-report` showed the held-out score, and the only
way to produce that screen without a human labelling session is to seed the labels from the
same DAB measurement the classifier is then fitted on — which makes the classes separable by
construction and puts F1 1.00, AUC 1.00 on the screen. That number is a property of how the
screenshot was made, not of the software, and it would be read as a claim about the software.
The step keeps its explanation and drops the picture; this test pins that choice so nobody
"fixes" the gap by regenerating it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "oasis/webui/index.html"
HELP = ROOT / "oasis/webui/help"
SPEC = ROOT / "packaging/OASIS.spec"


@pytest.fixture(scope="module")
def referenced():
    """Every filename the HELP steps name, in order."""
    html = INDEX.read_text()
    start = html.index("const HELP")
    block = html[start:html.index("\nfunction ", start)]
    return re.findall(r"img:\s*'([^']+)'", block)


def test_some_images_are_referenced(referenced):
    assert len(referenced) >= 15, (
        f"only {len(referenced)} help images referenced — the HELP block or this regex moved")


def test_every_referenced_image_exists(referenced):
    missing = [n for n in referenced if not (HELP / n).exists()]
    assert not missing, ("help steps name pictures that are not in oasis/webui/help: "
                         + ", ".join(missing))


def test_no_referenced_image_is_a_stub(referenced):
    """A truncated or zero-byte PNG renders exactly like a missing one."""
    # Absence is the test above's job; raising FileNotFoundError here would report the same
    # problem as an error rather than a failure, and hide this check's own result.
    tiny = [n for n in referenced
            if (HELP / n).exists() and (HELP / n).stat().st_size < 8_000]
    assert not tiny, f"suspiciously small help images (capture failed?): {tiny}"


def test_the_held_out_report_step_has_no_picture():
    html = INDEX.read_text()
    assert "cls-4-report.png" not in html, (
        "the held-out report step has a picture again. Any screenshot of that screen made "
        "without hand labelling shows a perfect score, because the labels come from the same "
        "measurement the rule is fitted on — see this module's docstring before restoring it")


def test_the_help_folder_is_bundled():
    """The spec carries it explicitly; nothing imports these files, so nothing else would."""
    spec = SPEC.read_text()
    assert re.search(r'"oasis",\s*"webui",\s*"help"', spec), (
        "packaging/OASIS.spec no longer bundles oasis/webui/help — the shipped app would "
        "show a broken frame on every help step while the source tree looks fine")


def test_no_orphan_images(referenced):
    """An image nothing references is dead weight in a bundle measured in hundreds of MB."""
    on_disk = {p.name for p in HELP.glob("*.png")}
    orphans = sorted(on_disk - set(referenced))
    assert not orphans, f"help images nothing references: {orphans}"
