"""No tracked file may hardcode a path that only exists on the machine that wrote it.

Two of these shipped. `validation/run_kamp_codex.R` sourced KAMP from an absolute path under
a temporary working directory, so the script was broken for everyone else and said so only
as a file-not-found halfway down. `tools/capture_help.py` did the same for its demo images.

Both were invisible in review for the same reason: an absolute path is syntactically fine,
the file it points at exists for the author, and nothing runs these two in CI. They fail for
the next person, on a different machine, with an error that reads like a missing dependency
rather than a bad constant.

The rule is the fix: a path outside the repo comes from an environment variable or an
argument, with a message naming what to set.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Spelled in pieces so this file is not its own tripwire.
BAD = [
    ("/private/" + "tmp/claude-", "a session scratchpad directory"),
    ("/var/" + "folders/", "a macOS per-user temporary directory"),
    (".claude/" + "projects/", "an assistant working directory"),
    ("Co-Authored" + "-By: Claude", "an assistant commit trailer"),
]

# Text files only. A binary match is meaningless and the help PNGs are large.
SUFFIXES = {".py", ".R", ".r", ".js", ".html", ".sh", ".yaml", ".yml", ".toml",
            ".cff", ".md", ".txt", ".spec", ".groovy"}


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    for rel in out.stdout.splitlines():
        p = ROOT / rel
        # legacy/ is kept deliberately as a record of removed work and is never run.
        if rel.startswith("legacy/") or p.suffix not in SUFFIXES or not p.is_file():
            continue
        yield rel, p


@pytest.mark.parametrize("needle,what", BAD, ids=[b[1] for b in BAD])
def test_no_tracked_file_hardcodes_a_local_path(needle, what):
    hits = []
    for rel, path in tracked_text_files():
        if rel == Path(__file__).relative_to(ROOT).as_posix():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if needle in line:
                hits.append(f"{rel}:{i}")
    assert not hits, (f"{what} is hardcoded in tracked files — it will not exist for anyone "
                      f"else, and it should not be published: {', '.join(hits[:10])}")


def test_the_scan_actually_sees_files():
    """A guard that silently matches nothing passes forever."""
    seen = list(tracked_text_files())
    assert len(seen) > 50, f"only {len(seen)} tracked text files found — git ls-files failed?"


def test_home_directory_is_not_hardcoded():
    """`~` and $HOME are fine; a literal /Users/<someone> in code is not.

    Scoped to things that RUN. The research notes and changelogs are a lab record: they say
    where a particular run's inputs sat on the day it was run, and rewriting that history to
    satisfy a lint would destroy the only account of it. A path in code is a dependency; a
    path in a note is a fact about the past.
    """
    code = {".py", ".R", ".r", ".js", ".html", ".sh", ".yaml", ".yml", ".spec", ".groovy"}
    pattern = re.compile(r"/Users/[A-Za-z0-9._-]+/")
    hits = []
    for rel, path in tracked_text_files():
        if rel == Path(__file__).relative_to(ROOT).as_posix() or path.suffix not in code:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            # A comment showing an example command is documentation, not a dependency.
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            if pattern.search(line):
                hits.append(f"{rel}:{i}")
    assert not hits, ("a specific user's home directory is hardcoded in tracked files: "
                      + ", ".join(hits[:10]))
