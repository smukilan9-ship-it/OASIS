"""The stain tokens the matcher knows, and the ordering that keeps them apart.

A missing token does not fail loudly: the files are simply reported "unmatched", with no
hint that the reason is a gap in a list rather than a problem with the filenames. CD45 —
leukocyte common antigen, one of the most-used markers in IHC — was absent, so a CD8/CD45
serial-section pair of folders matched zero pairs.

The ordering matters as much as the membership. Tokens are checked longest-first, so "cd45"
must be tried before "cd4"; if that sort were ever dropped, every CD45 file would normalize
to a stem still carrying a stray "5" and would pair with nothing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oasis.common.file_matcher import normalize_name


def test_cd45_is_recognised():
    stem, token = normalize_name("679_CD45_lm00")
    assert token == "cd45", f"CD45 not recognised as a stain: {token!r}"
    assert stem == "679_lm00"


def test_cd45_and_cd8_normalise_to_the_same_stem():
    """Which is the whole point: it is what makes them a pair."""
    a, _ = normalize_name("679_CD8_lm00")
    b, _ = normalize_name("679_CD45_lm00")
    assert a == b == "679_lm00"


def test_cd4_is_not_eaten_by_cd45():
    stem, token = normalize_name("case12_CD4_10x")
    assert token == "cd4"
    assert stem == "case12_10x"


def test_a_longer_token_wins_over_its_own_prefix():
    """cd163 over cd16-style prefixes, tim-3 over tim, cd45 over cd4 — one rule, three cases."""
    for name, expected in (("s_CD163_1", "cd163"), ("s_TIM3_1", "tim3"), ("s_CD45_1", "cd45")):
        _, token = normalize_name(name)
        assert token == expected, f"{name} matched {token!r}, expected {expected!r}"
