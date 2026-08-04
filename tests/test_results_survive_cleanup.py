"""A finished run must not report itself as an empty one.

The results screen — total cells, the per-image table, the methods paragraph — is built
from the per-image `*_summary.json` files. The worker deletes those at the end of the run
when "Per-image settings record" is switched off, and it did so BEFORE the API read them.

Observed: a four-image CD8 batch of 31,154 cells finished, wrote a complete results.csv and
four overlays, and rendered "TOTAL CELLS 0 · 0 image(s)", an empty table and "a cell called
positive above NaN OD". Nothing failed and nothing warned; the run was fine and only the
account of it was wrong, which is the worst shape this class of bug takes.

results.csv is not a substitute source here: it carries no `pixel_size_warning` and no
`membrane_quality_warning`, so rebuilding the table from it would have dropped exactly the
two flags that say the numbers may not be trustworthy.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _summary(name, cells, pos):
    return {"image": name, "total_cells": cells, "positive_cells": pos,
            "negative_cells": cells - pos, "positivity_pct": pos / cells * 100,
            "dab_threshold": 0.2, "pixel_size_um": 0.7519,
            "pixel_size_source": "scale_bar", "pixel_size_warning": True}


@pytest.fixture
def staged(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    for name, cells, pos in (("A.tif", 4521, 119), ("B.tif", 13073, 236)):
        (out / f"{Path(name).stem}_summary.json").write_text(json.dumps(_summary(name, cells, pos)), encoding="utf-8")
    return out


def test_the_table_is_built_before_the_summaries_are_deleted(staged):
    from oasis.webui.api import API
    cfg = {"output_dir": str(staged), "dashboard_dir": str(staged),
           "image_whitelist": ["A.tif", "B.tif"], "_defer_summary_cleanup": True}

    res = API()._load_results(cfg)

    names = sorted(m["name"] for m in res["metrics"])
    assert names == ["A", "B"], f"the run reported itself as empty: {res['metrics']}"
    assert sum(m["total_cells"] for m in res["metrics"]) == 17594
    # The flag that says the numbers may not be trustworthy has to survive with them.
    assert all(m["pixel_size_warning"] for m in res["metrics"])


def test_the_deletion_still_happens_afterwards(staged):
    from oasis.webui.api import API
    cfg = {"output_dir": str(staged), "dashboard_dir": str(staged),
           "_defer_summary_cleanup": True}

    API()._load_results(cfg)

    assert not list(staged.glob("*_summary.json")), (
        "deferring the cleanup must not cancel it — the operator asked for these to go")


def test_summaries_are_kept_when_the_operator_asked_to_keep_them(staged):
    from oasis.webui.api import API
    cfg = {"output_dir": str(staged), "dashboard_dir": str(staged)}

    API()._load_results(cfg)

    assert len(list(staged.glob("*_summary.json"))) == 2
