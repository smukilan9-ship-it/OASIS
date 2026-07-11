"""
validate_dense_scaffold_keren_external.py

Validate the dense morphology-conditioned null against an independent scaffold on
three Keren TNBC dense fields.

This is an artifact-check validation for the UI-style Keren pilot already run
outside the repo. It does not re-download or re-render the 4+ GB dataset. Instead
it verifies the produced comparison JSON:

  OASIS UI path with OASIS all-cell scaffold
    vs
  same positives/window with Keren mask-derived external scaffold

PASS means the existing pilot artifacts show the verdict was stable under this
external scaffold substitution. It does not mean dense-scaffold circularity is
globally solved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_ROOT = Path.home() / "Desktop" / "OASIS_keren_tnbc_validation"
REL = "external_scaffold_spatial_comparison/external_vs_oasis_scaffold_comparison.json"


def _root() -> Path:
    env = os.environ.get("OASIS_KEREN_TNBC_VALIDATION_DIR")
    return Path(env).expanduser() if env else DEFAULT_ROOT


def _fmt_p(v) -> str:
    try:
        return f"{float(v):.5g}"
    except Exception:
        return "NA"


def main() -> int:
    root = _root()
    path = root / REL
    if not path.exists():
        print("FAIL: Keren external-scaffold artifact is missing.")
        print(f"Expected: {path}")
        print("Set OASIS_KEREN_TNBC_VALIDATION_DIR to the pilot folder if it lives elsewhere.")
        return 1

    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        print(f"FAIL: {path} does not contain a non-empty comparison list.")
        return 1

    expected_samples = {"p13", "p16", "p32"}
    seen = {str(r.get("sample_id")) for r in rows}
    missing = expected_samples - seen
    if missing:
        print(f"FAIL: missing expected Keren samples: {', '.join(sorted(missing))}")
        return 1

    failures = []
    print("Dense scaffold external-source check: Keren TNBC pseudo-IHC fields")
    print("sample  oasis_p  external_p  direction  stable  support_oasis/external")
    for row in sorted(rows, key=lambda r: str(r.get("sample_id"))):
        sample = str(row.get("sample_id"))
        stable = bool(row.get("verdict_stable"))
        same_direction = row.get("ui_oasis_direction") == row.get("external_direction")
        both_robust = (
            row.get("ui_oasis_robustness") == "robust"
            and row.get("external_robustness") == "robust"
        )
        dense_selected = row.get("external_primary_null") == "dense_morphology"
        if not (stable and same_direction and both_robust and dense_selected):
            failures.append(sample)
        print(
            f"{sample:>5}  {_fmt_p(row.get('ui_oasis_p_dclf')):>7}  "
            f"{_fmt_p(row.get('external_p_dclf')):>10}  "
            f"{str(row.get('external_direction')):>10}  "
            f"{'yes' if stable else 'no ':>6}  "
            f"{row.get('ui_oasis_support_n')}/{row.get('external_support_n')}"
        )

    metrics = {
        "samples": len(rows),
        "stable_samples": sum(1 for r in rows if r.get("verdict_stable")),
        "external_scaffold_robust_samples": sum(
            1 for r in rows if r.get("external_robustness") == "robust"
        ),
        "min_external_p": min(float(r["external_p_dclf"]) for r in rows),
        "max_external_p": max(float(r["external_p_dclf"]) for r in rows),
    }
    print("##METRICS## " + json.dumps(metrics, sort_keys=True))

    if failures:
        print(f"FAIL: unstable external-scaffold verdict(s): {', '.join(failures)}")
        return 1

    print("PASS: all three Keren fields kept the same robust dense-null verdict with the external scaffold.")
    print("CAVEAT: this is a three-field pilot and excludes serial-section registration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
