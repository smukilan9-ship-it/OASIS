"""
validate_dense_scaffold_perturbation.py

Check the dense-null scaffold sensitivity harness on Keren TNBC pilot artifacts.

The harness keeps the same OASIS-observed CD8/PanCK positives and replaces only
the all-cell dense-null support scaffold: random thinning, density-biased
deletion, local dropout, and centroid jitter.

PASS here is deliberately nuanced:
  * p13 and p16 must remain stable across all variants.
  * p32 is expected to be flagged as scaffold-sensitive/borderline.

That means the validation proves the harness can distinguish strong dense calls
from fragile ones. It is not a claim that all dense calls are scaffold-invariant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_ROOT = Path.home() / "Desktop" / "OASIS_keren_tnbc_validation"
REL = "scaffold_perturbation_tests/scaffold_perturbation_summary.json"


def _root() -> Path:
    env = os.environ.get("OASIS_KEREN_TNBC_VALIDATION_DIR")
    return Path(env).expanduser() if env else DEFAULT_ROOT


def _fmt(v) -> str:
    try:
        return f"{float(v):.5g}"
    except Exception:
        return "NA"


def main() -> int:
    root = _root()
    path = root / REL
    if not path.exists():
        print("FAIL: Keren scaffold-perturbation summary is missing.")
        print(f"Expected: {path}")
        print("Set OASIS_KEREN_TNBC_VALIDATION_DIR to the pilot folder if it lives elsewhere.")
        return 1

    rows = json.loads(path.read_text())
    by_sample = {str(r.get("sample_id")): r for r in rows}
    required = {"p13", "p16", "p32"}
    missing = required - set(by_sample)
    if missing:
        print(f"FAIL: missing expected samples: {', '.join(sorted(missing))}")
        return 1

    failures = []
    print("Dense scaffold perturbation stress test")
    print("sample  variants  stable  significant  p_min    p_max    fail_closed")
    for sample in ("p13", "p16", "p32"):
        row = by_sample[sample]
        n = int(row.get("n_variants", 0))
        stable = int(row.get("stable_direction_verdict", 0))
        sig = int(row.get("significant_variants", 0))
        fail_closed = int(row.get("fail_closed_variants", 0))
        print(
            f"{sample:>5}  {n:>8}  {stable:>6}  {sig:>11}  "
            f"{_fmt(row.get('p_min')):>7}  {_fmt(row.get('p_max')):>7}  {fail_closed:>11}"
        )

        if sample in {"p13", "p16"} and not (n == 33 and stable == 33 and sig == 33 and fail_closed == 0):
            failures.append(f"{sample} was not fully stable")
        if sample == "p32":
            # p32 is the designed cautionary case: the harness should flag that a
            # borderline dense result can depend on scaffold quality.
            if not (n == 33 and stable < n and sig < n and fail_closed >= 1):
                failures.append("p32 was not flagged as scaffold-sensitive")

    p32 = by_sample["p32"]
    metrics = {
        "samples": len(rows),
        "variants_per_sample": int(by_sample["p13"].get("n_variants", 0)),
        "fully_stable_samples": sum(
            1 for r in rows
            if int(r.get("stable_direction_verdict", 0)) == int(r.get("n_variants", -1))
        ),
        "p32_stable_variants": int(p32.get("stable_direction_verdict", 0)),
        "p32_significant_variants": int(p32.get("significant_variants", 0)),
        "p32_max_p": float(p32.get("p_max")),
        "p32_fail_closed_variants": int(p32.get("fail_closed_variants", 0)),
    }
    print("##METRICS## " + json.dumps(metrics, sort_keys=True))

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1

    print("PASS: p13/p16 are stable; p32 is correctly exposed as borderline/scaffold-sensitive.")
    print("CAVEAT: publication claims must report scaffold sensitivity, especially for borderline dense calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
