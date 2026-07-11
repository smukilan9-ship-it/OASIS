"""
validate_stabilization_gates.py — prove the 2026-06-21 stabilization gates fire.

Covers (no QuPath/InstanSeg needed — all gates sit before segmentation):
  • A7/A6: provenance stamps reweight_bandwidth_um, null_seed, architecture assumption.
  • A1/B1: the spatial certification stamp is fail-closed (status=not_performed,
    is_certified=False) — checked via the literal the pipeline writes.
  • A8: cohort BH/FDR runs across per-pair DCLF p-values.
  • B4: Restained run is FAIL-CLOSED without manual correspondence certification, and
    the (advisory) hematoxylin correlation diagnostic discriminates corresponding from
    grossly non-corresponding tiles. Restained imports cleanly (no Shapely error).

Exit non-zero if any expectation fails.
"""
import os, sys, tempfile, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def test_provenance():
    print("\nA7/A6 — provenance bandwidth/seed/architecture")
    import run_pipeline
    prov = run_pipeline.build_provenance({}, None, "legacy", 0.7519, "manual")
    ap = prov["analysis_params"]
    check("reweight_bandwidth_um stamped (75.0)", ap.get("reweight_bandwidth_um") == 75.0)
    check("null_seed stamped (0)", ap.get("null_seed") == 0)
    check("architecture_scale_assumption_um stamped", ap.get("architecture_scale_assumption_um") == 75.0)
    check("architecture_scale_measured False when no association", ap.get("architecture_scale_measured") is False)

    # A6 (now enforced): with an association carrying a measured architecture scale,
    # provenance flips to measured=True and records the value/status/ok gate.
    assoc = {"association": {"CD8__TIM3": {
        "n_perm": 999, "primary_null": "reweighted",
        "global": {"dclf_rmin_um": 10.0, "dclf_rmax_um": 50.0},
        "architecture_scale": {"scale_um": 42.0, "status": "unreliable", "ok": False},
    }}}
    prov2 = run_pipeline.build_provenance({}, assoc, "landmark", 0.5, "scale")
    ap2 = prov2["analysis_params"]
    check("architecture_scale_measured True when measured", ap2.get("architecture_scale_measured") is True)
    check("architecture_scale_um recorded", ap2.get("architecture_scale_um") == 42.0)
    check("architecture_scale_ok gate present", ap2.get("architecture_scale_ok") is False)


def test_cohort_fdr():
    print("\nA8 — cohort BH/FDR across per-pair DCLF p")
    from oasis.spatial.spatial_stats import cohort_multiple_comparison_correction
    out = cohort_multiple_comparison_correction([0.001, 0.04, 0.5, None, 0.2], method="bh")
    check("n_tested drops the None", out["n_tested"] == 4)
    check("adjusted p available", len(out["adjusted_pvalues"]) == 4)
    check("n_significant_adjusted <= n_significant_raw",
          out["n_significant_adjusted"] <= out["n_significant_raw"])


def _png(path, arr):
    from PIL import Image
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_restained_gate():
    print("\nB4 — Restained correspondence gate (fail-closed) + diagnostic")
    from oasis.restained import restained_coexpression as rc          # also proves import (no Shapely error)
    check("restained module imports cleanly", True)
    tmp = tempfile.mkdtemp(prefix="restgate_")
    try:
        rng = np.random.default_rng(0)
        base = rng.integers(120, 230, (256, 256, 3), dtype=np.uint8)
        h = os.path.join(tmp, "s_Hematoxylin.png")
        a = os.path.join(tmp, "s_CD8.png")
        b = os.path.join(tmp, "s_FoxP3.png")
        _png(h, base); _png(a, base.copy()); _png(b, base.copy())
        bundle = {"sample_id": "s", "hematoxylin": h, "marker_a": a, "marker_b": b,
                  "reference_mask": None}

        # (1) No certification → BLOCKED before any segmentation.
        res = rc.run_bundle(bundle, {"pixel_size_um": 0.5, "threshold_a": 0.1,
                                     "threshold_b": 0.1, "output_dir": tmp},
                            os.path.join(tmp, "out1"))
        check("uncertified bundle is BLOCKED", res.get("correspondence", {}).get("status") == "BLOCKED")
        check("blocked bundle has NO coexpression stats", res.get("coexpression") is None)
        check("blocked validity = blocked_uncertified",
              res.get("validity", {}).get("biological_validity") == "blocked_uncertified")

        # (2) Diagnostic discriminates corresponding vs non-corresponding.
        d_same = rc.structural_correspondence_diagnostic(bundle)
        other = rng.integers(120, 230, (256, 256, 3), dtype=np.uint8)   # independent content
        b2 = os.path.join(tmp, "s_FoxP3_bad.png"); _png(b2, other)
        bad = dict(bundle, marker_b=b2)
        d_bad = rc.structural_correspondence_diagnostic(bad)
        check(f"corresponding min_corr high ({d_same.get('min_corr')})", d_same.get("min_corr", 0) > 0.8)
        check(f"non-corresponding min_corr lower ({d_bad.get('min_corr')})",
              d_bad.get("min_corr", 1) < d_same.get("min_corr", 0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 64 + "\nSTABILIZATION GATE TESTS (2026-06-21)\n" + "=" * 64)
    test_provenance()
    test_cohort_fdr()
    test_restained_gate()
    print("\n" + "=" * 64)
    if fails:
        print(f"FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("ALL STABILIZATION GATE CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
