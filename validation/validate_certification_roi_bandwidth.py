#!/usr/bin/env python
"""
validate_certification_roi_bandwidth.py

Decisive checks for the two reviewer-facing controls added to the spatial-association
pipeline:

  1. Per-image 75 µm BANDWIDTH pre-flight (spatial.precheck_bandwidth_within_window):
       coarse architecture → ok/caution (valid);
       fine/dense architecture → dense_tissue_bandwidth_invalid (invalid);
       too-few cells → underpowered_insufficient_positives (invalid).
       Dense/fine and underpowered both fail closed, but they are different failure
       modes: only dense/fine is eligible for the dense morphology-conditioned null.

  2. CERTIFICATION ROI (serial_registration.landmark_register_and_verify with
       user_roi_polygon):
       • landmarks OUTSIDE the ROI cannot drive the verdict (constrain);
       • a clean subset inside the ROI CERTIFIES and the ROI becomes the certified
         window (roi_polygon == ROI, certified_window_source == 'user_roi');
       • too-few-inside → NOT_CERTIFIABLE;  sliver window → NOT_CERTIFIABLE
         (never emit a tiny window);
       • coordinate round-trip: mov_roi = M^-1 · ref_roi is exact for a known M.

Run:  python validation/validate_certification_roi_bandwidth.py
Exit code 0 = all pass. This is a self-contained synthetic test (no datasets).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial import precheck_bandwidth_within_window          # noqa: E402
from oasis.spatial.serial_registration import (landmark_register_and_verify,  # noqa: E402
                                 _fit_similarity_ls, _apply_affine)

_FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


# ── 1. Bandwidth pre-flight ───────────────────────────────────────────────────
def test_bandwidth():
    print("\n1. 75 µm bandwidth pre-flight (within analysis window)")
    rng = np.random.default_rng(0)
    px = 0.5

    # Coarse: a few widely separated blobs → ℓ̂ well above the bandwidth.
    centers = np.array([[100, 100], [1500, 100], [100, 1500],
                        [1500, 1500], [800, 800]], float) / px
    coarse = np.vstack([c + rng.normal(0, 120 / px, size=(120, 2)) for c in centers])

    # Fine: dense fine-grained structure (~15 µm blobs) → ℓ̂ below the bandwidth.
    fc = rng.uniform(0, 3000 / px, size=(200, 2))
    fine = np.vstack([c + rng.normal(0, 15 / px, size=(6, 2)) for c in fc])

    few = rng.uniform(0, 3000 / px, size=(10, 2))    # 5≤n<30 → sparse (Q3)
    none3 = rng.uniform(0, 3000 / px, size=(3, 2))   # n<5 → absent (Q3)

    rc = precheck_bandwidth_within_window({"A": coarse, "B": coarse}, ["A", "B"], px, None)
    rf = precheck_bandwidth_within_window({"A": fine, "B": fine}, ["A", "B"], px, None)
    ru = precheck_bandwidth_within_window({"A": few, "B": few}, ["A", "B"], px, None)
    rabs = precheck_bandwidth_within_window({"A": none3, "B": none3}, ["A", "B"], px, None)
    rm = precheck_bandwidth_within_window({"A": fine, "B": few}, ["A", "B"], px, None)

    check("coarse architecture is valid (ok/caution)",
          rc["valid"] and rc["worst_status"] in ("ok", "caution"),
          f"worst={rc['worst_status']} ℓ̂={rc['per_image']['A']['scale_um']}")
    check("fine/dense architecture is labelled dense_tissue_bandwidth_invalid + fails closed",
          rf["worst_status"] == "dense_tissue_bandwidth_invalid" and rf["valid"] is False,
          f"worst={rf['worst_status']} ℓ̂={rf['per_image']['A']['scale_um']}")
    check("sparse marker (5≤n<30) → underpowered_sparse_marker (runs, not fail-closed)",
          ru["worst_status"] == "underpowered_sparse_marker" and ru["valid"] is False
          and ru["per_image"]["A"]["power"] == "sparse",
          f"worst={ru['worst_status']} n={ru['per_image']['A']['n']}")
    check("near-absent marker (n<5) → marker_absent (abundance finding)",
          rabs["worst_status"] == "marker_absent" and "A" in rabs["absent_markers"],
          f"worst={rabs['worst_status']} n={rabs['per_image']['A']['n']}")
    check("sparse beats dense when one marker is sparse",
          rm["worst_status"] == "underpowered_sparse_marker" and rm["valid"] is False,
          f"worst={rm['worst_status']} A_power={rm['per_image']['A']['power']} B_power={rm['per_image']['B']['power']}")
    check("window_scope names the window (not per-image)",
          rc["window_scope"] == "certified_analysis_window")


# ── 2. Certification ROI ──────────────────────────────────────────────────────
def _clean_and_corrupt(px=0.5):
    """8 clean inside-ROI landmarks (mov==ref) + 6 corrupted outside-ROI landmarks.
    Without the ROI the corrupted points wreck the fit; with it, only clean ones fit."""
    inside = np.array([[300, 300], [700, 300], [300, 700], [700, 700],
                       [400, 500], [600, 500], [500, 400], [500, 600]], float)
    outside = np.array([[60, 60], [940, 60], [60, 940], [940, 940],
                        [60, 500], [940, 500]], float)
    ref = np.vstack([inside, outside])
    mov = ref.copy()
    mov[len(inside):] += 45.0            # corrupt the OUTSIDE points (≫ 5 µm at 0.5 µm/px)
    return ref, mov, px


def test_roi_constrains_and_becomes_window():
    print("\n2. Certification ROI — constrain + become-window")
    ref, mov, px = _clean_and_corrupt()
    wh = (1000, 1000)
    roi = [[250, 250], [750, 250], [750, 750], [250, 750]]      # area frac 0.25

    no_roi = landmark_register_and_verify(ref, mov, px, image_wh=wh)
    with_roi = landmark_register_and_verify(ref, mov, px, image_wh=wh, user_roi_polygon=roi)

    check("without ROI the corrupted outside points block certification",
          no_roi["verdict"] != "CERTIFIED", f"verdict={no_roi['verdict']}")
    check("with ROI the clean inside subset CERTIFIES",
          with_roi["verdict"] == "CERTIFIED", f"verdict={with_roi['verdict']}")
    check("ROI-constrained fit uses only inside landmarks (n==8)",
          with_roi["n"] == 8, f"n={with_roi['n']}")
    check("drawn ROI becomes the certified window (source=user_roi)",
          with_roi.get("certified_window_source") == "user_roi"
          and with_roi.get("roi_polygon") is not None)
    # roi_polygon should be (approximately) the drawn square.
    if with_roi.get("roi_polygon"):
        poly = np.array(with_roi["roi_polygon"], float)
        area = _shoelace(poly)
        check("certified window area ≈ drawn ROI area",
              abs(area - 250000.0) / 250000.0 < 0.05, f"area={area:.0f}")


def test_roi_too_few_inside():
    print("\n3. Certification ROI — too few landmarks inside → NOT_CERTIFIABLE")
    ref, mov, px = _clean_and_corrupt()
    wh = (1000, 1000)
    tiny = [[280, 280], [360, 280], [360, 360], [280, 360]]     # contains ≤1 landmark
    r = landmark_register_and_verify(ref, mov, px, image_wh=wh, user_roi_polygon=tiny)
    check("too-few-inside → NOT_CERTIFIABLE",
          r["verdict"] == "NOT_CERTIFIABLE", f"verdict={r['verdict']}")
    check("reason names the Certification ROI",
          "Certification ROI" in (r.get("reason") or ""), r.get("reason"))


def test_roi_sliver_window():
    print("\n4. Certification ROI — sliver window fails closed (no tiny window emitted)")
    # 8 clean landmarks clustered so ≥6 fall inside a 100×100 (0.01 area-frac) ROI.
    cluster = np.array([[410, 410], [490, 410], [410, 490], [490, 490],
                        [450, 420], [420, 460], [470, 470], [430, 440]], float)
    ref = cluster
    mov = cluster.copy()
    px, wh = 0.5, (1000, 1000)
    sliver = [[400, 400], [500, 400], [500, 500], [400, 500]]   # area frac 0.01 < 0.10
    r = landmark_register_and_verify(ref, mov, px, image_wh=wh, user_roi_polygon=sliver)
    check("sliver ROI → NOT_CERTIFIABLE (no sliver window)",
          r["verdict"] == "NOT_CERTIFIABLE" and r.get("roi_polygon") is None,
          f"verdict={r['verdict']} roi={r.get('roi_polygon')}")


def test_coord_roundtrip():
    print("\n5. Coordinate round-trip — mov_roi = M^-1 · ref_roi is exact for a known M")
    import cv2
    # A known similarity M (mov → ref): scale 1.0, rotation 12°, translation (30, -20).
    th = np.deg2rad(12.0)
    s = 1.0
    M = np.array([[s * np.cos(th), -s * np.sin(th), 30.0],
                  [s * np.sin(th),  s * np.cos(th), -20.0]], float)
    ref_roi = np.array([[100, 100], [400, 120], [380, 500], [90, 460]], float)
    Minv = cv2.invertAffineTransform(M.astype(np.float32))
    mov_roi = _apply_affine(ref_roi, Minv)                      # ref → mov
    back = _apply_affine(mov_roi, M)                            # mov → ref (should recover)
    err = float(np.abs(back - ref_roi).max())
    check("M · (M^-1 · ref_roi) recovers ref_roi", err < 1e-3, f"max_err={err:.2e}")


def _shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


if __name__ == "__main__":
    print("=" * 74)
    print("Certification ROI + 75 µm bandwidth pre-flight — decisive validation")
    print("=" * 74)
    test_bandwidth()
    test_roi_constrains_and_becomes_window()
    test_roi_too_few_inside()
    test_roi_sliver_window()
    test_coord_roundtrip()
    print("\n" + "=" * 74)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} check(s) FAILED: {', '.join(_FAILS)}")
        sys.exit(1)
    print("RESULT: all checks PASSED")
    sys.exit(0)
