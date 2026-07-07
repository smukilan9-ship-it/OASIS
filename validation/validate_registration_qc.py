"""
validate_registration_qc.py — prove the registration fail-closed QC gate fires.

Builds synthetic H-DAB-like image pairs (no QuPath needed) and drives the REAL
functions used by the pipeline:

  • registration.compute_registration       — the unchanged transform cascade
  • registration.compute_registration_qc    — the new objective QC measurement
  • run_pipeline.evaluate_registration_qc    — the fail-closed gate

Three cases:
  1. WELL-ALIGNED pair (same tissue, small known shift) -> registration_qc.valid
     == True, status "valid".
  2. UNREGISTRABLE pair (independent structure / large unfixable misalignment) ->
     registration_qc.valid == False, status "invalid"  => statistics_valid False.
  3. FORCED IDENTITY fallback -> registration_qc.valid == False, status "invalid".

Exits non-zero if any expectation fails.
"""

import os, sys, tempfile, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIXEL_SIZE_UM = 0.5     # so the 5 µm valid / 10 µm warn thresholds map to 10 / 20 px


def _draw_pattern(rng, size=640, n_shapes=170, shift=(0, 0)):
    """A textured, feature-rich 'tissue' RGB image: a contiguous tissue mass with
    dark shapes on a light field, shifted by `shift`. The contiguous mass makes
    the QC tissue-overlap metric meaningful (real tissue is a connected region,
    not isolated dots); the shapes give corners/edges for ORB/SIFT and keypoint
    QC. Both the mass and the shapes move together with `shift`."""
    import cv2
    dx, dy = shift
    img = np.full((size, size, 3), 245, np.uint8)
    # Contiguous tissue region (darker than the bright background → segmented as
    # tissue by the Otsu mask the QC uses), translated with the pattern.
    cv2.ellipse(img, (size // 2 + dx, size // 2 + dy),
                (int(size * 0.40), int(size * 0.32)), 0, 0, 360,
                (190, 180, 185), -1)
    for _ in range(n_shapes):
        x = int(rng.integers(20, size - 20) + dx)
        y = int(rng.integers(20, size - 20) + dy)
        col = tuple(int(c) for c in rng.integers(30, 150, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.integers(4, 10)), col, -1)
        elif kind == 1:
            w, h = int(rng.integers(6, 16)), int(rng.integers(6, 16))
            cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-14, 14)),
                                   y + int(rng.integers(-14, 14))), col, 2)
    # mild noise → stable keypoints without overwhelming structure
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _save(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)


def main():
    from registration import compute_registration, compute_registration_qc
    from run_pipeline import evaluate_registration_qc

    tmp = tempfile.mkdtemp(prefix="regqc_")
    failures = []
    try:
        rng = np.random.default_rng(7)

        # Shared base tissue used for the WELL-ALIGNED pair.
        base       = _draw_pattern(rng, shift=(0, 0))
        base_shift = _draw_pattern(np.random.default_rng(7), shift=(6, 4))  # same shapes, +6,+4 px
        # Independent tissue (different seed) for the UNREGISTRABLE pair.
        other      = _draw_pattern(np.random.default_rng(999), shift=(0, 0))

        p_ref   = os.path.join(tmp, "ref.png")
        p_good  = os.path.join(tmp, "moving_aligned.png")
        p_bad   = os.path.join(tmp, "moving_independent.png")
        _save(p_ref, base); _save(p_good, base_shift); _save(p_bad, other)

        def run_case(name, ref, mov, expect_valid, expect_status):
            print(f"\n{'='*70}\n{name}\n{'='*70}")
            reg = compute_registration(ref, mov)
            qcm = compute_registration_qc(ref, mov, reg, PIXEL_SIZE_UM)
            qc  = evaluate_registration_qc(reg, qcm, PIXEL_SIZE_UM)
            stats_valid = qc["valid"]      # mirrors run_pipeline's statistics_valid
            print(f"  method            : {qc['method']}")
            print(f"  residual (µm)     : {qc.get('residual_error_um')}")
            print(f"  tissue overlap    : {qc.get('tissue_overlap_fraction')}")
            print(f"  qc inlier ratio   : {qcm.get('qc_inlier_ratio')} "
                  f"({qcm.get('n_qc_inliers')}/{qcm.get('n_qc_matches')})")
            print(f"  QC status / valid : {qc['status']} / {qc['valid']}")
            print(f"  reason            : {qc['reason']}")
            print(f"  -> statistics_valid would be: {stats_valid}")
            ok = (qc["valid"] is expect_valid) and (qc["status"] == expect_status)
            print(f"  EXPECT valid={expect_valid}, status={expect_status}  ->  "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(name)
            return qc

        # 1. Well-aligned → valid.
        run_case("CASE 1  WELL-ALIGNED (small known shift)",
                 p_ref, p_good, expect_valid=True, expect_status="valid")

        # 2. Unregistrable (independent structure) → invalid, stats not trustworthy.
        run_case("CASE 2  UNREGISTRABLE (independent tissue)",
                 p_ref, p_bad, expect_valid=False, expect_status="invalid")

        # 3. Forced identity fallback → invalid (gate keys off method directly).
        print(f"\n{'='*70}\nCASE 3  FORCED IDENTITY FALLBACK\n{'='*70}")
        identity = {"matrix": np.float32([[1, 0, 0], [0, 1, 0]]),
                    "scale_ref": 1.0, "scale_mov": 1.0,
                    "method": "identity", "success": False, "metric": None}
        qc3 = evaluate_registration_qc(identity, {}, PIXEL_SIZE_UM)
        print(f"  QC status / valid : {qc3['status']} / {qc3['valid']}")
        print(f"  reason            : {qc3['reason']}")
        ok3 = (qc3["valid"] is False) and (qc3["status"] == "invalid")
        print(f"  EXPECT valid=False, status=invalid  ->  {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            failures.append("CASE 3 forced identity")

        print(f"\n{'='*70}\nVERDICT\n{'='*70}")
        if failures:
            print(f"  FAILED: {len(failures)} case(s): {failures}")
            print("  The registration QC gate did NOT behave as required.")
            return 1
        print("  ALL CASES PASS — the fail-closed registration QC gate fires "
              "correctly:\n"
              "    • well-aligned pairs are marked valid\n"
              "    • identity fallback and unregistrable pairs are marked invalid\n"
              "      (their spatial statistics are flagged unreliable).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
