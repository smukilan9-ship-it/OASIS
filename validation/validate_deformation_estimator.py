"""
NEGATIVE RESULT — the patch-flow deformation estimator cannot measure model error.

WHAT WAS PROPOSED
-----------------
Certify a landmark registration on the error AT A CELL rather than at the landmarks:

    cell_registration_error = sqrt(estimation² + model²)

      estimation  prediction SE of the fitted similarity, σ·sqrt(fᵀ(XᵀX)⁻¹f)
      model       tissue-vs-tissue residual after the landmark transform, measured
                  on the images by `serial_registration.measure_deformation`
                  (Hann-windowed cv2.phaseCorrelate over dense tissue-overlap patches)

The motivation was sound: leave-one-out landmark TRE has a hard floor at the landmark
localisation noise σ (~4 µm on real H-DAB serial sections), so a perfectly registered
pair can still read TRE ≈ 6.5 µm and fail a ≤5 µm gate. Cells are not clicked, so the
error that matters at a cell is smaller than the error at a landmark.

WHY IT DOES NOT WORK
--------------------
The model term is unmeasurable with this estimator. `measure_deformation` operates on
`structural_channel`, which is hematoxylin OD Gaussian-blurred at σ ≈ 12 µm (16 px) —
deliberately, to suppress the non-corresponding nuclei of two different sections. That
blur destroys precisely the high-frequency content needed to localise a displacement.

At the 128 px patch scale, any two patches of blurred liver parenchyma look like the
same smooth blob, so the phase-correlation peak sits at zero whatever the true offset.
The estimator therefore cannot distinguish "perfectly aligned" from "no correspondence
at all". It is not attenuated. It is blind.

MEASURED, on the real pair LL477_CD8_x10_1 vs LL477_Tim3_x10_1 (0.7519 µm/px):

    transform                       reported median residual
    certified landmark similarity            0.14 µm
    IDENTITY (~106 µm true offset)           0.22 µm      <-- must have been ~106
    uniform 48.8 µm translation              0.18 µm      <-- must have been ~48.8
    Gaussian fold, 54 µm peak                0.20 µm      <-- must have been ~54

The estimator's response to true displacement is flat. Two replacements were tested and
also rejected on the same images:

    NCC template match (96 px template, ±80 px search, TM_CCOEFF_NORMED, peak ≥ 0.5)
        reports a 27 µm median residual for the CERTIFIED transform (truth ≈ 0): a
        smooth template matches nearly anywhere inside the search window.
    phase correlation on the gradient magnitude of the structural channel
        admits zero patches — the gradient of a σ=16 px blur has std < 5.

    `lumen_tre` also fails to discriminate (4.86 µm certified vs 5.85 µm for the
    unregistered pair) and is censored by construction: its tol_um=12 µm inlier gate
    caps what it can ever report, so it measures inlier consistency, not model error.

CONSEQUENCE
-----------
Because the model term reads ≈ 0 unconditionally, cell_registration_error collapses to
the estimation term, which shrinks like 1/√n. A CERTIFIED verdict would then be
obtainable by clicking more landmarks, regardless of whether the tissue agrees — the
exact opposite of fail-closed. The cell-level gate is therefore NOT ADOPTED, and
`measure_deformation` must never drive a verdict. Leave-one-out landmark TRE, though
conservative (σ-floored), remains the gate because it is a genuine held-out measurement.

This script is a regression guard: it fails if anyone re-wires the blind estimator into
the verdict, and it re-derives the blindness on real data so the claim stays checkable.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial import serial_registration as sr  # noqa: E402

PIXEL_SIZE_UM = 0.7519
REF = os.path.expanduser("~/Desktop/assets/cd8_input/LL477_CD8_x10_1.tif")
MOV = os.path.expanduser("~/Desktop/assets/tim3 input/LL477_Tim3_x10_1.tif")


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    if not (os.path.exists(REF) and os.path.exists(MOV)):
        print("SKIP: real LL477 pair not available on this machine")
        return 0

    ref, mov = _load(REF), _load(MOV)
    h, w = ref.shape[:2]
    prop = sr.propose_landmarks(ref, mov, PIXEL_SIZE_UM, max_points=12)
    ref_pts = np.asarray(prop["ref_points"], float)
    mov_pts = np.asarray(prop["mov_points"], float)
    matrix = np.asarray(sr._fit_similarity_robust(mov_pts, ref_pts), float)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    certified = sr.measure_deformation(ref, mov, matrix, PIXEL_SIZE_UM)
    unregistered = sr.measure_deformation(ref, mov, identity, PIXEL_SIZE_UM)

    shift_px = 64
    shifted = cv2.warpAffine(mov, np.float32([[1, 0, shift_px], [0, 1, 0]]), (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    true_shift_um = shift_px * PIXEL_SIZE_UM * sr._affine_scale(matrix)
    translated = sr.measure_deformation(ref, shifted, matrix, PIXEL_SIZE_UM)

    print(f"certified transform      median residual = {certified['median_um']} µm")
    print(f"identity (~106 µm off)   median residual = {unregistered['median_um']} µm")
    print(f"uniform {true_shift_um:.1f} µm shift    median residual = "
          f"{translated['median_um']} µm")

    failures = []

    # 1. The estimator is blind: it cannot separate a good transform from no transform.
    if unregistered["median_um"] is None or unregistered["median_um"] > 10.0:
        failures.append("measure_deformation now DETECTS the unregistered pair — the "
                        "estimator changed; re-derive its calibration before trusting it")
    if translated["median_um"] is None or translated["median_um"] > 0.5 * true_shift_um:
        failures.append("measure_deformation now TRACKS a known translation — the "
                        "estimator changed; re-derive its calibration before trusting it")

    # 2. The verdict must not depend on it. Same landmarks, wildly different deformation
    #    dicts -> identical verdict and identical accuracy basis.
    honest = sr.landmark_register_and_verify(ref_pts, mov_pts, PIXEL_SIZE_UM,
                                             image_wh=(w, h))
    poisoned = sr.landmark_register_and_verify(
        ref_pts, mov_pts, PIXEL_SIZE_UM, image_wh=(w, h),
        deformation={"measured": True, "median_um": 0.0, "max_um": 0.0,
                     "region_max_um": 0.0, "n_patches": 999, "verified_frac": 1.0,
                     "capture_range_um": 48.1})
    print(f"\nverdict without deformation dict : {honest['verdict']} "
          f"(basis {honest['accuracy_basis']})")
    print(f"verdict with a zeroed deformation: {poisoned['verdict']} "
          f"(basis {poisoned['accuracy_basis']})")

    if poisoned["verdict"] != honest["verdict"]:
        failures.append("a supplied deformation dict CHANGED the verdict — the blind "
                        "patch-flow estimator is gating certification again")
    if poisoned["accuracy_basis"] != "leave_one_out_landmark_tre":
        failures.append(f"accuracy basis is {poisoned['accuracy_basis']!r}, expected "
                        f"'leave_one_out_landmark_tre' — the cell-level gate is back")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: estimator is blind (as documented) and does not drive the verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
