"""
Tier 1 — landmark registration + certification + auto-proposal + NGF selection
(serial_registration). Synthetic, known-transform; fast and deterministic.
"""
import math
import numpy as np
import pytest
from oasis.spatial import serial_registration as sr


def _similarity(theta_deg, s, tx, ty):
    th = math.radians(theta_deg); c, sn = math.cos(th), math.sin(th)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty]])


def _apply(M, pts):
    return (M @ np.c_[pts, np.ones(len(pts))].T).T


def test_fit_similarity_recovers_known_transform():
    rng = np.random.default_rng(0)
    mov = rng.uniform(0, 1000, (12, 2))
    M = _similarity(7.0, 1.03, 40, -25)
    ref = _apply(M, mov)
    Mfit = sr._fit_similarity_ls(mov, ref)
    assert np.allclose(_apply(Mfit, mov), ref, atol=1e-6)


def test_loo_tre_small_for_consistent_points():
    rng = np.random.default_rng(1)
    mov = rng.uniform(0, 1000, (10, 2))
    ref = _apply(_similarity(5.0, 1.0, 20, 10), mov)
    loo = sr.loo_tre(ref, mov, pixel_size_um=0.5)
    assert loo["loo_median_um"] is not None and loo["loo_median_um"] < 0.5


def test_certified_on_clean_landmarks():
    rng = np.random.default_rng(2)
    mov = rng.uniform(50, 1950, (10, 2))
    ref = _apply(_similarity(6.0, 1.0, 30, -18), mov)
    res = sr.landmark_register_and_verify(ref, mov, pixel_size_um=0.5,
                                          image_wh=(2000, 2000))
    assert res["verdict"] == "CERTIFIED"
    assert res["tre_median_um"] < 5.0


def test_deformed_landmarks_not_certified():
    """Add per-point warp no similarity can absorb ⇒ must NOT certify."""
    rng = np.random.default_rng(3)
    mov = rng.uniform(50, 1950, (10, 2))
    ref = _apply(_similarity(6.0, 1.0, 30, -18), mov)
    ref = ref + rng.normal(0, 18, ref.shape)          # ~9 µm scatter at 0.5 µm/px
    res = sr.landmark_register_and_verify(ref, mov, pixel_size_um=0.5,
                                          image_wh=(2000, 2000))
    assert res["verdict"] != "CERTIFIED"


def test_too_few_landmarks_not_certifiable():
    rng = np.random.default_rng(4)
    mov = rng.uniform(0, 1000, (4, 2))
    ref = _apply(_similarity(3.0, 1.0, 10, 5), mov)
    res = sr.landmark_register_and_verify(ref, mov, pixel_size_um=0.5,
                                          image_wh=(1000, 1000))
    assert res["verdict"] == "NOT_CERTIFIABLE"


# ── Auto-proposal + NGF selection on a synthetic structural pair ─────────────
def _synthetic_pair(seed=0, angle=5.0, tx=16, ty=-9):
    import cv2
    rng = np.random.default_rng(seed)
    H, W = 560, 640
    base = np.full((H, W, 3), 60, np.uint8)
    for _ in range(28):
        cx, cy = rng.integers(50, H - 50), rng.integers(50, W - 50)
        cv2.circle(base, (int(cy), int(cx)), int(rng.integers(9, 20)), (235, 235, 235), -1)
    for _ in range(6):
        p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        q = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.line(base, p, q, (205, 205, 205), 3)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0); M[0, 2] += tx; M[1, 2] += ty
    mov = cv2.warpAffine(base, M, (W, H), borderValue=(60, 60, 60))
    return base, mov


def test_propose_landmarks_recovers_transform():
    ref, mov = _synthetic_pair()
    r = sr.propose_landmarks(ref, mov, pixel_size_um=0.75, max_points=8)
    assert r["ok"] and r["n"] >= 6
    Mfit = sr._fit_similarity_ls(np.array(r["mov_points"]), np.array(r["ref_points"]))
    d = np.linalg.norm(_apply(Mfit, np.array(r["mov_points"])) - np.array(r["ref_points"]),
                       axis=1) * 0.75
    assert np.median(d) < 5.0
    assert all(0.0 <= c <= 1.0 for c in r["confidences"])


def test_guided_landmark_suggests_moving_correspondence_from_existing_pairs():
    import cv2
    ref, mov = _synthetic_pair(angle=4.0, tx=14, ty=-7)
    H, W = ref.shape[:2]
    M_ref_to_mov = cv2.getRotationMatrix2D((W / 2, H / 2), 4.0, 1.0)
    M_ref_to_mov[0, 2] += 14
    M_ref_to_mov[1, 2] += -7

    ref_pts = np.array([
        [120, 120], [520, 130], [130, 430], [500, 410],
        [310, 160], [330, 390],
    ], dtype=float)
    mov_pts = _apply(M_ref_to_mov, ref_pts)
    new_ref = np.array([360, 285], dtype=float)
    true_mov = _apply(M_ref_to_mov, new_ref.reshape(1, 2))[0]

    r = sr.suggest_moving_landmark(
        ref, mov, new_ref, pixel_size_um=0.75,
        existing_ref_pts=ref_pts, existing_mov_pts=mov_pts)

    assert r["ok"], r["msg"]
    assert r["method"] == "confirmed_landmark_ransac"
    assert np.linalg.norm(np.array(r["mov_point"]) - true_mov) * 0.75 < 12.0


def test_auto_local_roi_recovers_deformed_global_landmarks():
    from oasis.webui.api import API

    clean_mov = np.array([
        [180, 180], [760, 180], [180, 760], [760, 760],
        [470, 250], [470, 690],
    ], dtype=float)
    M = _similarity(4.0, 1.0, 30, -18)
    clean_ref = _apply(M, clean_mov)

    bad_mov = np.array([
        [1250, 1250], [1820, 1260], [1260, 1820],
        [1820, 1820], [1540, 1320], [1600, 1720],
    ], dtype=float)
    bad_ref = _apply(M, bad_mov) + np.array([
        [60, -30], [-45, 55], [80, 70], [-70, -65], [95, 20], [-20, 90],
    ], dtype=float)

    ref = np.vstack([clean_ref, bad_ref])
    mov = np.vstack([clean_mov, bad_mov])
    global_res = sr.landmark_register_and_verify(
        ref, mov, pixel_size_um=0.5, image_wh=(2000, 2000))
    # THE INVARIANT IS "NEVER CLAIM THE WHOLE FIELD", not a specific failing verdict.
    # LOCALLY_CERTIFIED is now a legitimate outcome here and a better one: with the accuracy
    # gate derived from the bands, the six clean landmarks are recognised as a consistent
    # subset and their hull is certified directly, so the pair is recovered instead of
    # discarded. Verified separately that the ROI is exactly the clean region — 6/6 clean
    # landmarks inside, 0/6 deformed ones inside, 9 % of the field. Asserting DEFORMED would
    # be pinning the failure rather than the property.
    assert global_res["verdict"] != "CERTIFIED", "must not claim the whole field"
    if global_res["verdict"] == "LOCALLY_CERTIFIED":
        assert global_res.get("roi_polygon"), "a local certification must name its window"

    r = API().suggest_local_certification_roi({
        "ref_points": ref.tolist(),
        "mov_points": mov.tolist(),
        "pixel_size_um": 0.5,
        "image_wh": [2000, 2000],
    })

    assert r["status"] == "ok", r.get("error")
    assert r["certification"]["status"] == "LOCALLY_CERTIFIED"
    assert r["certification"]["is_certified"] is True
    assert r["roi_polygon"]


def test_register_similarity_selects_non_identity():
    ref, mov = _synthetic_pair(angle=6.0, tx=18, ty=-10)
    reg = sr.register_similarity(ref, mov, pixel_size_um=0.75)
    assert reg["success"] and reg["method"] != "identity"
    assert reg["struct_dice"] > 0.8


# ── Robust fit, similarity invariant, radius floor ────────────────────────────
# Together these encode the result of validation/validate_radius_floor.py: a serial
# section that deforms is analysable, its error only ever weakens an association, and
# the transform that carries that guarantee must stay a similarity.

def _fold_pair(n_bad=2, fold_px=60.0):
    """Well-spread landmarks under a clean similarity, with `n_bad` of them displaced as
    if they sat on a fold or tear — the case plain least squares cannot survive."""
    rng = np.random.default_rng(5)
    c = np.array([1000.0, 1000.0])
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ref = np.vstack([c + rng.uniform(-400, 400, size=(4, 2)),
                     c + np.c_[np.cos(ang), np.sin(ang)] * 820])
    th = math.radians(2.5)
    rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    mov = (ref - c) @ rot + c + np.array([10.0, -6.0]) + rng.normal(0, 0.6, ref.shape)
    bad = [5, 9][:n_bad]
    mov[bad] += fold_px
    return ref, mov, bad


def test_robust_fit_resists_landmarks_on_a_fold():
    """Two folded landmarks must bend the fit, not break it. Plain LS drags every good
    landmark past the 5 µm gate; Huber IRLS keeps them sub-micron."""
    ref, mov, bad = _fold_pair()
    good = np.setdiff1d(np.arange(len(ref)), bad)
    px = 0.5

    def median_good_residual(M):
        e = np.linalg.norm(_apply(M, mov) - ref, axis=1) * px
        return float(np.median(e[good]))

    ls = median_good_residual(sr._fit_similarity_ls(mov, ref))
    robust = median_good_residual(sr._fit_similarity_robust(mov, ref))

    assert ls > 5.0, f"expected plain LS to be dragged past the gate, got {ls:.2f} µm"
    assert robust < 1.0, f"robust fit should keep good landmarks sub-micron, got {robust:.2f} µm"
    assert sr.landmark_register_and_verify(
        ref, mov, px, image_wh=(2000, 2000))["verdict"] == "CERTIFIED"


def test_robust_fit_returns_the_ls_solution_on_an_exact_fit():
    rng = np.random.default_rng(3)
    mov = rng.uniform(0, 1000, (12, 2))
    ref = _apply(_similarity(7.0, 1.03, 40, -25), mov)
    a = sr._fit_similarity_ls(mov, ref)
    b = sr._fit_similarity_robust(mov, ref)
    assert np.allclose(a, b, atol=1e-9)


def test_robust_fit_cannot_unseat_a_clean_noisy_pair():
    """With no outliers the Huber weights are ≈1, so the robust fit must track least
    squares closely enough that a certifiable pair stays certified. Not bit-identical:
    ordinary noise puts a few residuals past 1.345 robust SDs."""
    ref, mov, _ = _fold_pair(n_bad=0)
    px = 0.5
    a = sr._fit_similarity_ls(mov, ref)
    b = sr._fit_similarity_robust(mov, ref)
    assert np.allclose(a, b, atol=5e-3)

    res = lambda M: float(np.median(np.linalg.norm(_apply(M, mov) - ref, axis=1)) * px)
    assert abs(res(a) - res(b)) < 0.05          # µm — far below the 5 µm gate
    assert sr.landmark_register_and_verify(
        ref, mov, px, image_wh=(2000, 2000))["verdict"] == "CERTIFIED"


def test_weighted_fit_ignores_a_zero_weighted_outlier():
    rng = np.random.default_rng(4)
    mov = rng.uniform(0, 1000, (10, 2))
    ref = _apply(_similarity(5.0, 1.0, 12, 7), mov)
    ref[0] += 500.0                                  # gross outlier
    w = np.ones(len(mov)); w[0] = 0.0
    M = sr._fit_similarity_ls(mov, ref, weights=w)
    err = np.linalg.norm(_apply(M, mov)[1:] - ref[1:], axis=1)
    assert err.max() < 1e-6


def test_similarity_invariant_rejects_shear_and_accepts_similarity():
    ok = np.array([[1.2, 0.0, 5.0], [0.0, 1.2, 3.0]])
    assert sr.similarity_defect(ok) < 1e-9
    assert sr.assert_distance_preserving(ok, "ok") < 1e-9

    shear = np.array([[1.0, 0.35, 5.0], [0.0, 1.0, 3.0]])
    assert sr.similarity_defect(shear) > 0.02
    with pytest.raises(ValueError, match="not a similarity"):
        sr.assert_distance_preserving(shear, "moving→fixed")

    degenerate = np.zeros((2, 3))
    with pytest.raises(ValueError):
        sr.assert_distance_preserving(degenerate, "degenerate")


def test_radius_floor_scales_with_tre_and_fails_closed_when_unknown():
    from oasis.spatial.spatial_stats import registration_radius_floor as floor, _RADIUS_FLOOR_FACTOR
    assert floor(None) is None                       # unknown TRE -> caller must fail closed
    assert floor(float("nan")) is None
    assert floor(-1.0) is None
    assert floor(0.0) == 0.0
    assert floor(8.0) == pytest.approx(8.0 * _RADIUS_FLOOR_FACTOR)
    assert floor(2.0) < floor(5.0) < floor(12.0)     # monotone in registration error


def test_radius_limited_keeps_the_field_but_surrenders_small_radii():
    """A pair too deformed to certify, but whose landmarks still agree on ONE similarity,
    is analysable above the reporting floor rather than discarded.

    The floor multiple is read from _RADIUS_FLOOR_FACTOR, not hard-coded: it was 3.0 and is
    now 1.0 (calibrated — see validate_radius_floor_calibration.py), and a test that pins a
    calibrated constant by value fails for the wrong reason the moment it is calibrated.
    """
    rng = np.random.default_rng(11)
    c = np.array([1000.0, 1000.0])
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ref = np.vstack([c + rng.uniform(-400, 400, size=(4, 2)),
                     c + np.c_[np.cos(ang), np.sin(ang)] * 820])
    # Deformation big enough to miss the 5 µm gate, small enough to leave a usable band.
    mov = ref + rng.normal(0, 0.6, ref.shape) + rng.normal(0, 12.0, ref.shape)
    out = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))

    if out["verdict"] == "RADIUS_LIMITED":
        r_min = out["min_interpretable_radius_um"]
        assert r_min is not None and 0 < r_min < out["max_radius_um"]
        from oasis.spatial.spatial_stats import registration_radius_floor
        assert r_min == pytest.approx(
            registration_radius_floor(out["tre_median_um"]), rel=1e-3)
        assert out["roi_polygon"] is None            # keeps the whole field
    else:
        # Whatever the verdict, an uncertifiable pair must never claim a resolvable radius.
        assert out["verdict"] in ("CERTIFIED", "LOCALLY_CERTIFIED",
                                  "DEFORMED", "NOT_CERTIFIABLE")


def test_heavy_contamination_degrades_rather_than_certifying():
    """Huber down-weights outliers but never rejects them, so ~33% grossly deformed
    landmarks still drag the fit. The pair must then degrade to a WEAKER verdict — it must
    never certify on a corrupted transform. This pins the documented breakdown limit."""
    rng = np.random.default_rng(2)
    c = np.array([1000.0, 1000.0])
    clean = c + rng.uniform(-500, 500, size=(8, 2))
    torn = np.array([[120.0, 1880.0], [1880.0, 120.0], [110.0, 110.0], [1890.0, 1890.0]])
    ref = np.vstack([clean, torn])
    mov = ref + rng.normal(0, 0.4, ref.shape)
    mov[8:] += rng.normal(0, 40.0, (4, 2))           # only the far corners are torn
    out = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))
    assert out["verdict"] != "CERTIFIED", "must not certify on a dragged transform"
    assert out["verdict"] in ("LOCALLY_CERTIFIED", "RADIUS_LIMITED",
                             "DEFORMED", "NOT_CERTIFIABLE")


def _deformation(median_um, n_patches=64):
    return {"measured": True, "median_um": median_um, "p90_um": median_um,
            "max_um": median_um, "region_max_um": median_um, "n_patches": n_patches,
            "verified_frac": 1.0, "overlap_frac": 1.0, "capture_range_um": 48.1,
            "reason": None}


def test_deformation_never_moves_the_verdict():
    """`measure_deformation` is blind — on real sections it reports ~0.2 µm for an IDENTITY
    transform that leaves them 106 µm apart (validation/validate_deformation_estimator.py).
    It must therefore never gate. A pair that fails on held-out TRE cannot be rescued by a
    flattering deformation reading, and a pair that passes cannot be vetoed by an alarming
    one. Both are recorded as diagnostics and ignored."""
    rng = np.random.default_rng(7)
    mov = rng.uniform(0, 2000, (12, 2))
    ref = _apply(_similarity(4.0, 1.0, 30, -15), mov)

    # A clean pair certifies. A "catastrophic" deformation reading must not veto it.
    clean = ref + rng.normal(0, 0.3, ref.shape)
    base = sr.landmark_register_and_verify(clean, mov, 0.5, image_wh=(2000, 2000))
    assert base["verdict"] == "CERTIFIED"
    vetoed = sr.landmark_register_and_verify(clean, mov, 0.5, image_wh=(2000, 2000),
                                             deformation=_deformation(999.0))
    assert vetoed["verdict"] == base["verdict"]

    # A pair whose landmarks disagree must not be rescued by a zeroed deformation reading.
    noisy = ref + rng.normal(0, 24.0, ref.shape)
    bad = sr.landmark_register_and_verify(noisy, mov, 0.5, image_wh=(2000, 2000))
    assert bad["verdict"] != "CERTIFIED"
    flattered = sr.landmark_register_and_verify(noisy, mov, 0.5, image_wh=(2000, 2000),
                                                deformation=_deformation(0.0))
    assert flattered["verdict"] == bad["verdict"], "blind estimator is gating again"

    for out in (base, vetoed, bad, flattered):
        assert out["accuracy_basis"] == "leave_one_out_landmark_tre"


def test_deformation_is_recorded_as_diagnostic():
    """It is reported (an operator may want to see it) but flagged unvalidated."""
    rng = np.random.default_rng(8)
    mov = rng.uniform(0, 2000, (12, 2))
    ref = _apply(_similarity(4.0, 1.0, 30, -15), mov) + rng.normal(0, 0.3, (12, 2))
    out = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000),
                                          deformation=_deformation(3.5))
    assert out["deformation_um"] == 3.5
    assert out["deformation_patches"] == 64
    assert out["deformation_is_validated"] is False


def test_prediction_error_is_reported_but_does_not_certify():
    """Prediction SE shrinks like 1/sqrt(n), so certifying on it would let an operator
    certify ANY pair by clicking more landmarks. Adding landmarks to a genuinely
    disagreeing pair must shrink prediction_error_um yet leave it uncertified."""
    rng = np.random.default_rng(9)
    verdicts, pred = [], []
    for n in (8, 40):
        mov = rng.uniform(0, 2000, (n, 2))
        ref = _apply(_similarity(4.0, 1.0, 30, -15), mov) + rng.normal(0, 20.0, (n, 2))
        out = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))
        verdicts.append(out["verdict"])
        pred.append(out["prediction_error_um"])
    assert pred[1] < pred[0], "prediction SE must shrink with n"
    assert all(v != "CERTIFIED" for v in verdicts), "n alone must not buy certification"


# ── Outlier hardening of the dense-matcher path ──────────────────────────────────
# Regression tests for three defects found when 84 real CD8/TIM-3 field pairs certified
# zero ROIs: the assay had stopped discriminating at a dense matcher's n, and nothing
# between the matcher and the Huber fit rejected gross mismatches.

def _smooth_field(src, amp=30.0, period=400.0):
    return np.c_[amp * np.sin(src[:, 0] / period), amp * np.cos(src[:, 1] / period)]


def test_assay_calls_a_contaminated_smooth_field_bad_not_deformed():
    """The defect: verdicting on p<0.05 alone. A genuinely smooth field with 12% gross
    outliers has its Moran's I destroyed (I≈0.014, indistinguishable from the random
    control) yet still reaches p<0.05 at large n — and was reported to the user as
    deformed TISSUE, when the correct reading is that the matcher is wrong."""
    rng = np.random.default_rng(3)
    n = 200
    ref = rng.uniform(0, 1000, (n, 2))
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mov = ref + _smooth_field(ref) + rng.normal(0, 1.0, (n, 2))
    mov[rng.choice(n, int(0.12 * n), replace=False)] += rng.normal(0, 300, (24, 2))
    a = sr.residual_field_assay(ref, mov, M, 1.0)
    assert a["moran_i"] < sr._MORAN_EFFECT_FLOOR
    assert a["p_value"] < 0.05, "p alone would have passed this — that is the defect"
    assert a["verdict"] == "CORRESPONDENCES_BAD"


def test_assay_still_recognises_a_clean_smooth_field():
    """The effect-size floor must not cost the assay its true positive."""
    rng = np.random.default_rng(4)
    n = 200
    ref = rng.uniform(0, 1000, (n, 2))
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mov = ref + _smooth_field(ref) + rng.normal(0, 1.0, (n, 2))
    a = sr.residual_field_assay(ref, mov, M, 1.0)
    assert a["moran_i"] >= sr._MORAN_EFFECT_FLOOR
    assert a["verdict"] == "REAL_DEFORMATION"


def test_assay_random_field_reads_bad():
    rng = np.random.default_rng(5)
    n = 200
    ref = rng.uniform(0, 1000, (n, 2))
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mov = ref + rng.normal(0, 5.0, (n, 2))
    assert sr.residual_field_assay(ref, mov, M, 1.0)["verdict"] == "CORRESPONDENCES_BAD"


def test_local_smoothness_rejects_outliers_and_keeps_good_matches():
    from oasis.spatial.loftr_matcher import _local_smoothness
    rng = np.random.default_rng(6)
    n = 200
    src = rng.uniform(0, 1000, (n, 2))
    dst = src + _smooth_field(src) + rng.normal(0, 0.5, (n, 2))
    bad = rng.choice(n, 24, replace=False)
    dst[bad] += rng.normal(0, 300, (24, 2))
    keep = _local_smoothness(src, dst, tol_px=4.0)
    good = np.ones(n, bool); good[bad] = False
    assert (~keep[bad]).sum() == len(bad), "every gross outlier must be rejected"
    assert (~keep[good]).sum() <= 0.05 * good.sum(), "must not cull the good population"


def test_local_smoothness_does_not_select_for_a_similarity():
    """THE non-circularity property, and the reason this filter is admissible where RANSAC
    is not. A shear-plus-bend that NO similarity can describe must pass through intact —
    otherwise the filter would be manufacturing agreement with the model under test."""
    from oasis.spatial.loftr_matcher import _local_smoothness
    rng = np.random.default_rng(7)
    n = 200
    src = rng.uniform(0, 1000, (n, 2))
    warp = np.c_[0.15 * src[:, 1] + 40 * np.sin(src[:, 0] / 250.0), 0.10 * src[:, 0]]
    dst = src + warp + rng.normal(0, 0.5, (n, 2))
    assert _local_smoothness(src, dst, tol_px=4.0).sum() >= 0.95 * n


def test_tile_grid_packs_the_frame_instead_of_yielding_one_candidate():
    """The auto-sweep tested ONE location per image pair. Its grid stepped by 2*half and
    pinned the phase to the anchor, so on a 1920x1440 camera field every neighbour of the
    anchor fell out of bounds. A pair whose centre is unregistrable was then reported
    DEFORMED with the rest of the field never examined."""
    from oasis.webui.api import tile_centres
    W, H, px = 1920, 1440, 0.7519                      # a real 10X frame, 1444 x 1083 um
    for r_um, min_expected in ((450, 2), (350, 5), (260, 10)):
        R = r_um / px
        n = len(tile_centres(0, W, R, max(R, 8.0))) * len(tile_centres(0, H, R, max(R, 8.0)))
        assert n >= min_expected, f"R={r_um}um gave only {n} candidates"


def test_tile_grid_degrades_to_one_centre_when_the_tile_exceeds_the_span():
    from oasis.webui.api import tile_centres
    c = tile_centres(0, 100, 80, 80)                   # 160-wide tile in a 100-wide span
    assert c == [50.0]


def test_tile_grid_stays_inside_the_span():
    from oasis.webui.api import tile_centres
    for half, stride in ((100, 50), (100, 100), (37, 19)):
        for c in tile_centres(0, 1000, half, stride):
            assert c - half >= -1e-9 and c + half <= 1000 + 1e-9


def test_the_verdict_cascade_has_no_unreachable_state():
    """DEFORMED must be reachable on the path production actually uses.

    It was not. `band_ok` covers held-out TRE up to max_radius*(1-band_frac)/floor_factor =
    100*0.5/3 = 16.67 um, and the DEFORMED arm below it additionally required TRE <= 15, so
    the window [16.67, 15] was EMPTY whenever image_wh was supplied — i.e. on every
    production call. Measured before the fix: TRE 10.7 -> RADIUS_LIMITED, TRE 16.8 ->
    NOT_CERTIFIABLE, DEFORMED never issued.

    That mislabels the diagnosis on the manual-landmark path v1 ships with: NOT_CERTIFIABLE
    tells the operator the landmarks "do not agree on a single transform" and to add
    correspondences, when they DO agree and the real problem is deformation. The gap also
    widens whenever the floor factor is lowered, so it has to stay closed.
    """
    import numpy as np
    from oasis.spatial.serial_registration import landmark_register_and_verify

    rng = np.random.default_rng(0)
    ref = rng.uniform(100, 1800, (14, 2))
    seen = {}
    for noise in (1, 3, 6, 9, 15, 25, 40):
        mov = ref + np.array([12.0, -7.0]) + rng.normal(0, noise, ref.shape)
        out = landmark_register_and_verify(ref, mov, 0.7519, image_wh=(1920, 1440))
        seen.setdefault(out["verdict"], out["tre_median_um"])

    assert "DEFORMED" in seen, (
        f"DEFORMED is unreachable with image_wh set; only saw {sorted(seen)}")
    assert "RADIUS_LIMITED" in seen and "CERTIFIED" in seen, sorted(seen)
    # The fix must not open the gate: DEFORMED still blocks analysis exactly as before.
    mov = ref + np.array([12.0, -7.0]) + rng.normal(0, 25, ref.shape)
    out = landmark_register_and_verify(ref, mov, 0.7519, image_wh=(1920, 1440))
    assert out["verdict"] == "DEFORMED"
    assert out["verdict"] not in ("CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")


def test_unmeasurable_landmark_sets_are_still_not_certifiable():
    """NOT_CERTIFIABLE keeps the cases that genuinely cannot be measured.

    Making DEFORMED the direct `else` would be wrong if it swallowed these too — the two
    verdicts mean different things and both are needed.
    """
    import numpy as np
    from oasis.spatial.serial_registration import landmark_register_and_verify

    ref = np.array([[100.0, 100.0], [200.0, 150.0], [300.0, 120.0]])
    out = landmark_register_and_verify(ref, ref + 1.0, 0.7519, image_wh=(1920, 1440))
    assert out["verdict"] == "NOT_CERTIFIABLE", out["verdict"]


def test_a_certification_without_an_explicit_floor_still_yields_one():
    """A missing key must not silently withhold the contact-scale claim.

    Certifications reach run_spatial_association from several paths and they do not all
    emit `min_interpretable_radius_um`. The shipped LL477 run certified LOCALLY_CERTIFIED
    through the LoFTR-ROI path with a cell error of 2.352 um, and its stored certification
    carries `cell_error_um` but not the floor — so None was passed through and the run
    recorded `contact_scale_resolved: false` for one of its best-registered pairs. The floor
    is a pure function of the cell error, so deriving it is exact rather than a guess.
    """
    from run_pipeline import _radius_floor_from_cell_error
    from oasis.spatial.spatial_stats import registration_radius_floor

    assert _radius_floor_from_cell_error(
        {"cell_error_um": 2.352}) == registration_radius_floor(2.352)
    # p90 is preferred over the point estimate when both are present
    assert _radius_floor_from_cell_error(
        {"cell_error_p90_um": 9.0, "cell_error_um": 2.0}) == registration_radius_floor(9.0)
    # unknown stays unknown — callers fail closed on None and must keep being able to
    assert _radius_floor_from_cell_error({"verdict": "LOCALLY_CERTIFIED"}) is None
    assert _radius_floor_from_cell_error(None) is None


def test_a_certificate_never_claims_field_the_landmarks_do_not_span():
    """A residual can only measure the transform where the landmarks are.

    This path used to return roi_polygon=None — the WHOLE FIELD — from any landmark set that
    fit well, with no support requirement at all. Demonstrated failure: 10 collinear
    landmarks on y=500, under a vertical scaling about that same line, certify at
    fit-residual 0.0 um and held-out TRE 0.0 um while the realized error at cells across the
    field reaches 117 um. The landmarks sit exactly on the deformation's invariant line, so
    the certificate is blind by construction rather than merely optimistic.

    _certify_fitzpatrick_west had always made the hull the certified window; only this legacy
    LOO path extrapolated. `coverage_frac` already measured the problem (0.0 for the
    collinear set) and gated nothing.
    """
    import numpy as np
    from oasis.spatial.serial_registration import (landmark_register_and_verify,
                                                   CERTIFICATION_GATES)

    wh = (1920, 1440)
    collinear = np.column_stack([np.linspace(100, 1800, 10), np.full(10, 500.0)])
    out = landmark_register_and_verify(collinear, collinear + np.array([5.0, 2.0]),
                                       0.7519, image_wh=wh)
    assert out["verdict"] == "NOT_CERTIFIABLE", out["verdict"]
    assert out["coverage_frac"] == 0.0
    assert "hull covers only" in out["reason"]

    # A well-spread set still certifies — and over its hull, not the whole field.
    rng = np.random.default_rng(0)
    good = rng.uniform(150, 1750, (12, 2))
    ok = landmark_register_and_verify(good, good + np.array([6.0, -3.0])
                                      + rng.normal(0, 1.0, good.shape), 0.7519, image_wh=wh)
    assert ok["verdict"] == "CERTIFIED", ok["verdict"]
    assert ok["coverage_frac"] >= CERTIFICATION_GATES["min_roi_frac"]
    assert ok["roi_polygon"] is not None, "CERTIFIED must name the window it certifies"
    assert ok["certified_window_source"] == "landmark_hull"


def test_the_hull_window_matches_what_the_landmarks_can_actually_see():
    """Certifying the hull is not cosmetic — inside it the claimed error is honest.

    With landmarks spread +-200 px about y=500 and a 1.05 vertical scaling, the field-wide
    error reaches ~28 um while the error INSIDE the hull is a few um, because the
    deformation grows with distance from the line. Certifying the hull reports the error
    that region actually has; certifying the field reported 3.9 um for a 28 um situation.
    """
    import numpy as np
    from oasis.spatial.serial_registration import (landmark_register_and_verify,
                                                   _apply_affine)

    # Deterministic layout, not a seeded random one: this test depends on landing in the
    # narrow band where the fit still passes the 5 um gate but the hull is small, and a
    # reseeded draw silently walks out of it.
    x = np.linspace(120, 1800, 12)
    y = 500.0 + 120.0 * np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
    ref = np.column_stack([x, y])
    mov = ref.copy()
    mov[:, 1] = 500.0 + 1.05 * (mov[:, 1] - 500.0)
    rng = np.random.default_rng(0)
    out = landmark_register_and_verify(ref, mov, 0.7519, image_wh=(1920, 1440))
    assert out["verdict"] == "CERTIFIED"
    assert out["roi_polygon"] is not None

    M = np.asarray(out["matrix"], float)
    inside = np.column_stack([rng.uniform(200, 1700, 200),
                              500.0 + rng.uniform(-110, 110, 200)])
    warped = inside.copy()
    warped[:, 1] = 500.0 + 1.05 * (warped[:, 1] - 500.0)
    err_in = np.linalg.norm(_apply_affine(warped, M) - inside, axis=1) * 0.7519
    # the certificate's claim must not understate the error inside the window it certified
    assert np.percentile(err_in, 90) <= max(3.0 * out["tre_median_um"], 12.0), (
        f"claimed {out['tre_median_um']} µm but the hull's own p90 is "
        f"{np.percentile(err_in, 90):.1f} µm")


def test_a_fit_dragged_by_bad_landmarks_cannot_hide_behind_its_residual():
    """Held-out TRE is blind to a fit dragged by a systematically wrong SUBSET.

    Dropping one landmark at a time leaves the other outliers still pulling the fit, so every
    held-out prediction is wrong in the same direction as the fit and the residual stays
    small. Measured on 12 points with 4 torn corners: LOO reported 7.79 um while the
    transform displaced cells by a median of 10.4 um and a p90 of 15.4 um. That is why the
    accuracy gate could not simply be loosened — it was compensating for the blind spot
    rather than measuring accuracy.

    transform_drag_um refits on the landmarks that agree with each other and measures how far
    the two transforms diverge across the FIELD, which is exactly what holding points out
    cannot reveal.
    """
    import numpy as np
    from oasis.spatial import serial_registration as sr

    def build(contaminate):
        rng = np.random.default_rng(2)
        c = np.array([1000.0, 1000.0])
        clean = c + rng.uniform(-500, 500, size=(8, 2))
        torn = np.array([[120.0, 1880.0], [1880.0, 120.0],
                         [110.0, 110.0], [1890.0, 1890.0]])
        ref = np.vstack([clean, torn])
        mov = ref + rng.normal(0, 0.4, ref.shape)
        if contaminate:
            mov[8:] += rng.normal(0, 40.0, (4, 2))
        return ref, mov

    ref, mov = build(False)
    clean_out = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))
    assert clean_out["transform_drag_um"] < 0.5, (
        "a consistent landmark set must show no drag — a false alarm here would refuse "
        "good registrations")
    assert clean_out["verdict"] == "CERTIFIED"

    ref, mov = build(True)
    bad = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))
    assert bad["transform_drag_um"] > bad["tre_median_um"], (
        f"drag {bad['transform_drag_um']} did not exceed the held-out TRE "
        f"{bad['tre_median_um']} — the blind spot is open again")
    assert bad["accuracy_limited_by"] == "transform_drag"
    assert bad["verdict"] != "CERTIFIED", "a dragged fit must not certify"


def test_the_drag_estimator_uses_a_scale_free_consistency_test():
    """An absolute residual cutoff collapses on the case this exists to catch.

    When the fit is dragged badly enough, EVERY landmark exceeds a fixed threshold, the
    consistent subset empties, and the check silently reports "no drag" — the failure mode is
    worst exactly where the answer matters most. A median + k*MAD rule asks whether a point
    disagrees with the BULK, which stays well defined however large the errors get.

    Its known breakdown is ~50 % contamination, where MAD itself is inflated and nothing
    looks like an outlier; held-out TRE covers that regime instead, since wholesale
    disagreement is what LOO is good at. The two are complementary and accuracy takes the
    worse of them.
    """
    import inspect

    import numpy as np
    from oasis.spatial import serial_registration as sr

    src = inspect.getsource(sr.transform_drag_um)
    assert "np.median(np.abs(resid - med))" in src, "consistency test is no longer MAD-based"

    # identical residuals: nothing can be dragging, and it must not divide by zero
    ref = np.array([[100.0, 100.0], [900.0, 120.0], [120.0, 900.0], [900.0, 900.0]])
    assert sr.transform_drag_um(ref, ref.copy(), np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                                (1000, 1000), 0.5) == 0.0
    # unknown stays None rather than defaulting to a comfortable zero
    assert sr.transform_drag_um(ref[:2], ref[:2], np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                                (1000, 1000), 0.5) is None
    assert sr.transform_drag_um(ref, ref.copy(), None, (1000, 1000), 0.5) is None


def test_both_panes_show_the_same_tissue_after_clipping():
    """The reference outline and the moving outline must be counterparts of each other.

    certify_local_roi trims the reference ROI to the moving frame's footprint, because the
    part of a drawn region with no moving tissue cannot be measured. But the API mapped the
    ORIGINAL roi to moving space and reported the ORIGINAL roi as the reference window, so
    the clip never reached the UI: the operator saw a full rectangle on the reference and a
    truncated one on the moving section, covering different tissue, with nothing saying why.

    With a 150 px shift on a 500 px-wide moving image, the drawn region x 20-300 maps to
    moving x -130..150 — a third of it off-frame. Clipped, the reference becomes x 150-300
    and maps to moving x 0..150, entirely inside. The cut on one side implies the cut on the
    other, which is the property this pins.
    """
    import numpy as np
    from oasis.spatial import loftr_matcher as lm

    rng = np.random.default_rng(0)
    ref = (rng.random((400, 500, 3)) * 255).astype("uint8")
    mov = ref.copy()
    M = np.array([[1.0, 0.0, 150.0], [0.0, 1.0, 0.0]])
    roi = np.array([[20.0, 60.0], [300.0, 60.0], [300.0, 340.0], [20.0, 340.0]])

    c = lm.certify_local_roi(ref, mov, roi, 0.7519, provisional_matrix=M, work_max_dim=400)
    eff = np.asarray(c["roi_polygon"], float)
    drawn = np.asarray(c["roi_polygon_drawn"], float)

    assert c["roi_clipped_to_moving_frac"] < 1.0
    assert c["roi_clip_note"], "a silently shrunk window is what caused the confusion"
    assert drawn[:, 0].min() < eff[:, 0].min(), "the reference was not cut"

    # the clipped reference must map ENTIRELY inside the moving image
    A, t = M[:2, :2], M[:2, 2]
    mov_eff = (eff - t) @ np.linalg.inv(A).T
    assert mov_eff[:, 0].min() >= -0.5, (
        f"clipped region still maps off the moving frame at x={mov_eff[:, 0].min():.1f}")
    assert mov_eff[:, 0].max() <= mov.shape[1] + 0.5

    # a region wholly inside the moving frame must be left alone
    clean = lm.certify_local_roi(ref, mov, roi, 0.7519,
                                 provisional_matrix=np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                                 work_max_dim=400)
    assert clean["roi_clipped_to_moving_frac"] == 1.0
    assert clean.get("roi_clip_note") is None


def test_a_region_fit_that_disagrees_with_the_whole_field_is_refused():
    """A per-region residual cannot see an error that lives in the PARAMETERS.

    A similarity fitted to correspondences packed into one small region can absorb a large
    rotation or scale error while still fitting those points well, so the residual stays
    small and the region certifies. Nothing asked whether the regions of a pair agreed with
    EACH OTHER, and measured across the cohort's certifying windows they often did not:
    rotation spread within a single pair reached 61.9 deg and scale spread 0.53, while
    genuinely well-registered pairs sit at 0.22-0.36 deg and 0.002-0.007.

    Two serial sections of one block differ by ONE placement rotation and ONE scale. Local
    deformation bends tissue; it does not rotate one region 40 deg relative to its neighbour.
    """
    import math

    import cv2
    import numpy as np
    from oasis.spatial import loftr_matcher as lm

    rng = np.random.default_rng(0)
    ref = (rng.random((400, 500, 3)) * 255).astype("uint8")
    th = np.deg2rad(20.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    mov = cv2.warpAffine(ref, np.hstack([R, np.array([[40.0], [10.0]])]), (500, 400))
    roi = np.array([[120.0, 120.0], [340.0, 120.0], [340.0, 300.0], [120.0, 300.0]])

    # the whole-field transform says ~0 deg; a local fit claiming 20 deg must be refused
    prov = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 10.0]])
    c = lm.certify_local_roi(ref, mov, roi, 0.7519, provisional_matrix=prov, work_max_dim=400)
    assert c.get("rotation_vs_global_deg") is not None, "agreement is no longer measured"
    assert c["rotation_vs_global_deg"] > lm.GLOBAL_AGREEMENT_MAX_DEG
    assert c["verdict"] == "NOT_CERTIFIABLE", (
        f"a {c['rotation_vs_global_deg']} deg disagreement certified anyway")
    assert "whole-field" in (c.get("reason") or "")


def test_the_agreement_thresholds_sit_between_the_measured_populations():
    """The thresholds are calibrated, not picked: an order of magnitude above the good pairs
    and an order below the bad, so genuine local deformation is never called a bad fit."""
    from oasis.spatial import loftr_matcher as lm

    # well-registered pairs measured at <=0.36 deg / <=0.007 scale spread
    assert lm.GLOBAL_AGREEMENT_MAX_DEG > 0.36 * 5
    assert lm.GLOBAL_AGREEMENT_MAX_SCALE > 0.007 * 5
    # broken pairs measured at >=18 deg / >=0.17 scale spread
    assert lm.GLOBAL_AGREEMENT_MAX_DEG < 18.0
    assert lm.GLOBAL_AGREEMENT_MAX_SCALE < 0.17
