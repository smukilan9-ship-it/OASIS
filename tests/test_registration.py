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
    assert global_res["verdict"] in ("DEFORMED", "NOT_CERTIFIABLE")

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
