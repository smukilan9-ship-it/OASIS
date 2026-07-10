"""
Tier 1 — landmark registration + certification + auto-proposal + NGF selection
(serial_registration). Synthetic, known-transform; fast and deterministic.
"""
import math
import numpy as np
import pytest
import serial_registration as sr


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
    from webui.api import API

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
