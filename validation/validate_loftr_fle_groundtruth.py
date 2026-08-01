#!/usr/bin/env python
"""
validate_loftr_fle_groundtruth.py — Phase A of research/registration.md.

THE QUESTION. Every certification OASIS issues rests on one variance identity:

    sigma_fit^2 = 2*FLE^2 + deformation^2

sigma_fit is measured from the fit residuals, so whatever FLE is *not*, the identity books
as deformation. On the disputed LL477 CD8/TIM-3 ROI, sigma_fit = 4.073 um and the shipped
FLE was 0.199 um: 4.06 um of the 4.07 was charged to deformation, and the pair was refused.
But the residual distribution on that same ROI has p90/median = 1.83 against a Rayleigh
1.90 -- it is localisation noise, not a deformation field. So the FLE is wrong.

It is wrong for a reason that is visible in the code. `loftr_fle` re-localises matches under
added image noise, which measures REPEATABILITY (precision), not distance from truth
(accuracy). A dense matcher on repetitive tissue texture can be perfectly repeatable while
systematically mis-localising. `ihc.md` s 3.5 already calls it "a conservative lower bound",
and a lower-bound FLE is mechanically an upper-bound deformation.

THE MEASUREMENT. Apply a KNOWN warp W to one real H-DAB section and match the section
against its own warped copy. For any point X in the reference, the moving image was
CONSTRUCTED so that its partner is exactly W(X) -- not approximately, not to within an
annotator's click, exactly. So for a returned correspondence (p, q):

    error = q - W(p)

is the true localisation error of that correspondence. No annotator, no second section, no
circularity: the ground truth is the warp we chose.

WHAT IS AND IS NOT MEASURED. Matching a section against a resampled copy of itself removes
the biological difference between two serial sections -- different cells, different stain,
different tears. That is deliberate: it isolates the matcher's own localisation error, which
is precisely the FLE term. The residual difference between this number and sigma_fit on a
real pair IS the deformation, which is what the identity claims and what Phase B tests. The
harness cannot, and does not try to, measure how LoFTR behaves across two different stains.

TWO CONVENTIONS THAT MATTER, both of which would silently corrupt the answer:

  1. The error reported per match is the PAIRWISE error, which is what serial_registration
     calls `fle_combined_um`; the per-point FLE the budget wants is that over sqrt(2)
     (since 2*FLE^2 = FLE_combined^2). Both are reported. Mixing them up is a factor of 1.41
     on a gate whose whole dispute is a factor of ~2.5.

  2. Certification does not run LoFTR at full resolution. `certify_local_roi` downscales the
     crop to `work_max_dim` and passes px_work = px/r, so the FLE that enters the budget is
     whatever LoFTR does at THAT scale. This harness therefore sweeps work_max_dim and
     reports the error in both um and working pixels. If the error is constant in working
     pixels, FLE is a property of the matcher's grid and scales with the downsample -- which
     would explain the entire dispute, because the disputed ROI ran at r ~ 0.41.

WHAT MAKES THIS SCRIPT FAIL. Only the controls: the resampling direction convention, the
identity warp returning ~0, the dense-field inversion converging, and enough surviving
matches to say anything. The FLE values themselves are the unknown being measured -- they
are reported, never asserted. A harness that asserts its own answer cannot discover one.

Run:
    python validation/validate_loftr_fle_groundtruth.py                 # full sweep, ~10 min
    python validation/validate_loftr_fle_groundtruth.py --quick         # controls + one scale
    python validation/validate_loftr_fle_groundtruth.py --image PATH --pixel-size 0.7519

Writes validation/loftr_fle_groundtruth_results.json for Phase B.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.loftr_matcher import loftr_correspondences, clear_loftr_caches  # noqa: E402
from oasis.spatial.serial_registration import tissue_mask                          # noqa: E402

# LL477: the cohort every disputed number comes from. 0.7519 um/px is the corrected
# scale-bar calibration (research/ihc.md s 12.1); the older 0.6802 under-states every
# distance by 10.5 %.
DEFAULT_IMAGE = os.path.expanduser("~/Desktop/cd8_input/LL477_CD8_x10_1.tif")
DEFAULT_PX = 0.7519

# The disputed ROI was ~1299 um across. Matching that keeps the measurement in the regime
# the dispute is about rather than in a regime chosen for convenience.
CROP_UM = 1300.0

# What certify_local_roi actually uses. The rest of the sweep exists to show how the answer
# depends on it.
CERT_WORK_MAX_DIM = 800

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "loftr_fle_groundtruth_results.json")

_RAYLEIGH_MEDIAN = np.sqrt(2.0 * np.log(2.0))       # median|e| = 1.1774 sigma, 2-D Gaussian
_RAYLEIGH_P90 = np.sqrt(2.0 * np.log(10.0))         # p90|e|    = 2.1460 sigma
_RAYLEIGH_RATIO = _RAYLEIGH_P90 / _RAYLEIGH_MEDIAN  # 1.8226 -- pure isotropic noise

_FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)
    return bool(cond)


# ---------------------------------------------------------------------------------------
# Coordinates
#
# cv2.resize maps a full-resolution coordinate X to the downscaled coordinate
# (X + 0.5)*r - 0.5, NOT X*r. The half-pixel term cancels for pure translation but not for
# rotation or scale, where it is worth ~0.05 working px -- small, but this harness exists to
# measure a sub-pixel quantity, so it is not a term to leave lying around.
# ---------------------------------------------------------------------------------------
def to_work(X, r):
    return (np.asarray(X, float) + 0.5) * r - 0.5


def to_full(x, r):
    return (np.asarray(x, float) + 0.5) / r - 0.5


# ---------------------------------------------------------------------------------------
# Warps. Each returns (name, kind, magnitude_um, forward(X)->W(X), image builder).
# ---------------------------------------------------------------------------------------
def _affine_matrix(kind, amount, centre, px):
    """2x3 forward map (reference -> moving), rotating/scaling about the crop centre."""
    cx, cy = float(centre[0]), float(centre[1])
    if kind == "identity":
        A, t = np.eye(2), np.zeros(2)
    elif kind == "translation":
        A, t = np.eye(2), np.array([amount / px, 0.0])          # um -> px, along x
    elif kind == "rotation":
        th = np.deg2rad(amount)
        A, t = np.array([[np.cos(th), -np.sin(th)],
                         [np.sin(th), np.cos(th)]]), np.zeros(2)
    elif kind == "scale":
        A, t = np.eye(2) * (1.0 + amount / 100.0), np.zeros(2)
    else:
        raise ValueError(kind)
    # about the centre: X -> A(X - c) + c + t
    c = np.array([cx, cy])
    M = np.zeros((2, 3))
    M[:, :2] = A
    M[:, 2] = c - A @ c + t
    return M


def _apply_affine_pts(P, M):
    P = np.asarray(P, float).reshape(-1, 2)
    return P @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]


def _bspline_field(shape, target_p90_px, seed=0, ctrl=6):
    """A smooth displacement field d(X) with a known p90 magnitude, on the REFERENCE grid.

    Coarse random control points upsampled with INTER_CUBIC: smooth by construction, which
    is what serial-section deformation is. `ctrl` sets the wavelength -- 6 control points
    across a 1300 um crop is a ~220 um wavelength, the scale at which sections actually
    stretch, rather than a high-frequency field no similarity could ever track.
    """
    import cv2
    H, W = shape
    rng = np.random.default_rng(seed)
    cp = rng.normal(size=(ctrl, ctrl, 2))
    d = cv2.resize(cp.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    mag = np.linalg.norm(d, axis=2)
    p90 = float(np.percentile(mag, 90))
    if p90 > 1e-9:
        d *= float(target_p90_px) / p90
    return d.astype(np.float32)


def _invert_field_to_maps(d):
    """cv2.remap needs, for each DESTINATION pixel y, the source coordinate X with
    X + d(X) = y. Fixed-point: X <- y - d(X). Converges while ||grad d|| < 1, which every
    field here satisfies by construction; convergence is checked, not assumed.
    """
    import cv2
    H, W = d.shape[:2]
    yy, xx = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32), indexing="xy")
    gx, gy = yy.astype(np.float32), xx.astype(np.float32)     # destination grid (x, y)
    mx, my = gx.copy(), gy.copy()
    for _ in range(12):
        dx = cv2.remap(d[:, :, 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dy = cv2.remap(d[:, :, 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        nx, ny = gx - dx, gy - dy
        step = float(np.max(np.abs(nx - mx) + np.abs(ny - my)))
        mx, my = nx, ny
        if step < 1e-4:
            break
    # residual of the inversion: does map(y) + d(map(y)) come back to y?
    dx = cv2.remap(d[:, :, 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(d[:, :, 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    resid = float(np.max(np.hypot(mx + dx - gx, my + dy - gy)))
    return mx, my, resid


def _sample_field(d, P):
    """Bilinear d(X) at sub-pixel reference points."""
    import cv2
    P = np.asarray(P, np.float32).reshape(-1, 2)
    mx = P[:, 0].reshape(-1, 1).copy()
    my = P[:, 1].reshape(-1, 1).copy()
    dx = cv2.remap(d[:, :, 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(d[:, :, 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.concatenate([dx, dy], axis=1).astype(float)


def build_warp(kind, amount, crop, px):
    """(moving_image, forward(X)->W(X), valid_mask, extra) for one known warp.

    ONE resample, INTER_LINEAR, white border. White because an H-DAB slide's background IS
    white: a black border would be an unnaturally high-contrast edge that LoFTR could latch
    onto, manufacturing matches that exist only because the harness made them.
    """
    import cv2
    H, W = crop.shape[:2]
    centre = ((W - 1) / 2.0, (H - 1) / 2.0)
    white = (255, 255, 255)
    ones = np.full((H, W), 255, np.uint8)
    extra = {}

    if kind == "bspline":
        d = _bspline_field((H, W), float(amount) / px, seed=7)
        mx, my, resid = _invert_field_to_maps(d)
        extra["inversion_residual_px"] = round(resid, 6)
        extra["field_p90_px"] = round(float(np.percentile(np.linalg.norm(d, axis=2), 90)), 4)
        mov = cv2.remap(crop, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=white)
        valid = cv2.remap(ones, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        forward = lambda P: np.asarray(P, float).reshape(-1, 2) + _sample_field(d, P)  # noqa: E731
        return mov, forward, valid, extra

    M = _affine_matrix(kind, amount, centre, px)
    mov = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=white)
    valid = cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    forward = lambda P: _apply_affine_pts(P, M)                                       # noqa: E731
    extra["matrix"] = M.tolist()
    return mov, forward, valid, extra


# ---------------------------------------------------------------------------------------
# Appearance
#
# WHY THIS AXIS EXISTS, and it is the important one. Warping a section and matching it back
# gives LoFTR two images with IDENTICAL texture: same nuclei, same stain, same focus. It
# measures the matcher's localisation floor and nothing else, and the first run of this
# harness measured that floor at ~0.19 um -- indistinguishable from the FLE the app already
# ships. So geometry alone does not explain the disputed 4.07 um, and the explanation, if
# the matcher is responsible at all, has to be in what makes a real serial pair HARDER than
# a resampled copy of itself.
#
# These transforms change how the moving image LOOKS while leaving where everything IS
# untouched, so the ground truth stays exact. A different marker's DAB density, a weaker
# counterstain, a focal-plane difference, an illumination gradient, sensor noise -- each is
# a real difference between two slides scanned on two days.
#
# WHAT THIS STILL CANNOT REPRODUCE, and it must not be glossed over: two serial sections are
# 4 um apart in the block, so they contain DIFFERENT CELLS. The "same point" in two sections
# is genuinely ambiguous by something like a nuclear diameter, and no single-section harness
# can manufacture that. Whatever gap remains between the number measured here and sigma_fit
# on a real pair is either that ambiguity or true deformation -- and telling those two apart
# needs a real pair with expert landmarks (Phase B).
# ---------------------------------------------------------------------------------------
def _deconvolve(rgb, background):
    from oasis.quant.segment import QUPATH_HEMATOXYLIN, QUPATH_DAB, _norm_vec
    h = _norm_vec(QUPATH_HEMATOXYLIN)
    d = _norm_vec(QUPATH_DAB)
    M = np.array([h, d, _norm_vec(np.cross(h, d))])
    bg = np.asarray(background, float).reshape(1, 1, 3)
    od = -np.log10((np.asarray(rgb, float)[..., :3] + 1.0) / bg)
    return od @ np.linalg.inv(M), M, bg


def _recompose(conc, M, bg):
    od = conc @ M
    return np.clip(bg * (10.0 ** (-od)) - 1.0, 0, 255).astype(np.uint8)


def restain(rgb, hem_gain=1.0, dab_gain=1.0):
    """Re-mix the stain concentrations. Geometry untouched, so the truth is unaffected."""
    from oasis.quant.segment import QUPATH_WHITE
    c, M, bg = _deconvolve(rgb, QUPATH_WHITE)
    c[..., 0] *= float(hem_gain)
    c[..., 1] *= float(dab_gain)
    return _recompose(c, M, bg)


def apply_appearance(rgb, kind):
    """One appearance perturbation, applied to the MOVING image after warping."""
    import cv2
    if kind in (None, "none"):
        return rgb
    if kind == "dab_weak":
        return restain(rgb, dab_gain=0.45)
    if kind == "dab_strong":
        return restain(rgb, dab_gain=1.9)
    if kind == "hem_weak":
        return restain(rgb, hem_gain=0.6)
    if kind == "blur":                       # a focal-plane difference between two slides
        return cv2.GaussianBlur(rgb, (0, 0), 1.5)
    if kind == "illum":                      # smooth uneven illumination, +/-15 %
        H, W = rgb.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        g = 1.0 + 0.15 * ((xx / max(W - 1, 1)) - 0.5 + (yy / max(H - 1, 1)) - 0.5)
        return np.clip(rgb.astype(float) * g[..., None], 0, 255).astype(np.uint8)
    if kind == "noise":
        rng = np.random.default_rng(11)
        return np.clip(rgb.astype(float) + rng.normal(0, 6.0, rgb.shape),
                       0, 255).astype(np.uint8)
    if kind == "cross_stain":                # everything at once: the closest analogue of a
        out = restain(rgb, hem_gain=0.6, dab_gain=0.45)     # real CD8 -> TIM-3 pair that a
        out = apply_appearance(out, "blur")                 # single section can produce
        out = apply_appearance(out, "illum")
        return apply_appearance(out, "noise")
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------------------
def densest_tissue_crop(rgb, px, side_um=CROP_UM):
    """The window of the requested size holding the most tissue.

    Deterministic, and better than "the centre", which on these fields can be glass. A crop
    of background would measure LoFTR's behaviour on empty slide, which is not the question.
    """
    import cv2
    H, W = rgb.shape[:2]
    side = int(round(side_um / px))
    side = min(side, H, W)
    m = (tissue_mask(rgb, px) > 0).astype(np.float64)
    ii = cv2.integral(m)
    best, best_xy = -1.0, (0, 0)
    step = max(side // 8, 1)
    for y0 in range(0, H - side + 1, step):
        for x0 in range(0, W - side + 1, step):
            s = (ii[y0 + side, x0 + side] - ii[y0, x0 + side]
                 - ii[y0 + side, x0] + ii[y0, x0])
            if s > best:
                best, best_xy = s, (x0, y0)
    x0, y0 = best_xy
    frac = best / float(side * side)
    return rgb[y0:y0 + side, x0:x0 + side].copy(), (x0, y0, side), frac


# ---------------------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------------------
def measure(crop, mov, forward, valid, px, work_max_dim, ref_tissue,
            margin_px=8.0, gross_um=25.0):
    """Per-match true error for one (warp, working-scale) pair.

    Reproduces certify_local_roi's geometry exactly: one shared downscale to `work_max_dim`
    with INTER_AREA, px_work = px/r, and loftr_correspondences at its shipped defaults. The
    FLE that reaches the error budget is the FLE of the population the certification path
    actually selects, so the selection has to be the same one.
    """
    import cv2
    H, W = crop.shape[:2]
    r = min(1.0, float(work_max_dim) / float(max(H, W)))

    def _rs(im, interp=cv2.INTER_AREA):
        if r >= 1.0:
            return im
        return cv2.resize(im, (max(int(im.shape[1] * r), 8), max(int(im.shape[0] * r), 8)),
                          interpolation=interp)

    small_r, small_m = _rs(crop), _rs(mov)
    px_work = float(px) / r

    t0 = time.time()
    c = loftr_correspondences(small_r, small_m, pixel_size_um=px_work)
    dt = time.time() - t0
    out = {"work_max_dim": int(work_max_dim), "r": round(r, 6),
           "px_work_um": round(px_work, 4), "seconds": round(dt, 1),
           "n_raw": c.get("n_raw"), "n_selected": c.get("n"), "n_used": 0,
           "loftr_msg": c.get("msg")}
    if not c["ok"]:
        return out

    p = np.asarray(c["ref_points"], float)          # working px, reference
    q = np.asarray(c["mov_points"], float)          # working px, moving
    p_full = to_full(p, r)
    truth_full = forward(p_full)
    truth = to_work(truth_full, r)

    # Admissibility. Not outlier rejection -- these drop matches whose ground truth the
    # harness cannot state, never matches whose error it dislikes.
    #   * the reference end must be on tissue (LoFTR on blank glass is not the question);
    #   * the truth must land inside the region the warp actually filled, clear of the
    #     border, or "error" would be measured against invented white.
    keep = np.ones(len(p), bool)
    ph, pw = ref_tissue.shape[:2]
    pi = np.clip(np.round(p_full).astype(int), [0, 0], [pw - 1, ph - 1])
    keep &= ref_tissue[pi[:, 1], pi[:, 0]] > 0
    vh, vw = valid.shape[:2]
    ti = np.round(truth_full).astype(int)
    inside = ((ti[:, 0] >= margin_px) & (ti[:, 0] < vw - margin_px) &
              (ti[:, 1] >= margin_px) & (ti[:, 1] < vh - margin_px))
    keep &= inside
    ok_ti = np.clip(ti, [0, 0], [vw - 1, vh - 1])
    keep &= valid[ok_ti[:, 1], ok_ti[:, 0]] > 0
    out["n_dropped_offtissue_or_border"] = int((~keep).sum())

    p, q, truth = p[keep], q[keep], truth[keep]
    if len(p) < 10:
        out["loftr_msg"] = f"only {len(p)} matches admissible ({out['loftr_msg']})"
        return out

    e = q - truth                                    # working px
    mag_px = np.linalg.norm(e, axis=1)
    mag_um = mag_px * px_work
    n = len(mag_um)

    # NO trimming. registration.md s 6: "do not treat it as outliers again." A blunder --
    # a match on the wrong instance of a repeated structure -- is a different error mode
    # from FLE, so it is COUNTED and reported rather than deleted, and both a
    # blunder-sensitive (RMS) and a blunder-insensitive (median-derived) sigma are given.
    med = float(np.median(mag_um))
    p90 = float(np.percentile(mag_um, 90))
    bias = e.mean(axis=0) * px_work
    ec = e - e.mean(axis=0)

    def _sigma(v):                                   # per-axis sigma, um -- the convention
        return float(np.sqrt((v ** 2).sum() / (2 * len(v))))   # serial_registration uses

    fle_comb = _sigma(e * px_work)
    fle_comb_centred = _sigma(ec * px_work)
    fle_comb_robust = med / _RAYLEIGH_MEDIAN         # insensitive to the blunder tail

    out.update(
        n_used=int(n),
        err_med_um=round(med, 4),
        err_p75_um=round(float(np.percentile(mag_um, 75)), 4),
        err_p90_um=round(p90, 4),
        err_max_um=round(float(mag_um.max()), 4),
        rayleigh_ratio=round(p90 / med, 3) if med > 1e-9 else None,
        n_gross=int((mag_um > gross_um).sum()),
        gross_frac=round(float((mag_um > gross_um).mean()), 4),
        bias_um=round(float(np.linalg.norm(bias)), 4),
        bias_xy_um=[round(float(bias[0]), 4), round(float(bias[1]), 4)],
        # the pairwise sigma (== serial_registration's fle_combined_um) ...
        fle_combined_um=round(fle_comb, 4),
        fle_combined_centred_um=round(fle_comb_centred, 4),
        fle_combined_robust_um=round(fle_comb_robust, 4),
        # ... and the per-point FLE that goes into sigma_fit^2 = 2*FLE^2 + model^2
        fle_um=round(fle_comb / np.sqrt(2.0), 4),
        fle_robust_um=round(fle_comb_robust / np.sqrt(2.0), 4),
        # the same answer in WORKING PIXELS: constant here => a matcher-grid property that
        # scales with the downsample, which is the scale-dependence hypothesis.
        err_med_px=round(med / px_work, 4),
        fle_px=round(fle_comb / np.sqrt(2.0) / px_work, 4),
    )
    return out


# ---------------------------------------------------------------------------------------
# Controls -- the only things that can fail this script
# ---------------------------------------------------------------------------------------
def control_resample_direction(crop, px):
    """cv2.warpAffine(src, M) must move content FORWARD by M, not backward by M.

    OpenCV inverts M internally unless WARP_INVERSE_MAP is set, so `M` is the source->dest
    map -- which is the convention this harness assumes when it calls forward(p) the truth.
    Getting it backwards would flip the sign of every measured error while leaving every
    magnitude plausible, so it is checked against the pixels rather than against the docs.
    """
    import cv2
    shift = 17                                        # integer px: resampling is then exact
    M = np.array([[1.0, 0.0, float(shift)], [0.0, 1.0, 0.0]])
    H, W = crop.shape[:2]
    mov = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    a = crop[:, :W - shift].astype(np.int32)
    b = mov[:, shift:].astype(np.int32)
    mad = float(np.abs(a - b).mean())
    return check("cv2.warpAffine moves content forward by M "
                 "(a sign error here would invert every truth)",
                 mad < 0.5, f"mean|diff| over the overlap = {mad:.4f} grey levels")


def control_identity(crop, px, ref_tissue, work_max_dim):
    """Identity warp: coordinate bookkeeping, end to end.

    Tests the harness, not the matcher -- with W = I the matcher is handed the same image
    twice, so any error is the harness's own. An off-by-one in the crop, a swapped x/y, or
    a to_work/to_full mistake shows up here and nowhere else.
    """
    mov, forward, valid, _ = build_warp("identity", 0.0, crop, px)
    m = measure(crop, mov, forward, valid, px, work_max_dim, ref_tissue)
    ok = m.get("n_used", 0) >= 20 and (m.get("err_med_um") or 9e9) < 0.1
    check("identity warp returns ~0 error (coordinate bookkeeping)",
          ok, f"n={m.get('n_used')} median={m.get('err_med_um')} um "
              f"max={m.get('err_max_um')} um")
    return m


def control_field_inversion(crop, px):
    """The B-spline image is built by inverting the field numerically. If that inversion has
    not converged, the generated image does not correspond to the warp we then call truth,
    and every B-spline number is quietly wrong."""
    worst = 0.0
    for amount in (5.0, 15.0):
        _, _, _, extra = build_warp("bspline", amount, crop, px)
        worst = max(worst, float(extra.get("inversion_residual_px", 9e9)))
    return check("dense-field inversion converges (B-spline truth is exact)",
                 worst < 0.01, f"worst round-trip residual {worst:.2e} px")


# ---------------------------------------------------------------------------------------
def run(image_path, px, quick=False, work_dims=None, out_json=OUT_JSON):
    import cv2

    print("=" * 78)
    print("LoFTR FLE against ground truth - Phase A (research/registration.md)")
    print("=" * 78)
    print(f"  image      : {image_path}")
    print(f"  pixel size : {px} um/px")

    if not os.path.exists(image_path):
        print(f"\nERROR: image not found: {image_path}")
        print("  Pass --image PATH. Phase A needs one real H-DAB field and nothing else.")
        return 2

    rgb = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    crop, (cx, cy, side), tfrac = densest_tissue_crop(rgb, px)
    ref_tissue = tissue_mask(crop, px)
    print(f"  full field : {rgb.shape[1]} x {rgb.shape[0]} px")
    print(f"  crop       : {side} px = {side * px:.0f} um at ({cx}, {cy}), "
          f"tissue {tfrac:.0%}")
    if side * px < CROP_UM - 1:
        # Said out loud rather than clamped quietly: the disputed ROI was ~1300 um, and a
        # field that cannot hold one is measuring a smaller window than the dispute is about.
        print(f"  NOTE       : field is too small for the requested {CROP_UM:.0f} um crop; "
              f"using {side * px:.0f} um.")

    print("\n1. Controls")
    control_resample_direction(crop, px)
    control_field_inversion(crop, px)
    ident = control_identity(crop, px, ref_tissue, CERT_WORK_MAX_DIM)

    # The warp table. Translation in um along x; rotation in degrees; scale in percent;
    # bspline by its p90 displacement in um.
    warps = [
        ("translation", 2.0), ("translation", 5.0), ("translation", 10.0),
        ("translation", 30.0),                       # the aliasing regime of memory
        ("rotation", 1.0), ("rotation", 3.0), ("rotation", 10.0),
        ("scale", 2.0), ("scale", -2.0), ("scale", 5.0), ("scale", -5.0),
        ("bspline", 5.0), ("bspline", 15.0),
    ]
    if quick:
        warps = [("translation", 5.0), ("rotation", 3.0), ("bspline", 5.0)]

    if work_dims is None:
        work_dims = [CERT_WORK_MAX_DIM] if quick else [400, CERT_WORK_MAX_DIM, 1200, 1600]

    print(f"\n2. Known warps at work_max_dim={CERT_WORK_MAX_DIM} "
          f"(what certify_local_roi uses)")
    print(f"   {'warp':>18}  {'n':>4}  {'med um':>7}  {'p90 um':>7}  "
          f"{'p90/med':>7}  {'bias':>6}  {'FLE um':>7}  {'gross':>5}")
    rows = []
    for kind, amount in warps:
        mov, forward, valid, extra = build_warp(kind, amount, crop, px)
        m = measure(crop, mov, forward, valid, px, CERT_WORK_MAX_DIM, ref_tissue)
        m.update(warp=kind, amount=amount, warp_extra={k: v for k, v in extra.items()
                                                       if k != "matrix"})
        rows.append(m)
        label = f"{kind} {amount:g}"
        if m.get("n_used"):
            print(f"   {label:>18}  {m['n_used']:>4}  {m['err_med_um']:>7.3f}  "
                  f"{m['err_p90_um']:>7.3f}  {m['rayleigh_ratio']:>7.2f}  "
                  f"{m['bias_um']:>6.2f}  {m['fle_um']:>7.3f}  {m['n_gross']:>5}")
        else:
            print(f"   {label:>18}  {'--':>4}  {m.get('loftr_msg')}")
        clear_loftr_caches()

    print(f"\n3. Scale dependence - is FLE a tissue property (constant um) or a "
          f"matcher-grid\n   property (constant working px)?")
    print(f"   {'warp':>14} {'workdim':>7} {'um/px':>6}  {'n':>4}  "
          f"{'med um':>7}  {'med px':>7}  {'FLE um':>7}  {'FLE px':>7}")
    scale_rows = []
    scale_warps = warps if quick else [("translation", 5.0), ("rotation", 3.0),
                                       ("bspline", 5.0)]
    for kind, amount in scale_warps:
        mov, forward, valid, _ = build_warp(kind, amount, crop, px)
        for wd in work_dims:
            m = measure(crop, mov, forward, valid, px, wd, ref_tissue)
            m.update(warp=kind, amount=amount)
            scale_rows.append(m)
            if m.get("n_used"):
                print(f"   {kind + ' ' + format(amount, 'g'):>14} {wd:>7} "
                      f"{m['px_work_um']:>6.2f}  {m['n_used']:>4}  "
                      f"{m['err_med_um']:>7.3f}  {m['err_med_px']:>7.3f}  "
                      f"{m['fle_um']:>7.3f}  {m['fle_px']:>7.3f}")
            else:
                print(f"   {kind + ' ' + format(amount, 'g'):>14} {wd:>7} "
                      f"{'--':>6}  {m.get('loftr_msg')}")
            clear_loftr_caches()

    print("\n4. Appearance mismatch - the term a geometry-only warp cannot see.")
    print("   Same warp (rotation 3 deg), moving image made to LOOK like a different slide.")
    print(f"   {'appearance':>14}  {'n':>4}  {'med um':>7}  {'p90 um':>7}  "
          f"{'p90/med':>7}  {'FLE um':>7}  {'gross':>5}")
    appearances = ["none", "dab_weak", "hem_weak", "blur", "cross_stain"] if quick else \
                  ["none", "dab_weak", "dab_strong", "hem_weak", "blur", "illum",
                   "noise", "cross_stain"]
    app_rows = []
    base_mov, base_forward, base_valid, _ = build_warp("rotation", 3.0, crop, px)
    for app in appearances:
        mov_a = apply_appearance(base_mov, app)
        m = measure(crop, mov_a, base_forward, base_valid, px, CERT_WORK_MAX_DIM, ref_tissue)
        m.update(warp="rotation", amount=3.0, appearance=app)
        app_rows.append(m)
        if m.get("n_used"):
            print(f"   {app:>14}  {m['n_used']:>4}  {m['err_med_um']:>7.3f}  "
                  f"{m['err_p90_um']:>7.3f}  {m['rayleigh_ratio']:>7.2f}  "
                  f"{m['fle_um']:>7.3f}  {m['n_gross']:>5}")
        else:
            print(f"   {app:>14}  {'--':>4}  {m.get('loftr_msg')}")
        clear_loftr_caches()

    ok_rows = [r for r in rows if r.get("n_used")]
    check("enough admissible matches to measure anything",
          len(ok_rows) >= max(1, len(warps) // 2),
          f"{len(ok_rows)}/{len(warps)} warps produced a usable set")

    summary = _summarise(rows, scale_rows, ident, px, app_rows)
    _report(summary, rows)

    payload = {
        "image": image_path, "pixel_size_um": px,
        "crop": {"x": cx, "y": cy, "side_px": side, "side_um": round(side * px, 1),
                 "tissue_frac": round(tfrac, 4)},
        "cert_work_max_dim": CERT_WORK_MAX_DIM,
        "identity_control": ident,
        "warp_rows": rows, "scale_rows": scale_rows, "appearance_rows": app_rows,
        "summary": summary,
        "controls_failed": list(_FAILS),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")
    return 1 if _FAILS else 0


def _summarise(rows, scale_rows, ident, px, app_rows=None):
    """The numbers Phase B needs, and the one arithmetic line that settles the dispute."""
    rigid = [r for r in rows if r.get("n_used") and r.get("warp") != "bspline"]
    out = {"n_rigid_warps": len(rigid)}
    if not rigid:
        return out

    fle = np.array([r["fle_um"] for r in rigid])
    fle_rob = np.array([r["fle_robust_um"] for r in rigid])
    ratio = np.array([r["rayleigh_ratio"] for r in rigid if r.get("rayleigh_ratio")])
    bias = np.array([r["bias_um"] for r in rigid])
    amounts = np.array([abs(r["amount"]) for r in rigid])

    out.update(
        fle_um_median=round(float(np.median(fle)), 4),
        fle_um_range=[round(float(fle.min()), 4), round(float(fle.max()), 4)],
        fle_robust_um_median=round(float(np.median(fle_rob)), 4),
        rayleigh_ratio_median=round(float(np.median(ratio)), 3) if len(ratio) else None,
        rayleigh_reference=round(float(_RAYLEIGH_RATIO), 3),
        bias_um_median=round(float(np.median(bias)), 4),
        bias_um_max=round(float(bias.max()), 4),
    )
    # Does error grow with warp magnitude? If it does, part of it is systematic (the warp
    # itself defeats the matcher); if not, it is the matcher's fixed localisation noise.
    if len(rigid) >= 4 and np.ptp(amounts) > 0:
        med = np.array([r["err_med_um"] for r in rigid])
        out["err_vs_magnitude_pearson_r"] = round(
            float(np.corrcoef(amounts, med)[0, 1]), 3)

    # Scale dependence: constant in um, or constant in working px?
    if scale_rows:
        by = {}
        for r in scale_rows:
            if r.get("n_used"):
                by.setdefault((r["warp"], r["amount"]), []).append(r)
        cv_um, cv_px = [], []
        for _, rs in by.items():
            if len(rs) < 2:
                continue
            u = np.array([x["err_med_um"] for x in rs])
            p = np.array([x["err_med_px"] for x in rs])
            cv_um.append(float(u.std() / u.mean()) if u.mean() else np.nan)
            cv_px.append(float(p.std() / p.mean()) if p.mean() else np.nan)
        if cv_um:
            out["scale_cv_um"] = round(float(np.nanmedian(cv_um)), 3)
            out["scale_cv_px"] = round(float(np.nanmedian(cv_px)), 3)
            out["scale_verdict"] = ("constant in working pixels (matcher-grid property)"
                                    if out["scale_cv_px"] < out["scale_cv_um"] else
                                    "constant in um (tissue property)")

    # Appearance: how much does FLE grow when the two images stop looking identical?
    if app_rows:
        good = {r["appearance"]: r for r in app_rows if r.get("n_used")}
        out["appearance_fle_um"] = {k: v["fle_um"] for k, v in good.items()}
        out["appearance_n"] = {k: v["n_used"] for k, v in good.items()}
        base = good.get("none")
        worst = max(good.values(), key=lambda v: v["fle_um"]) if good else None
        if base and worst:
            out["appearance_worst"] = worst["appearance"]
            out["appearance_worst_fle_um"] = worst["fle_um"]
            out["appearance_inflation"] = round(worst["fle_um"] / base["fle_um"], 2) \
                if base["fle_um"] else None
            # Matches lost is itself a finding: a perturbation that halves the surviving
            # correspondence count has changed the population, not just its precision.
            out["appearance_worst_n_frac"] = round(
                worst["n_used"] / base["n_used"], 3) if base["n_used"] else None

    # The disputed ROI, from research/registration.md s 1 and s 5. Evaluated at BOTH the
    # geometry-only floor and the hardest appearance mismatch, because those bracket what
    # this harness can say: neither is the real cross-section FLE, but the real one cannot
    # be smaller than the floor, and this harness gives no reason to think it as large as
    # sigma_fit.
    sigma_fit, shipped_fle = 4.073, 0.199
    out["disputed_roi"] = {"sigma_fit_um": sigma_fit, "shipped_fle_um": shipped_fle,
                           "shipped_deformation_um": round(float(np.sqrt(max(
                               sigma_fit ** 2 - 2 * shipped_fle ** 2, 0.0))), 4)}
    for tag, fle_used in (("geometry_only", out["fle_um_median"]),
                          ("worst_appearance", out.get("appearance_worst_fle_um"))):
        if fle_used is None:
            continue
        comb2 = 2.0 * float(fle_used) ** 2
        out["disputed_roi"][tag] = {
            "fle_um": round(float(fle_used), 4),
            "fle_combined_um": round(float(np.sqrt(comb2)), 4),
            "deformation_um": round(float(np.sqrt(max(sigma_fit ** 2 - comb2, 0.0))), 4),
            "explains_residual_frac": round(float(min(comb2 / sigma_fit ** 2, 1.0)), 4),
        }
    # What FLE WOULD be needed to explain the whole residual? The number Phase B has to
    # decide is plausible or not, stated once instead of left to be re-derived.
    out["disputed_roi"]["fle_um_that_would_explain_all"] = round(
        float(sigma_fit / np.sqrt(2.0)), 4)
    out["identity_median_um"] = ident.get("err_med_um")
    return out


def _report(s, rows):
    print("\n" + "=" * 78)
    print("MEASURED FLE")
    print("=" * 78)
    if not s.get("n_rigid_warps"):
        print("  no rigid warp produced a usable correspondence set - nothing measured.")
        return
    print(f"  FLE (per point, rigid warps)   : {s['fle_um_median']} um  "
          f"[{s['fle_um_range'][0]} - {s['fle_um_range'][1]}]")
    print(f"  FLE, blunder-insensitive       : {s['fle_robust_um_median']} um")
    print(f"  systematic bias                : {s['bias_um_median']} um "
          f"(max {s['bias_um_max']})")
    print(f"  p90/median                     : {s['rayleigh_ratio_median']} "
          f"(pure isotropic noise = {s['rayleigh_reference']})")
    if "err_vs_magnitude_pearson_r" in s:
        r = s["err_vs_magnitude_pearson_r"]
        print(f"  error vs warp magnitude        : r = {r} "
              f"({'grows with the warp - systematic' if r > 0.5 else 'flat - noise, not warp-driven'})")
    if "scale_verdict" in s:
        print(f"  across working scales          : {s['scale_verdict']} "
              f"(CV {s['scale_cv_um']} in um vs {s['scale_cv_px']} in px)")

    if "appearance_worst" in s:
        print(f"  appearance mismatch            : worst is '{s['appearance_worst']}' at "
              f"{s['appearance_worst_fle_um']} um "
              f"({s['appearance_inflation']}x the matched-appearance floor, "
              f"keeping {s['appearance_worst_n_frac']:.0%} of the matches)")

    d = s["disputed_roi"]
    print("\n" + "-" * 78)
    print("APPLIED TO THE DISPUTED ROI (registration.md s 1)")
    print("-" * 78)
    print(f"  sigma_fit measured on that ROI : {d['sigma_fit_um']} um")
    print(f"  shipped FLE {d['shipped_fle_um']} um -> deformation "
          f"{d['shipped_deformation_um']} um   (pair REFUSED)")
    for tag, label in (("geometry_only", "matched appearance"),
                       ("worst_appearance", "worst appearance  ")):
        if tag in d:
            v = d[tag]
            print(f"  measured FLE, {label}: {v['fle_um']:>6} um -> deformation "
                  f"{v['deformation_um']} um  (explains "
                  f"{v['explains_residual_frac']:.0%} of the residual variance)")
    print(f"  an FLE of {d['fle_um_that_would_explain_all']} um would explain ALL of it - "
          f"that is the number\n  Phase B must judge plausible or not.")
    print("\n  Phase A's deliverable. It does NOT re-derive the gate - that is Phase B")
    print("  (validate_fw_anhir_calibration.py with this FLE), and the gate threshold")
    print("  must not be touched before that runs.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--pixel-size", type=float, default=DEFAULT_PX)
    ap.add_argument("--quick", action="store_true",
                    help="controls + 3 warps at one scale (~2 min)")
    ap.add_argument("--work-dims", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    code = run(a.image, a.pixel_size, quick=a.quick, work_dims=a.work_dims, out_json=a.out)
    print("\n" + "=" * 78)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} control(s) FAILED: {', '.join(_FAILS)}")
        print("The measured FLE above is NOT trustworthy until these pass.")
    elif code == 0:
        print("RESULT: all controls PASSED - the measured FLE above is usable for Phase B.")
    sys.exit(code)


if __name__ == "__main__":
    main()
