"""
serial_registration.py
Serial-section-appropriate registration + structural QC + local-residual TRE
for paired H-DAB CD8 / TIM-3 sections (Phase A certification).

WHY THIS EXISTS (see ihc.md "Phase A — registration redesign"):
The legacy path (registration.py: rigid Euler2D MI, then nuclear ORB/SIFT) cannot
register serial sections — individual nuclei are different physical objects across
the z-gap, so nuclear texture does not correspond — and its QC fails *closed* on
genuinely well-aligned tissue (residual measured on the same non-repeatable nuclear
features). This module instead:

  1. Registers on a LOW-FREQUENCY STRUCTURAL hematoxylin signal (vessels, lumens,
     sinusoids, tissue boundaries) at σ≈12 µm — the morphology shared across serial
     sections; single nuclei are blurred away.
  2. Uses a SIMILARITY transform (rotation + translation + uniform scale) so any
     within-pair scale difference is absorbed (rigid cannot), and exports the
     estimated scale for cross-check against the scale-bar ratio.
  3. EVALUATES multiple candidate transforms (multi-init multi-resolution MI +
     phase correlation + identity) and SELECTS by LOCAL STRUCTURAL CONSENSUS —
     dense patch phase-correlation, rewarding many confident locally-aligned
     patches with low residual. (A global NCC/MI is too flat on near-uniform liver
     parenchyma and was the cause of the spurious identity fallbacks.)
  4. CERTIFIES by LOCAL residual measured directly on structure (patch
     phase-correlation residual flow → median / p90 / per-region max), cross-checked
     by independent LUMEN-CENTROID TRE, and produces green/magenta + checkerboard
     overlays for a human 2-minute visual confirmation.

HONESTY: the patch-flow residual is measured on the structural channel that MI was
optimised on (so it is a consistency check, not a fully independent gold standard);
the lumen-centroid TRE (independent objects) and the human visual overlays are what
close the independence gap, as the scope requires.
"""

import os
import math
import numpy as np

from registration import (
    extract_hematoxylin,
    _rgb_to_gray,
    _load_rgb_thumbnail,
    _sitk_to_affine,
)


# The landmark-certification gates. Fixed (≤5 µm criterion + serial-section z-gap
# floor), not tuned per dataset. Every caller — interactive certification, guided
# candidate scoring, local-ROI recovery — must use these same numbers, and every
# output reports them, so a reader can see exactly what a verdict was measured against.
CERTIFICATION_GATES = {
    "min_n": 6,
    "target_n": 12,
    "loo_max_um": 5.0,
    "fit_max_um": 5.0,
    "deformed_loo_um": 15.0,
    "min_roi_frac": 0.10,
    # RADIUS_LIMITED: a pair whose landmarks agree on ONE similarity, but only to within
    # TRE > loo_max_um. The cross-K test remains correctly sized under that error — it
    # loses power, not validity (validation/validate_radius_floor.py) — so the pair is
    # analysable; it simply cannot resolve distances below ~3×TRE. Accepted only if a
    # useful stretch of the evaluated radius range survives that floor; otherwise there
    # is no readable curve and the pair fails closed.
    "min_interpretable_band_frac": 0.5,
    "max_radius_um": 100.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Scale-bar self-calibration (burned-in 100 µm bar)
# ──────────────────────────────────────────────────────────────────────────────
def detect_scale_bar_px(image_path: str, bar_um: float = 100.0):
    """
    Robustly measure the burned-in scale bar length (px) in the bottom strip.

    The bar is a SOLID horizontal segment; the "100 µm" label above it is thin
    text. The legacy extractor mis-measured by merging the two; here we take the
    longest CONTIGUOUS solid dark run (fill≈1, short height), voted across a small
    threshold sweep so anti-aliasing / text-merge artefacts are rejected.

    Returns dict: {bar_px, pixel_size_um, bbox=(x,y,w,h), source}; bar_px None on
    failure.
    """
    import cv2
    from PIL import Image
    try:
        g = np.array(Image.open(image_path).convert("L"))
    except Exception as e:
        return {"bar_px": None, "pixel_size_um": None, "bbox": None,
                "source": f"load_failed:{e}"}

    h, w = g.shape
    y0 = int(h * 0.80)
    strip = g[y0:, :]
    widths, boxes = {}, {}
    for thr in (60, 90, 120):
        dark = (strip < thr).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        op = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(op, connectivity=8)
        best = None
        for i in range(1, n):
            cw = int(stats[i, cv2.CC_STAT_WIDTH]); ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if cw < 40 or ch > 20 or area / float(cw * ch) < 0.6:
                continue
            if best is None or cw > best[0]:
                best = (cw, ch, int(stats[i, cv2.CC_STAT_LEFT]),
                        y0 + int(stats[i, cv2.CC_STAT_TOP]))
        if best is not None:
            widths[thr] = best[0]; boxes[thr] = best

    if not widths:
        return {"bar_px": None, "pixel_size_um": None, "bbox": None,
                "source": "no_bar_detected"}
    bar_px = int(round(np.median(list(widths.values()))))
    box = next((boxes[t] for t in (90, 60, 120) if widths.get(t) == bar_px),
               list(boxes.values())[0])
    cw, ch, bx, by = box
    return {"bar_px": bar_px, "pixel_size_um": round(bar_um / bar_px, 4),
            "bbox": (bx, by, cw, ch), "source": "scale_bar"}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Structural representations (low-frequency hematoxylin, tissue mask, lumens)
# ──────────────────────────────────────────────────────────────────────────────
def structural_channel(rgb: np.ndarray, pixel_size_um: float):
    """Low-frequency STRUCTURAL channel: hematoxylin density blurred at σ≈12 µm so
    individual (non-corresponding) nuclei are suppressed and tissue architecture
    (vessels, sinusoids, lumens, boundaries) dominates. Returns uint8."""
    import cv2
    try:
        hema = extract_hematoxylin(rgb)
    except Exception:
        hema = _rgb_to_gray(rgb)
    sigma_px = max(12.0 / float(pixel_size_um), 4.0)
    k = int(sigma_px * 3) | 1
    return cv2.GaussianBlur(hema, (k, k), sigma_px).astype(np.uint8)


def tissue_mask(rgb: np.ndarray, pixel_size_um: float):
    """Binary tissue mask from hematoxylin density (Otsu; tissue = stained).
    Holes/lumens NOT filled (an empty lumen is not tissue)."""
    import cv2
    struct = structural_channel(rgb, pixel_size_um)
    _, m = cv2.threshold(struct, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2
    h, w = mask.shape
    ff = mask.copy()
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def lumen_centroids(mask: np.ndarray, pixel_size_um: float):
    """Centroids of lumens/holes inside the tissue (sinusoids, veins, vessels,
    glandular lumens) — genuine structural OBJECTS that correspond across serial
    sections, used for the independent TRE cross-check. Returns Nx2 (x,y) px."""
    import cv2
    filled = _fill_holes(mask)
    holes = ((filled > 0) & (mask == 0)).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(holes, connectivity=8)
    min_area = (8.0 / pixel_size_um) ** 2
    max_area = 0.05 * mask.size
    pts = [[float(cent[i][0]), float(cent[i][1])] for i in range(1, n)
           if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Local-residual patch flow (the TRE engine)
# ──────────────────────────────────────────────────────────────────────────────
def patch_residual_flow(ref_struct, warped_struct, overlap, pixel_size_um,
                        patch=128, stride=96, resp_min=0.06, min_std=5.0):
    """
    Dense LOCAL residual: for each tissue-overlap patch, the residual translation
    that still best aligns ref vs registered-moving structure (cv2.phaseCorrelate,
    Hann-windowed). If registration is good the residual ≈ 0 everywhere; a locally
    deformed region shows up as a large residual in that patch only.

    Returns list of (residual_um, cx, cy, response).
    """
    import cv2
    H, W = ref_struct.shape
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    recs = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            if overlap[y:y + patch, x:x + patch].mean() < 0.7:
                continue
            rp = ref_struct[y:y + patch, x:x + patch].astype(np.float32)
            wp = warped_struct[y:y + patch, x:x + patch].astype(np.float32)
            if rp.std() < min_std or wp.std() < min_std:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(rp, wp, win)
            if resp < resp_min:
                continue
            recs.append((float(np.hypot(dx, dy)) * float(pixel_size_um),
                         x + patch // 2, y + patch // 2, float(resp)))
    return recs


def flow_stats(recs, shape, grid=5):
    """Summarise patch-flow residuals. `region_max_um` is the worst LOCAL REGION
    (median residual within a grid cell), robust to a single noisy patch; `max_um`
    is the raw worst single patch (reported for transparency)."""
    if not recs:
        return {"n": 0, "median_um": None, "p90_um": None, "max_um": None,
                "region_max_um": None}
    H, W = shape
    r = np.array([v[0] for v in recs])
    cells = {}
    for resid, cx, cy, _resp in recs:
        gx = min(int(cx / W * grid), grid - 1)
        gy = min(int(cy / H * grid), grid - 1)
        cells.setdefault((gx, gy), []).append(resid)
    region_med = [float(np.median(v)) for v in cells.values()]
    return {"n": len(recs),
            "median_um": round(float(np.median(r)), 3),
            "p90_um": round(float(np.percentile(r, 90)), 3),
            "max_um": round(float(r.max()), 3),
            "region_max_um": round(float(max(region_med)), 3)}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Similarity registration: multi-init candidates + local-consensus selection
# ──────────────────────────────────────────────────────────────────────────────
def _affine_scale(matrix):
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    return float(math.sqrt(abs(a * d - b * c)))


def _make_inits(fixed, moving):
    import SimpleITK as sitk
    inits = []
    for mode in (sitk.CenteredTransformInitializerFilter.GEOMETRY,
                 sitk.CenteredTransformInitializerFilter.MOMENTS):
        try:
            base = sitk.Similarity2DTransform(sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Similarity2DTransform(), mode))
        except Exception:
            continue
        base_angle = base.GetAngle()
        for da in (math.radians(a) for a in (-10, -5, 0, 5, 10)):
            t = sitk.Similarity2DTransform(base)
            t.SetAngle(base_angle + da)
            inits.append(t)
    return inits


def _run_similarity(fixed, moving, init_tf, metric="mi"):
    """Optimise a Similarity2D transform. Returns (final_transform, metric_value) or
    (None, None); lower metric value = better for both options.

    metric="mi"  → Mattes mutual information on the intensity structural channel
                   (localises well but its histogram can ALIAS on quasi-periodic
                   tissue, per ihc.md §18.4).
    metric="ngf" → normalized cross-correlation on the GRADIENT-MAGNITUDE (edge)
                   image. SimpleITK has no true NGF/MIND optimiser metric (only the
                   six built-ins), so this is the closest edge-DRIVEN optimiser the
                   framework supports; it keys on structural boundaries rather than
                   absolute intensity and does not alias the way the intensity
                   histogram does. The genuine NGF criterion is applied at SELECTION
                   and REFINEMENT time (see register_similarity / _ngf_score), which
                   is where the aliasing failure actually manifests.
    """
    import SimpleITK as sitk
    try:
        R = sitk.ImageRegistrationMethod()
        if metric == "ngf":
            R.SetMetricAsCorrelation()                       # NCC on the edge image
        else:
            R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.25, seed=42)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300,
            gradientMagnitudeTolerance=1e-6)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        R.SetInitialTransform(sitk.Similarity2DTransform(init_tf), inPlace=False)
        final = R.Execute(fixed, moving)
        return final, float(R.GetMetricValue())
    except Exception:
        return None, None


# ── Normalized Gradient Field (NGF): edge-alignment selection / refinement ──────
def _grad_xy(img):
    """Sobel gradient (gx, gy, magnitude) of a uint8/float image, as float32."""
    import cv2
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def grad_magnitude_image(img):
    """Edge (gradient-magnitude) image, rescaled to 0–255 float32, for edge-driven
    registration. Suppresses the flat intensity that makes MI/NCC alias on near-
    uniform parenchyma; keeps the vessel/lumen/boundary edges that actually
    correspond across serial sections."""
    _gx, _gy, mag = _grad_xy(img)
    mx = float(mag.max())
    return (mag * (255.0 / mx)).astype(np.float32) if mx > 0 else mag.astype(np.float32)


def _ngf_unit(img, eta):
    """Normalized gradient field: ∇img / sqrt(|∇img|² + η²). η is the edge/noise
    scale — gradients below η are damped toward 0 (treated as noise, contributing
    neither agreement nor disagreement)."""
    gx, gy, _ = _grad_xy(img)
    denom = np.sqrt(gx * gx + gy * gy + eta * eta)
    return gx / denom, gy / denom


def _eta(img):
    """Robust edge/noise scale = median of non-zero gradient magnitude."""
    _gx, _gy, mag = _grad_xy(img)
    nz = mag[mag > 0]
    return max(float(np.median(nz)) if nz.size else 1.0, 1e-3)


def _ngf_score(ref_struct, warped_struct, mask=None):
    """Normalized Gradient Field alignment in [0, 1] (higher = better): the mean of
    <n_ref, n_warp>² over `mask`. Squared so anti-parallel edges (a section imaged
    with inverted contrast) still count as aligned; keys purely on edge GEOMETRY, so
    it does not reward the period-shifted alias that fools intensity MI."""
    ax, ay = _ngf_unit(ref_struct, _eta(ref_struct))
    bx, by = _ngf_unit(warped_struct, _eta(warped_struct))
    val = (ax * bx + ay * by) ** 2
    if mask is not None:
        return float(val[mask].mean()) if mask.any() else 0.0
    return float(val.mean())


def _ngf_refine(ref_struct, mov_struct, M, ref_mask, mov_mask, max_shift=8):
    """Small translation search that maximises the NGF score around M — corrects the
    sub-period translation aliasing MI is prone to. Rotation/scale are left to the
    optimiser; only integer-pixel translation is refined (cheap and robust)."""
    import cv2
    Hh, Ww = ref_struct.shape
    best_M, best_s = M.copy(), -1.0
    for dy in range(-max_shift, max_shift + 1, 2):
        for dx in range(-max_shift, max_shift + 1, 2):
            Mt = M.copy(); Mt[0, 2] += dx; Mt[1, 2] += dy
            warped = cv2.warpAffine(mov_struct, Mt, (Ww, Hh))
            wmask = cv2.warpAffine(mov_mask, Mt, (Ww, Hh), flags=cv2.INTER_NEAREST)
            overlap = (ref_mask > 0) & (wmask > 0)
            s = _ngf_score(ref_struct, warped, overlap)
            if s > best_s:
                best_s, best_M = s, Mt
    return best_M, best_s


def _mi_eval(fixed, moving, tf):
    """Evaluate Mattes MI for a fixed transform (no optimisation) → comparable
    score across MI / identity candidates. Lower = better."""
    import SimpleITK as sitk
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.3, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tf)
    try:
        return float(R.MetricEvaluate(fixed, moving))
    except Exception:
        return float("inf")


def register_similarity(ref_rgb, mov_rgb, pixel_size_um):
    """
    Register mov→ref (similarity) on the structural channel, then SELECT the transform
    by the Normalized Gradient Field (NGF) edge-alignment score — not by the optimiser
    metric value.

    WHY (ihc.md §18.4): the intensity MI histogram, and dense NCC/phase-correlation,
    SATURATE or ALIAS on this quasi-periodic tissue, which is what produced spurious
    period-shifted and identity picks. So we now:
      1. Generate candidates from TWO complementary optimisers per init — Mattes MI on
         the intensity structural channel AND correlation on the GRADIENT-MAGNITUDE
         (edge) image — plus identity.
      2. SELECT among the sane candidates by NGF (edge geometry), which does not reward
         a period-shifted alias, then REFINE the winner's translation by NGF.
    SimpleITK exposes no true NGF/MIND optimiser metric, so NGF is applied at the
    selection/refinement stage (where the aliasing failure actually shows up); the
    edge-image correlation candidate gives the optimiser an edge-driven proposal too.

    patch-flow / lumen residuals remain DIAGNOSTICS only — manual landmark TRE gates
    certification. Returns dict with the chosen transform + diagnostics.
    """
    import cv2
    import SimpleITK as sitk

    ref_struct = structural_channel(ref_rgb, pixel_size_um)
    mov_struct = structural_channel(mov_rgb, pixel_size_um)
    ref_mask = tissue_mask(ref_rgb, pixel_size_um)
    mov_mask = tissue_mask(mov_rgb, pixel_size_um)
    Hh, Ww = ref_struct.shape
    diag = float(np.hypot(Hh, Ww))

    fixed = sitk.GetImageFromArray(ref_struct.astype(np.float32))
    moving = sitk.GetImageFromArray(mov_struct.astype(np.float32))
    fixed_g = sitk.GetImageFromArray(grad_magnitude_image(ref_struct))
    moving_g = sitk.GetImageFromArray(grad_magnitude_image(mov_struct))

    cand = []   # (label, sitk_transform, optimiser_value)
    for i, init in enumerate(_make_inits(fixed, moving)):
        final, mi = _run_similarity(fixed, moving, init, metric="mi")
        if final is not None:
            cand.append((f"sim_mi_{i}", final, mi))
        final_g, cval = _run_similarity(fixed_g, moving_g, init, metric="ngf")
        if final_g is not None:
            cand.append((f"sim_ngf_{i}", final_g, cval))
    identity_tf = sitk.Similarity2DTransform()
    cand.append(("identity", identity_tf, _mi_eval(fixed, moving, identity_tf)))

    # Sanity-gate candidates, then SCORE each by NGF edge alignment over tissue overlap.
    sane = []
    for label, tf, opt in cand:
        try:
            M = _sitk_to_affine(tf)
        except Exception:
            continue
        s = _affine_scale(M)
        tx, ty = float(M[0, 2]), float(M[1, 2])
        ang = abs(math.degrees(math.atan2(M[1, 0], M[0, 0])))
        if label != "identity" and (
                ang > 45 or abs(s - 1.0) > 0.30 or np.hypot(tx, ty) > diag):
            continue
        warped = cv2.warpAffine(mov_struct, M, (Ww, Hh))
        wmask = cv2.warpAffine(mov_mask, M, (Ww, Hh), flags=cv2.INTER_NEAREST)
        overlap = (ref_mask > 0) & (wmask > 0)
        ngf = _ngf_score(ref_struct, warped, overlap)
        sane.append({"label": label, "matrix": M, "opt": opt,
                     "est_scale": s, "ngf": ngf})
    sane.sort(key=lambda d: -d["ngf"])                      # higher NGF = better
    best = sane[0]

    # NGF translation refinement of the winner (corrects sub-period MI aliasing).
    M, refined_ngf = _ngf_refine(ref_struct, mov_struct, best["matrix"],
                                 ref_mask, mov_mask)
    best_ngf = max(best["ngf"], refined_ngf)
    if best["label"].startswith("sim_mi"):
        method = "similarity_mi"
    elif best["label"].startswith("sim_ngf"):
        method = "similarity_ngf"
    else:
        method = best["label"]

    # Diagnostics (NOT gating): structural NCC/Dice + patch-flow + lumen residual.
    warped = cv2.warpAffine(mov_struct, M, (Ww, Hh))
    wmask = cv2.warpAffine(mov_mask, M, (Ww, Hh), flags=cv2.INTER_NEAREST)
    overlap = (ref_mask > 0) & (wmask > 0)
    ncc = float(np.corrcoef(ref_struct[overlap], warped[overlap])[0, 1]) \
        if overlap.sum() > 10 else 0.0
    dice = 2.0 * overlap.sum() / float((ref_mask > 0).sum() + (wmask > 0).sum())
    recs = patch_residual_flow(ref_struct, warped, overlap, pixel_size_um)
    return {
        "matrix": np.asarray(M, dtype=np.float32),
        "scale_ref": 1.0, "scale_mov": 1.0,
        "method": method, "success": method != "identity",
        "est_scale": round(best["est_scale"], 4),
        "mi_value": round(best["opt"], 5),
        "select_metric": "ngf", "ngf_score": round(best_ngf, 4),
        "struct_ncc": round(ncc, 4), "struct_dice": round(dice, 4),
        "flow": flow_stats(recs, (Hh, Ww)), "recs": recs,
        "n_candidates": len(sane),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Independent lumen-centroid TRE cross-check
# ──────────────────────────────────────────────────────────────────────────────
def _apply_affine(pts, matrix):
    if len(pts) == 0:
        return pts
    return (matrix @ np.hstack([pts, np.ones((len(pts), 1))]).T).T


def lumen_tre(ref_mask, mov_mask, matrix, pixel_size_um, tol_um=12.0):
    """Independent object-based TRE: match lumen centroids by mutual nearest
    neighbour (tight tolerance) in the common frame. Returns dict."""
    from scipy.spatial import cKDTree
    ref_pts = lumen_centroids(ref_mask, pixel_size_um)
    mov_pts = lumen_centroids(mov_mask, pixel_size_um)
    out = {"n_ref": len(ref_pts), "n_mov": len(mov_pts), "n_corr": 0,
           "median_um": None, "p90_um": None,
           "ref_matched": np.zeros((0, 2)), "mapped_matched": np.zeros((0, 2))}
    if len(ref_pts) == 0 or len(mov_pts) == 0:
        return out
    mapped = _apply_affine(mov_pts, matrix)
    tol_px = tol_um / float(pixel_size_um)
    tr, tm = cKDTree(ref_pts), cKDTree(mapped)
    d_rm, idx_rm = tr.query(mapped)
    _d_mr, idx_mr = tm.query(ref_pts)
    res, rm_, mp_ = [], [], []
    for j, (d, i) in enumerate(zip(d_rm, idx_rm)):
        if d <= tol_px and idx_mr[i] == j:
            res.append(d * float(pixel_size_um)); rm_.append(ref_pts[i]); mp_.append(mapped[j])
    if res:
        res = np.array(res)
        out.update(n_corr=len(res), median_um=round(float(np.median(res)), 3),
                   p90_um=round(float(np.percentile(res, 90)), 3),
                   ref_matched=np.array(rm_), mapped_matched=np.array(mp_))
    return out


def manual_tre(ref_pts, mov_pts, matrix, pixel_size_um):
    """
    GOLD-STANDARD TRE from manual corresponding landmarks (full-res px).
    `matrix` is the automated similarity transform (mov→ref). We measure how well
    that automated transform maps the HUMAN-identified moving points onto their
    human-identified reference points (the target registration error), and — as an
    internal-consistency check — fit a similarity directly from the clicks and
    report its residual (click + biological scatter) and its discrepancy from the
    automated transform.
    """
    import cv2
    ref = np.asarray(ref_pts, dtype=np.float64)
    mov = np.asarray(mov_pts, dtype=np.float64)
    out = {"n": len(ref), "median_um": None, "p90_um": None, "max_um": None,
           "per_point_um": [], "fit_residual_med_um": None, "fit_scale": None,
           "fit_vs_mi_med_um": None}
    if len(ref) == 0:
        return out
    d = np.linalg.norm(_apply_affine(mov, matrix) - ref, axis=1) * float(pixel_size_um)
    out.update(median_um=round(float(np.median(d)), 3),
               p90_um=round(float(np.percentile(d, 90)), 3),
               max_um=round(float(d.max()), 3),
               per_point_um=[round(float(x), 3) for x in d])
    if len(ref) >= 3:
        Mfit, _inl = cv2.estimateAffinePartial2D(
            mov.astype(np.float32), ref.astype(np.float32), method=cv2.LMEDS)
        if Mfit is not None:
            df = np.linalg.norm(_apply_affine(mov, Mfit) - ref, axis=1) * float(pixel_size_um)
            disc = np.linalg.norm(_apply_affine(mov, Mfit) - _apply_affine(mov, matrix),
                                  axis=1) * float(pixel_size_um)
            out.update(fit_residual_med_um=round(float(np.median(df)), 3),
                       fit_scale=round(_affine_scale(Mfit), 4),
                       fit_vs_mi_med_um=round(float(np.median(disc)), 3))
    return out


def similarity_defect(matrix):
    """How far a 2x3 affine is from a similarity (s·R), as a relative quantity.

    A similarity satisfies AᵀA = s²·I. The defect is ‖AᵀA − s²I‖ / s², i.e. 0 for a
    pure rotation+uniform-scale, and >0 in the presence of shear or anisotropic scale.
    Returns +inf for a degenerate matrix.
    """
    A = np.asarray(matrix, float)[:, :2]
    G = A.T @ A
    s2 = 0.5 * float(np.trace(G))
    if not np.isfinite(s2) or s2 <= 0:
        return float("inf")
    return float(np.linalg.norm(G - s2 * np.eye(2)) / s2)


def assert_distance_preserving(matrix, name="transform", tol=0.02):
    """Fail closed unless `matrix` is a similarity, i.e. preserves distance ratios.

    THIS IS A SCIENTIFIC INVARIANT, not a sanity check. Every cross-K claim rests on
    two properties that a similarity guarantees and a general warp does not:

      • distances between cells are preserved up to one global scale, so a radius r in
        the reference frame means the same physical distance everywhere;
      • the transform has 4 degrees of freedom and is fitted to anatomical landmarks,
        so it is blind to the stained cells. It CANNOT locally pull marker-A-rich
        tissue onto marker-B-rich tissue and manufacture the association under test.

    An intensity-driven non-rigid warp (B-spline / TPS on stain intensity) breaks both:
    it optimises on a signal correlated with cell density, with enough freedom to move
    cells relative to one another. validation/validate_radius_floor.py establishes that
    registration error only ever weakens a true association — that result is conditional
    on this invariant, so the code enforces it rather than trusting a comment.
    """
    d = similarity_defect(matrix)
    if not np.isfinite(d) or d > tol:
        raise ValueError(
            f"{name} is not a similarity (defect {d:.4g} > {tol}): it does not preserve "
            f"inter-cell distances, so cross-K radii would be meaningless. A shear or "
            f"anisotropic/non-rigid warp is never applied before a spatial-association "
            f"test — see serial_registration.assert_distance_preserving.")
    return d


def _fit_similarity_ls(src, dst, weights=None):
    """Closed-form least-squares similarity (rotation + uniform scale + translation)
    mapping src→dst (Umeyama). Deterministic — no RANSAC randomness. Returns 2x3 or None.

    `weights` (length-N, non-negative) makes this the weighted Umeyama fit, which is what
    the robust IRLS estimator below iterates on.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 2:
        return None
    if weights is None:
        w = np.ones(n, float)
    else:
        w = np.asarray(weights, float).reshape(-1)
        if len(w) != n or not np.isfinite(w).all() or w.sum() <= 0:
            return None
    wsum = w.sum()
    mx = (w[:, None] * src).sum(0) / wsum
    my = (w[:, None] * dst).sum(0) / wsum
    Xc, Yc = src - mx, dst - my
    cov = (Yc * w[:, None]).T @ Xc / wsum
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var_x = (w * (Xc ** 2).sum(1)).sum() / wsum
    s = float(np.trace(np.diag(S) @ D) / var_x) if var_x > 0 else 1.0
    t = my - s * (R @ mx)
    M = np.zeros((2, 3))
    M[:, :2] = s * R
    M[:, 2] = t
    return M


def _robust_scale(resid):
    """MAD-about-zero scale of non-negative residual magnitudes."""
    s = float(np.median(resid)) / 0.6745
    return s if np.isfinite(s) and s > 1e-12 else None


def _fit_similarity_robust(src, dst, max_iter=30, tune=1.345, tol=1e-9):
    """Huber-weighted IRLS similarity fit. Deterministic — no RANSAC resampling — so two
    runs on identical landmarks reproduce a certification exactly.

    WHY. Plain least squares has a breakdown point of zero: a single landmark on a fold or
    tear drags the whole similarity, inflating the residual at every GOOD landmark and
    failing an otherwise certifiable pair. Huber caps a landmark's influence beyond ~1.345
    robust standard deviations, so a few locally-deformed correspondences bend the fit
    rather than break it. On an exact fit it returns the least-squares (Umeyama) solution
    unchanged; on clean but noisy landmarks the weights are all ≈1 and the two fits agree
    to ~1e-3 px, well below any certification threshold — it cannot unseat a pair that
    already certifies.

    KNOWN LIMIT — DO NOT MISTAKE THIS FOR HIGH-BREAKDOWN. Huber's ψ is bounded but does
    not redescend: an outlier is down-weighted, never rejected. Measured behaviour on
    12 well-spread landmarks: 2 grossly deformed points (17%) are absorbed and the pair
    still CERTIFIES; 4 (33%) drag the fit anyway and the pair degrades to RADIUS_LIMITED.
    A redescending (Tukey/MM) stage does NOT rescue that case either, because with 33%
    contamination the MAD scale is itself inflated and the cutoff never reaches the bad
    points — it needs a high-breakdown scale estimator (LTS concentration or exhaustive
    consensus), which is deliberately not built here.

    That is the right trade. In the guided workflow the researcher validates every
    correspondence before it is accepted, so gross outliers are rare; this is a safety net
    for a landmark that is slightly off, not a substitute for that judgement. And when it
    is exceeded the pair degrades to a weaker verdict — the fail-safe direction — rather
    than certifying on a corrupted transform.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    M = _fit_similarity_ls(src, dst)
    if M is None or len(src) < 4:
        return M
    for _ in range(max_iter):
        resid = np.linalg.norm(_apply_affine(src, M) - dst, axis=1)
        scale = _robust_scale(resid)
        if scale is None:
            return M                                  # already an exact fit
        u = resid / (tune * scale)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        M_new = _fit_similarity_ls(src, dst, weights=w)
        if M_new is None:
            break
        converged = np.max(np.abs(M_new - M)) < tol
        M = M_new
        if converged:
            break
    return M


def loo_tre(ref_pts, mov_pts, pixel_size_um):
    """
    Leave-one-out TRE: for each landmark, refit the similarity on the other N−1 (robustly)
    and predict the held-out one. Scoring a point that helped fit the transform is
    optimistic; LOO removes that.

    KNOWN, MEASURED BIAS (validation/validate_radius_floor.py): LOO over-states the error
    a typical CELL experiences, by ~1.4× at n=12 and ~1.15× at n=20. Two causes — holding
    a point out perturbs the fit (variance that a cell never sees), and guided landmarking
    maximises spread, so landmarks sit at the periphery where deformation is largest while
    cells fill the interior. The bias shrinks as n grows, and it errs toward OVER-stating
    error: the certification gate and the radius floor both under-claim rather than
    over-claim. That is the correct direction for a fail-closed tool, so it is documented
    and left in place rather than corrected by a model that could under-state it.
    """
    ref = np.asarray(ref_pts, float)
    mov = np.asarray(mov_pts, float)
    n = len(ref)
    out = {"n": n, "loo_median_um": None, "loo_p90_um": None, "loo_max_um": None,
           "per_point_um": []}
    if n < 3:                                    # need ≥3 so the N−1 fit is determined
        return out
    errs = []
    for i in range(n):
        m = np.arange(n) != i
        M = _fit_similarity_robust(mov[m], ref[m])
        if M is None:
            continue
        pred = (M @ np.array([mov[i, 0], mov[i, 1], 1.0]))[:2]
        errs.append(float(np.linalg.norm(pred - ref[i])) * float(pixel_size_um))
    if errs:
        e = np.array(errs)
        out.update(loo_median_um=round(float(np.median(e)), 3),
                   loo_p90_um=round(float(np.percentile(e, 90)), 3),
                   loo_max_um=round(float(e.max()), 3),
                   per_point_um=[round(float(x), 3) for x in e])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Registration error a CELL experiences — DIAGNOSTIC ONLY, NOT A GATE
# ──────────────────────────────────────────────────────────────────────────────
# Leave-one-out TRE answers "how far off is the NEXT LANDMARK?" — so it carries that
# landmark's own picking noise σ and can never fall below σ, however perfectly the
# sections are registered. On a real LL477 CD8/TIM-3 pair, LOO reports 6.6 µm because a
# lumen centroid can only be located to σ ≈ 4 µm. Cells are not clicked, so they do not
# carry that noise. What a cell experiences is:
#
#   ESTIMATION error  — how precisely n noisy landmarks pin down the 4-parameter
#                       transform at the cell's location. Standard prediction SE of the
#                       fit: σ·sqrt(fᵀ(XᵀX)⁻¹f). Shrinks like 1/√n.
#   MODEL error       — how far the tissue departs from a single similarity, i.e. real
#                       deformation.
#
# Total would be sqrt(estimation² + model²). We report the estimation term because it is
# correct and useful, but we DO NOT certify on it, because the model term is not
# measurable here. `measure_deformation` — dense phase-correlation patch flow — was built
# for it and is BLIND: on the real pair it reports 0.14 µm for the certified transform
# and 0.22 µm for an IDENTITY transform that leaves the sections 106 µm apart. The
# structural channel is blurred at σ ≈ 12 µm to suppress non-corresponding nuclei, which
# destroys the high-frequency content a displacement estimator needs; two patches of
# blurred parenchyma correlate at zero offset whatever their true separation. NCC template
# matching and gradient-magnitude phase correlation were tried and fail on the same images.
# See validation/validate_deformation_estimator.py.
#
# Certifying on the estimation term alone would be actively unsafe: it shrinks like 1/√n,
# so an operator could certify any pair, however deformed, by clicking more landmarks.
# Until an independent model-error measurement exists, the gate stays on leave-one-out
# TRE — conservative and σ-floored, but a genuine held-out measurement.
_SIMILARITY_DOF = 4


def _similarity_design(pts):
    """Design rows of the similarity model [x' ; y'] = [[x,-y,1,0],[y,x,0,1]]·(a,b,tx,ty).
    Returns (n, 2, 4)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    o, z = np.ones(len(p)), np.zeros(len(p))
    return np.stack([np.c_[x, -y, o, z], np.c_[y, x, z, o]], axis=1)


def landmark_noise_sigma(ref_pts, mov_pts, matrix, pixel_size_um):
    """Per-coordinate landmark localisation noise σ (µm), from the fit residuals.

    σ² = Σ‖residual‖² / (2n − 4): the residual variance per coordinate, corrected for the
    4 parameters the fit consumed. This is how imprecisely a correspondence can be placed
    — a lumen's outline differs between a CD8- and a TIM-3-stained section, the structural
    channel is blurred at σ≈12 µm, and clicks land on a finite pixel grid. It is a property
    of the ANNOTATION, not of the registration. Returns None when underdetermined.
    """
    ref = np.asarray(ref_pts, float).reshape(-1, 2)
    mov = np.asarray(mov_pts, float).reshape(-1, 2)
    if matrix is None or len(ref) < 2:
        return None
    dof = 2 * len(ref) - _SIMILARITY_DOF
    if dof <= 0:
        return None
    resid = np.linalg.norm(_apply_affine(mov, np.asarray(matrix, float)) - ref, axis=1)
    resid_um = resid * float(pixel_size_um)
    return float(np.sqrt((resid_um ** 2).sum() / dof))


def transform_prediction_error(ref_pts, sigma_um, eval_pts):
    """Standard error (µm) of the fitted similarity's prediction at each of `eval_pts`.

    σ·sqrt(trace(J(u)(XᵀX)⁻¹J(u)ᵀ)) — the ordinary prediction SE of the least-squares fit,
    evaluated where the CELLS are rather than where the landmarks are. Returns None if the
    design is degenerate (landmarks collinear or too few).
    """
    if sigma_um is None:
        return None
    X = _similarity_design(ref_pts)
    if len(X) < 2:
        return None
    XtX = np.einsum('nij,nik->jk', X, X)
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inv).all():
        return None
    G = _similarity_design(eval_pts)
    var = np.einsum('mij,jk,mik->mi', G, inv, G).sum(axis=1)   # trace of the 2x2 pred-cov
    return float(sigma_um) * np.sqrt(np.maximum(var, 0.0))


def prediction_error_stats(ref_pts, sigma_um, image_wh, roi_polygon=None, grid=24):
    """Prediction SE of the transform over the ANALYSIS WINDOW (µm): where the cells are."""
    if sigma_um is None or not image_wh:
        return {"median_um": None, "p90_um": None, "max_um": None, "n_eval": 0}
    w, h = float(image_wh[0]), float(image_wh[1])
    xs = np.linspace(0.02 * w, 0.98 * w, grid)
    ys = np.linspace(0.02 * h, 0.98 * h, grid)
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    poly = _polygon_from_points(roi_polygon)
    if poly is not None:
        keep = _points_inside(pts, poly)
        if keep.sum() >= 4:
            pts = pts[keep]
    se = transform_prediction_error(ref_pts, sigma_um, pts)
    if se is None or not len(se):
        return {"median_um": None, "p90_um": None, "max_um": None, "n_eval": 0}
    return {"median_um": round(float(np.median(se)), 3),
            "p90_um": round(float(np.percentile(se, 90)), 3),
            "max_um": round(float(se.max()), 3),
            "n_eval": int(len(se))}


# ──────────────────────────────────────────────────────────────────────────────
# Fitzpatrick–West error budget: FLE → predicted TRE at the CELLS + deformation
# ──────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS. The legacy gate is the median leave-one-out landmark TRE. That
# statistic is a measure of how self-consistent a landmark SET is, and it is the wrong
# thing to certify on, for two independently fatal reasons (both measured, see
# validation/validate_fw_certification.py):
#
#   1. It cannot pass good work. LOO error at a held-out landmark contains that
#      landmark's own localisation noise in BOTH images, amplified by its leverage. On a
#      PERFECT transform with zero deformation, an operator clicking to σ = 3 µm sees a
#      LOO median of 5.8 µm and certifies 26% of the time. Nobody clicks a lumen centroid
#      across two differently-stained sections to the ~2.5 µm this gate demands. Worse,
#      LOO barely improves with n (5.81 µm at n=8 → 5.33 µm at n=20), because the held-out
#      point's own noise never averages away — so "click more landmarks" does not help.
#
#   2. It cannot fail bad work. On landmark sets produced by propose_landmarks — which
#      RANSAC-selects correspondences for agreement with a single similarity — LOO is
#      near-constant (4.9 → 6.2 µm) as true elastic deformation goes 0 → 62 µm, and the
#      DEFORMED verdict never fires. The selection step discards exactly the points that
#      carry the deformation signal.
#
# This is Fitzpatrick's classical result: fiducial registration error and target
# registration error are uncorrelated (Fitzpatrick, Med. Imag. 2009; Fitzpatrick, West &
# Maurer, IEEE TMI 17(5):694, 1998). The fix is theirs too — do not try to drive landmark
# residuals to zero. Instead:
#
#   FLE  fiducial localisation error. A property of the ANNOTATOR (or detector), not of
#        the registration. Measured by REPEAT ANNOTATION (`fle_from_repeat`) or, for an
#        automatic detector, by re-localisation under image perturbation
#        (`fle_by_relocalization`). It CANNOT be recovered from fit residuals, because
#        there it is confounded with real deformation — which is precisely the
#        decomposition below.
#
#   TRE  target registration error, PREDICTED at the points that matter (the cells) from
#        FLE and the landmark geometry: σ·sqrt(fᵀ(XᵀX)⁻¹f). Falls like 1/√n, so adding
#        landmarks genuinely and honestly buys accuracy.
#
#   MODEL error (real deformation) is then recoverable by variance decomposition, because
#        the residual mean square about the fitted similarity estimates the SUM:
#
#            E[SSR] / (2n − 4)  =  FLE_ref² + FLE_mov²  +  model²        (per coordinate)
#
#        With FLE known independently, model² falls out. This is the "model term" that
#        measure_deformation (dense patch flow) was built to find and cannot see; it is
#        not unmeasurable, it is only unmeasurable from residuals ALONE.
#
# The certified quantity is then the error a CELL experiences:
#
#            cell_error(p) = sqrt( TRE_pred(p)²  +  deformation_rms² )
#
# gated at the same fixed ≤5 µm criterion, on the p90 over the analysis window. Both terms
# are conservative: TRE_pred uses the FLE-driven parameter covariance, and deformation
# enters through the UPPER confidence bound, so an undetectable deformation still costs
# the pair its noise-floor's worth of budget rather than nothing.


_CHI2_CONF = 0.95


def fle_from_repeat(pass1, pass2, pixel_size_um, robust=True):
    """Fiducial localisation error σ (µm, per coordinate, per image) from two independent
    annotation passes over the SAME landmarks.

    `robust=True` (default) estimates σ from the MEDIAN disagreement magnitude rather than
    its mean square. Two annotators marking "the same" anatomical point sometimes mark
    genuinely different structures: on ANHIR mammary-gland the PS-vs-JB disagreement has a
    median of 55 µm and a p90 of 270 µm. The mean square is dominated by that tail and
    returns an FLE (209 µm) that describes no landmark in the set. |Δ| is Rayleigh with
    median 1.1774·σ_Δ and σ_Δ = √2·FLE, hence σ = median|Δ| / (1.1774·√2).

    `discordant_frac` reports the share of landmarks whose disagreement exceeds 3× the
    median — pairs where the two experts did not mark the same thing. Those rows are not
    landmarks and callers should drop them; doing so uses only the ANNOTATIONS, never the
    transform, so it cannot bias a certification.

    This is the only honest way to get FLE. The two passes differ by the sum of two
    independent draws of the localisation noise, so Var(difference) = 2σ² per coordinate.
    A constant offset between passes is a systematic bias, not localisation noise, so the
    differences are centred before the variance is taken.

    Use two annotators (inter-observer, the ANHIR convention) if you want the FLE that
    bounds what a reader can reproduce; use one annotator twice (intra-observer) for the
    FLE that bounds this operator's own precision. The former is larger and is the one to
    quote in a paper.

    Returns {'fle_um', 'n', 'bias_um', 'per_axis_um'} or fle_um=None if underdetermined.
    """
    a = np.asarray(pass1, float).reshape(-1, 2)
    b = np.asarray(pass2, float).reshape(-1, 2)
    out = {"fle_um": None, "n": int(len(a)), "bias_um": None, "per_axis_um": None,
           "discordant_frac": None, "concordant": None, "method": None,
           "source": "repeat_annotation"}
    if len(a) < 2 or len(a) != len(b):
        return out
    d = (b - a) * float(pixel_size_um)            # (n, 2) µm
    bias = np.median(d, axis=0) if robust else d.mean(axis=0)
    dc = d - bias                                  # centre out systematic offset
    mag = np.linalg.norm(dc, axis=1)
    med = float(np.median(mag))
    keep = mag <= 3.0 * med if med > 0 else np.ones(len(mag), bool)
    out.update(discordant_frac=round(float(1.0 - keep.mean()), 4),
               concordant=keep.tolist(),
               bias_um=round(float(np.linalg.norm(bias)), 4))
    if robust:
        out.update(method="robust_median",
                   fle_um=round(med / (_RAYLEIGH_MEDIAN * np.sqrt(2.0)), 4))
        return out
    # (n-1) per axis: one dof consumed by the mean. Var(dc) = 2σ² per coordinate.
    var_per_axis = (dc ** 2).sum(axis=0) / max(len(a) - 1, 1)
    sig_per_axis = np.sqrt(np.maximum(var_per_axis / 2.0, 0.0))
    out.update(method="mean_square",
               fle_um=round(float(np.sqrt(np.mean(sig_per_axis ** 2))), 4),
               per_axis_um=[round(float(s), 4) for s in sig_per_axis])
    return out


def fle_by_relocalization(ref_rgb, mov_rgb, ref_pts, mov_pts, pixel_size_um,
                          n_trials=12, noise_frac=0.02, jitter_px=1.5, seed=0,
                          search_um=12.0, patch_um=45.0):
    """FLE (µm, per coordinate, per image) for landmarks placed by the AUTOMATIC matcher.

    The machine analogue of asking an annotator to click twice. Each trial perturbs BOTH
    sections by an independent known sub-pixel shift plus sensor-scale intensity noise,
    re-derives the structural channel, and re-runs the same local NCC snap that
    propose_landmarks uses. The scatter of the re-localised point about its known true
    position is the localisation error of the whole detect-and-snap pipeline.

    This is the right FLE for an auto-proposed set and the WRONG one for a hand-clicked
    set: a matcher's repeatability under image noise is far better than a human's
    repeatability under the ambiguity of what "the same point" means across two stains.
    For hand-clicked landmarks use `fle_from_repeat`, which measures that ambiguity.

    Returns {'fle_um', 'per_point_um', 'n_trials', 'source'}.
    """
    import cv2
    rng = np.random.default_rng(seed)
    ref_pts = np.asarray(ref_pts, float).reshape(-1, 2)
    mov_pts = np.asarray(mov_pts, float).reshape(-1, 2)
    px = float(pixel_size_um)
    search = max(int(round(search_um / px)), 3)
    patch = max(int(round(patch_um / px)) | 1, 9)
    H, W = mov_rgb.shape[:2]
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def perturb(img, shift):
        n = rng.normal(0, noise_frac * 255.0, img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + n, 0, 255)
        return cv2.remap(out, xx - shift[0], yy - shift[1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.uint8)

    errs = [[] for _ in range(len(ref_pts))]
    for _ in range(int(n_trials)):
        s_r = rng.uniform(-jitter_px, jitter_px, 2).astype(np.float32)
        s_m = rng.uniform(-jitter_px, jitter_px, 2).astype(np.float32)
        rs = structural_channel(perturb(ref_rgb, s_r), px)
        ms = structural_channel(perturb(mov_rgb, s_m), px)
        for i, (rp, mp) in enumerate(zip(ref_pts, mov_pts)):
            true_r, true_m = rp + s_r, mp + s_m
            got, _ = _local_ncc_refine(rs, ms, tuple(true_r), tuple(true_m), search, patch)
            errs[i].append(np.asarray(got, float) - true_m)

    per_point, all_d = [], []
    for e in errs:
        if not e:
            per_point.append(None)
            continue
        d = (np.array(e) - np.mean(e, axis=0)) * px           # centre out any bias
        all_d.append(d)
        per_point.append(round(float(np.sqrt((d ** 2).sum() / (2 * len(d)))), 4))
    if not all_d:
        return {"fle_um": None, "per_point_um": [], "n_trials": int(n_trials),
                "source": "relocalization"}
    D = np.concatenate(all_d, axis=0)
    # Per-coordinate σ of ONE section: the snap sees both sections' noise, hence /√2.
    sigma_combined = float(np.sqrt((D ** 2).sum() / (2 * len(D))))
    return {"fle_um": round(sigma_combined / np.sqrt(2.0), 4),
            "fle_combined_um": round(sigma_combined, 4),
            "per_point_um": per_point, "n_trials": int(n_trials),
            "source": "relocalization"}


def _leverage_traces(ref_pts):
    """trace(H_ii) for each landmark under the similarity design. Σ_i trace(H_ii) = 4."""
    X = _similarity_design(ref_pts)
    XtX = np.einsum('nij,nik->jk', X, X)
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inv).all():
        return None
    Hii = np.einsum('nij,jk,nlk->nil', X, inv, X)          # (n, 2, 2)
    return np.einsum('nii->n', Hii)


_RAYLEIGH_MEDIAN = np.sqrt(2.0 * np.log(2.0))       # median |e| = 1.1774·σ for 2-D Gaussian
_RAYLEIGH_P90 = np.sqrt(2.0 * np.log(10.0))         # p90    |e| = 2.1460·σ


def deformation_from_landmarks(ref_pts, mov_pts, matrix, pixel_size_um, fle_um,
                               conf=_CHI2_CONF, method="robust", n_boot=999, seed=0):
    """Separate real tissue deformation from landmark localisation noise.

    The residual mean square about the fitted similarity estimates the SUM of the two
    per-coordinate variances, so with FLE measured independently the model term falls out:

        σ_fit²  =  FLE_ref² + FLE_mov² + model²   ⇒   model² = σ_fit² − 2·FLE²

    Reported as 2-D vector RMS (×√2), the units a µm-scale gate is written in.

    method='classical'  σ_fit² = SSR/(2n−4); upper bound from SSR/σ² ~ χ²(2n−4).
    method='robust'     (DEFAULT) σ_fit from the MEDIAN of leverage-standardised residual
                        magnitudes, and the upper bound by bootstrap.

    WHY ROBUST IS THE DEFAULT. SSR has a breakdown point of zero, and every automatic
    matcher has a tail. On LL477 with LoFTR at confidence ≥ 0.8 the residual median is
    3.8 µm and the RMS is 24 µm: a handful of gross mismatches, and the classical estimator
    reported 34 µm of "deformation" that `residual_field_assay` showed to be spatially
    random, i.e. matcher error. The robust estimator reads the bulk of the distribution.

    This does NOT hide real deformation. Real deformation is a smooth field — it shifts the
    whole residual distribution, it does not add a heavy tail — so the median moves with it
    (verified: injected 18.7 µm reads back as 20.1 µm, 37.3 µm as 32.6 µm). A heavy tail is
    diagnostic of bad correspondences, and `residual_field_assay` is what adjudicates that.

    Leverage standardisation: E‖e_i‖² = σ²·trace(I − H_ii), so magnitudes are divided by
    sqrt(trace(I − H_ii)/2) before the median is taken. Without it, high-leverage landmarks
    (the ones at the periphery, which guided placement deliberately creates) read low.

    ASSUMPTIONS that bound what this can claim: (a) FLE is isotropic and equal in both
    sections; (b) smooth low-order deformation is partly ABSORBED by the similarity fit, so
    this under-reports it — the analysis window and landmark spread matter; (c) the
    bootstrap (and the χ² dof) treat residuals as independent, but a dense matcher's
    matches are spatially correlated, so the interval is somewhat too tight.

    Returns deformation_rms_um (0 when below the noise floor), deformation_rms_ub_um (the
    number to gate on), `detectable`, and `fle_consistent`.
    """
    from scipy.stats import chi2
    ref = np.asarray(ref_pts, float).reshape(-1, 2)
    mov = np.asarray(mov_pts, float).reshape(-1, 2)
    n = len(ref)
    out = {"n": n, "dof": 0, "method": method, "sigma_fit_um": None, "fle_um": None,
           "fle_combined_um": None, "deformation_rms_um": None,
           "deformation_rms_ub_um": None, "detectable": False, "p_value": None,
           "fle_consistent": None, "p_value_fle_too_high": None, "conf": float(conf),
           "residual_median_um": None, "residual_rms_um": None, "tail_ratio": None,
           "deformation_p90_um": None, "deformation_p90_ub_um": None,
           "residual_p90_um": None}
    dof = 2 * n - _SIMILARITY_DOF
    if matrix is None or fle_um is None or dof <= 0:
        return out
    resid_um = np.linalg.norm(_apply_affine(mov, np.asarray(matrix, float)) - ref,
                              axis=1) * float(pixel_size_um)
    ssr = float((resid_um ** 2).sum())
    fle = float(fle_um)
    fle_comb2 = 2.0 * fle ** 2
    med = float(np.median(resid_um))
    rms = float(np.sqrt((resid_um ** 2).mean()))
    out.update(residual_median_um=round(med, 4), residual_rms_um=round(rms, 4),
               tail_ratio=round(rms / med, 3) if med > 1e-9 else None)

    # Both branches must answer three questions: what is σ_fit, what is its upper bound,
    # and is σ_fit distinguishable from the declared FLE (in either direction)?
    #   p                 residual is LARGER than FLE alone explains → real deformation.
    #   p_value_fle_too_high  residual is SMALLER than the declared FLE can explain. Then
    #                     the FLE does not belong to these landmarks — either it was
    #                     borrowed from elsewhere, or an inlier gate censored the residuals.
    #                     Without this test, overstating FLE attributes all residual to
    #                     noise, shrinks the deformation bound, and buys a certification.
    fle_comb = float(np.sqrt(fle_comb2))
    if method == "robust":
        h = _leverage_traces(ref)
        if h is None:
            return out
        w = np.sqrt(np.maximum(2.0 - h, 1e-6) / 2.0)        # sqrt(trace(I - H_ii)/2)
        u = resid_um / w                                    # ~ Rayleigh(σ_fit)
        s_fit = float(np.median(u)) / _RAYLEIGH_MEDIAN
        rng = np.random.default_rng(seed)
        boot = np.array([np.median(u[rng.integers(0, n, n)])
                         for _ in range(int(n_boot))]) / _RAYLEIGH_MEDIAN
        s_ub = float(np.percentile(boot, 100.0 * conf))
        s2_fit, s2_ub = s_fit ** 2, s_ub ** 2
        # Distribution-free: the median's sampling law is not χ², so read both tails off
        # the bootstrap rather than borrowing a χ² that does not apply.
        p = float((np.sum(boot <= fle_comb) + 1) / (len(boot) + 1))
        p_low = float((np.sum(boot >= fle_comb) + 1) / (len(boot) + 1))
    else:
        s2_fit = ssr / dof
        s2_ub = ssr / chi2.ppf(1.0 - conf, dof)
        chi_stat = ssr / fle_comb2 if fle_comb2 > 0 else np.inf
        p = float(chi2.sf(chi_stat, dof)) if np.isfinite(chi_stat) else 0.0
        p_low = float(chi2.cdf(chi_stat, dof)) if np.isfinite(chi_stat) else 1.0

    # ── The quantity the GATE needs: a high quantile of deformation, not its RMS ──────
    # Real deformation is a smooth field, so its magnitude across landmarks is nowhere near
    # Rayleigh — some regions barely move, others move a lot. An RMS summary under-states
    # the p90 by ~1.6x on ANHIR (validate_fw_anhir_calibration.py), which is the unsafe
    # direction. Read the quantile off the empirical residual distribution instead, and
    # subtract the noise's OWN p90 in quadrature so a pure-noise residual returns exactly
    # zero deformation (q90 of a Rayleigh(σ) is σ·sqrt(2·ln10) = 2.146σ).
    h_q = _leverage_traces(ref)
    if h_q is not None:
        u_q = resid_um / np.sqrt(np.maximum(2.0 - h_q, 1e-6) / 2.0)
        rng_q = np.random.default_rng(seed + 1)
        q_obs = float(np.percentile(u_q, 90))
        q_boot = np.array([np.percentile(u_q[rng_q.integers(0, n, n)], 90)
                           for _ in range(int(n_boot))])
        q_ub = float(np.percentile(q_boot, 100.0 * conf))
        # The subtrahend must be the p90 of pure noise ESTIMATED THE SAME WAY — the
        # empirical 90th percentile of n samples, not the theoretical Rayleigh quantile.
        # At n=8 the empirical p90 is nearly the maximum and is badly biased; subtracting
        # the theoretical value would leave that bias in the answer, and it does not cancel.
        sim = rng_q.normal(0.0, fle_comb, size=(200, n, 2))
        q90_noise = float(np.mean(np.percentile(np.linalg.norm(sim, axis=2), 90, axis=1)))
        out["deformation_p90_um"] = round(float(np.sqrt(max(q_obs ** 2 - q90_noise ** 2, 0.0))), 4)
        out["deformation_p90_ub_um"] = round(float(np.sqrt(max(q_ub ** 2 - q90_noise ** 2, 0.0))), 4)
        out["residual_p90_um"] = round(q_obs, 4)
        out["noise_p90_um"] = round(q90_noise, 4)

    m2 = max(s2_fit - fle_comb2, 0.0)
    m2_ub = max(s2_ub - fle_comb2, 0.0)
    out.update(dof=int(dof),
               sigma_fit_um=round(float(np.sqrt(s2_fit)), 4),
               fle_um=round(fle, 4),
               fle_combined_um=round(float(np.sqrt(fle_comb2)), 4),
               deformation_rms_um=round(float(np.sqrt(m2) * np.sqrt(2.0)), 4),
               deformation_rms_ub_um=round(float(np.sqrt(m2_ub) * np.sqrt(2.0)), 4),
               detectable=bool(p < (1.0 - conf)),
               p_value=round(p, 6),
               fle_consistent=bool(p_low >= (1.0 - conf)),
               p_value_fle_too_high=round(p_low, 6))
    return out


def cell_error_budget(ref_pts, mov_pts, matrix, pixel_size_um, fle_um, image_wh,
                      roi_polygon=None, conf=_CHI2_CONF):
    """The error a CELL experiences, decomposed and summed in quadrature.

        cell_error(p) = sqrt( TRE_pred(p)²  +  deformation_rms_ub² )

    TRE_pred uses σ_eff = √2·FLE — the noise that actually perturbs the fitted similarity
    (both sections contribute) — NOT the fit residual, which already contains deformation
    and would double-count it against the second term.

    Evaluated on a grid over the analysis window (the drawn ROI when there is one), and
    summarised at the p90: the gate should hold almost everywhere a cell can sit, not just
    at the median.
    """
    out = {"tre_pred_median_um": None, "tre_pred_p90_um": None,
           "cell_error_median_um": None, "cell_error_p90_um": None,
           "deformation_p90_ub_um": None, "deformation": None, "sigma_eff_um": None}
    if matrix is None or fle_um is None or not image_wh:
        return out
    deform = deformation_from_landmarks(ref_pts, mov_pts, matrix, pixel_size_um,
                                        fle_um, conf=conf)
    out["deformation"] = deform
    sigma_eff = float(fle_um) * np.sqrt(2.0)
    out["sigma_eff_um"] = round(sigma_eff, 4)
    pred = prediction_error_stats(ref_pts, sigma_eff, image_wh, roi_polygon=roi_polygon)
    if pred["median_um"] is None:
        return out
    # Gate on the LARGER of two deformation bounds, because they fail in opposite regimes.
    #
    #   p90 bound   the quantile read straight off the empirical residuals. Calibrated
    #               against a second annotator on ANHIR (ratio 0.96–1.10, coverage 89–95%)
    #               at n = 44–70. But the 90th percentile of 8 points is essentially the
    #               maximum: at small n it is biased LOW and would certify too easily.
    #   RMS bound   bootstrap/χ² on the scale. Well behaved at small n — its width is
    #               exactly what makes n=8 uncertifiable, correctly — but under-states a
    #               smooth field's p90 by ~1.6× at large n, the unsafe direction.
    #
    # max() takes the RMS bound where the quantile is unreliable and the quantile where the
    # RMS under-states. Both are 95% one-sided, so the max is conservative in both regimes.
    d_ub = max(deform.get("deformation_p90_ub_um") or 0.0,
               deform.get("deformation_rms_ub_um") or 0.0)
    d_med = max(deform.get("deformation_p90_um") or 0.0,
                deform.get("deformation_rms_um") or 0.0)
    out.update(tre_pred_median_um=pred["median_um"], tre_pred_p90_um=pred["p90_um"],
               deformation_p90_ub_um=round(float(d_ub), 3),
               cell_error_median_um=round(float(np.hypot(pred["median_um"], d_med)), 3),
               cell_error_p90_um=round(float(np.hypot(pred["p90_um"], d_ub)), 3))
    return out


def measure_deformation(ref_rgb, mov_rgb, matrix, pixel_size_um, patch=128, stride=96):
    """DEPRECATED — DOES NOT WORK. Kept only so its failure stays documented and tested.

    Intended to measure how far the TISSUE departs from the landmark similarity: apply the
    landmark transform to the moving structural channel, then phase-correlate ref-vs-warped
    over dense tissue-overlap patches.

    It is BLIND. It reports a near-zero residual for ANY transform, including none at all.
    Measured on the real LL477 CD8/TIM-3 pair (0.7519 µm/px):

        certified landmark similarity   ->  0.14 µm
        IDENTITY, ~106 µm true offset   ->  0.22 µm
        uniform 48.8 µm translation     ->  0.18 µm
        Gaussian fold, 54 µm peak       ->  0.20 µm

    The cause is `structural_channel`: hematoxylin OD blurred at σ ≈ 12 µm (16 px) to
    suppress the non-corresponding nuclei of two different sections. That blur removes the
    high-frequency content a displacement estimator needs. At the 128 px patch scale any
    two patches of blurred parenchyma look like the same smooth blob, so the correlation
    peak sits at zero regardless of true offset — the response to displacement is flat, not
    merely attenuated. (Phase correlation itself is fine: it recovers a synthetic np.roll of
    64 px on this same channel exactly. The failure needs two patches of DIFFERENT tissue.)

    NCC template matching and phase correlation on the gradient magnitude were tried as
    replacements and fail on the same images; so does `lumen_tre`, which is additionally
    censored by its tol_um inlier gate. See validation/validate_deformation_estimator.py.

    Callers must NOT gate on this. `landmark_register_and_verify` ignores it for the
    verdict. The returned numbers are retained as an unvalidated diagnostic only.
    """
    import cv2
    out = {"median_um": None, "p90_um": None, "max_um": None, "region_max_um": None,
           "n_patches": 0, "verified_frac": None, "overlap_frac": None,
           "capture_range_um": round(0.5 * patch * float(pixel_size_um), 1),
           "measured": False, "reason": None}
    if matrix is None or ref_rgb is None or mov_rgb is None:
        out["reason"] = "no transform or images available"
        return out
    M = np.asarray(matrix, float)[:2]
    rs = structural_channel(ref_rgb, pixel_size_um)
    ms = structural_channel(mov_rgb, pixel_size_um)
    rmask = tissue_mask(ref_rgb, pixel_size_um)
    mmask = tissue_mask(mov_rgb, pixel_size_um)
    hw = (rs.shape[1], rs.shape[0])
    warped = cv2.warpAffine(ms, M, hw)
    wmask = cv2.warpAffine(mmask, M, hw)
    overlap = ((rmask > 0) & (wmask > 0)).astype(np.uint8)
    recs = patch_residual_flow(rs, warped, overlap, pixel_size_um,
                               patch=patch, stride=stride)

    tissue = rmask > 0
    tissue_px = max(int(tissue.sum()), 1)
    out["overlap_frac"] = round(float((overlap.astype(bool) & tissue).sum() / tissue_px), 3)
    if not recs:
        out["reason"] = ("no tissue-overlap patch carried enough structure to measure "
                         "deformation — the sections cannot be confirmed to align")
        return out

    verified = np.zeros(rs.shape, np.uint8)
    half = patch // 2
    for _resid, cx, cy, _resp in recs:
        y0, x0 = max(int(cy) - half, 0), max(int(cx) - half, 0)
        verified[y0:y0 + patch, x0:x0 + patch] = 1
    out["verified_frac"] = round(float((verified.astype(bool) & tissue).sum() / tissue_px), 3)

    stats = flow_stats(recs, rs.shape)
    out.update(median_um=stats["median_um"], p90_um=stats["p90_um"],
               max_um=stats["max_um"], region_max_um=stats["region_max_um"],
               n_patches=stats["n"], measured=True)
    return out


def cell_registration_error(prediction_um, deformation_um):
    """Registration error at a cell: estimation and model error added in quadrature.

    NOT USED FOR CERTIFICATION. The formula is right, but the model term has no working
    measurement (see measure_deformation), and with `deformation_um` stuck near zero this
    reduces to the estimation term, which shrinks like 1/√n — certifiable by clicking.
    Retained for callers that supply an independently validated deformation estimate.
    """
    if prediction_um is None:
        return None
    d = 0.0 if deformation_um is None else float(deformation_um)
    return round(float(np.hypot(float(prediction_um), d)), 3)


def _hull_area(pts):
    import cv2
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts)))


def _polygon_from_points(poly_pts):
    """Build a valid shapely Polygon from an Nx2 point list, or None."""
    if poly_pts is None or len(poly_pts) < 3:
        return None
    try:
        from shapely.geometry import Polygon
        p = Polygon([(float(x), float(y)) for x, y in poly_pts])
        if not p.is_valid:
            p = p.buffer(0)                      # repair self-intersections
        return p if (p.is_valid and not p.is_empty and p.area > 0) else None
    except Exception:
        return None


def _points_inside(pts, poly):
    """Boolean mask of the Nx2 points inside (or on) the shapely polygon `poly`."""
    from shapely.geometry import Point
    pts = np.asarray(pts, float)
    if not len(pts) or poly is None:
        return np.zeros(len(pts), bool)
    return np.array([poly.covers(Point(float(x), float(y))) for x, y in pts], bool)


def _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac):
    """When the operator drew a Certification ROI, the certified analysis window is the
    ROI intersected with any tighter LOCALLY_CERTIFIED hull. Fail closed (downgrade to
    NOT_CERTIFIABLE) on an empty / invalid / sliver window — never emit a sliver."""
    if roi_poly is None:
        return out
    local_hull = _polygon_from_points(out.get("roi_polygon"))
    if local_hull is not None:
        final, source = roi_poly.intersection(local_hull), "local_hull_within_roi"
    else:
        final, source = roi_poly, "user_roi"
    ok = final is not None and final.is_valid and (not final.is_empty) and final.area > 0
    frac = (final.area / float(image_wh[0] * image_wh[1])) if (ok and image_wh) else 0.0
    if not ok or (image_wh and frac < min_roi_frac):
        out.update(verdict="NOT_CERTIFIABLE", roi_polygon=None,
                   certified_window_source=None,
                   reason="certified region (drawn ROI ∩ landmark-supported area) is too "
                          "small to analyse — enlarge the Certification ROI or improve the "
                          "landmarks inside it")
        return out
    geom = final
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    out["roi_polygon"] = [[float(x), float(y)] for x, y in geom.exterior.coords[:-1]]
    out["certified_window_source"] = source
    return out


def _certify_fitzpatrick_west(out, ref, mov, M, pixel_size_um, fle_um, image_wh,
                              roi_poly, user_roi_polygon, n, min_n, target_n,
                              loo_max_um, deformed_loo_um, min_roi_frac, max_radius_um,
                              min_interpretable_band_frac, model_selected, censor_um=None):
    """Verdict from the FLE-based error budget rather than from held-out landmark TRE.

    Gate: p90 over the analysis window of sqrt(TRE_pred² + deformation_ub²) ≤ loo_max_um.

    Precedence, and why:
      DEFORMED           deformation RMS alone exceeds deformed_loo_um — the sections do
                         not share a similarity, and no ROI or radius floor rescues that.
      NOT_CERTIFIABLE    the landmark set was RANSAC-selected for agreement with a
                         similarity (model_selected). Its residuals cannot test the model
                         they were selected under, so no deformation estimate — and hence
                         no certification — is possible from it. Fail closed.
      CERTIFIED          budget holds across the window.
      LOCALLY_CERTIFIED  budget holds on a spatially-coherent subset whose residuals are
                         consistent with FLE alone (≤3σ_combined); analyse that hull.
      RADIUS_LIMITED     budget exceeded but the pair still shares one similarity; the
                         error is random w.r.t. the cells, so surrender small radii.
    """
    import cv2
    from spatial_stats import registration_radius_floor
    out["gate"] = "fitzpatrick_west"
    out["fle_um"] = round(float(fle_um), 4)
    out["landmarks_are_model_selected"] = bool(model_selected)

    # The analysis window defaults to the LANDMARK HULL, not the whole field. Outside the
    # hull the similarity is extrapolating and its prediction SE grows without bound; a
    # cell there is not covered by any evidence we hold. The legacy gate never noticed
    # this because held-out landmark TRE is computed only AT the landmarks.
    eval_roi = user_roi_polygon
    if eval_roi is None and len(ref) >= 3:
        eval_roi = cv2.convexHull(ref.astype(np.float32)).reshape(-1, 2).tolist()
        out["certified_window_source"] = "landmark_hull"

    budget = cell_error_budget(ref, mov, M, pixel_size_um, fle_um, image_wh,
                               roi_polygon=eval_roi)
    out["cell_error_budget"] = budget
    deform = budget["deformation"] or {}
    out["deformation_rms_um"] = deform.get("deformation_rms_um")
    out["deformation_rms_ub_um"] = deform.get("deformation_rms_ub_um")
    out["deformation_detectable"] = deform.get("detectable")
    out["tre_pred_p90_um"] = budget["tre_pred_p90_um"]
    out["cell_error_p90_um"] = budget["cell_error_p90_um"]

    accuracy_um = budget["cell_error_p90_um"]
    if accuracy_um is None:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason="the landmark layout is degenerate (collinear or too few) — "
                          "the transform's prediction error is undefined, so no error "
                          "budget and no verdict is possible")
        return out
    out["min_interpretable_radius_um"] = registration_radius_floor(accuracy_um)
    tier = "" if n >= target_n else f" (n={n} < {target_n} preferred — provisional)"
    d_rms = deform.get("deformation_rms_um") or 0.0
    d_ub = deform.get("deformation_rms_ub_um") or 0.0
    budget_txt = (f"cell-error p90 {accuracy_um} µm = √(TRE_pred {budget['tre_pred_p90_um']}² "
                  f"+ deformation≤{d_ub}²), FLE {out['fle_um']} µm, n={n}{tier}")

    if d_rms > deformed_loo_um:
        out.update(verdict="DEFORMED",
                   reason=f"tissue deformation RMS {d_rms} µm (residual scatter far beyond "
                          f"the {out['fle_um']} µm localisation noise, p={deform.get('p_value')}) "
                          f"exceeds the {deformed_loo_um} µm limit — the sections do not "
                          f"share a single similarity (no warp applied)")
        return out

    # The proposer could only pair structures within `censor_um`; deformation beyond that
    # was never in the search window and cannot be seen. If the estimate climbs toward the
    # limit, the estimate itself is untrustworthy (censored from above) — fail closed.
    if censor_um is not None and d_rms > 0.5 * float(censor_um):
        out.update(verdict="NOT_CERTIFIABLE",
                   reason=f"estimated deformation {d_rms} µm approaches the "
                          f"{float(censor_um):.0f} µm pairing window of the correspondence "
                          f"finder — the estimate is censored from above and the true "
                          f"deformation may be larger. Re-pair with a wider window.")
        return out

    if deform.get("fle_consistent") is False:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason=f"the declared FLE ({out['fle_um']} µm) is larger than these "
                          f"landmarks' own residual scatter ({deform.get('sigma_fit_um')} µm "
                          f"per coordinate, p={deform.get('p_value_fle_too_high')}) can "
                          f"support — the FLE does not belong to this landmark set, or the "
                          f"residuals were censored by an inlier gate. Re-measure FLE by "
                          f"repeat annotation of THESE landmarks.")
        return out

    if model_selected:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason="these correspondences were selected by RANSAC for agreement "
                          "with a single similarity, so their residuals cannot test that "
                          "model — deformation is unmeasurable from them and the pair "
                          "cannot be certified. Verify/replace them with independently "
                          "placed landmarks (they may still seed the canvas).")
        return out

    if accuracy_um <= loo_max_um:
        out.update(verdict="CERTIFIED", reason=budget_txt)
        if roi_poly is None:
            out["roi_polygon"] = eval_roi          # the hull IS the certified window
            return out
        return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)

    # Locally certified: keep the landmarks whose residual is consistent with FLE alone
    # (≥3σ_combined ⇒ that landmark sits on a fold/tear, or its correspondence is wrong).
    resid_um = np.linalg.norm(_apply_affine(mov, M) - ref, axis=1) * float(pixel_size_um)
    keep = resid_um <= 3.0 * float(fle_um) * np.sqrt(2.0)
    out["n_good"] = int(keep.sum())
    if keep.sum() >= min_n and keep.sum() < n and image_wh:
        gref, gmov = ref[keep], mov[keep]
        roi_frac = _hull_area(gref) / float(image_wh[0] * image_wh[1])
        Mloc = _fit_similarity_robust(gmov, gref)
        if roi_frac >= min_roi_frac and Mloc is not None:
            hull = cv2.convexHull(gref.astype(np.float32)).reshape(-1, 2)
            lb = cell_error_budget(gref, gmov, Mloc, pixel_size_um, fle_um, image_wh,
                                   roi_polygon=hull.tolist())
            if lb["cell_error_p90_um"] is not None and lb["cell_error_p90_um"] <= loo_max_um:
                out.update(verdict="LOCALLY_CERTIFIED", matrix=Mloc.tolist(),
                           cell_error_budget=lb,
                           cell_error_p90_um=lb["cell_error_p90_um"],
                           roi_polygon=[[float(x), float(y)] for x, y in hull],
                           min_interpretable_radius_um=registration_radius_floor(
                               lb["cell_error_p90_um"]),
                           reason=f"{int(keep.sum())} of {n} landmarks have residuals "
                                  f"consistent with localisation noise, spanning "
                                  f"~{roi_frac*100:.0f}% of the field; that ROI holds the "
                                  f"≤{loo_max_um} µm budget (cell-error p90 "
                                  f"{lb['cell_error_p90_um']} µm) — analyse it only")
                return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)

    r_min = out["min_interpretable_radius_um"]
    band_ok = (r_min is not None and image_wh is not None and r_min < max_radius_um
               and (max_radius_um - r_min) >= min_interpretable_band_frac * max_radius_um)
    if band_ok:
        contact = " Direct cell-cell contact (~10–20 µm) is NOT resolved." if r_min > 20 else ""
        out.update(verdict="RADIUS_LIMITED",
                   reason=f"{budget_txt} — above the ≤{loo_max_um} µm gate, so inter-cell "
                          f"distances below {r_min} µm cannot be resolved; the curve there "
                          f"is unmeasurable, not null.{contact} The landmarks still agree "
                          f"on one distance-preserving transform and this error weakens "
                          f"association rather than creating it, so the pair is reported "
                          f"over {r_min}–{max_radius_um:.0f} µm with reduced sensitivity.")
        return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)

    out.update(verdict="DEFORMED",
               reason=f"{budget_txt} — the resulting {r_min} µm resolution limit leaves "
                      f"under {min_interpretable_band_frac*100:.0f}% of the "
                      f"0–{max_radius_um:.0f} µm range readable; no interpretable curve "
                      f"remains (no warp applied)")
    return out


def landmark_register_and_verify(ref_pts, mov_pts, pixel_size_um,
                                 val_ref_pts=None, val_mov_pts=None, image_wh=None,
                                 min_n=CERTIFICATION_GATES["min_n"],
                                 target_n=CERTIFICATION_GATES["target_n"],
                                 loo_max_um=CERTIFICATION_GATES["loo_max_um"],
                                 fit_max_um=CERTIFICATION_GATES["fit_max_um"],
                                 deformed_loo_um=CERTIFICATION_GATES["deformed_loo_um"],
                                 min_roi_frac=CERTIFICATION_GATES["min_roi_frac"],
                                 max_radius_um=CERTIFICATION_GATES["max_radius_um"],
                                 min_interpretable_band_frac=CERTIFICATION_GATES[
                                     "min_interpretable_band_frac"],
                                 deformation=None,
                                 user_roi_polygon=None,
                                 fle_um=None,
                                 landmarks_are_model_selected=False,
                                 censor_um=None):
    """
    GOLD-STANDARD, landmark-DRIVEN registration + verification (Phase A).

    The operator's confident anatomical landmarks DEFINE the registration: a
    least-squares similarity (distance-preserving, so the downstream cross-K stays
    valid — we never non-rigidly warp). Accuracy is measured on HELD-OUT points:

      • if an independent validation set (ideally a SECOND annotator) is supplied,
        TRE is its error under the fit set's transform — annotator-independent.
      • otherwise leave-one-out (LOO). NOTE: LOO is *fit-unbiased* (a point is never
        in the transform that predicts it) but NOT annotator-independent — all points
        share one annotator's selection bias. It is the limited-data fallback, not an
        ANHIR-grade gold standard.

    Five-state verdict (a failed pair is reported, never warped or forced):
      CERTIFIED         n≥min_n, held-out TRE median ≤loo_max_um, fit-residual ≤fit_max_um
      LOCALLY_CERTIFIED only a spatial subset passes (≥min_n, hull ≥min_roi_frac of
                        field) → analyse that ROI only
      RADIUS_LIMITED    the landmarks DO agree on one similarity, but only to within
                        TRE > loo_max_um. Serial sections deform; this is expected. Such
                        error blurs cross-K toward the null: the test stays correctly
                        sized and loses only power (validate_radius_floor.py), so the
                        pair is analysable — it simply cannot resolve inter-cell
                        distances below ~3·TRE. Accepted while a useful stretch of the
                        radius range remains readable above that floor.
      DEFORMED          error too large for any usable radius band to remain
      NOT_CERTIFIABLE   too few unambiguous correspondences to measure accuracy — this
                        is NOT positive evidence the sections are unrelated

    Verdict precedence is deliberate: a field-wide CERTIFIED result beats a locally
    certified one, which beats a radius-limited one. RADIUS_LIMITED keeps the whole
    field but surrenders the smallest radii — i.e. it can no longer speak to direct
    cell-cell contact (~10–20 µm), only to neighbourhood co-localisation. Where a local
    ROI retains the contact scale, that is the stronger scientific claim.

    TWO GATES. Pass `fle_um` (measured by `fle_from_repeat` — two annotation passes) to
    get the Fitzpatrick–West gate: the p90 cell-error budget sqrt(TRE_pred² +
    deformation_ub²) over the analysis window. That is the gate to use. Deformation is
    recovered by variance decomposition against the independently-measured FLE, so
    certifying on the estimation term alone — unsafe, since prediction SE shrinks like
    1/√n and any pair could be certified by clicking more landmarks — is not what happens:
    the deformation term does not shrink with n and holds the gate honest.

    Without `fle_um` this falls back to the LEGACY gate, held-out (leave-one-out) landmark
    TRE. Keep in mind what that measures — the self-consistency of the landmark SET, not
    the accuracy of the registration, and the two are uncorrelated (Fitzpatrick 2009). It
    inherits each landmark's own localisation noise σ (~3–4 µm on real H-DAB serial
    sections), so it FAILS well-registered pairs annotated by hand (σ=3 µm ⇒ LOO 5.8 µm on
    a perfect transform), and it PASSES badly-deformed pairs whose landmarks were
    RANSAC-selected for similarity-consistency (LOO stays ~5 µm as true deformation goes
    0→62 µm). Measured in validation/validate_fw_certification.py. The fallback exists for
    back-compatibility, not because it is defensible.

    Set `landmarks_are_model_selected=True` for any set that came from `propose_landmarks`
    without independent human re-placement: such residuals cannot test the similarity they
    were selected under, and the pair fails closed.

    Thresholds follow the ≤5 µm criterion + serial-section z-gap floor; fixed, not
    tuned. ~target_n well-spread points are wanted for paper-grade; min_n only fits.
    """
    import cv2
    ref = np.asarray(ref_pts, float)
    mov = np.asarray(mov_pts, float)
    out = {"n": len(ref), "matrix": None, "est_scale": None, "fit_residual_um": None,
           "tre_median_um": None, "tre_p90_um": None, "tre_max_um": None,
           "validation": None, "coverage_frac": None, "n_good": 0,
           "roi_polygon": None, "certified_window_source": None,
           "verdict": None, "reason": None,
           "min_interpretable_radius_um": None, "max_radius_um": float(max_radius_um),
           "transform_model": "similarity (rotation + translation + uniform scale)",
           # Reported diagnostics. NOT gates — see the docstring and _SIMILARITY_DOF notes.
           "landmark_noise_um": None, "prediction_error_um": None,
           "gate": "loo", "fle_um": None, "landmarks_are_model_selected": False,
           "cell_error_budget": None, "cell_error_p90_um": None,
           "tre_pred_p90_um": None, "deformation_rms_um": None,
           "deformation_rms_ub_um": None, "deformation_detectable": None,
           "prediction_error_p90_um": None, "deformation_um": None,
           "deformation_max_um": None, "deformation_region_max_um": None,
           "deformation_patches": 0, "deformation_verified_frac": None,
           "deformation_capture_range_um": None,
           "deformation_is_validated": False,
           "accuracy_basis": "leave_one_out_landmark_tre",
           "gates": {"min_n": int(min_n), "target_n": int(target_n),
                     "loo_max_um": float(loo_max_um), "fit_max_um": float(fit_max_um),
                     "deformed_loo_um": float(deformed_loo_um),
                     "min_roi_frac": float(min_roi_frac),
                     "max_radius_um": float(max_radius_um),
                     "min_interpretable_band_frac": float(min_interpretable_band_frac)}}

    # Certification ROI (operator-drawn, full-res ref coords): landmarks OUTSIDE the
    # trusted region cannot drive the verdict — drop them before fitting (fail closed).
    roi_poly = _polygon_from_points(user_roi_polygon)
    if roi_poly is not None:
        keep = _points_inside(ref, roi_poly)
        ref, mov = ref[keep], mov[keep]
        if val_ref_pts is not None and val_mov_pts is not None and len(val_ref_pts):
            vr, vm = np.asarray(val_ref_pts, float), np.asarray(val_mov_pts, float)
            vk = _points_inside(vr, roi_poly)
            val_ref_pts, val_mov_pts = vr[vk], vm[vk]
        if len(ref) < min_n:
            out.update(n=len(ref), verdict="NOT_CERTIFIABLE",
                       reason=f"only {len(ref)} landmark(s) fall inside the drawn "
                              f"Certification ROI (need ≥{min_n}) — enlarge the ROI or "
                              f"add landmarks inside it")
            return out
    n = len(ref)
    out["n"] = n

    # Robust (Huber IRLS) fit: a landmark on a fold or tear bends the transform instead
    # of breaking it. Plain least squares has zero breakdown point — one bad point drags
    # the whole similarity and fails an otherwise certifiable pair.
    M = _fit_similarity_robust(mov, ref) if n >= 2 else None
    if M is not None:
        out["matrix"] = M.tolist()
        out["est_scale"] = round(_affine_scale(M), 4)
        d = np.linalg.norm(_apply_affine(mov, M) - ref, axis=1) * float(pixel_size_um)
        out["fit_residual_um"] = round(float(np.median(d)), 3)
    if image_wh and M is not None:
        out["coverage_frac"] = round(_hull_area(ref) / float(image_wh[0] * image_wh[1]), 4)

    # ── Reported diagnostics (never gates) ───────────────────────────────────────
    # σ is the operator's landmark picking noise; the prediction SE is how precisely n
    # such landmarks pin the transform down where the CELLS are. Together they explain why
    # a well-aligned pair can still show a large held-out TRE. They cannot certify: the
    # model term they would need (real tissue deformation) is not measurable here.
    sigma = landmark_noise_sigma(ref, mov, M, pixel_size_um)
    out["landmark_noise_um"] = None if sigma is None else round(float(sigma), 3)
    pred = prediction_error_stats(ref, sigma, image_wh, roi_polygon=user_roi_polygon)
    out["prediction_error_um"] = pred["median_um"]
    out["prediction_error_p90_um"] = pred["p90_um"]

    # ── Fitzpatrick–West gate (only when FLE has been measured independently) ────
    if fle_um is not None and M is not None and n >= min_n:
        return _certify_fitzpatrick_west(
            out, ref, mov, M, pixel_size_um, float(fle_um), image_wh, roi_poly,
            user_roi_polygon, n, min_n, target_n, loo_max_um, deformed_loo_um,
            min_roi_frac, max_radius_um, min_interpretable_band_frac,
            landmarks_are_model_selected, censor_um)

    # Recorded verbatim, used for nothing. measure_deformation is blind (it returns a
    # near-zero residual even for an identity transform), so a verdict must never move
    # because of it. See validation/validate_deformation_estimator.py.
    deform = deformation or {}
    if deform.get("measured"):
        out["deformation_um"] = deform.get("median_um")
        out["deformation_max_um"] = deform.get("max_um")
        out["deformation_region_max_um"] = deform.get("region_max_um")
        out["deformation_patches"] = int(deform.get("n_patches") or 0)
        out["deformation_verified_frac"] = deform.get("verified_frac")
        out["deformation_capture_range_um"] = deform.get("capture_range_um")

    # Held-out accuracy: independent validation set if supplied, else LOO.
    if val_ref_pts is not None and len(val_ref_pts) >= 1 and M is not None:
        vr, vm = np.asarray(val_ref_pts, float), np.asarray(val_mov_pts, float)
        err = np.linalg.norm(_apply_affine(vm, M) - vr, axis=1) * float(pixel_size_um)
        out["validation"] = f"independent validation set (n={len(err)})"
        local_ok = False
    else:
        loo = loo_tre(ref, mov, pixel_size_um)
        err = np.array(loo["per_point_um"]) if loo["per_point_um"] else np.array([])
        out["validation"] = ("leave-one-out (single-annotator; fit-unbiased, "
                             "NOT annotator-independent)")
        local_ok = True

    if err.size:
        out.update(tre_median_um=round(float(np.median(err)), 3),
                   tre_p90_um=round(float(np.percentile(err, 90)), 3),
                   tre_max_um=round(float(err.max()), 3))
    good = (err <= loo_max_um) if err.size else np.array([], bool)
    out["n_good"] = int(good.sum())
    tier = "" if n >= target_n else f" (n={n} < {target_n} preferred — provisional)"

    if n < min_n or not err.size:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason=f"only {n} confident landmarks — too few unambiguous "
                          f"correspondences to measure accuracy (NOT evidence the "
                          f"sections are unrelated)")
        return out
    med, fr = out["tre_median_um"], out["fit_residual_um"]
    from spatial_stats import registration_radius_floor

    # The verdict is decided on held-out landmark TRE. This over-states a cell's true
    # registration error (it carries the landmarks' own picking noise σ), and we accept
    # that: erring toward more error is the only safe direction for a fail-closed tool.
    accuracy_um, accuracy_max_um = med, loo_max_um

    # Smallest radius whose cross-K value survives this pair's registration error.
    out["min_interpretable_radius_um"] = registration_radius_floor(accuracy_um)

    if accuracy_um is None:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason="registration accuracy could not be estimated from these "
                          "landmarks (degenerate layout) — no verdict is possible")
        return out

    if accuracy_um <= accuracy_max_um:
        reason = (f"held-out TRE median {med} µm (p90 {out['tre_p90_um']}), "
                  f"fit-residual {fr} µm, n={n}{tier}")
        out.update(verdict="CERTIFIED", reason=reason)
        # A drawn ROI becomes the certified analysis window (whole field otherwise).
        return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)
    # Locally certified? a spatially-coherent subset of good points (LOO case only).
    # Preferred over RADIUS_LIMITED: a smaller window that keeps the contact scale says
    # more than the whole field with the contact scale removed.
    if local_ok and out["n_good"] >= min_n:
        gref = ref[:len(err)][good]
        roi_frac = (_hull_area(gref) / float(image_wh[0] * image_wh[1])) if image_wh else 0.0
        if roi_frac >= min_roi_frac:
            Mloc = _fit_similarity_robust(mov[:len(err)][good], gref)
            out.update(verdict="LOCALLY_CERTIFIED",
                       matrix=(Mloc.tolist() if Mloc is not None else out["matrix"]),
                       roi_polygon=[[float(x), float(y)] for x, y in
                                    cv2.convexHull(gref.astype(np.float32)).reshape(-1, 2)],
                       min_interpretable_radius_um=registration_radius_floor(
                           float(np.median(err[good]))),
                       reason=f"{out['n_good']} of {n} landmarks pass within an ROI "
                              f"(~{roi_frac*100:.0f}% of field); analyse that ROI only")
            # Tighten to (drawn ROI ∩ local hull) when the operator drew one.
            return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)

    # Radius-limited: the landmarks agree on ONE distance-preserving similarity, just
    # not to ≤loo_max_um. The error is random with respect to the stained cells, so the
    # cross-K test stays correctly sized and only loses power. Keep the field; surrender
    # the distances the error cannot resolve, provided enough curve remains to read.
    r_min = out["min_interpretable_radius_um"]
    band_ok = (r_min is not None and image_wh is not None
               and r_min < max_radius_um
               and (max_radius_um - r_min) >= min_interpretable_band_frac * max_radius_um)
    if band_ok:
        contact = " Direct cell-cell contact (~10–20 µm) is NOT resolved." if r_min > 20 else ""
        basis = "held-out landmark TRE"
        out.update(verdict="RADIUS_LIMITED",
                   reason=f"{basis} is {accuracy_um} µm, above the ≤{accuracy_max_um} µm "
                          f"gate, so inter-cell distances below {r_min} µm (≈3× that error) "
                          f"cannot be resolved — the curve there is unmeasurable, not "
                          f"null.{contact} The landmarks still agree on one distance-"
                          f"preserving transform and this error weakens association rather "
                          f"than creating it, so the pair is analysable and reported over "
                          f"{r_min}–{max_radius_um:.0f} µm, with reduced sensitivity.")
        return _apply_certification_roi(out, roi_poly, image_wh, min_roi_frac)

    if accuracy_um <= deformed_loo_um:
        out.update(verdict="DEFORMED",
                   reason=f"registration error {accuracy_um} µm exceeds the "
                          f"≤{accuracy_max_um} µm gate, and the resulting {r_min} µm "
                          f"resolution limit leaves under "
                          f"{min_interpretable_band_frac*100:.0f}% of the "
                          f"0–{max_radius_um:.0f} µm range readable — no interpretable "
                          f"curve remains (no warp applied)")
    else:
        out.update(verdict="NOT_CERTIFIABLE",
                   reason=f"registration error {accuracy_um} µm ≫ tolerance — the "
                          f"landmarks do not agree on a single transform; insufficient "
                          f"correspondence to certify")
    return out


def registration_perturbation_sensitivity(stat_fn, base_matrix, tre_um, pixel_size_um,
                                          field_um, n_samples=50, seed=0):
    """
    Phase-B robustness (Codex recommendation): perturb the certified transform within
    its MEASURED landmark uncertainty and re-run the spatial statistic. If the verdict
    (direction + significance) is stable across perturbations the conclusion is
    supported; if it flips it is inconclusive. (A registration error comparable to a
    tested radius means that radius is not interpretable.)

    stat_fn(matrix_2x3) -> dict with 'significant' (bool) and 'direction' (str).
    tre_um: held-out registration TRE; field_um: field half-extent (for rotation jitter).
    Returns {'base', 'n', 'agree_frac', 'stable', 'tre_um'}.
    """
    import math
    rng = np.random.default_rng(seed)
    B = np.vstack([np.asarray(base_matrix, float), [0, 0, 1]])
    base = stat_fn(np.asarray(base_matrix, float))
    sigma_px = tre_um / float(pixel_size_um)
    sigma_rot = tre_um / max(field_um, 1.0)              # rad: arc ≈ TRE at field edge
    agree = 0
    for _ in range(n_samples):
        th = rng.normal(0, sigma_rot)
        c, s = math.cos(th), math.sin(th)
        J = np.array([[c, -s, rng.normal(0, sigma_px)],
                      [s, c, rng.normal(0, sigma_px)], [0, 0, 1]])
        r = stat_fn((J @ B)[:2])
        if (r.get("significant") == base.get("significant")
                and r.get("direction") == base.get("direction")):
            agree += 1
    return {"base": base, "n": n_samples, "agree_frac": round(agree / n_samples, 3),
            "stable": (agree / n_samples) >= 0.9, "tre_um": tre_um}


# ──────────────────────────────────────────────────────────────────────────────
# 5b. Auto-PROPOSE corresponding landmarks for human verification
# ──────────────────────────────────────────────────────────────────────────────
def _structural_corners(struct, mask, pixel_size_um, max_corners=150):
    """Distinctive structural corners on the σ≈12 µm channel (vessel junctions,
    lumen edges, tissue boundaries) — never single nuclei (they are blurred away).
    Spacing ≥18 µm so proposals stay well-separated."""
    import cv2
    md = max(int(18.0 / float(pixel_size_um)), 8)
    c = cv2.goodFeaturesToTrack(struct, maxCorners=max_corners, qualityLevel=0.02,
                                minDistance=md, mask=mask.astype(np.uint8))
    return c.reshape(-1, 2).astype(np.float64) if c is not None else np.zeros((0, 2))


def _grid_seed(ref_lum, mov_lum, center, tol_px, max_shift_px, rot_deg=(-6, -3, 0, 3, 6)):
    """Coarse translation×rotation search (uniform scale = 1, per the scale bars)
    that maps the most moving lumens onto a reference lumen. Independent of any MI
    transform, so it also re-checks pairs the intensity metric mis-aligns."""
    from scipy.spatial import cKDTree
    if len(ref_lum) == 0 or len(mov_lum) == 0:
        return None
    tree = cKDTree(ref_lum)
    step = max(int(max_shift_px / 22), 6)
    best = (-1, None)
    for th in np.radians(rot_deg):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        rot = (R @ (mov_lum - center).T).T + center     # rotate once per angle
        for dx in range(-int(max_shift_px), int(max_shift_px) + 1, step):
            for dy in range(-int(max_shift_px), int(max_shift_px) + 1, step):
                mapped = rot + np.array([dx, dy])
                d, _ = tree.query(mapped)
                h = int((d <= tol_px).sum())
                if h > best[0]:
                    M = np.array([[c, -s, dx + center[0] - (R @ center)[0]],
                                  [s, c, dy + center[1] - (R @ center)[1]]])
                    best = (h, M)
    return best[1]


def _mutual_matches(ref, mov, M, tol_px):
    """Geometrically-consistent mutual-nearest-neighbour matches under transform M."""
    from scipy.spatial import cKDTree
    if len(ref) == 0 or len(mov) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    mapped = _apply_affine(mov, M)
    tr, tm = cKDTree(ref), cKDTree(mapped)
    d_rm, i_rm = tr.query(mapped)
    _d, i_mr = tm.query(ref)
    rr, mm = [], []
    for j, (d, i) in enumerate(zip(d_rm, i_rm)):
        if d <= tol_px and i_mr[i] == j:
            rr.append(ref[i]); mm.append(mov[j])
    return (np.array(rr), np.array(mm)) if rr else (np.zeros((0, 2)), np.zeros((0, 2)))


def _spread_select_idx(pts, k):
    """Farthest-point indices so the selection is well-spread (a good spatial spread
    is what makes the fitted transform — and its TRE — trustworthy)."""
    n = len(pts)
    if n <= k:
        return list(range(n))
    idx = [0]
    while len(idx) < k:
        rest = [i for i in range(n) if i not in idx]
        nxt = max(rest, key=lambda i: min(np.linalg.norm(pts[i] - pts[j]) for j in idx))
        idx.append(nxt)
    return idx


def _spread_select(ref_pts, mov_pts, k):
    sel = _spread_select_idx(ref_pts, k)
    return ref_pts[sel], mov_pts[sel]


def _estimate_landmark_guidance_transform(ref_rgb, mov_rgb, pixel_size_um,
                                          existing_ref_pts=None, existing_mov_pts=None,
                                          seed_transform=None):
    """Estimate a moving→reference similarity for interactive landmark guidance.

    Existing human-confirmed pairs get first priority and are fit with RANSAC so a
    bad earlier click does not dominate the next suggestion. If too few pairs are
    available, fall back to the same structural lumen/corner RANSAC used by
    auto-proposal. Returns (M, meta, ref_struct, mov_struct).
    """
    import cv2

    rs = structural_channel(ref_rgb, pixel_size_um)
    ms = structural_channel(mov_rgb, pixel_size_um)

    eref_src = [] if existing_ref_pts is None else existing_ref_pts
    emov_src = [] if existing_mov_pts is None else existing_mov_pts
    eref = np.asarray(eref_src, dtype=np.float64).reshape(-1, 2)
    emov = np.asarray(emov_src, dtype=np.float64).reshape(-1, 2)
    n = min(len(eref), len(emov))
    if n >= 2:
        eref, emov = eref[:n], emov[:n]
        threshold_px = max(8.0 / float(pixel_size_um), 3.0)
        M = None
        inliers = None
        if n >= 3:
            M, inliers = cv2.estimateAffinePartial2D(
                emov.astype(np.float32), eref.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=threshold_px)
        if M is None:
            M = _fit_similarity_ls(emov, eref)
        if M is not None:
            nin = int(inliers.sum()) if inliers is not None else n
            return M.astype(float), {
                "method": "confirmed_landmark_ransac",
                "n_existing": int(n),
                "n_inliers": int(nin),
            }, rs, ms

    rmask = tissue_mask(ref_rgb, pixel_size_um)
    mmask = tissue_mask(mov_rgb, pixel_size_um)
    H, W = rs.shape
    center = np.array([W / 2.0, H / 2.0])
    tol_px = 12.0 / float(pixel_size_um)
    max_shift_px = 0.15 * max(H, W)

    ref_lum = lumen_centroids(rmask, pixel_size_um)
    mov_lum = lumen_centroids(mmask, pixel_size_um)
    seed = None
    if len(ref_lum) >= 3 and len(mov_lum) >= 3:
        seed = _grid_seed(ref_lum, mov_lum, center, tol_px, max_shift_px)
    if seed is None and seed_transform is not None:
        seed = np.asarray(seed_transform, float)[:2]
    if seed is None:
        return None, {
            "method": "structural_ransac",
            "reason": f"too few structural lumens (ref {len(ref_lum)}, mov {len(mov_lum)})",
        }, rs, ms

    M = seed
    rr, mm = _mutual_matches(ref_lum, mov_lum, seed, 1.4 * tol_px)
    if len(rr) >= 3:
        Mr, _ = cv2.estimateAffinePartial2D(mm.astype(np.float32),
                                            rr.astype(np.float32),
                                            method=cv2.RANSAC,
                                            ransacReprojThreshold=8.0)
        if Mr is not None:
            M = Mr.astype(float)

    ref_all = np.vstack([ref_lum, _structural_corners(rs, rmask, pixel_size_um)]) \
        if len(ref_lum) else _structural_corners(rs, rmask, pixel_size_um)
    mov_all = np.vstack([mov_lum, _structural_corners(ms, mmask, pixel_size_um)]) \
        if len(mov_lum) else _structural_corners(ms, mmask, pixel_size_um)
    rr2, mm2 = _mutual_matches(ref_all, mov_all, M, tol_px)
    inlier_count = len(rr2)
    if len(rr2) >= 3:
        Mf, inl = cv2.estimateAffinePartial2D(mm2.astype(np.float32),
                                              rr2.astype(np.float32),
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=6.0)
        if Mf is not None:
            M = Mf.astype(float)
        if inl is not None:
            inlier_count = int(inl.sum())
    return M, {
        "method": "structural_ransac",
        "n_lumen_ref": int(len(ref_lum)),
        "n_lumen_mov": int(len(mov_lum)),
        "n_inliers": int(inlier_count),
    }, rs, ms


def _local_ncc_refine(ref_struct, mov_struct, ref_xy, mov_xy, search, patch):
    """Snap the moving point to the local zero-mean-NCC maximum around its current
    position, on the structural channel. The search window includes (0,0), so the
    match never gets WORSE; it only nudges the point (≤`search` px) onto the locally
    best-corresponding structure, tightening per-landmark accuracy. Returns
    (refined_mov_xy, confidence in [-1,1]); keeps the point unchanged if the patch is
    featureless (low variance) or falls off the image edge."""
    H, W = ref_struct.shape
    h = patch // 2
    rx, ry = int(round(ref_xy[0])), int(round(ref_xy[1]))
    if rx - h < 0 or ry - h < 0 or rx + h >= W or ry + h >= H:
        return mov_xy, 0.0
    rp = ref_struct[ry - h:ry + h, rx - h:rx + h].astype(np.float32)
    rp = rp - rp.mean()
    rn = float(np.sqrt((rp * rp).sum()))
    if rp.std() < 4.0 or rn < 1e-6:
        return mov_xy, 0.0
    mx0, my0 = int(round(mov_xy[0])), int(round(mov_xy[1]))
    best_ncc, best = -2.0, (mx0, my0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            mx, my = mx0 + dx, my0 + dy
            if mx - h < 0 or my - h < 0 or mx + h >= W or my + h >= H:
                continue
            mp = mov_struct[my - h:my + h, mx - h:mx + h].astype(np.float32)
            mp = mp - mp.mean()
            mn = float(np.sqrt((mp * mp).sum()))
            if mn < 1e-6:
                continue
            ncc = float((rp * mp).sum() / (rn * mn))
            if ncc > best_ncc:
                best_ncc, best = ncc, (float(mx), float(my))
    return best, max(best_ncc, 0.0)


def suggest_moving_landmark(ref_rgb, mov_rgb, ref_point, pixel_size_um,
                            existing_ref_pts=None, existing_mov_pts=None,
                            roi_polygon=None, seed_transform=None):
    """Suggest the moving-image mate for one newly placed reference landmark.

    This is the semi-automated/manual bridge: the user chooses the anatomical
    point in fixed/reference tissue, then RANSAC estimates the current
    moving→reference geometry and the inverse transform predicts the moving-side
    location. A local NCC search snaps the suggestion to the best nearby
    structural match. The returned point is still a proposal; certification later
    uses the unchanged held-out/LOO landmark TRE gates.
    """
    import cv2

    out = {"ok": False, "mov_point": None, "confidence": 0.0, "method": None,
           "n_inliers": 0, "msg": ""}
    try:
        ref_xy = np.asarray(ref_point, dtype=np.float64).reshape(2)
        if roi_polygon is not None and len(roi_polygon) >= 3:
            cnt = np.asarray(roi_polygon, np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(cnt, (float(ref_xy[0]), float(ref_xy[1])), False) < 0:
                out["msg"] = "the fixed-tissue point is outside the Certification ROI"
                return out

        M, meta, rs, ms = _estimate_landmark_guidance_transform(
            ref_rgb, mov_rgb, pixel_size_um,
            existing_ref_pts=existing_ref_pts, existing_mov_pts=existing_mov_pts,
            seed_transform=seed_transform)
        out.update({k: v for k, v in meta.items() if k != "reason"})
        if M is None:
            out["msg"] = meta.get("reason") or "could not estimate a guidance transform"
            return out

        Minv = cv2.invertAffineTransform(np.asarray(M, np.float32))
        pred = _apply_affine(ref_xy.reshape(1, 2), Minv)[0]
        search = max(int(10.0 / float(pixel_size_um)), 4)
        patch = max(int(44.0 / float(pixel_size_um)), 24)
        patch += patch % 2
        refined, conf = _local_ncc_refine(rs, ms, ref_xy, pred, search, patch)
        mov_xy = np.asarray(refined, dtype=np.float64).reshape(2)
        Hm, Wm = ms.shape
        if not (0 <= mov_xy[0] < Wm and 0 <= mov_xy[1] < Hm):
            out["msg"] = "suggested moving point falls outside the moving image"
            return out

        out.update(ok=True,
                   mov_point=[round(float(mov_xy[0]), 2), round(float(mov_xy[1]), 2)],
                   confidence=round(float(conf), 3),
                   msg=("RANSAC-guided moving landmark proposed; verify before "
                        "certifying."))
        return out
    except Exception as e:
        out["msg"] = f"guided landmark failed: {e}"
        return out


def residual_field_assay(ref_pts, mov_pts, matrix, pixel_size_um, fle_um=None,
                         n_perm=999, seed=0):
    """Judge a correspondence SET without ground truth and without trusting the transform.

    Fit a similarity, take the residual VECTORS, and ask whether they are spatially
    autocorrelated (Moran's I on vectors, permutation-tested). Real tissue deformation is a
    smooth field: neighbouring landmarks displace in similar directions. Correspondence
    ERROR — a landmark matched to the wrong structure, or localised badly — is spatially
    random. Three regimes, and they are distinguishable:

        residual ≈ √2·FLE                  → correspondences good, no deformation
        residual ≫ √2·FLE, I > 0, p small  → correspondences good, REAL deformation
        residual ≫ √2·FLE, I ≈ 0, p ≈ 0.5  → correspondences BAD. Says nothing about
                                             deformation. Must not be certified.

    This is the only tool here that separates "the tissue is bent" from "the matcher is
    wrong" using no transform-dependent selection and no ground truth, which makes it the
    acceptance test for any new correspondence source. Controls: a synthetic smooth field
    gives I=0.331 (p=0.001); pure random vectors give I=-0.006 (p=0.153). Needs n ≳ 10 for
    the permutation test to have power.

    Returns {'n', 'residual_median_um', 'moran_i', 'p_value', 'verdict'}.
    """
    ref = np.asarray(ref_pts, float).reshape(-1, 2)
    mov = np.asarray(mov_pts, float).reshape(-1, 2)
    out = {"n": len(ref), "residual_median_um": None, "moran_i": None,
           "p_value": None, "verdict": "UNDETERMINED"}
    if matrix is None or len(ref) < 6:
        return out
    resid = (_apply_affine(mov, np.asarray(matrix, float)) - ref) * float(pixel_size_um)
    out["residual_median_um"] = round(float(np.median(np.linalg.norm(resid, axis=1))), 3)

    D = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    scale = 0.25 * max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1]))
    if scale <= 0:
        return out
    W = np.exp(-(D / scale) ** 2)
    np.fill_diagonal(W, 0.0)
    v = resid - resid.mean(axis=0)
    denom = W.sum() * np.mean((v ** 2).sum(axis=1))
    if denom <= 0:
        return out

    def moran(u):
        return np.einsum('ij,ik,jk->', W, u, u) / denom

    obs = float(moran(v))
    rng = np.random.default_rng(seed)
    null = np.array([moran(v[rng.permutation(len(v))]) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    out.update(moran_i=round(obs, 4), p_value=round(p, 4))

    floor = 2.0 * float(fle_um) if fle_um else None
    if floor is not None and out["residual_median_um"] <= floor:
        out["verdict"] = "CORRESPONDENCES_GOOD_NO_DEFORMATION"
    elif p < 0.05:
        out["verdict"] = "REAL_DEFORMATION"
    else:
        out["verdict"] = "CORRESPONDENCES_BAD"
    return out


def correspondences_for_certification(ref_rgb, mov_rgb, pixel_size_um, seed_transform=None,
                                      pair_tol_um=50.0, lowe_ratio=0.75, min_ncc=0.35,
                                      area_ratio_max=3.0, roi_polygon=None):
    """Model-INDEPENDENT lumen correspondences, suitable for certification.

    MEASURED RESULT: THIS DOES NOT WORK ON CD8/TIM-3 SERIAL SECTIONS. It is kept because
    the negative result is load-bearing for the design, and because the approach is sound
    for any feature type that IS identifiable by appearance.

    On the real LL477 pair it returns 18 correspondences (vs 8 from propose_landmarks) with
    a residual median of 19.3 µm and a maximum of 49.6 µm. `residual_field_assay` calls
    that field spatially RANDOM (Moran I = −0.086, p = 0.63) — so those residuals are
    correspondence error, not deformation. Two independent probes agree that lumen
    centroids cannot be matched across these stains by appearance:

      • no patch descriptor separates correct from incorrect pairings (blurred-structure
        NCC, sharp-hematoxylin NCC, and gradient-magnitude NCC all give AUC 0.48–0.64);
      • SIFT with mutual nearest-neighbour + Lowe ratio and NO RANSAC yields ZERO matches
        at ratio ≤ 0.7, and nonsense at 0.8 (residual median 277 µm on a 1443 µm field).

    THE CONSEQUENCE, and it is the central constraint of this whole module: the RANSAC step
    in `propose_landmarks` is not a lazy filter, it is what ESTABLISHES the correspondence.
    Geometry is doing the matching because appearance cannot. And a correspondence set that
    needed the similarity model to exist cannot then be used to test that model. There is no
    way around this with lumens — only a correspondence source that is identifiable without
    geometry escapes it. A human is one (they match on whole-tissue architecture, not a
    45 µm patch). A learned detector-free matcher (LoFTR, DISK+LightGlue) may be another;
    neither torch nor kornia is installed here, and `residual_field_assay` is the test to
    run on any candidate before trusting it.

    THE ONE RULE. A correspondence may be rejected because it is AMBIGUOUS — a property of
    the images, computable without reference to any transform. It may NEVER be rejected
    because it disagrees with the fitted similarity. `propose_landmarks` breaks that rule
    (cv2.estimateAffinePartial2D + RANSAC, 8 px inlier threshold), which is why its output
    always agrees with a similarity and can therefore never test one: on a pair warped by
    0→55 µm the deformation it implies saturates at ~6 µm, the inlier threshold, and the
    old gate certified a 31 µm-deformed pair. See validation/validate_fw_certification.py.

    So here: pair on APPEARANCE and UNIQUENESS, keep everything that passes, fit nothing.

      1. Lumen centroids in both sections (the same detector propose_landmarks uses).
      2. A seed similarity from lumen overlap, used ONLY to make the search tractable —
         with `pair_tol_um` set far above the DEFORMED gate (default 50 µm vs 15 µm), so a
         deformation large enough to matter is still inside the search window and can still
         be seen. This is a CENSORING LIMIT, returned as `censor_um`: deformation beyond it
         is invisible to this proposer, and the caller must fail closed rather than certify
         when the estimate approaches it.
      3. For each reference lumen, score every candidate within the window by local NCC of
         the structural patch, plus an area-compatibility check (a 12 µm sinusoid is not a
         200 µm vein). Accept only if (a) the match is MUTUAL, (b) the second-best NCC is
         below `lowe_ratio` × best — the classic uniqueness test — and (c) best ≥ min_ncc.
      4. Refine each moving point by local NCC snap centred on the MATCHED LUMEN, never on
         the seed's prediction, so the refinement carries no geometric prior either.

    Returns dict: ref_points, mov_points, ncc, n, censor_um, n_lumen_ref, n_lumen_mov,
    n_rejected_ambiguous, ok, msg.
    """
    import cv2
    out = {"ref_points": [], "mov_points": [], "ncc": [], "n": 0,
           "censor_um": float(pair_tol_um), "n_lumen_ref": 0, "n_lumen_mov": 0,
           "n_rejected_ambiguous": 0, "ok": False, "msg": ""}
    px = float(pixel_size_um)
    rs = structural_channel(ref_rgb, px)
    ms = structural_channel(mov_rgb, px)
    rmask, mmask = tissue_mask(ref_rgb, px), tissue_mask(mov_rgb, px)
    H, W = rs.shape

    def _areas(mask):
        filled = _fill_holes(mask)
        holes = ((filled > 0) & (mask == 0)).astype(np.uint8)
        nn, _lab, st, cen = cv2.connectedComponentsWithStats(holes, connectivity=8)
        lo, hi = (8.0 / px) ** 2, 0.05 * mask.size
        keep = [i for i in range(1, nn) if lo <= st[i, cv2.CC_STAT_AREA] <= hi]
        pts = np.array([[cen[i][0], cen[i][1]] for i in keep], float) if keep else np.zeros((0, 2))
        ar = np.array([st[i, cv2.CC_STAT_AREA] for i in keep], float) if keep else np.zeros(0)
        return pts, ar

    ref_lum, ref_area = _areas(rmask)
    mov_lum, mov_area = _areas(mmask)
    out["n_lumen_ref"], out["n_lumen_mov"] = len(ref_lum), len(mov_lum)
    if len(ref_lum) < 3 or len(mov_lum) < 3:
        out["msg"] = (f"too few lumens to pair (ref {len(ref_lum)}, mov {len(mov_lum)}) — "
                      f"this tissue needs a different correspondence source")
        return out

    center = np.array([W / 2.0, H / 2.0])
    seed = _grid_seed(ref_lum, mov_lum, center, 12.0 / px, 0.15 * max(H, W))
    if seed is None and seed_transform is not None:
        seed = np.asarray(seed_transform, float)[:2]
    if seed is None:
        out["msg"] = "could not seed a search window from lumen overlap"
        return out

    tol_px = pair_tol_um / px
    search = max(int(round(6.0 / px)), 3)
    patch = max(int(round(45.0 / px)) | 1, 9)
    pred = _apply_affine(mov_lum, seed)                 # mov lumens in ref frame

    def _ncc(rxy, mxy):
        h = patch // 2
        rx, ry, mx, my = int(rxy[0]), int(rxy[1]), int(mxy[0]), int(mxy[1])
        if not (h <= rx < W - h and h <= ry < H - h and h <= mx < W - h and h <= my < H - h):
            return -1.0
        a = rs[ry - h:ry + h + 1, rx - h:rx + h + 1].astype(np.float32)
        b = ms[my - h:my + h + 1, mx - h:mx + h + 1].astype(np.float32)
        a, b = a - a.mean(), b - b.mean()
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / d) if d > 1e-6 else -1.0

    # best + second-best appearance match for every reference lumen, inside the window
    best = {}
    for i, rp in enumerate(ref_lum):
        cand = np.where(np.linalg.norm(pred - rp, axis=1) <= tol_px)[0]
        cand = [j for j in cand
                if 1.0 / area_ratio_max <= (ref_area[i] / max(mov_area[j], 1.0)) <= area_ratio_max]
        if not cand:
            continue
        scores = sorted(((_ncc(rp, mov_lum[j]), j) for j in cand), reverse=True)
        if scores[0][0] < min_ncc:
            continue
        if len(scores) > 1 and scores[1][0] > lowe_ratio * scores[0][0]:
            out["n_rejected_ambiguous"] += 1        # two candidates look alike → unusable
            continue
        best[i] = scores[0]

    # mutual consistency: j's best reference must be i (appearance only, no geometry)
    back = {}
    for i, (sc, j) in best.items():
        if j not in back or sc > back[j][0]:
            back[j] = (sc, i)

    poly = _polygon_from_points(roi_polygon)
    rpts, mpts, nccs = [], [], []
    for i, (sc, j) in best.items():
        if back[j][1] != i:
            continue
        rp, mp = ref_lum[i], mov_lum[j]
        # localisation refinement centred on the MATCHED lumen — no seed, no prior
        snapped, conf = _local_ncc_refine(rs, ms, tuple(rp), tuple(mp), search, patch)
        rpts.append(rp)
        mpts.append(np.asarray(snapped, float))
        nccs.append(round(float(conf), 4))
    if not rpts:
        out["msg"] = "no unambiguous mutual lumen correspondences"
        return out
    rpts, mpts = np.array(rpts), np.array(mpts)
    if poly is not None:
        k = _points_inside(rpts, poly)
        rpts, mpts, nccs = rpts[k], mpts[k], [c for c, t in zip(nccs, k) if t]

    out.update(ref_points=rpts.tolist(), mov_points=mpts.tolist(), ncc=nccs,
               n=len(rpts), ok=len(rpts) >= CERTIFICATION_GATES["min_n"],
               msg=(f"{len(rpts)} unambiguous correspondences from {len(ref_lum)}/"
                    f"{len(mov_lum)} lumens ({out['n_rejected_ambiguous']} rejected as "
                    f"ambiguous); deformation above {pair_tol_um:.0f} µm is not visible "
                    f"to this proposer"))
    return out


def propose_landmarks(ref_rgb, mov_rgb, pixel_size_um, max_points=8, seed_transform=None,
                      roi_polygon=None):
    """
    Auto-PROPOSE consistent corresponding landmarks (mov↔ref) for HUMAN verification.

    Machine proposes, human disposes: these correspondences are consistent BY
    CONSTRUCTION (a single similarity relates them), but they are only VALID once a
    human confirms them on the numbered overlays. They pre-load the landmark canvas
    so the operator verifies/nudges ~8 dots instead of clicking 12 from scratch; the
    downstream certification (landmark_register_and_verify: LOO / independent-set
    TRE) is UNCHANGED — proposal never certifies.

    Method (independent of the MI transform):
      1. Detect lumen/sinusoid centroids (holes in the tissue mask) + structural
         corners in BOTH sections on the σ≈12 µm structural channel.
      2. Data-driven seed: translation×rotation grid maximising lumen overlap
         (scale fixed = 1, per the scale bars). Falls back to `seed_transform`.
      3. RANSAC-refine a similarity from lumen matches, re-match the full candidate
         set → geometrically-consistent inlier correspondences.
      4. LOCAL NCC refinement: snap each moving point onto the locally best-matching
         structure (sub-µm) and score its confidence.
      5. COVERAGE-FIRST → ROI cascade: keep the self-consistent points; if their hull
         spans the field return them as a `global` proposal, otherwise return the
         coherent region they DO cover as an `roi` proposal (with roi_polygon) — the
         same coverage-vs-ROI logic the certifier uses, applied at proposal time so
         the operator starts from the set most likely to certify.

    `roi_polygon` (optional, thumbnail/array-pixel REFERENCE coords) is an operator-drawn
    Certification ROI: proposals are restricted to correspondences whose reference point
    falls inside it (fail closed with a clear message if too few remain — never silently
    fall back to a field-wide proposal). The ROI is also mapped into moving space via the
    fitted transform's inverse and returned as `mov_roi_polygon` for the moving-pane crop.

    Returns dict: ref_points/mov_points (Nx2 lists, array-pixel coords), confidences,
    n, n_lumen_ref, n_lumen_mov, mode ('global'|'roi'), coverage_frac,
    fit_residual_um, roi_polygon, mov_roi_polygon, ok, msg.
    """
    import cv2
    out = {"ref_points": [], "mov_points": [], "confidences": [], "n": 0,
           "n_lumen_ref": 0, "n_lumen_mov": 0, "mode": None, "coverage_frac": None,
           "fit_residual_um": None, "roi_polygon": None, "mov_roi_polygon": None,
           "ok": False, "msg": ""}
    try:
        rs = structural_channel(ref_rgb, pixel_size_um)
        ms = structural_channel(mov_rgb, pixel_size_um)
        rmask = tissue_mask(ref_rgb, pixel_size_um)
        mmask = tissue_mask(mov_rgb, pixel_size_um)
        H, W = rs.shape
        center = np.array([W / 2.0, H / 2.0])
        tol_px = 12.0 / float(pixel_size_um)               # ~12 µm matching tolerance
        max_shift_px = 0.15 * max(H, W)                     # serial sections co-locate

        ref_lum = lumen_centroids(rmask, pixel_size_um)
        mov_lum = lumen_centroids(mmask, pixel_size_um)
        out["n_lumen_ref"], out["n_lumen_mov"] = len(ref_lum), len(mov_lum)

        # 2. seed
        seed = None
        if len(ref_lum) >= 3 and len(mov_lum) >= 3:
            seed = _grid_seed(ref_lum, mov_lum, center, tol_px, max_shift_px)
        if seed is None and seed_transform is not None:
            seed = np.asarray(seed_transform, float)[:2]
        if seed is None:
            out["msg"] = ("too few structural lumens to propose landmarks "
                          f"(ref {len(ref_lum)}, mov {len(mov_lum)}); place points manually")
            return out

        # 3. refine on lumen matches, then on the full candidate set
        M = seed
        rr, mm = _mutual_matches(ref_lum, mov_lum, seed, 1.4 * tol_px)
        if len(rr) >= 3:
            Mr, _ = cv2.estimateAffinePartial2D(mm.astype(np.float32),
                                                rr.astype(np.float32),
                                                method=cv2.RANSAC, ransacReprojThreshold=8.0)
            if Mr is not None:
                M = Mr.astype(float)

        ref_all = np.vstack([ref_lum, _structural_corners(rs, rmask, pixel_size_um)]) \
            if len(ref_lum) else _structural_corners(rs, rmask, pixel_size_um)
        mov_all = np.vstack([mov_lum, _structural_corners(ms, mmask, pixel_size_um)]) \
            if len(mov_lum) else _structural_corners(ms, mmask, pixel_size_um)
        rr2, mm2 = _mutual_matches(ref_all, mov_all, M, tol_px)
        if len(rr2) >= 3:
            Mf, inl = cv2.estimateAffinePartial2D(mm2.astype(np.float32),
                                                  rr2.astype(np.float32),
                                                  method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if inl is not None:
                keep = inl.ravel().astype(bool)
                rr2, mm2 = rr2[keep], mm2[keep]

        # Certification ROI: keep only correspondences whose REFERENCE point is inside the
        # operator's trusted region. Fail closed rather than proposing outside it.
        roi_cnt = None
        if roi_polygon is not None and len(roi_polygon) >= 3:
            roi_cnt = np.asarray(roi_polygon, np.float32).reshape(-1, 1, 2)
            inside = np.array([cv2.pointPolygonTest(roi_cnt, (float(x), float(y)), False) >= 0
                               for x, y in rr2], bool)
            rr2, mm2 = rr2[inside], mm2[inside]
            if len(rr2) < 3:
                out["msg"] = ("not enough structural landmarks inside the Certification "
                              "ROI — enlarge the ROI or place landmarks manually inside it")
                return out

        if len(rr2) < 3:
            out["msg"] = ("could not find enough geometrically-consistent "
                          "correspondences; place landmarks manually")
            return out

        # 4. local NCC refinement of each moving point (+ per-point confidence)
        srch = max(int(6.0 / float(pixel_size_um)), 3)     # ±~6 µm snap window
        pch = max(int(40.0 / float(pixel_size_um)), 24); pch += pch % 2
        refined, conf = [], []
        for (rx, ry), (mx, my) in zip(rr2, mm2):
            m2, c = _local_ncc_refine(rs, ms, (rx, ry), (mx, my), srch, pch)
            refined.append(m2); conf.append(c)
        mm2 = np.asarray(refined, float); conf = np.asarray(conf, float)

        # 5. keep self-consistent points, then coverage-first → ROI cascade
        Mfit = _fit_similarity_ls(mm2, rr2)
        resid = (np.linalg.norm(_apply_affine(mm2, Mfit) - rr2, axis=1) * float(pixel_size_um)
                 if Mfit is not None else np.full(len(rr2), np.inf))
        gi = np.where(resid <= 8.0)[0]                      # proposal-stage tolerance
        if len(gi) < 3:
            gi = np.arange(len(rr2))                        # best effort if too few
        gref, gmov, gconf, gres = rr2[gi], mm2[gi], conf[gi], resid[gi]

        area = float(H * W)
        cover = (_hull_area(gref) / area) if area > 0 else 0.0
        mode = "global" if cover >= 0.30 else "roi"        # field-wide vs coherent ROI

        sel = _spread_select_idx(gref, max_points)
        sref, smov = gref[sel], gmov[sel]
        med_res = float(np.median(gres)) if len(gres) and np.isfinite(gres).all() else None

        # Map the reference ROI into moving space (Mfit maps mov→ref, so use its inverse)
        # so the UI can show the cropped moving region the operator is certifying within.
        mov_roi = None
        if roi_polygon is not None and Mfit is not None:
            try:
                Minv = cv2.invertAffineTransform(np.asarray(Mfit, np.float32))
                mroi = _apply_affine(np.asarray(roi_polygon, float), Minv)
                mov_roi = [[round(float(x), 1), round(float(y), 1)] for x, y in mroi]
            except Exception:
                mov_roi = None
        out.update(
            ref_points=[[round(float(x), 2), round(float(y), 2)] for x, y in sref],
            mov_points=[[round(float(x), 2), round(float(y), 2)] for x, y in smov],
            confidences=[round(float(gconf[i]), 3) for i in sel],
            n=len(sel), ok=True, mode=mode, coverage_frac=round(float(cover), 3),
            fit_residual_um=(round(med_res, 3) if med_res is not None else None),
            roi_polygon=([[round(float(x), 1), round(float(y), 1)] for x, y in
                          cv2.convexHull(gref.astype(np.float32)).reshape(-1, 2)]
                         if mode == "roi" and len(gref) >= 3 else None),
            mov_roi_polygon=mov_roi,
            msg=(f"proposed {len(sel)} correspondences · "
                 f"{'field-wide' if mode == 'global' else 'ROI'} coverage "
                 f"{cover*100:.0f}%"
                 + (f" · median self-residual {med_res:.1f} µm" if med_res is not None else "")
                 + " — verify before certifying"))
        return out
    except Exception as e:                                   # never crash the UI
        out["msg"] = f"proposal failed: {e}"
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 6. Human-verifiable QC visualisations (green/magenta + checkerboard)
# ──────────────────────────────────────────────────────────────────────────────
def save_qc_overlays(ref_rgb, mov_rgb, matrix, out_prefix, recs=None, lumens=None):
    """Save a two-colour overlay (ref=green, registered mov=magenta; grey where
    they agree, with patch-flow residual vectors) and a checkerboard. Returns the
    two file paths."""
    import cv2
    ref_g = _rgb_to_gray(ref_rgb)
    mov_g = _rgb_to_gray(mov_rgb)
    Hh, Ww = ref_g.shape
    warped = cv2.warpAffine(mov_g, matrix, (Ww, Hh))

    ov = np.zeros((Hh, Ww, 3), np.uint8)
    ov[..., 1] = ref_g
    ov[..., 0] = warped
    ov[..., 2] = warped
    if recs:
        for resid, cx, cy, _resp in recs:                 # exaggerate ×10 to see
            cv2.circle(ov, (int(cx), int(cy)), 3, (255, 255, 0), -1)
    if lumens is not None and len(lumens.get("ref_matched", [])):
        for (rx, ry), (mx, my) in zip(lumens["ref_matched"], lumens["mapped_matched"]):
            cv2.line(ov, (int(rx), int(ry)), (int(mx), int(my)), (0, 255, 255), 1)
    overlay_path = f"{out_prefix}_overlay.png"
    cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    tile = max(Hh, Ww) // 12
    cb = ref_g.copy()
    for y in range(0, Hh, tile):
        for x in range(0, Ww, tile):
            if ((x // tile) + (y // tile)) % 2 == 1:
                cb[y:y + tile, x:x + tile] = warped[y:y + tile, x:x + tile]
    cb_path = f"{out_prefix}_checkerboard.png"
    cv2.imwrite(cb_path, cb)
    return overlay_path, cb_path


# ──────────────────────────────────────────────────────────────────────────────
# 7. Orchestrator: certify one pair
# ──────────────────────────────────────────────────────────────────────────────
def certify_pair(sample_id, ref_path, mov_path, pixel_size_um, out_dir,
                 ref_bar_px=None, mov_bar_px=None, pixel_size_source="manual",
                 tre_median_max_um=5.0, region_max_um=10.0, min_patches=10,
                 scale_xcheck_tol=0.03):
    """SUPERSEDED — DO NOT USE TO CERTIFY. Kept for validation/phase_a_certify.py only.

    This is the original fully-automatic Phase-A certifier. It decides on
    `register_similarity(...)["flow"]`, i.e. the same phase-correlation patch flow that
    `measure_deformation` uses, and that statistic is BLIND: it reports a near-zero median
    residual even for an identity transform (validation/validate_deformation_estimator.py).
    So this function's CERTIFIED status is close to unconditional and means nothing.

    Production certification goes through `landmark_register_and_verify`, which gates on
    held-out leave-one-out TRE over operator-verified correspondences. Nothing in
    run_pipeline.py or webui/ calls this.
    """
    os.makedirs(out_dir, exist_ok=True)
    row = {"sample_id": sample_id, "pixel_size_um": pixel_size_um,
           "pixel_size_source": pixel_size_source,
           "ref_bar_px": ref_bar_px, "mov_bar_px": mov_bar_px}

    ref_rgb, _ = _load_rgb_thumbnail(ref_path, max_side=1920)
    mov_rgb, _ = _load_rgb_thumbnail(mov_path, max_side=1920)
    if ref_rgb is None or mov_rgb is None:
        row.update(status="NOT CERTIFIED", reason="image load failed")
        return row

    reg = register_similarity(ref_rgb, mov_rgb, pixel_size_um)
    fs = reg["flow"]
    row.update(method=reg["method"], est_scale=reg["est_scale"],
               struct_ncc=reg["struct_ncc"], struct_dice=reg["struct_dice"],
               n_patches=fs["n"], tre_median_um=fs["median_um"],
               tre_p90_um=fs["p90_um"], tre_max_um=fs["max_um"],
               region_max_um=fs["region_max_um"])

    if ref_bar_px and mov_bar_px:
        bar_ratio = ref_bar_px / float(mov_bar_px)
        row["bar_scale_expected"] = round(bar_ratio, 4)
        row["scale_xcheck_delta"] = round(abs(reg["est_scale"] - bar_ratio), 4)
        row["scale_xcheck_ok"] = row["scale_xcheck_delta"] <= scale_xcheck_tol
    else:
        row["bar_scale_expected"] = None
        row["scale_xcheck_delta"] = None
        row["scale_xcheck_ok"] = None

    ref_mask = tissue_mask(ref_rgb, pixel_size_um)
    mov_mask = tissue_mask(mov_rgb, pixel_size_um)
    lum = lumen_tre(ref_mask, mov_mask, reg["matrix"], pixel_size_um)
    row.update(lumen_n_corr=lum["n_corr"], lumen_tre_median_um=lum["median_um"])

    prefix = os.path.join(out_dir, sample_id)
    ov, cb = save_qc_overlays(ref_rgb, mov_rgb, reg["matrix"], prefix,
                              recs=reg["recs"], lumens=lum)
    row["overlay_path"] = ov
    row["checkerboard_path"] = cb

    # ── Gate-A decision ───────────────────────────────────────────────────────
    reasons = []
    if reg["method"] == "identity":
        reasons.append("identity fallback (no structural alignment found)")
    if fs["n"] < min_patches:
        row["status"] = "NEEDS-MY-INPUT"
        row["reason"] = (f"only {fs['n']} confident structural patches "
                         f"(< {min_patches}); supply a few manual landmark points")
        return row

    tre_ok = fs["median_um"] is not None and fs["median_um"] <= tre_median_max_um
    region_ok = fs["region_max_um"] is not None and fs["region_max_um"] < region_max_um
    if not tre_ok:
        reasons.append(f"median local residual {fs['median_um']} µm "
                       f"> {tre_median_max_um} µm")
    if not region_ok:
        reasons.append(f"worst-region residual {fs['region_max_um']} µm "
                       f"≥ {region_max_um} µm")
    if row["scale_xcheck_ok"] is False:
        reasons.append(f"scale cross-check failed "
                       f"(est {reg['est_scale']} vs bar {row['bar_scale_expected']})")

    if (reg["method"] != "identity" and tre_ok and region_ok
            and row["scale_xcheck_ok"] is not False):
        row["status"] = "CERTIFIED"
        row["reason"] = (f"median local residual {fs['median_um']} µm "
                         f"(p90 {fs['p90_um']}, worst-region {fs['region_max_um']}, "
                         f"n={fs['n']} patches); lumen TRE {lum['median_um']} µm "
                         f"(n={lum['n_corr']})")
    else:
        row["status"] = "NOT CERTIFIED"
        row["reason"] = "; ".join(reasons) if reasons else "failed certification"
    return row
