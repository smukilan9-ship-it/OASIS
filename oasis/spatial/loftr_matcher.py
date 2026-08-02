"""
LoFTR correspondences for certification — a MODEL-FREE matcher.

WHY THIS MODULE EXISTS. Certifying a registration means testing a similarity transform
against correspondences. Any correspondence set that was *selected* for agreeing with a
similarity cannot perform that test (see serial_registration.correspondences_for_
certification for the measured proof). `propose_landmarks` selects with RANSAC, so its
output always agrees and the old gate certified a pair carrying 31 µm of deformation.

Lumen centroids cannot be matched by appearance — no patch descriptor separates correct
from incorrect pairings (AUC 0.48–0.64), and SIFT with mutual-NN + Lowe returns zero
matches at any sane ratio. Geometry was doing the matching, which is why it could not then
test geometry.

LoFTR is detector-free and attends over the whole image, so it carries the global
architectural context that a 45 µm patch does not — the same context a human uses to match
a vessel across two stains. On the real LL477 CD8/TIM-3 pair it returns ~750 raw matches
where lumen matching returned 8.

SELECTION IS MODEL-FREE, AND DELIBERATELY NOT A CONFIDENCE THRESHOLD. Picking a confidence
cut by watching the residual tail shrink is a mild form of the circularity this module
exists to avoid: the residual is a function of the transform under test. Instead we filter
on two properties of the MATCHER, computable with no transform at all:

  CYCLE CONSISTENCY   match ref→mov and mov→ref independently. A correspondence survives
                      only if the round trip returns to where it started, within `tol_um`.
                      A mismatch in a repetitive field almost never round-trips, because
                      the backward pass has a different set of distractors.
  SCALE CONSISTENCY   match at two image scales. A correspondence survives only if both
                      scales agree within `tol_um`. Real structure is scale-stable;
                      texture aliasing is not.
  LOCAL SMOOTHNESS    a correspondence survives only if its displacement agrees with the
                      median displacement of its nearest neighbours. The first two filters
                      test the MATCHER against itself, and a match to the wrong instance of
                      a repeated structure passes both — the reverse pass and the coarser
                      scale make the same mistake for the same reason. This one asks a
                      question neither can: is the displacement field continuous here?

None of the three can see the similarity, so the surviving residuals are admissible evidence
about it. The third constrains only LOCAL continuity, never global form, so a tissue that no
similarity describes survives it intact — see `_local_smoothness`.

`serial_registration.residual_field_assay` is the acceptance test for the output
(and note it catches RANDOM error, not SYSTEMATIC error: LoFTR's `indoor` weights produce
confidently wrong but spatially SMOOTH matches on this data, residual median 77–170 µm on a
1443 µm field, which the assay happily labels REAL_DEFORMATION. Weight choice matters and
must be validated externally.)

Requires torch + kornia. Weight download may need SSL_CERT_FILE=$(python -m certifi).
"""
import math
import numpy as np

_MATCHER = {}
_DEVICE = None

# Content-addressed memoization of the DETERMINISTIC (noise==0) passes. Two crops with the
# same pixels produce the same hematoxylin prep and the same LoFTR output, so a re-probed
# region or a re-run pair costs nothing. Keyed on image CONTENT (not id(), which Python
# reuses for freshly-allocated crops). Bypassed entirely when noise>0 so the FLE trials stay
# stochastic. Bounded so a long batch cannot grow memory without limit.
_PREP_CACHE = {}
_RAW_CACHE = {}
# Prep entries are whole downsampled images (MBs each) but are only reused WITHIN a call
# (forward+reverse pass of the same crop), so a small bound suffices. Raw entries are tiny
# keypoint arrays reused ACROSS calls (re-probed crop, re-run pair), so they get a big bound.
_PREP_CACHE_MAX = 48
_RAW_CACHE_MAX = 512

# LoFTR's fiducial localisation error, in WORKING pixels — measured, not assumed.
#
# validation/validate_loftr_fle_groundtruth.py warps a real H-DAB field by a transform we
# choose and matches it against itself, so the true partner of every point is known
# analytically. Across thirteen warps and two working scales the per-point error is
# 0.121-0.195 working px (median 0.16), and it is far more nearly constant in WORKING PIXELS
# than in µm (CV 0.09 vs 0.41) — it is a property of the matcher's grid, so it scales with
# px_work. Two independent estimates agree: `loftr_fle` reports 0.199 µm on the disputed ROI
# (0.11 working px there) and the semivariogram nugget of that pair's real residuals gives
# 0.409 µm (0.22 working px).
#
# WHY A CONSTANT INSTEAD OF MEASURING PER REGION. `loftr_fle` re-runs the whole pipeline under
# image noise, which bypasses the content cache by design, and it costs 14.3 s per region —
# measured. What it buys is a sharper value for a term that research/registration.md § 11.3
# shows explains 1 % of the residual variance on real serial sections. Paying 14 s a region to
# refine 1 % is a bad trade, and it was the dominant cost of a certification run.
#
# 0.16 is the self-matched floor, which is the CONSERVATIVE end: a smaller declared FLE charges
# more of the residual to deformation and so certifies less. The variogram's 0.22 would be
# slightly more permissive. The guard against a constant that is too large for a given pair
# already exists — `p_value_fle_too_high` in the FW certification rejects a declared FLE the
# residuals cannot support, which is what matters when a pair is genuinely well registered and
# σ_fit approaches FLE.
# Agreement required between a REGION's local transform and the whole-field provisional one.
# Calibrated from the cohort: well-registered pairs hold their regions to 0.22-0.36 deg of
# rotation spread and 0.002-0.007 in scale, while broken ones reach 61.9 deg and 0.53. The
# thresholds sit an order of magnitude above the good pairs and an order below the bad, so
# genuine local deformation is never mistaken for a bad fit.
GLOBAL_AGREEMENT_MAX_DEG = 5.0
GLOBAL_AGREEMENT_MAX_SCALE = 0.05

FLE_WORKING_PX = 0.16


def _arr_key(a):
    import hashlib
    a = np.ascontiguousarray(a)
    return (a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest())


def clear_loftr_caches():
    """Drop the memoized prep/inference. Call between unrelated image pairs to cap memory."""
    _PREP_CACHE.clear()
    _RAW_CACHE.clear()


def _device():
    """CPU by default. Measured: MPS gives NO speedup for LoFTR here (its attention ops
    fall back), so CPU is the reliable default. CUDA is used when present; MPS only if
    explicitly requested with LOFTR_GPU=1."""
    global _DEVICE
    if _DEVICE is None:
        import os
        import torch
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif os.environ.get("LOFTR_GPU") and torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def _get(weights):
    if weights not in _MATCHER:
        from kornia.feature import LoFTR
        _MATCHER[weights] = LoFTR(pretrained=weights).eval().to(_device())
    return _MATCHER[weights]


def _prep(rgb, scale, pixel_size_um, noise=0.0, rng=None):
    """Hematoxylin — the one channel both stains share — CLAHE-equalised, for LoFTR.

    Deterministic results (noise==0) are memoized: every correspondences call preps the same
    image for BOTH its forward and reverse pass at scale 0.75, so caching removes that
    duplicate hematoxylin+CLAHE work outright."""
    import cv2, torch
    ck = None
    if not noise:
        ck = (_arr_key(rgb), round(float(scale), 6))
        hit = _PREP_CACHE.get(ck)
        if hit is not None:
            return hit
    from oasis.common.registration import extract_hematoxylin
    h = extract_hematoxylin(rgb).astype(np.float32)
    if noise:
        h = h + rng.normal(0, noise * 255.0, h.shape)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(h)
    H, W = h.shape
    nh, nw = max(int(H * scale) // 8 * 8, 8), max(int(W * scale) // 8 * 8, 8)
    small = cv2.resize(h, (nw, nh), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(small).float()[None, None] / 255.0
    out = (t, np.array([W / nw, H / nh]))
    if ck is not None:
        if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
            _PREP_CACHE.clear()
        _PREP_CACHE[ck] = out
    return out


def _raw(a_rgb, b_rgb, scale, pixel_size_um, weights, conf_floor, noise=0.0, rng=None):
    """One directed LoFTR pass, returned in full-resolution pixel coordinates.

    Deterministic passes (noise==0) are memoized by image content, so a crop that was already
    probed — or a whole pair re-run — returns instantly instead of re-running the transformer."""
    global _DEVICE, _MATCHER
    import torch
    ck = None
    if not noise:
        ck = (_arr_key(a_rgb), _arr_key(b_rgb), round(float(scale), 6),
              weights, round(float(conf_floor), 4))
        hit = _RAW_CACHE.get(ck)
        if hit is not None:
            return hit
    ta, sa = _prep(a_rgb, scale, pixel_size_um, noise, rng)
    tb, sb = _prep(b_rgb, scale, pixel_size_um, noise, rng)
    dev = _device()
    try:
        with torch.inference_mode():
            o = _get(weights)({"image0": ta.to(dev), "image1": tb.to(dev)})
        k0 = o["keypoints0"].detach().cpu().numpy() * sa
        k1 = o["keypoints1"].detach().cpu().numpy() * sb
        cf = o["confidence"].detach().cpu().numpy()
    except Exception:
        # a GPU op LoFTR uses may be unsupported on this backend — fall back to CPU for good.
        _DEVICE = torch.device("cpu"); _MATCHER.clear()
        with torch.inference_mode():
            o = _get(weights)({"image0": ta, "image1": tb})
        k0 = o["keypoints0"].numpy() * sa
        k1 = o["keypoints1"].numpy() * sb
        cf = o["confidence"].numpy()
    k = cf >= conf_floor
    # stride of the coarse matching grid, in full-resolution pixels
    out = (k0[k], k1[k], cf[k], float(8.0 / scale))
    if ck is not None:
        if len(_RAW_CACHE) >= _RAW_CACHE_MAX:
            _RAW_CACHE.clear()
        _RAW_CACHE[ck] = out
    return out


def _disp_agree(src_a, dst_a, src_b, dst_b, tol_px, lookup_px):
    """Keep matches of A whose nearest neighbour in B predicts the same DISPLACEMENT.

    Compare (q − p) against (q' − p'), not q against q'. The two passes place their
    keypoints on different coarse grids (a reverse pass grids the moving image; a coarser
    scale grids at a different stride), so absolute positions are offset by up to a grid
    cell even when the two agree perfectly. Differencing cancels that offset. `lookup_px`
    is the neighbour-search radius and must therefore exceed the grid stride, while
    `tol_px` is the actual agreement tolerance and stays tight.
    """
    if not len(src_b):
        return np.zeros(len(src_a), bool)
    from scipy.spatial import cKDTree
    src_a = np.asarray(src_a, float); dst_a = np.asarray(dst_a, float)
    src_b = np.asarray(src_b, float); dst_b = np.asarray(dst_b, float)
    disp_b = dst_b - src_b
    # exact nearest neighbour of every A point in B — same selection as the argmin loop,
    # O((N+M)logM) via the tree instead of O(N·M) pairwise distances.
    dist, idx = cKDTree(src_b).query(src_a, k=1)
    disp_a = dst_a - src_a
    agree = np.linalg.norm(disp_b[idx] - disp_a, axis=1) <= tol_px
    return (np.asarray(dist) <= lookup_px) & agree


def _local_smoothness(src, dst, tol_px, k=8):
    """Keep matches whose displacement agrees with their k nearest NEIGHBOURS' displacement.

    THE THIRD FILTER, AND WHY IT IS NEEDED. Cycle and scale test the MATCHER's self-
    consistency, and a match to the wrong instance of a repeated structure can pass both:
    the reverse pass makes the same mistake for the same reason, and so does the coarser
    scale. Measured on real CD8/TIM-3 pairs, roughly 10-20% of the survivors are gross
    errors — one field had a 7 µm median residual and a 631 µm maximum. Nothing downstream
    removes them: `_fit_similarity_robust` is Huber, which down-weights but never rejects,
    and its docstring's assumption that "the researcher validates every correspondence"
    holds for hand-placed landmarks, not for a dense matcher's output. Those few points then
    set the residual p90 that `cell_error_budget` gates on, and they destroy the Moran's I
    that `residual_field_assay` needs, so the pair is failed AND misdiagnosed as deformed
    tissue.

    WHY THIS IS NOT THE CIRCULARITY THE MODULE EXISTS TO AVOID. RANSAC is barred here
    because it selects for agreement with the similarity under test, leaving residuals that
    cannot test it. This filter never forms a similarity. It compares each match only with
    its immediate spatial neighbours — tens of µm away in a dense field — so it constrains
    the displacement field to be LOCALLY continuous and says nothing about whether that
    field is globally a rotation, a scale, or a translation. A smoothly warped tissue that
    no similarity can describe passes this filter completely intact, which is exactly why
    the certification that follows retains its power to reject.

    THE ASSUMPTION IT DOES ADD, stated plainly: the true displacement field is continuous at
    the k-neighbour scale. Serial sections satisfy this away from tears and folds; ACROSS a
    genuine tear it would discard the correct correspondences on the minority side. Callers
    get `local_drop_frac` so an implausible cull is visible rather than silent.

    Threshold is `max(tol_px, median + 3·MAD)` of the neighbour deviation: never stricter
    than the module's declared agreement tolerance, and above that a robust adaptive cut
    that needs no per-image tuning. Deterministic — no resampling.
    """
    from scipy.spatial import cKDTree
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = len(src)
    if n <= k + 1:
        # Too few points for the neighbourhood to be LOCAL: with k ≈ n the median
        # displacement is the field's global median, i.e. a translation, and rejecting
        # against it would start selecting for a motion model — the very circularity this
        # filter is designed to avoid. Shrinking k instead would be worse (fewer points per
        # median = noisier). Pass everything through and let the certification gate decide.
        return np.ones(n, bool)
    d = dst - src
    _, idx = cKDTree(src).query(src, k=k + 1)
    med = np.median(d[idx[:, 1:]], axis=1)          # neighbours only, never self
    dev = np.linalg.norm(d - med, axis=1)
    m = float(np.median(dev))
    mad = float(np.median(np.abs(dev - m))) * 1.4826
    return dev <= max(float(tol_px), m + 3.0 * mad)


def loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um, weights="outdoor",
                          scales=(0.75, 0.5), tol_um=4.0, conf_floor=0.2,
                          noise=0.0, rng=None, local_k=8, scale_tol_stride=0.5):
    """Cycle-, scale- and locally-smooth LoFTR correspondences. No RANSAC, no residuals.

    `noise`/`rng` perturb both images identically-in-distribution; used by `loftr_fle` to
    re-run this WHOLE pipeline under noise, so the FLE it measures belongs to the selected
    population rather than to the raw matcher.

    `local_k` sets the neighbourhood of the third filter (`_local_smoothness`); 0 disables
    it, which is how the shipped behaviour before it was added can be reproduced.

    Returns dict: ref_points, mov_points, confidence, n, n_raw, n_after_cycle,
    n_after_scale, n_after_local, local_drop_frac, tol_um, weights, ok, msg.
    """
    tol_px = float(tol_um) / float(pixel_size_um)
    s0 = scales[0]

    fk0, fk1, fcf, stride0 = _raw(ref_rgb, mov_rgb, s0, pixel_size_um, weights,
                                  conf_floor, noise, rng)
    out = {"ref_points": [], "mov_points": [], "confidence": [], "n": 0,
           "n_raw": int(len(fk0)), "n_after_cycle": 0, "n_after_scale": 0,
           "n_after_local": 0, "local_drop_frac": 0.0,
           "tol_um": float(tol_um), "weights": weights, "ok": False, "msg": ""}
    if len(fk0) < 6:
        out["msg"] = f"LoFTR returned only {len(fk0)} raw matches"
        return out

    # CYCLE: the reverse pass is an independent run whose distractor set differs.
    bk1, bk0, _, _ = _raw(mov_rgb, ref_rgb, s0, pixel_size_um, weights, conf_floor, noise, rng)
    keep = _disp_agree(fk0, fk1, bk0, bk1, tol_px, stride0)
    fk0, fk1, fcf = fk0[keep], fk1[keep], fcf[keep]
    out["n_after_cycle"] = int(len(fk0))
    if len(fk0) < 6:
        out["msg"] = f"only {len(fk0)} matches survive cycle consistency"
        return out

    # SCALE: a coarser pass must predict the same displacement at the same place.
    if len(scales) > 1:
        gk0, gk1, _, stride1 = _raw(ref_rgb, mov_rgb, scales[1], pixel_size_um,
                                    weights, conf_floor, noise, rng)
        # THE TOLERANCE AND THE GRID IT IS APPLIED TO. `tol_px` is an absolute distance
        # derived from tol_um, but the coarse pass matches on a stride1 = 8/scale grid — 16
        # full-resolution pixels at scale 0.5. Requiring it to agree to a fraction of its own
        # cell is a demand it structurally cannot meet, and it shows: measured on real
        # 270 µm ROIs, 224 cycle-consistent matches fall to 46 here, and 5 survive inside the
        # polygon, which the UI then reports as NO_MATCHES on a region that had 283 raw
        # matches.
        #
        # `scale_tol_stride` floors the tolerance at a multiple of the coarse stride, which
        # keeps the filter's PURPOSE (a match whose displacement the coarse pass contradicts
        # beyond what that pass can resolve is aliasing) while dropping a precision demand
        # that was never justified.
        #
        # 0.5 IS EARNED, NOT CHOSEN. Relaxing a filter is the over-certifying direction, so
        # validate_scale_filter_anhir.py A/B'd four arms over 29 ANHIR training pairs against
        # expert landmarks LoFTR never sees — the only non-circular test, since a filter
        # selects the points its own residual is measured on. Overall the arms are
        # indistinguishable, because ANHIR's median pair already has 427 correspondences and
        # was never near the threshold. The answer is in the 10 pairs with fewer than 200,
        # which is the regime that produces NO_MATCHES:
        #
        #                       n gain   TRE med   paired Δ mean   WORST
        #   filter OFF           4.20x     71.09          +8.85   +47.98
        #   stride-aware 0.5x    1.64x     36.04          −2.39    +2.13
        #   stride-aware 1.0x    1.84x     49.32          +1.77   +25.27
        #
        # So the filter IS doing real work — off degrades hard pairs badly — and 1.0x is too
        # loose. 0.5x gains coverage, improves accuracy, and bounds the worst case. 0.0
        # restores the absolute-only behaviour for anyone reproducing an older result.
        tol_scale_px = tol_px
        if scale_tol_stride:
            tol_scale_px = max(tol_px, float(scale_tol_stride) * float(stride1))
        out["tol_scale_px"] = round(float(tol_scale_px), 3)
        out["coarse_stride_px"] = round(float(stride1), 3)
        keep = _disp_agree(fk0, fk1, gk0, gk1, tol_scale_px, stride1)
        fk0, fk1, fcf = fk0[keep], fk1[keep], fcf[keep]
    out["n_after_scale"] = int(len(fk0))

    # LOCAL SMOOTHNESS: the survivors must agree with their own neighbours. Still no
    # transform — see `_local_smoothness` for why this does not reintroduce circularity.
    if local_k and len(fk0) > int(local_k) + 1:
        keep = _local_smoothness(fk0, fk1, tol_px, k=int(local_k))
        n_before = len(fk0)
        fk0, fk1, fcf = fk0[keep], fk1[keep], fcf[keep]
        out["local_drop_frac"] = round(1.0 - len(fk0) / float(n_before), 4)
    out["n_after_local"] = int(len(fk0))
    if len(fk0) < 6:
        out["msg"] = f"only {len(fk0)} matches survive local smoothness"
        return out

    out.update(ref_points=fk0.tolist(), mov_points=fk1.tolist(),
               confidence=[round(float(c), 4) for c in fcf], n=int(len(fk0)),
               ok=len(fk0) >= 6,
               msg=(f"{out['n_raw']} raw → {out['n_after_cycle']} cycle-consistent → "
                    f"{out['n_after_scale']} scale-consistent → {len(fk0)} locally smooth "
                    f"(dropped {out['local_drop_frac']:.0%}, tol {tol_um} µm); "
                    f"no transform used"))
    return out


def _roi_bbox(poly, W, H, pad):
    x0 = max(int(np.floor(poly[:, 0].min() - pad)), 0)
    y0 = max(int(np.floor(poly[:, 1].min() - pad)), 0)
    x1 = min(int(np.ceil(poly[:, 0].max() + pad)), W)
    y1 = min(int(np.ceil(poly[:, 1].max() + pad)), H)
    return x0, y0, x1, y1


def certify_local_roi(ref_rgb, mov_rgb, roi_polygon_ref, pixel_size_um,
                      provisional_matrix=None, fallback_ref_lm=None, fallback_mov_lm=None,
                      weights="outdoor", tol_um=4.0, min_matches=8, work_max_dim=800,
                      return_correspondences=False, fle_fast=True, loftr_kw=None,
                      matcher="loftr"):
    """Certify a user-drawn ROI by a LOCAL rigid fit from LoFTR correspondences inside it.

    THE WHOLE POINT. Serial-section deformation is smooth, so a similarity fit CONFINED to a
    small region leaves far less error than a global one (validated: mammary 335->117 um as
    the window shrinks). And LoFTR's one weakness -- choking on huge, coarse whole-slides --
    disappears inside a small ROI, because that patch is matched at (near) full resolution.

    roi_polygon_ref : Nx2 polygon in FIXED (reference) image pixels -- the shape the user drew.
    provisional_matrix : the current moving->reference similarity (codebase convention). Used
                         ONLY to locate the corresponding patch on the moving image so LoFTR
                         has something to match; the FINAL fit is recomputed locally and does
                         not trust it. Identity if None.
    fallback_*_lm : optional corresponding landmarks (full-res px) used if LoFTR cannot find
                    enough matches in the ROI -- the graceful degradation path.

    The gate is NOT relaxed: correspondences go through the ordinary Fitzpatrick-West
    certification with the user ROI as the window. The user chooses WHERE; the pass is still
    earned. Returns verdict, local matrix, cell-error, and which source supplied the matches.
    """
    import cv2
    from matplotlib.path import Path as _MplPath
    from oasis.spatial import serial_registration as sr

    roi = np.asarray(roi_polygon_ref, float).reshape(-1, 2)
    Hr, Wr = ref_rgb.shape[:2]
    Hm, Wm = mov_rgb.shape[:2]
    pad = max(80.0 / float(pixel_size_um), 40.0)

    # map the ROI onto the moving image via the provisional transform (invert mov->ref)
    if provisional_matrix is not None:
        M0 = np.asarray(provisional_matrix, float)
        A = M0[:2, :2] if M0.shape == (3, 3) else M0[:, :2]
        t = M0[:2, 2] if M0.shape == (3, 3) else M0[:, 2]
        Ainv = np.linalg.inv(A)
        roi_mov = (roi - t) @ Ainv.T
    else:
        roi_mov = roi.copy()

    # CUT THE REFERENCE ROI TO WHERE MOVING TISSUE ACTUALLY EXISTS.
    #
    # _roi_bbox clamps the MOVING crop to the moving image, but the reference crop was taken
    # from the full drawn ROI — so when the transform pushes part of the region off the
    # moving frame, LoFTR matched a full reference patch against a truncated moving one and
    # the whole reference polygon was still reported as certified. The strip with no moving
    # counterpart got certified on evidence that could not cover it, and every cell in it
    # entered the analysis window with no possible partner.
    #
    # run_spatial_association removes that strip later via the A∩B tissue intersection, so
    # the certified area and the analysed area silently disagreed. Cutting it here makes the
    # certificate describe the window that is actually analysed.
    roi_eff, roi_clip_frac = roi, 1.0
    try:
        from shapely.geometry import Polygon as _P
        mov_rect = np.array([[0.0, 0.0], [Wm, 0.0], [Wm, Hm], [0.0, Hm]], float)
        # moving frame -> reference space (M0 maps moving onto reference)
        if provisional_matrix is not None:
            footprint = mov_rect @ A.T + t
        else:
            footprint = mov_rect
        pr, pf = _P(roi).buffer(0), _P(footprint).buffer(0)
        inter = pr.intersection(pf)
        if (not inter.is_empty) and inter.area > 0 and pr.area > 0:
            roi_clip_frac = float(inter.area / pr.area)
            if roi_clip_frac < 0.999:
                g = max(inter.geoms, key=lambda x: x.area) if \
                    inter.geom_type == "MultiPolygon" else inter
                roi_eff = np.asarray(g.exterior.coords[:-1], float)
                roi = roi_eff
                if provisional_matrix is not None:
                    roi_mov = (roi - t) @ Ainv.T
                else:
                    roi_mov = roi.copy()
    except Exception:
        pass                      # geometry unavailable: fall back to the drawn ROI

    rx0, ry0, rx1, ry1 = _roi_bbox(roi, Wr, Hr, pad)
    mx0, my0, mx1, my1 = _roi_bbox(roi_mov, Wm, Hm, pad)
    crop_r = ref_rgb[ry0:ry1, rx0:rx1]
    crop_m = mov_rgb[my0:my1, mx0:mx1]
    if crop_r.size == 0 or crop_m.size == 0:
        return {"ok": False, "verdict": "ROI_EMPTY", "msg": "ROI maps outside an image"}

    # downscale both crops by ONE factor (shared working um/px keeps LoFTR scale-matched)
    long_side = max(crop_r.shape[0], crop_r.shape[1], crop_m.shape[0], crop_m.shape[1])
    r = min(1.0, float(work_max_dim) / float(long_side))
    def _rs(im):
        return cv2.resize(im, (max(int(im.shape[1] * r), 8), max(int(im.shape[0] * r), 8)),
                          interpolation=cv2.INTER_AREA) if r < 1.0 else im
    small_r, small_m = _rs(crop_r), _rs(crop_m)
    px_work = float(pixel_size_um) / r

    source = "loftr_in_roi"
    ref_pts = mov_pts = None
    fle_um = None
    n_in_roi = None      # survivors that actually fell inside the polygon, for the failure path
    # `loftr_kw` reaches the matcher's own knobs (e.g. local_k=0 to reproduce the pre-
    # local-smoothness selection) so a validation run can A/B the filter through the real
    # certification path instead of a reimplementation of it.
    # DISPATCHED, and the default is LoFTR because it is the only matcher that works on the
    # target cohort. validate_matchers_on_cohort.py benchmarked five on LL477 both ways:
    # against a synthetic warp (exact truth, same tissue) EVERY matcher is sub-micron with no
    # gross errors, so they all handle spindle-cell H-DAB texture. On the REAL cross-stain
    # pairs only LoFTR survives — DISK and SIFT return 0 matches, DeDoDe returns thousands of
    # which 37-67 % are gross, KeyNet returns ~90 of which 100 % are gross. `matcher="disk"`
    # or "auto" remain reachable for cohorts like ANHIR where DISK is measurably cleaner.
    from oasis.spatial.sparse_matcher import correspondences as _match
    c = _match(small_r, small_m, pixel_size_um=px_work, matcher=matcher,
               weights=weights, tol_um=tol_um, **(loftr_kw or {}))
    if c["ok"]:
        rp = np.asarray(c["ref_points"], float) / r + np.array([rx0, ry0])
        mp = np.asarray(c["mov_points"], float) / r + np.array([mx0, my0])
        inside = _MplPath(roi).contains_points(rp)     # keep only matches truly in the ROI
        rp, mp = rp[inside], mp[inside]
        # The crop is the ROI bbox PLUS `pad` on every side, so on a small ROI most of the
        # crop is padding and most matches land outside the polygon by construction (a 270 µm
        # ROI is ~40 % of its own crop). Recorded so a NO_MATCHES can distinguish "the filters
        # took them" from "they were all in the margin".
        n_in_roi = int(len(rp))
        keep_conf = c["ref_points"]                    # for FLE re-localization
        if len(rp) >= min_matches:
            ref_pts, mov_pts = rp, mp
            if fle_fast:
                # The CALIBRATED FLE — no longer a placeholder. This used to declare a flat
                # 0.7 µm, which was never measured and is ~3x the real value; because a larger
                # declared FLE books less of the residual as deformation, it was LENIENT, and
                # it was reported as a certification by the auto path (3.2 µm) while the drawn
                # path measured 0.199 µm and said 14.5 µm about the identical polygon.
                #
                # FLE_WORKING_PX is measured against a known warp and scales with the working
                # pixel size, which is what the measurement says it does.
                fle_um = FLE_WORKING_PX * px_work
            else:
                # Same `loftr_kw` as the selection above: loftr_fle re-runs this pipeline, and
                # its whole point is that the FLE belongs to the population actually
                # certified — measuring a differently-filtered set would reintroduce exactly
                # the mismatch that tripped the FLE-consistency audit.
                fl = loftr_fle(small_r, small_m, c["ref_points"], c["mov_points"],
                               pixel_size_um=px_work, n_trials=2,  # lower bound; 2 is enough
                               **(loftr_kw or {}))
                fle_um = fl["fle_um"]

    # A VALIS-rigid recovery branch lived here for cross-modal ROIs where LoFTR finds nothing.
    # It was removed: its structural-residual certification over-certified 2-3x against held-out
    # expert landmarks, and no automatic method supplies anatomically-faithful correspondences on
    # cross-modal pairs. Manual landmarks remain the only honest path there. See research/valis.md.

    if ref_pts is None:                                # graceful fallback to landmarks
        if fallback_ref_lm is None or fallback_mov_lm is None:
            # Carry the funnel out with the failure. "NO_MATCHES" alone reads as "the matcher
            # found nothing here", which on this data is essentially never true — a region
            # that reports it typically had 190-280 RAW matches and lost them to the filters
            # (measured: 224 raw → 46 after scale-consistency → 5 inside the ROI). Without
            # these counts an operator cannot tell an untextured region from an over-tight
            # filter, and neither can the next person to read a bug report.
            return {"ok": False, "verdict": "NO_MATCHES",
                    "msg": (f"LoFTR found <{min_matches} matches in ROI and no landmark "
                            f"fallback ({c.get('msg') or 'no funnel'})"),
                    "n_loftr": int(c.get("n") or 0), "source": "none",
                    "loftr_funnel": {k: c.get(k) for k in
                                     ("n_raw", "n_after_cycle", "n_after_scale",
                                      "n_after_local")},
                    "n_in_roi": n_in_roi}
        fr = np.asarray(fallback_ref_lm, float).reshape(-1, 2)
        fm = np.asarray(fallback_mov_lm, float).reshape(-1, 2)
        insl = _MplPath(roi).contains_points(fr)
        ref_pts, mov_pts = fr[insl], fm[insl]
        source = "landmark_fallback"
        if len(ref_pts) < min_matches:
            return {"ok": False, "verdict": "NO_MATCHES",
                    "msg": f"only {len(ref_pts)} landmarks inside ROI (need {min_matches})",
                    "source": source}

    # BLUNDER REJECTION, before the fit that gets certified.
    #
    # LoFTR is the only matcher that works on this material, and 8-22 % of its correspondences
    # on real pairs are gross (validate_matchers_on_cohort.py). The gate reads a p90, so those
    # few set the reported cell error — 39 matches with a 6.55 um median residual produced a
    # 58 um cell error because ~4 were wrong. Huber down-weights them without rejecting, and
    # the local-smoothness filter compares displacements before any fit, which a blunder in a
    # locally-consistent wrong place survives.
    #
    # `reject_local_residual_outliers` drops a correspondence whose residual disagrees with its
    # NEIGHBOURS' residuals, which is not the circular "reject large residuals" — real
    # deformation is spatially continuous (nugget/sill 0.021, Moran's I +0.60, p <= 0.001) so a
    # displaced neighbourhood survives intact and only an isolated one is cut. Validated on 36
    # 10X pairs: gross fraction 50.4 % -> 47.0 % (Wilcoxon p = 0.0095), median residual
    # 34.3 -> 31.1 um (p = 6.6e-05), 94 % of correspondences kept, and against a synthetic warp
    # where the truth is exact the true p90 error improved on 36 of 36 pairs.
    _n_before = len(ref_pts)
    _keep = sr.reject_local_residual_outliers(
        ref_pts, mov_pts, sr._fit_similarity_robust(mov_pts, ref_pts))
    if _keep.sum() >= max(min_matches, 6):
        ref_pts, mov_pts = np.asarray(ref_pts)[_keep], np.asarray(mov_pts)[_keep]

    # local rigid fit + ordinary FW certification, windowed to the user's ROI
    M_local = sr._fit_similarity_robust(mov_pts, ref_pts)
    cert = sr.landmark_register_and_verify(
        ref_pts, mov_pts, float(pixel_size_um),
        image_wh=(Wr, Hr), user_roi_polygon=roi.tolist(),
        fle_um=fle_um, landmarks_are_model_selected=False)
    matrix = cert.get("matrix")
    cert["matrix"] = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    cert["local_matrix"] = M_local.tolist()
    cert["source"] = source
    cert["n_correspondences"] = int(len(ref_pts))
    cert["fle_um_loftr"] = fle_um
    cert["matcher"] = c.get("matcher")
    cert["n_blunders_rejected"] = int(_n_before - len(ref_pts))
    # Surfaced so an implausible cull by the local-smoothness filter is visible to the
    # caller rather than silently shaping the certified set.
    cert["local_drop_frac"] = c.get("local_drop_frac")
    cert["loftr_funnel"] = {k: c.get(k) for k in
                            ("n_raw", "n_after_cycle", "n_after_scale", "n_after_local")}
    # Report the trim explicitly. A silently shrunk window is how the certified area and the
    # analysed area came apart in the first place; the operator drew a region and should be
    # told that part of it has no counterpart on the moving section.
    # DOES THIS REGION'S FIT AGREE WITH THE WHOLE-FIELD ONE?
    #
    # A local fit is certified on its own residual, and a similarity fitted to
    # correspondences packed into one small region can absorb a large rotation or scale error
    # while still fitting those points well — the residual cannot see it, because the error
    # is in the parameters rather than in the points. Nothing asked whether the regions of a
    # pair agreed with EACH OTHER, and they frequently do not: measured across the cohort's
    # certifying windows, rotation spread within a single pair reached 61.9 deg and scale
    # spread 0.53, while genuinely well-registered pairs sit at 0.22-0.36 deg and 0.002-0.007.
    #
    # Two serial sections of one block differ by ONE placement rotation and ONE scale. Local
    # deformation bends tissue; it does not rotate one region 40 deg relative to its
    # neighbour. So a local fit that disagrees with the provisional whole-field transform is
    # wrong, however small its residual, and the disagreement is reported as the diagnostic
    # it is rather than hidden behind a passing verdict.
    if provisional_matrix is not None and cert.get("local_matrix") is not None:
        try:
            Lm = np.asarray(cert["local_matrix"], float)
            La, Lb = float(Lm[0, 0]), float(Lm[0, 1])
            Ga, Gb = float(A[0, 0]), float(A[0, 1])
            rot = abs(math.degrees(math.atan2(-Lb, La) - math.atan2(-Gb, Ga)))
            rot = min(rot, 360.0 - rot)
            sg = math.hypot(Ga, Gb)
            sl = math.hypot(La, Lb)
            cert["rotation_vs_global_deg"] = round(rot, 2)
            cert["scale_vs_global"] = round(sl / sg, 4) if sg > 0 else None
            if rot > GLOBAL_AGREEMENT_MAX_DEG or (
                    sg > 0 and abs(sl / sg - 1.0) > GLOBAL_AGREEMENT_MAX_SCALE):
                cert["verdict"] = "NOT_CERTIFIABLE"
                cert["reason"] = (
                    f"this region's own transform disagrees with the whole-field one by "
                    f"{rot:.1f} deg and a factor of {sl / max(sg, 1e-9):.3f} in scale. Two "
                    f"serial sections of one block share a single rotation and scale, so a "
                    f"local fit this different is wrong however well it fits its own "
                    f"correspondences — a small patch of matches can absorb a large "
                    f"rotation error without raising the residual.")
        except Exception:
            pass
    cert["roi_clipped_to_moving_frac"] = round(float(roi_clip_frac), 4)
    if roi_clip_frac < 0.999:
        cert["roi_polygon_drawn"] = np.asarray(roi_polygon_ref, float).reshape(-1, 2).tolist()
        cert["roi_polygon"] = np.asarray(roi_eff, float).tolist()
        cert["roi_clip_note"] = (
            f"{(1 - roi_clip_frac) * 100:.0f}% of the drawn region maps off the moving "
            f"section and was excluded — it has no corresponding tissue, so nothing there "
            f"could be measured. The certificate and the analysis window are the remaining "
            f"{roi_clip_frac * 100:.0f}%.")
    cert["ok"] = cert.get("verdict") in ("CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")
    if return_correspondences:            # the LoFTR points used for the fit (image coords in)
        cert["corr_ref"] = np.asarray(ref_pts, float).tolist()
        cert["corr_mov"] = np.asarray(mov_pts, float).tolist()
    return cert


def loftr_fle(ref_rgb, mov_rgb, ref_pts, mov_pts, pixel_size_um, n_trials=5,
              noise=0.02, seed=0, match_px=4.0, **kw):
    """FLE of the SELECTED correspondences: re-run the whole pipeline under image noise.

    It must re-run the pipeline, not just the raw matcher. The FLE that belongs in the error
    budget is the localisation error of the correspondences we actually certify on, and the
    filters change that population drastically — measuring the raw matcher instead returned
    13.9 µm where the selected set is sub-micron, which then tripped the FLE-consistency
    audit (correctly: residuals far smaller than the declared FLE can explain).

    Image-noise only, therefore a LOWER bound on FLE. That is the CONSERVATIVE direction:
    a smaller FLE charges more of the residual to deformation.
    """
    rng = np.random.default_rng(seed)
    base_r = np.asarray(ref_pts, float)
    base_m = np.asarray(mov_pts, float)
    devs = []
    for _ in range(int(n_trials)):
        c = loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um, noise=noise, rng=rng, **kw)
        if not c["n"]:
            continue
        r, m = np.array(c["ref_points"]), np.array(c["mov_points"])
        for p, q in zip(r, m):                       # pair back to the base set by position
            d = np.linalg.norm(base_r - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= match_px:
                devs.append((q - base_m[j]) - (p - base_r[j]))
    if len(devs) < 20:
        return {"fle_um": None, "n": len(devs), "source": "loftr_relocalization"}
    d = np.asarray(devs, float)
    d = d - d.mean(axis=0)
    comb = float(np.sqrt((d ** 2).sum() / (2 * len(d))) * float(pixel_size_um))
    return {"fle_um": round(comb / np.sqrt(2.0), 4), "fle_combined_um": round(comb, 4),
            "n": len(d), "source": "loftr_relocalization",
            "note": "image-noise only — a lower bound on FLE"}
