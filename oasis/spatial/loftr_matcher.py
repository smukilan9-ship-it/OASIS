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

Neither filter can see the similarity, so the surviving residuals are admissible evidence
about it. `serial_registration.residual_field_assay` is the acceptance test for the output
(and note it catches RANDOM error, not SYSTEMATIC error: LoFTR's `indoor` weights produce
confidently wrong but spatially SMOOTH matches on this data, residual median 77–170 µm on a
1443 µm field, which the assay happily labels REAL_DEFORMATION. Weight choice matters and
must be validated externally.)

Requires torch + kornia. Weight download may need SSL_CERT_FILE=$(python -m certifi).
"""
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


def loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um, weights="outdoor",
                          scales=(0.75, 0.5), tol_um=4.0, conf_floor=0.2,
                          noise=0.0, rng=None):
    """Cycle- and scale-consistent LoFTR correspondences. No RANSAC, no residuals.

    `noise`/`rng` perturb both images identically-in-distribution; used by `loftr_fle` to
    re-run this WHOLE pipeline under noise, so the FLE it measures belongs to the selected
    population rather than to the raw matcher.

    Returns dict: ref_points, mov_points, confidence, n, n_raw, n_after_cycle,
    n_after_scale, tol_um, weights, ok, msg.
    """
    tol_px = float(tol_um) / float(pixel_size_um)
    s0 = scales[0]

    fk0, fk1, fcf, stride0 = _raw(ref_rgb, mov_rgb, s0, pixel_size_um, weights,
                                  conf_floor, noise, rng)
    out = {"ref_points": [], "mov_points": [], "confidence": [], "n": 0,
           "n_raw": int(len(fk0)), "n_after_cycle": 0, "n_after_scale": 0,
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
        keep = _disp_agree(fk0, fk1, gk0, gk1, tol_px, stride1)
        fk0, fk1, fcf = fk0[keep], fk1[keep], fcf[keep]
    out["n_after_scale"] = int(len(fk0))

    out.update(ref_points=fk0.tolist(), mov_points=fk1.tolist(),
               confidence=[round(float(c), 4) for c in fcf], n=int(len(fk0)),
               ok=len(fk0) >= 6,
               msg=(f"{out['n_raw']} raw → {out['n_after_cycle']} cycle-consistent → "
                    f"{len(fk0)} scale-consistent (tol {tol_um} µm); no transform used"))
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
                      return_correspondences=False, fle_fast=False, valis_fallback=False):
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
    c = loftr_correspondences(small_r, small_m, pixel_size_um=px_work,
                              weights=weights, tol_um=tol_um)
    fle_um = None
    if c["ok"]:
        rp = np.asarray(c["ref_points"], float) / r + np.array([rx0, ry0])
        mp = np.asarray(c["mov_points"], float) / r + np.array([mx0, my0])
        inside = _MplPath(roi).contains_points(rp)     # keep only matches truly in the ROI
        rp, mp = rp[inside], mp[inside]
        keep_conf = c["ref_points"]                    # for FLE re-localization
        if len(rp) >= min_matches:
            ref_pts, mov_pts = rp, mp
            if fle_fast:
                # Skip the noise-relocalization (its 3 extra pipeline runs dominate cost).
                # A small fixed sub-pixel FLE is CONSERVATIVE: it charges more residual to
                # deformation, so the fast sweep never over-certifies. Re-certify a chosen
                # region without fast mode for the principled measured FLE.
                fle_um = 0.7
            else:
                fl = loftr_fle(small_r, small_m, c["ref_points"], c["mov_points"],
                               pixel_size_um=px_work, n_trials=2)  # lower bound; 2 is enough
                fle_um = fl["fle_um"]

    # VALIS-rigid fallback — recover CROSS-MODAL ROIs where LoFTR found too few matches
    # (H&E<->IHC: LoFTR returns 0, VALIS's deep matcher recovers hundreds). VALIS registers
    # in the isolated env; certification is the stain-robust STRUCTURAL patch-residual (not the
    # LoFTR->landmark gate, which needs correspondences we do not have here). Only rigid is used.
    if ref_pts is None and valis_fallback:
        try:
            from oasis.spatial import valis_engine as _ve
            if _ve.valis_available():
                vr = _ve.register_crops_and_certify(crop_r, crop_m, float(pixel_size_um))
                if vr.get("matrix") is not None:
                    Ac = np.asarray(vr["matrix"], float)[:2]
                    A_, t_ = Ac[:, :2], Ac[:, 2]
                    off_m = np.array([mx0, my0], float); off_r = np.array([rx0, ry0], float)
                    t_full = A_ @ (-off_m) + t_ + off_r     # crop_mov -> crop_ref -> full frame
                    M_full = np.hstack([A_, t_full.reshape(2, 1)])
                    return {"ok": bool(vr.get("ok")), "verdict": vr.get("verdict"),
                            "matrix": M_full.tolist(), "local_matrix": Ac.tolist(),
                            "source": "valis_rigid_in_roi", "cert_method": "structural_patch_residual",
                            "median_um": vr.get("median_um"), "region_max_um": vr.get("region_max_um"),
                            "lumen_tre_um": vr.get("lumen_tre_um"), "n_correspondences": vr.get("n_matches"),
                            "overlap_frac": vr.get("overlap_frac"), "msg": vr.get("reason")}
        except Exception as _e:
            pass  # VALIS unavailable/failed -> fall through to landmark fallback

    if ref_pts is None:                                # graceful fallback to landmarks
        if fallback_ref_lm is None or fallback_mov_lm is None:
            return {"ok": False, "verdict": "NO_MATCHES",
                    "msg": f"LoFTR found <{min_matches} matches in ROI and no landmark fallback",
                    "n_loftr": int(c.get("n") or 0), "source": "none"}
        fr = np.asarray(fallback_ref_lm, float).reshape(-1, 2)
        fm = np.asarray(fallback_mov_lm, float).reshape(-1, 2)
        insl = _MplPath(roi).contains_points(fr)
        ref_pts, mov_pts = fr[insl], fm[insl]
        source = "landmark_fallback"
        if len(ref_pts) < min_matches:
            return {"ok": False, "verdict": "NO_MATCHES",
                    "msg": f"only {len(ref_pts)} landmarks inside ROI (need {min_matches})",
                    "source": source}

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
