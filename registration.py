"""
registration.py
Computes a 2D rigid alignment transform between two IHC images
(same tissue block, different stains, serial sections).

Returns a 2x3 affine matrix to map coordinates from the moving image
(stain B) into the reference image space (stain A).
"""

import os
import numpy as np


def _load_thumbnail(image_path: str, max_side: int, mode: str):
    """
    Shared thumbnail loader. Tries openslide first so pyramidal whole-slide
    formats (SVS / NDPI / etc.) load correctly — PIL cannot read those and would
    silently fail, leaving registration with no usable image. Falls back to PIL
    for standard formats or when openslide is unavailable.

    Args:
        mode: PIL conversion mode — "L" (grayscale) or "RGB".

    Returns:
        (np.ndarray, scale_factor) where scale_factor maps full-resolution
        coordinates → thumbnail coordinates, or (None, 1.0) on failure.
    """
    # ── openslide (whole-slide pyramidal formats) ─────────────────────────────
    try:
        import openslide
        slide = openslide.OpenSlide(image_path)
        w, h  = slide.dimensions
        scale = min(max_side / max(w, h), 1.0)
        tw, th = max(int(w * scale), 1), max(int(h * scale), 1)
        thumb = slide.get_thumbnail((tw, th))
        slide.close()
        return np.array(thumb.convert(mode)), scale
    except Exception:
        pass  # not a WSI, or openslide unavailable → PIL fallback

    # ── PIL fallback (standard formats) ───────────────────────────────────────
    try:
        from PIL import Image
        img = Image.open(image_path).convert(mode)
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(img), scale
    except Exception as e:
        print(f"  Registration: could not load {os.path.basename(image_path)}: {e}")
        return None, 1.0


def load_thumbnail(image_path: str, max_side: int = 1024):
    """Load image as grayscale thumbnail. Returns (array, scale_factor).

    Tries openslide (pyramidal WSI: SVS/NDPI/...) then falls back to PIL.
    """
    return _load_thumbnail(image_path, max_side, "L")


def _load_rgb_thumbnail(image_path: str, max_side: int = 1024):
    """RGB thumbnail (needed for stain deconvolution). Returns (array, scale)."""
    return _load_thumbnail(image_path, max_side, "RGB")


# H-DAB reference stain vectors (Ruifrok-Johnston colour deconvolution).
_STAIN_H   = np.array([0.65, 0.70, 0.29], dtype=np.float64)
_STAIN_DAB = np.array([0.27, 0.57, 0.78], dtype=np.float64)


def extract_hematoxylin(rgb: np.ndarray) -> np.ndarray:
    """
    Extract the hematoxylin channel from an H-DAB RGB image via colour
    deconvolution (Ruifrok-Johnston).

    Hematoxylin counterstains nuclei in *both* the CD8 and TIM-3 sections, so it
    is the shared structural signal across serial stains — registering on it is
    far more robust than on raw grayscale or on the (differing) DAB channel.

    Args:
        rgb: HxWx3 (or HxWx4) uint8/float RGB array.

    Returns:
        HxW uint8 grayscale image of hematoxylin density (bright = more stain).
    """
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]

    # Orthonormal stain basis: H, DAB, and a residual vector (their cross product)
    res = np.cross(_STAIN_H, _STAIN_DAB)

    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    stain_matrix = np.array([_norm(_STAIN_H), _norm(_STAIN_DAB), _norm(res)])

    # Optical density, then deconvolve into stain concentrations
    od   = -np.log10((rgb + 1.0) / 256.0)          # +1 avoids log(0)
    conc = od @ np.linalg.inv(stain_matrix)        # (H, W, 3)
    h    = np.clip(conc[..., 0], 0.0, None)         # hematoxylin concentration

    hi = np.percentile(h, 99) if h.size else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = h.max() if h.size and h.max() > 0 else 1.0
    h8 = np.clip(h / hi, 0.0, 1.0) * 255.0
    return h8.astype(np.uint8)


def _sitk_to_affine(transform) -> np.ndarray:
    """
    Convert a SimpleITK 2D transform into a 2x3 cv2-style affine matrix mapping
    *moving* thumbnail coordinates → *reference* thumbnail coordinates.

    SimpleITK registration yields a transform T that maps fixed(reference) →
    moving coordinates (the resampling convention), so we invert it and sample
    three basis points to recover the affine that transform_centroids expects
    (moving → reference), identical in convention to the ORB output.
    """
    inv = transform.GetInverse()
    o   = np.array(inv.TransformPoint((0.0, 0.0)), dtype=np.float64)
    ex  = np.array(inv.TransformPoint((1.0, 0.0)), dtype=np.float64)
    ey  = np.array(inv.TransformPoint((0.0, 1.0)), dtype=np.float64)
    col_x = ex - o
    col_y = ey - o
    return np.float32([
        [col_x[0], col_y[0], o[0]],
        [col_x[1], col_y[1], o[1]],
    ])


def _register_simpleitk(ref_h, mov_h, scale_ref, scale_mov):
    """
    Primary method: rigid (Euler2D) registration driven by Mattes mutual
    information on the hematoxylin channels. MI is robust to the different DAB
    distributions between a CD8 and a TIM-3 section. Returns a result dict on a
    sane solution, or None to fall through to the next cascade stage.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  Registration: SimpleITK not installed — falling back to ORB+SIFT")
        return None

    try:
        fixed  = sitk.GetImageFromArray(ref_h.astype(np.float32))
        moving = sitk.GetImageFromArray(mov_h.astype(np.float32))

        initial = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.2, seed=42)   # fixed seed → reproducible
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4,
            numberOfIterations=200, gradientMagnitudeTolerance=1e-6,
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(initial, inPlace=False)

        final  = R.Execute(fixed, moving)
        metric = float(R.GetMetricValue())
        stop   = R.GetOptimizerStopConditionDescription()

        # Sanity-gate: reject implausible solutions (lets ORB/phase take over)
        angle  = float(final.GetParameters()[0])
        tx, ty = float(final.GetParameters()[1]), float(final.GetParameters()[2])
        diag   = float(np.hypot(*ref_h.shape))
        if abs(angle) > np.radians(45) or np.hypot(tx, ty) > diag:
            print(f"  Registration (SimpleITK): rejected implausible transform "
                  f"(angle={np.degrees(angle):.1f}°, "
                  f"shift={np.hypot(tx, ty):.0f}px) — falling back")
            return None

        print(f"  Registration (SimpleITK): MI metric={metric:.4f}  "
              f"angle={np.degrees(angle):.1f}°  [{stop}]")
        return {
            "matrix": _sitk_to_affine(final),
            "scale_ref": scale_ref, "scale_mov": scale_mov,
            "method": "simpleitk", "success": True, "metric": metric,
        }
    except Exception as e:
        print(f"  Registration SimpleITK failed: {e}")
        return None


def _register_features(ref_img, mov_img, scale_ref, scale_mov):
    """
    Fallback: ORB then SIFT feature matching + RANSAC partial-affine, run on the
    hematoxylin channel. Tries ORB first (fast, binary) and escalates to SIFT
    (slower, stronger) only if ORB fails its inlier gate. `method` is reported as
    "orb" for this stage; the actual detector used is recorded under "detector".
    """
    try:
        import cv2
    except ImportError:
        print("  Registration: opencv-python not installed — skipping feature matching")
        return None

    ref_img = ref_img.astype(np.uint8)
    mov_img = mov_img.astype(np.uint8)

    detectors = [("ORB", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING)]
    if hasattr(cv2, "SIFT_create"):
        detectors.append(("SIFT", cv2.SIFT_create(nfeatures=3000), cv2.NORM_L2))

    for name, det, norm in detectors:
        try:
            kp_ref, des_ref = det.detectAndCompute(ref_img, None)
            kp_mov, des_mov = det.detectAndCompute(mov_img, None)
            if des_ref is None or des_mov is None \
                    or len(kp_ref) < 10 or len(kp_mov) < 10:
                continue

            bf      = cv2.BFMatcher(norm, crossCheck=True)
            matches = sorted(bf.match(des_ref, des_mov), key=lambda x: x.distance)
            good    = matches[:min(200, len(matches))]
            if len(good) < 10:
                continue

            pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good])
            pts_mov = np.float32([kp_mov[m.trainIdx].pt for m in good])
            matrix, inliers = cv2.estimateAffinePartial2D(
                pts_mov, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0,
            )
            if matrix is None:
                continue
            n_inliers = int(inliers.sum()) if inliers is not None else 0
            if n_inliers < 10:
                continue

            print(f"  Registration ({name}): {n_inliers}/{len(good)} inliers")
            return {
                "matrix": matrix.astype(np.float32),
                "scale_ref": scale_ref, "scale_mov": scale_mov,
                "method": "orb", "success": True,
                "metric": n_inliers, "detector": name,
            }
        except Exception as e:
            print(f"  Registration {name} failed: {e}")
            continue
    return None


def _register_phase(ref_img, mov_img, scale_ref, scale_mov):
    """Fallback: phase correlation (translation only)."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        h = min(ref_img.shape[0], mov_img.shape[0])
        w = min(ref_img.shape[1], mov_img.shape[1])
        shift, response = cv2.phaseCorrelate(
            np.float32(ref_img[:h, :w]) / 255.0,
            np.float32(mov_img[:h, :w]) / 255.0,
        )
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        print(f"  Registration (phase): shift=({shift[0]:.1f}, {shift[1]:.1f})  "
              f"response={response:.3f}")
        return {
            "matrix": matrix,
            "scale_ref": scale_ref, "scale_mov": scale_mov,
            "method": "phase", "success": True, "metric": float(response),
        }
    except Exception as e:
        print(f"  Registration phase failed: {e}")
        return None


def compute_registration(ref_path: str, mov_path: str) -> dict:
    """
    Compute a rigid (translation + rotation) transform to align mov_path onto
    ref_path, using a cascade of methods on the hematoxylin channel:

        SimpleITK (Mattes MI)  → primary
        ORB + SIFT             → fallback
        phase correlation      → fallback
        identity               → last resort

    Each stage logs why it was chosen and its quality metric.

    Returns:
        {
          "matrix":    2x3 np.float32 affine matrix (moving → reference thumbnail),
          "scale_ref": float,   thumbnail scale for ref image
          "scale_mov": float,   thumbnail scale for moving image
          "method":    "simpleitk" | "orb" | "phase" | "identity",
          "success":   bool,
          "metric":    float | int | None,   quality of the chosen method
        }
    """
    identity = {
        "matrix": np.float32([[1, 0, 0], [0, 1, 0]]),
        "scale_ref": 1.0, "scale_mov": 1.0,
        "method": "identity", "success": False, "metric": None,
    }

    ref_rgb, scale_ref = _load_rgb_thumbnail(ref_path)
    mov_rgb, scale_mov = _load_rgb_thumbnail(mov_path)
    if ref_rgb is None or mov_rgb is None:
        return identity

    # Hematoxylin = signal common to both serial stains → register on it
    try:
        ref_feat = extract_hematoxylin(ref_rgb)
        mov_feat = extract_hematoxylin(mov_rgb)
    except Exception as e:
        print(f"  Registration: hematoxylin extraction failed: {e} — using luminance")
        ref_feat = _rgb_to_gray(ref_rgb)
        mov_feat = _rgb_to_gray(mov_rgb)

    # ── 1. SimpleITK Mattes MI (primary) ──────────────────────────────────────
    result = _register_simpleitk(ref_feat, mov_feat, scale_ref, scale_mov)
    if result is not None:
        return result

    # ── 2. ORB + SIFT feature matching (fallback) ─────────────────────────────
    result = _register_features(ref_feat, mov_feat, scale_ref, scale_mov)
    if result is not None:
        return result

    # ── 3. Phase correlation (fallback) ───────────────────────────────────────
    result = _register_phase(ref_feat, mov_feat, scale_ref, scale_mov)
    if result is not None:
        return result

    # ── 4. Identity (last resort) ─────────────────────────────────────────────
    print("  Registration: all methods failed — using identity transform")
    return identity


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance grayscale fallback when hematoxylin deconvolution is unavailable."""
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]
    g   = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(g, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Registration quality control (QC) — MEASUREMENT ONLY
#
# These helpers never alter the transform or the cascade order; they only measure
# how good the transform that compute_registration() already chose actually is, so
# the pipeline can refuse to present spatial statistics computed on a bad
# alignment. Serial-section local deformation / z-gap can be 10–50 µm — exactly the
# DCLF analysis band — so an unverified alignment must be treated as unreliable.
# ──────────────────────────────────────────────────────────────────────────────

def _tissue_mask_from_rgb(rgb: np.ndarray):
    """Binary tissue mask from an RGB thumbnail (Otsu on luminance, tissue darker
    than the bright background — same convention as estimate_tissue_mask)."""
    import cv2
    gray = _rgb_to_gray(rgb)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    return m


def _qc_match_points(ref_img, mov_img):
    """
    Detect + cross-check feature correspondences between two hematoxylin channels,
    INDEPENDENTLY of whatever method compute_registration used. Returns
    (pts_ref, pts_mov, n_matches) in thumbnail coordinates (the detector with the
    most matches wins), or (None, None, 0) when no usable matches exist.
    """
    import cv2
    ref_img = ref_img.astype(np.uint8)
    mov_img = mov_img.astype(np.uint8)
    detectors = [("ORB", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING)]
    if hasattr(cv2, "SIFT_create"):
        detectors.append(("SIFT", cv2.SIFT_create(nfeatures=3000), cv2.NORM_L2))

    best = (None, None, 0)
    for name, det, norm in detectors:
        try:
            kp_ref, des_ref = det.detectAndCompute(ref_img, None)
            kp_mov, des_mov = det.detectAndCompute(mov_img, None)
            if des_ref is None or des_mov is None \
                    or len(kp_ref) < 10 or len(kp_mov) < 10:
                continue
            bf      = cv2.BFMatcher(norm, crossCheck=True)
            matches = sorted(bf.match(des_ref, des_mov), key=lambda x: x.distance)
            good    = matches[:min(300, len(matches))]
            if len(good) < 10:
                continue
            pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good])
            pts_mov = np.float32([kp_mov[m.trainIdx].pt for m in good])
            if len(good) > best[2]:
                best = (pts_ref, pts_mov, len(good))
        except Exception:
            continue
    return best


def _tissue_overlap_fraction(ref_rgb, mov_rgb, reg_result):
    """
    Fraction of the MOVING tissue that lands inside the FIXED tissue after the
    chosen transform is applied. The 2×3 affine maps moving-thumbnail →
    reference-thumbnail coordinates, so we warp the moving tissue mask into the
    reference thumbnail frame and intersect. Low overlap ⇒ registration shoved the
    tissue off-target. Returns a float in [0, 1] or None on failure.
    """
    import cv2
    ref_mask = _tissue_mask_from_rgb(ref_rgb)
    mov_mask = _tissue_mask_from_rgb(mov_rgb)
    Hr, Wr   = ref_mask.shape[:2]
    matrix   = np.asarray(reg_result["matrix"], dtype=np.float32)
    # warpAffine treats `matrix` as the forward (moving→reference) map by default.
    warped   = cv2.warpAffine(mov_mask, matrix, (Wr, Hr), flags=cv2.INTER_NEAREST)
    mov_total = int((warped > 0).sum())
    if mov_total == 0:
        return None
    inter = int(np.logical_and(warped > 0, ref_mask > 0).sum())
    return float(inter) / float(mov_total)


def compute_registration_qc(ref_path: str, mov_path: str, reg_result: dict,
                            pixel_size_um: float,
                            residual_inlier_px: float = 5.0,
                            min_qc_inliers: int = 10,
                            max_side: int = 1024) -> dict:
    """
    Objective, method-agnostic quality metrics for an already-computed
    registration. MEASUREMENT ONLY — it neither recomputes nor changes the
    transform. Returns a dict with:

      method                     : the method compute_registration actually used
      quality_metric             : method-appropriate score
                                   (SimpleITK → Mattes MI value;
                                    ORB/SIFT   → inlier count + ratio;
                                    phase      → peak response;
                                    identity   → none)
      residual_error_px / _um    : post-registration alignment error, estimated by
                                   detecting matched feature points in both
                                   hematoxylin channels, mapping the moving points
                                   through the CHOSEN transform (the exact
                                   transform_centroids path the pipeline uses on
                                   cells), and measuring median / 90th-percentile
                                   residual distance to their matched fixed points.
                                   None when no reliable matches exist (→ failed QC).
      tissue_overlap_fraction    : fraction of moving tissue inside fixed tissue
                                   after the transform (None on failure)
      n_qc_matches / n_qc_inliers / qc_inlier_ratio : feature-matching diagnostics

    The residual is reported in reference-image pixels and converted to microns
    via pixel_size_um for thresholding against the analysis scale.
    """
    method     = (reg_result or {}).get("method")
    raw_metric = (reg_result or {}).get("metric")
    out = {
        "method":                  method,
        "quality_metric":          {"kind": "none", "value": None},
        "residual_error_px":       None,
        "residual_error_p90_px":   None,
        "residual_error_um":       None,
        "residual_error_p90_um":   None,
        "tissue_overlap_fraction": None,
        "n_qc_matches":            0,
        "n_qc_inliers":            0,
        "qc_inlier_ratio":         None,
        "pixel_size_um":           float(pixel_size_um) if pixel_size_um else None,
    }
    if not reg_result:
        return out
    try:
        import cv2
    except Exception:
        print("  Registration QC: opencv unavailable — cannot verify alignment")
        return out

    ref_rgb, scale_ref = _load_rgb_thumbnail(ref_path, max_side)
    mov_rgb, scale_mov = _load_rgb_thumbnail(mov_path, max_side)
    if ref_rgb is None or mov_rgb is None:
        return out

    try:
        ref_feat = extract_hematoxylin(ref_rgb)
        mov_feat = extract_hematoxylin(mov_rgb)
    except Exception:
        ref_feat = _rgb_to_gray(ref_rgb)
        mov_feat = _rgb_to_gray(mov_rgb)

    # ── Residual alignment error via independent feature correspondences ───────
    pts_ref, pts_mov, n_matches = _qc_match_points(ref_feat, mov_feat)
    out["n_qc_matches"] = int(n_matches)
    qc_ratio, n_inliers = None, 0
    if pts_ref is not None and n_matches >= min_qc_inliers:
        # RANSAC here only identifies geometrically-consistent correspondences
        # (which matches are real); it is NOT used as the alignment transform.
        _, inl = cv2.estimateAffinePartial2D(
            pts_mov, pts_ref, method=cv2.RANSAC,
            ransacReprojThreshold=residual_inlier_px)
        if inl is not None:
            inl       = inl.ravel().astype(bool)
            n_inliers = int(inl.sum())
            qc_ratio  = (float(n_inliers) / float(n_matches)) if n_matches else None
            if n_inliers >= min_qc_inliers:
                # Map the moving inlier points through the CHOSEN transform exactly
                # as the pipeline maps cell centroids, then measure the residual.
                mov_full    = pts_mov[inl] / max(float(scale_mov), 1e-9)
                mapped_full = transform_centroids(mov_full, reg_result)
                ref_full    = pts_ref[inl] / max(float(scale_ref), 1e-9)
                d   = np.linalg.norm(mapped_full - ref_full, axis=1)
                med = float(np.median(d))
                p90 = float(np.percentile(d, 90))
                out["residual_error_px"]     = round(med, 3)
                out["residual_error_p90_px"] = round(p90, 3)
                if pixel_size_um:
                    out["residual_error_um"]     = round(med * float(pixel_size_um), 3)
                    out["residual_error_p90_um"] = round(p90 * float(pixel_size_um), 3)
    out["n_qc_inliers"]    = n_inliers
    out["qc_inlier_ratio"] = round(qc_ratio, 3) if qc_ratio is not None else None

    # ── Method-appropriate quality metric ─────────────────────────────────────
    if method == "simpleitk":
        out["quality_metric"] = {"kind": "mattes_mutual_information",
                                 "value": raw_metric}
    elif method == "orb":
        out["quality_metric"] = {"kind": "ransac_inliers",
                                 "inlier_count": raw_metric,
                                 "inlier_ratio": out["qc_inlier_ratio"],
                                 "detector": (reg_result or {}).get("detector")}
    elif method == "phase":
        out["quality_metric"] = {"kind": "phase_correlation_peak",
                                 "value": raw_metric}
    else:
        out["quality_metric"] = {"kind": "none", "value": None}

    # ── Tissue overlap after the transform ────────────────────────────────────
    try:
        out["tissue_overlap_fraction"] = _tissue_overlap_fraction(
            ref_rgb, mov_rgb, reg_result)
    except Exception as e:
        print(f"  Registration QC: tissue-overlap measurement failed: {e}")
        out["tissue_overlap_fraction"] = None

    return out


def transform_centroids(centroids: np.ndarray, reg_result: dict) -> np.ndarray:
    """
    Apply a registration transform to full-resolution XY cell centroids.

    The transform was computed on thumbnails; this function accounts for the
    thumbnail scale factors so the output is in full-resolution ref coordinates.

    Args:
        centroids:  Nx2 array of (x, y) in full-resolution moving-image pixels
        reg_result: dict returned by compute_registration()

    Returns:
        Nx2 array of (x, y) in full-resolution reference-image pixels
    """
    if len(centroids) == 0:
        return centroids

    matrix    = reg_result["matrix"]      # 2x3 affine
    scale_ref = reg_result["scale_ref"]
    scale_mov = reg_result["scale_mov"]

    # Scale centroids down to thumbnail space
    pts = centroids.astype(np.float32) * scale_mov
    # Homogeneous coordinates
    ones  = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])          # Nx3
    # Apply affine transform: result is Nx2
    transformed = (matrix @ pts_h.T).T
    # Scale back up to full-resolution reference space
    return transformed / scale_ref
