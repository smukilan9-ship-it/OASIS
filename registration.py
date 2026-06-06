"""
registration.py
Computes a 2D rigid alignment transform between two IHC images
(same tissue block, different stains, serial sections).

Returns a 2x3 affine matrix to map coordinates from the moving image
(stain B) into the reference image space (stain A).
"""

import os
import numpy as np


def load_thumbnail(image_path: str, max_side: int = 1024):
    """Load image as grayscale thumbnail. Returns (array, scale_factor)."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(img), scale
    except Exception as e:
        print(f"  Registration: could not load {os.path.basename(image_path)}: {e}")
        return None, 1.0


def compute_registration(ref_path: str, mov_path: str) -> dict:
    """
    Compute rigid (translation + rotation) transform to align mov_path onto ref_path.

    Returns:
        {
          "matrix":    2x3 np.float32 affine matrix,
          "scale_ref": float,   thumbnail scale for ref image
          "scale_mov": float,   thumbnail scale for moving image
          "method":    "orb" | "phase" | "identity",
          "success":   bool,
        }
    """
    identity = {
        "matrix": np.float32([[1, 0, 0], [0, 1, 0]]),
        "scale_ref": 1.0, "scale_mov": 1.0,
        "method": "identity", "success": False,
    }

    ref_img, scale_ref = load_thumbnail(ref_path)
    mov_img, scale_mov = load_thumbnail(mov_path)
    if ref_img is None or mov_img is None:
        return identity

    try:
        import cv2
    except ImportError:
        print("  Registration: opencv-python not installed — using identity transform")
        return identity

    # ── ORB feature matching ──────────────────────────────────────────────────
    try:
        orb = cv2.ORB_create(nfeatures=3000)
        kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
        kp_mov, des_mov = orb.detectAndCompute(mov_img, None)

        if des_ref is not None and des_mov is not None \
                and len(kp_ref) >= 10 and len(kp_mov) >= 10:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des_ref, des_mov), key=lambda x: x.distance)
            good = matches[:min(200, len(matches))]

            if len(good) >= 10:
                pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good])
                pts_mov = np.float32([kp_mov[m.trainIdx].pt for m in good])
                matrix, inliers = cv2.estimateAffinePartial2D(
                    pts_mov, pts_ref,
                    method=cv2.RANSAC, ransacReprojThreshold=5.0,
                )
                if matrix is not None:
                    n_inliers = int(inliers.sum()) if inliers is not None else 0
                    print(f"  Registration (ORB): {n_inliers}/{len(good)} inliers")
                    return {
                        "matrix": matrix,
                        "scale_ref": scale_ref,
                        "scale_mov": scale_mov,
                        "method": "orb",
                        "success": True,
                    }
    except Exception as e:
        print(f"  Registration ORB failed: {e}")

    # ── Phase correlation fallback (translation only) ─────────────────────────
    try:
        h = min(ref_img.shape[0], mov_img.shape[0])
        w = min(ref_img.shape[1], mov_img.shape[1])
        shift, response = cv2.phaseCorrelate(
            np.float32(ref_img[:h, :w]) / 255.0,
            np.float32(mov_img[:h, :w]) / 255.0,
        )
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        print(f"  Registration (phase): shift=({shift[0]:.1f}, {shift[1]:.1f})  response={response:.3f}")
        return {
            "matrix": matrix,
            "scale_ref": scale_ref,
            "scale_mov": scale_mov,
            "method": "phase",
            "success": True,
        }
    except Exception as e:
        print(f"  Registration phase failed: {e}")

    print("  Registration: all methods failed — using identity transform")
    return identity


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
