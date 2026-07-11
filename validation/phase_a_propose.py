"""
phase_a_propose.py — auto-PROPOSE consistent corresponding landmarks per pair for
human verification (scope A4: auto-detect + visual confirm).

Method (independent of the MI transform, so it also re-checks pairs like Liver_1):
  1. Detect distinctive structures in BOTH images: lumen/sinusoid centroids (holes
     in the tissue mask) + structural corners (goodFeaturesToTrack on the σ≈12 µm
     channel — never single nuclei).
  2. Data-driven SEED: grid-search translation × small rotation (uniform scale=1,
     per the scale bars) that maps the most moving lumens onto a reference lumen.
  3. RANSAC-refine a similarity from lumen matches, then re-match the full candidate
     set under the refined transform → geometrically-consistent inlier set.
  4. Propose up to 8 well-spread inliers per pair, render numbered verification
     overlays, and write them as the tool's pre-load (landmarks.json).

These are consistent BY CONSTRUCTION; the human visual check (numbered overlays /
the tool) is what makes them valid correspondences — that is stated plainly.
"""
import os, sys, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial.serial_registration import (structural_channel, tissue_mask, lumen_centroids,  # noqa
                                  _load_rgb_thumbnail)
from oasis.common.registration import _rgb_to_gray

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_a_qc")
PX = 0.7519


def corners(struct, mask, px):
    md = max(int(18.0 / px), 8)
    c = cv2.goodFeaturesToTrack(struct, maxCorners=150, qualityLevel=0.02,
                                minDistance=md, mask=mask.astype(np.uint8))
    return c.reshape(-1, 2).astype(np.float64) if c is not None else np.zeros((0, 2))


def make_M(theta, dx, dy, center):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = center - R @ center + np.array([dx, dy])
    return np.array([[c, -s, t[0]], [s, c, t[1]]])


def apply(M, pts):
    if len(pts) == 0:
        return pts
    return (M @ np.c_[pts, np.ones(len(pts))].T).T


def count_hits(ref, mov_mapped, tol):
    from scipy.spatial import cKDTree
    if len(ref) == 0 or len(mov_mapped) == 0:
        return 0
    d, _ = cKDTree(ref).query(mov_mapped)
    return int((d <= tol).sum())


def grid_seed(ref_lum, mov_lum, center, px):
    """Translation×rotation grid maximising lumen overlap (scale fixed = 1)."""
    best = (-1, None)
    tol = 10.0 / px * 0  # placeholder
    tol = 12.0
    for th in np.radians([-6, -3, 0, 3, 6]):
        for dx in range(-260, 261, 12):
            for dy in range(-260, 261, 12):
                M = make_M(th, dx, dy, center)
                h = count_hits(ref_lum, apply(M, mov_lum), tol)
                if h > best[0]:
                    best = (h, M)
    return best[1]


def mutual_matches(ref, mov, M, tol):
    from scipy.spatial import cKDTree
    if len(ref) == 0 or len(mov) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    mapped = apply(M, mov)
    tr, tm = cKDTree(ref), cKDTree(mapped)
    d_rm, i_rm = tr.query(mapped)
    _d, i_mr = tm.query(ref)
    rr, mm = [], []
    for j, (d, i) in enumerate(zip(d_rm, i_rm)):
        if d <= tol and i_mr[i] == j:
            rr.append(ref[i]); mm.append(mov[j])
    return np.array(rr), np.array(mm)


def spread_select(ref_pts, mov_pts, k=8):
    if len(ref_pts) <= k:
        return ref_pts, mov_pts
    idx = [0]
    while len(idx) < k:
        rest = [i for i in range(len(ref_pts)) if i not in idx]
        best = max(rest, key=lambda i: min(np.linalg.norm(ref_pts[i] - ref_pts[j])
                                           for j in idx))
        idx.append(best)
    return ref_pts[idx], mov_pts[idx]


def main():
    reg = json.load(open(os.path.join(OUT, "registration.json")))
    proposals, summary = {}, []
    for sid, r in reg.items():
        ref_rgb, _ = _load_rgb_thumbnail(r["ref_path"], 1920)
        mov_rgb, _ = _load_rgb_thumbnail(r["mov_path"], 1920)
        rs, ms = structural_channel(ref_rgb, PX), structural_channel(mov_rgb, PX)
        rmask, mmask = tissue_mask(ref_rgb, PX), tissue_mask(mov_rgb, PX)
        H, W = rs.shape
        center = np.array([W / 2.0, H / 2.0])

        ref_lum, mov_lum = lumen_centroids(rmask, PX), lumen_centroids(mmask, PX)
        # data-driven seed (fallback to MI transform if too few lumens)
        if len(ref_lum) >= 3 and len(mov_lum) >= 3:
            seed = grid_seed(ref_lum, mov_lum, center, PX)
        else:
            seed = np.array(r["matrix"], float)

        # refine on lumen matches
        rr, mm = mutual_matches(ref_lum, mov_lum, seed, 16.0)
        M = seed
        if len(rr) >= 3:
            Mr, _ = cv2.estimateAffinePartial2D(mm.astype(np.float32),
                                                rr.astype(np.float32),
                                                method=cv2.RANSAC, ransacReprojThreshold=8.0)
            if Mr is not None:
                M = Mr.astype(float)

        # full candidate set (lumens + corners), match under refined M, RANSAC
        ref_all = np.vstack([ref_lum, corners(rs, rmask, PX)]) if len(ref_lum) else corners(rs, rmask, PX)
        mov_all = np.vstack([mov_lum, corners(ms, mmask, PX)]) if len(mov_lum) else corners(ms, mmask, PX)
        rr2, mm2 = mutual_matches(ref_all, mov_all, M, 10.0)
        if len(rr2) >= 3:
            Mf, inl = cv2.estimateAffinePartial2D(mm2.astype(np.float32),
                                                  rr2.astype(np.float32),
                                                  method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if inl is not None:
                m = inl.ravel().astype(bool)
                rr2, mm2 = rr2[m], mm2[m]

        rr2, mm2 = spread_select(rr2, mm2, 8)
        pts = [[float(rr2[i, 0]), float(rr2[i, 1]), float(mm2[i, 0]), float(mm2[i, 1])]
               for i in range(len(rr2))]
        proposals[sid] = {"orig_w": W, "points": pts}

        # verification overlay: CD8 | TIM3 with numbered matched dots
        rg, mg = _rgb_to_gray(ref_rgb), _rgb_to_gray(mov_rgb)
        canvas = np.zeros((H, W * 2 + 10, 3), np.uint8) + 255
        canvas[:, :W] = cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)
        canvas[:, W + 10:] = cv2.cvtColor(mg, cv2.COLOR_GRAY2BGR)
        for i, p in enumerate(pts):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 9, (0, 140, 0), 2)
            cv2.putText(canvas, str(i + 1), (int(p[0]) + 8, int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 0), 2)
            cv2.circle(canvas, (int(p[2]) + W + 10, int(p[3])), 9, (200, 0, 0), 2)
            cv2.putText(canvas, str(i + 1), (int(p[2]) + W + 18, int(p[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
        cv2.imwrite(os.path.join(OUT, f"propose_{sid}.png"),
                    cv2.resize(canvas, (1400, int(H * 1400 / (W * 2 + 10)))))
        summary.append((sid, len(pts)))
        print(f"{sid}: proposed {len(pts)} correspondences "
              f"(lumens ref/mov={len(ref_lum)}/{len(mov_lum)})")

    with open(os.path.join(OUT, "landmarks.json"), "w") as f:
        f.write("LANDMARKS " + json.dumps(proposals))
    print("\nWrote proposals → landmarks.json + propose_<pair>.png verification images")
    print("Summary:", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
