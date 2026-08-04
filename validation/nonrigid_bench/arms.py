"""
arms.py — STEP 2. Does a non-rigid warp change what a cross-type spatial test concludes?

THE SETUP, and why it is fair.

For each registered ANHIR pair we draw two marker patterns that have NO interaction
whatsoever:

    A       ~ the FIXED section's own tissue density   (stays in the fixed frame)
    B       ~ the MOVING section's own tissue density  (must be registered in)

They are independent draws. B's positions are generated without any reference to A's,
so the true cross-type association is EXACTLY nil, by construction, under every
transform — a deterministic map applied to B alone cannot know where A is. Any
rejection is therefore a false positive, and no ground-truth registration is needed to
say so. That is what makes this measurable at all for serial sections.

B is then carried into A's frame three ways, from ONE VALIS registration:

    rigid     : VALIS rigid only          — distance-preserving
    nonrigid  : VALIS rigid + micro-reg   — the warp OASIS forbids
    random    : rigid + a smooth RANDOM displacement of the SAME RMS as the
                micro-registration's, which is tissue-BLIND  (the ARM-3 control)

Everything else is held identical across arms: same A, same B in the moving frame,
same analysis window, same null seed, same radii. The transform is the only thing
that moves.

MECHANISM UNDER TEST. A non-rigid warp pushes moving tissue onto fixed tissue. A lives
in fixed tissue. So the warp aligns B's fine-scale density with A's, at scales below any
null model's bandwidth, and the test reads co-densification as cell-scale association.
If that is the mechanism, `nonrigid` inflates the false-positive rate and the
magnitude-matched `random` control does not — distortion per se is not the problem,
distortion that is aligned with the thing being measured is.

Run:  .venv/bin/python -m validation.nonrigid_bench.arms [--arm 1|2|3|all]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from oasis.spatial import spatial_stats as ss           # noqa: E402
from validation.nonrigid_bench import warpfield as wf   # noqa: E402

OUT = os.path.join(_HERE, "results")

# radii in MICRONS; converted per pair with that pair's µm/px
R_UM = np.arange(0.0, 101.0, 4.0)
# ROI-scale analysis window. The real pipeline certifies regions of R = 260-450 µm
# (ihc.md §3.5), so an 800 µm square is the scale at which this question is actually
# asked — and at this size a few hundred cells give a mean nearest-neighbour spacing
# of ~25 µm, so the 10-20 µm contact band carries real content.
WINDOW_UM = 800.0
N_A = 250
N_B = 250
OVERSAMPLE = 6            # draw extra in the moving frame; the window keeps N_B
N_SUPPORT = 2500          # all-cell morphology field for the dense null
ALPHA = 0.05
N_PERM = 299
# tissues whose 25pc pixel size cannot resolve a 0-20 µm contact band at all
MAX_PX_UM = 2.5


# ─────────────────────────── tissue density sampling ───────────────────────────

class Density:
    """Sampling weights ∝ real tissue/nuclear darkness, at thumbnail resolution.

    This is what gives A and B genuine architecture from the ACTUAL sections rather
    than a uniform scatter inside a coarse polygon. `draw` can be restricted to a
    full-resolution bounding box, which is how ROI-scale windows are populated at a
    realistic cell density (a whole 25pc ANHIR section is ~10^2 mm²; putting a few
    hundred points in it would leave the 10-20 µm contact band literally empty, and
    the real pipeline analyses certified sub-regions of a few hundred µm anyway).
    """

    def __init__(self, image_path, max_side=4000):
        from oasis.common.registration import _load_thumbnail
        import cv2
        gray, scale = _load_thumbnail(image_path, max_side=max_side, mode="L")
        if gray is None or scale <= 0:
            self.ok = False
            return
        g = np.asarray(gray, dtype=np.float64)
        thr, _ = cv2.threshold(np.ascontiguousarray(g.astype(np.uint8)), 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        w = np.where(g <= thr, g.max() - g, 0.0)   # background contributes nothing
        self.ok = w.sum() > 0
        self.w = w
        self.scale = scale
        self.H, self.W = w.shape

    def draw(self, n, rng, bbox=None):
        """n points as FULL-RESOLUTION coords; bbox=(x0,y0,x1,y1) in full-res px."""
        w = self.w
        if bbox is not None:
            x0, y0, x1, y1 = [v * self.scale for v in bbox]
            m = np.zeros_like(w)
            c0, c1 = max(0, int(x0)), min(self.W, int(np.ceil(x1)) + 1)
            r0, r1 = max(0, int(y0)), min(self.H, int(np.ceil(y1)) + 1)
            if c1 <= c0 or r1 <= r0:
                return np.empty((0, 2))
            m[r0:r1, c0:c1] = w[r0:r1, c0:c1]
            w = m
        tot = w.sum()
        if tot <= 0:
            return np.empty((0, 2))
        flat = (w / tot).ravel()
        pick = rng.choice(flat.size, size=int(n), replace=True, p=flat)
        yy, xx = np.divmod(pick, self.W)
        xx = xx + rng.random(len(xx))          # jitter inside the thumbnail pixel
        yy = yy + rng.random(len(yy))
        return np.column_stack([xx / self.scale, yy / self.scale])


# ────────────────────────────── association truth ──────────────────────────────

def attach_association(A, draw_b, rng, frac=0.5, sigma_px=8.0):
    """ARM 1 truth: a REAL association, imposed in the fixed frame. `frac` of B sits
    within ~sigma of a random A cell; the rest follows tissue density independently."""
    n_near = int(round(len(A) * frac))
    anchors = A[rng.integers(0, len(A), n_near)]
    near = anchors + rng.normal(0.0, sigma_px, size=(n_near, 2))
    rest = draw_b(N_B - n_near, rng)
    return np.vstack([near, rest])


# ──────────────────────────────── the statistic ────────────────────────────────

def _digest(summ):
    """The three numbers a practitioner actually reads off one null."""
    g = summ.get("global") or {}
    br = summ.get("bands_ring") or {}
    co = br.get("colocalization") or {}      # 10-20 µm  — the CELL-SCALE claim
    ci = br.get("coinfiltration") or {}      # 20-50 µm
    return {
        "p_dclf": _f(g.get("global_p_dclf")),
        "p_assoc": _f(g.get("global_p_association")),
        "significant": bool(g.get("significant")),
        "direction": g.get("direction"),
        "contact_p": _f(co.get("global_p_dclf")),
        "contact_sig": bool(co.get("significant")),
        "contact_dir": co.get("direction"),
        "coinfil_p": _f(ci.get("global_p_dclf")),
        "coinfil_sig": bool(ci.get("significant")),
        "coinfil_dir": ci.get("direction"),
    }


def verdicts(A, B, support, radii_px, area_px, px_um, poly, seed):
    """Run the pipeline's own nulls and return the numbers a practitioner would read."""
    out = {}
    try:
        res = ss.cross_k_all_nulls(A, B, radii_px, area_px, px_um,
                                   n_perm=N_PERM, seed=seed, tissue_polygon=poly)
        for name, summ in (res.get("nulls") or {}).items():
            out[name] = _digest(summ)
        out["robustness"] = (res.get("robustness") or {}).get("verdict")
    except Exception as e:
        out["error_allnulls"] = str(e)[:200]
    try:
        d = ss.cross_k_dense_morphology_test(A, B, support, radii_px, area_px, px_um,
                                             n_perm=N_PERM, seed=seed, tissue_polygon=poly)
        out["dense_morphology"] = _digest(d)
    except Exception as e:
        out["error_dense"] = str(e)[:200]
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────── driver ───────────────────────────────────

def run_pair(pw, arm, n_rep, seed0, cache):
    """One pair; n_rep ROI-scale windows; three transform arms per window."""
    from shapely.geometry import box
    from oasis.spatial.spatial_stats import estimate_tissue_polygon

    px_um = pw.px_um
    radii_px = R_UM / px_um
    half = 0.5 * WINDOW_UM / px_um            # window half-side, px

    key = pw.pair_id
    if key not in cache:
        fixed_img = _find_image(pw, "fixed")
        moving_img = _find_image(pw, "moving")
        if not fixed_img or not moving_img:
            return None
        _, poly = estimate_tissue_polygon(fixed_img, px_um)
        df, dm = Density(fixed_img), Density(moving_img)
        cache[key] = (poly, df if df.ok else None, dm if dm.ok else None)
    poly, dens_f, dens_m = cache[key]
    if poly is None or dens_f is None or dens_m is None:
        return None

    rows = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed0 + 1000 * rep + hash(key) % 997)
        # window centre drawn from tissue, so the window sits ON tissue
        c = dens_f.draw(1, rng)
        if not len(c):
            continue
        cx, cy = c[0]
        win = box(cx - half, cy - half, cx + half, cy + half).intersection(poly)
        if win.is_empty or win.area <= 0:
            continue
        area_px = float(win.area)
        wbb = (cx - half, cy - half, cx + half, cy + half)

        A = dens_f.draw(N_A, rng, bbox=wbb)
        support = dens_f.draw(N_SUPPORT, rng, bbox=wbb)
        if not len(A):
            continue

        if arm == 1:
            # truth lives in the FIXED frame; pull it back so `rigid` reproduces it exactly
            B_fixed = attach_association(A, lambda n, r: dens_f.draw(n, r, bbox=wbb), rng)
            B_mov = pw.inverse_rigid(B_fixed)
        else:
            # sample in the moving frame over the window's rigid pre-image, generously
            # margined so the non-rigid displacement cannot empty the window
            corners = pw.inverse_rigid(np.array(
                [[wbb[0], wbb[1]], [wbb[2], wbb[1]], [wbb[0], wbb[3]], [wbb[2], wbb[3]]]))
            m = 2.0 * max(pw.residual_rms_px, half)
            mbb = (corners[:, 0].min() - m, corners[:, 1].min() - m,
                   corners[:, 0].max() + m, corners[:, 1].max() + m)
            B_mov = dens_m.draw(N_B * OVERSAMPLE, rng, bbox=mbb)
        if not len(B_mov):
            continue

        B_rig, B_non = pw.rigid(B_mov), pw.nonrigid(B_mov)
        variants = {"rigid": B_rig, "nonrigid": B_non}
        d = B_non - B_rig
        d = d[np.isfinite(d).all(axis=1)]
        real_rms = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))) if len(d) else 0.0
        if arm == 3:
            variants["random"] = pw.random_smooth(B_mov, seed=seed0 + rep,
                                                  target_rms_px=real_rms)

        Ak = ss.filter_points_in_polygon(A, win)[0]
        row = {"pair_id": pw.pair_id, "set": pw.set, "rep": rep, "arm": arm,
               "px_um": px_um, "n_a": int(len(Ak)),
               "window_um": WINDOW_UM, "window_area_mm2": area_px * px_um ** 2 / 1e6,
               "microreg_rms_px": real_rms, "microreg_rms_um": real_rms * px_um}
        if len(Ak) < 40:
            continue
        for name, Bw in variants.items():
            Bw = Bw[np.isfinite(Bw).all(axis=1)]
            # NB: filter_points_in_polygon returns (points, n_excluded)
            Bk = ss.filter_points_in_polygon(Bw, win)[0] if len(Bw) else Bw
            if len(Bk) > N_B:                 # equalise n across arms; order is random
                Bk = Bk[rng.permutation(len(Bk))[:N_B]]
            if len(Bk) < 40:
                row[name] = {"n_b": int(len(Bk)), "skipped": True}
                continue
            r = verdicts(Ak, Bk, support, radii_px, area_px, px_um, win,
                         seed=seed0 + rep)
            r["n_b"] = int(len(Bk))
            row[name] = r
        rows.append(row)
    return rows


def _find_image(pw, which):
    """Recover the source image path for this pair from the valis_bench enumeration."""
    from validation.valis_bench import common as C
    for p in C.get_pairs():
        if p["pair_id"].replace("/", "_").replace(" ", "_") == pw.pair_id:
            return p[f"{which}_img"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-px-um", type=float, default=MAX_PX_UM)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pairs = wf.load_all()
    pairs = [p for p in pairs if p.px_um and p.px_um <= a.max_px_um]
    print(f"[arms] {len(pairs)} pairs at <= {a.max_px_um} µm/px", flush=True)
    if not pairs:
        print("[arms] nothing to do — run export_warps.py first")
        return

    arms = [1, 2, 3] if a.arm == "all" else [int(a.arm)]
    poly_cache = {}
    for arm in arms:
        t0 = time.time()
        allrows = []
        for i, pw in enumerate(pairs, 1):
            rows = run_pair(pw, arm, a.reps, a.seed, poly_cache)
            if rows:
                allrows += rows
                print(f"  arm{arm} [{i}/{len(pairs)}] {pw.pair_id} "
                      f"{len(rows)} reps ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  arm{arm} [{i}/{len(pairs)}] {pw.pair_id} SKIP", flush=True)
            json.dump(allrows, open(os.path.join(OUT, f"arm{arm}.json"), "w", encoding="utf-8"), indent=1)
        print(f"[arms] arm {arm}: {len(allrows)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
