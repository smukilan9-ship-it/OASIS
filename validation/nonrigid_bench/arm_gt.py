"""
arm_gt.py — THE DECISIVE ARM. Which transform gives the RIGHT statistical answer,
judged against expert landmarks rather than against an assumption.

Arm 1 in arms.py defines the truth by pulling it back through the rigid map, so rigid
reproduces it exactly by construction. That is fine for showing that micro-registration
CHANGES the answer, but it cannot show which answer is CORRECT — and since VALIS's
non-rigid stage is the more accurate one at expert landmarks (MMrTRE 0.0015 vs 0.0037,
ihc.md §7.1), assuming rigid is right stacks the deck.

Here the truth is external. ANHIR ships expert landmark correspondences that no
registration ever sees. `_predict_local_affine` turns them into a ground-truth local
displacement field, and we only place points where a landmark is genuinely nearby, so
the field is interpolated rather than extrapolated.

    B_true = T_gt(B_moving)      where the cells actually belong
    B_rigid = T_rigid(B_moving)  distance-preserving, less accurate at landmarks
    B_nonrig = T_nonrig(B_moving) more accurate at landmarks, not distance-preserving

A is then placed in the fixed frame with a KNOWN association to B_true. Running the
statistic on (A, B_true) gives the answer a perfect registration would produce. The
question is which of the two real transforms reproduces it.

Run:  .venv/bin/python -m validation.nonrigid_bench.arm_gt [--reps 10]
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

from oasis.spatial import spatial_stats as ss            # noqa: E402
from validation.nonrigid_bench import arms as A          # noqa: E402
from validation.nonrigid_bench import warpfield as wf    # noqa: E402
from validation.valis_bench import common as C           # noqa: E402

OUT = os.path.join(_HERE, "results")
MAX_NEAR_FRAC = 0.06      # a landmark must be within 6% of the image diagonal


def pair_meta(pair_id):
    for p in C.get_pairs():
        if p["pair_id"].replace("/", "_").replace(" ", "_") == pair_id:
            return p
    return None


def run_pair(pw, n_rep, seed0, cache):
    from shapely.geometry import box
    p = pair_meta(pw.pair_id)
    if p is None:
        return None
    lm_ref = np.asarray(p["fixed_lm"], float)     # fixed-image landmarks
    lm_mov = np.asarray(p["moving_lm"], float)    # moving-image landmarks
    if len(lm_ref) < 8:
        return None

    px_um = pw.px_um
    radii_px = A.R_UM / px_um
    half = 0.5 * A.WINDOW_UM / px_um
    diag = float(np.hypot(*pw.moving_wh))
    max_near = MAX_NEAR_FRAC * diag

    if pw.pair_id not in cache:
        mi = A._find_image(pw, "moving")
        fi = A._find_image(pw, "fixed")
        if not mi or not fi:
            return None
        dm, df = A.Density(mi), A.Density(fi)
        cache[pw.pair_id] = (dm if dm.ok else None, df if df.ok else None)
    dens_m, dens_f = cache[pw.pair_id]
    if dens_m is None or dens_f is None:
        return None

    rows = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed0 + 977 * rep + hash(pw.pair_id) % 991)
        # centre the window ON a landmark, so the ground-truth field is supported
        c = lm_mov[rng.integers(0, len(lm_mov))]
        wbb = (c[0] - half, c[1] - half, c[0] + half, c[1] + half)
        B_mov = dens_m.draw(A.N_B, rng, bbox=wbb)
        if len(B_mov) < 60:
            continue

        B_true = C._predict_local_affine(B_mov, lm_mov, lm_ref, k=6, max_near_px=max_near)
        keep = np.isfinite(B_true).all(axis=1)
        B_mov, B_true = B_mov[keep], B_true[keep]
        if len(B_mov) < 60:
            continue
        B_rig, B_non = pw.rigid(B_mov), pw.nonrigid(B_mov)
        ok = np.isfinite(B_rig).all(1) & np.isfinite(B_non).all(1)
        B_mov, B_true, B_rig, B_non = B_mov[ok], B_true[ok], B_rig[ok], B_non[ok]
        if len(B_mov) < 60:
            continue

        # how accurate is each transform, in µm, against the expert field?
        e_rig = np.linalg.norm(B_rig - B_true, axis=1) * px_um
        e_non = np.linalg.norm(B_non - B_true, axis=1) * px_um

        # A: a KNOWN association to where the cells actually belong
        n_near = int(round(0.5 * len(B_true)))
        anch = B_true[rng.integers(0, len(B_true), n_near)]
        sigma = 8.0 / px_um
        A_near = anch + rng.normal(0.0, sigma, size=(n_near, 2))
        tb = (B_true[:, 0].min(), B_true[:, 1].min(), B_true[:, 0].max(), B_true[:, 1].max())
        A_bg = dens_f.draw(len(B_true) - n_near, rng, bbox=tb)
        A_pts = np.vstack([A_near, A_bg]) if len(A_bg) else A_near

        win = box(*tb).buffer(0)
        if win.is_empty or win.area <= 0:
            continue
        area_px = float(win.area)
        support = dens_f.draw(A.N_SUPPORT, rng, bbox=tb)
        Ak = ss.filter_points_in_polygon(A_pts, win)[0]
        if len(Ak) < 40:
            continue

        row = {"pair_id": pw.pair_id, "set": pw.set, "rep": rep, "px_um": px_um,
               "n_a": int(len(Ak)), "n_lm": int(len(lm_ref)),
               "tre_rigid_um": float(np.median(e_rig)),
               "tre_nonrigid_um": float(np.median(e_non)),
               "window_area_mm2": area_px * px_um ** 2 / 1e6}
        for name, Bw in (("truth", B_true), ("rigid", B_rig), ("nonrigid", B_non)):
            Bk = ss.filter_points_in_polygon(Bw, win)[0]
            if len(Bk) < 40:
                row[name] = {"n_b": int(len(Bk)), "skipped": True}
                continue
            r = A.verdicts(Ak, Bk, support, radii_px, area_px, px_um, win,
                           seed=seed0 + rep)
            r["n_b"] = int(len(Bk))
            row[name] = r
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pairs = [p for p in wf.load_all() if p.px_um and p.px_um <= A.MAX_PX_UM]
    print(f"[gt] {len(pairs)} pairs", flush=True)
    cache, allrows = {}, []
    t0 = time.time()
    for i, pw in enumerate(pairs, 1):
        rows = run_pair(pw, a.reps, a.seed, cache)
        n = len(rows) if rows else 0
        allrows += rows or []
        print(f"  [{i}/{len(pairs)}] {pw.pair_id[:40]} {n} reps ({time.time()-t0:.0f}s)",
              flush=True)
        json.dump(allrows, open(os.path.join(OUT, "arm_gt.json"), "w"), indent=1)
    print(f"[gt] {len(allrows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
