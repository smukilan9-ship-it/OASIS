"""
route3.py — can a NON-RIGID transform be certified without ground truth?

THE PROBLEM. §23 showed the statistic's correctness is driven by cell-scale registration
error, and that the non-rigid warp is the more accurate transform. But §3.5's budget,
`TRE_pred = σ·√(fᵀ(XᵀX)⁻¹f)`, needs a parametric model with a design matrix, and VALIS's
micro-registration is a dense field with none. So adopting it means losing the bound.

THE QUESTION, stated so it can fail. A certificate is only worth anything if it is
computable from what an operator HAS — the images, the correspondences, the fitted
displacement field — and predicts what they CANNOT see, namely the true error at cells.
So: does any field-derived quantity predict the realized TRE against expert landmarks?

CANDIDATES (all computable with no ground truth whatsoever):

  distort_*   the ORIGINAL route-3 idea — local distance distortion. Sample point pairs
              in the window, carry them through the warp, and measure |Δd|/d in the
              analysis band. This is a direct empirical bound on how much the warp
              violates distance preservation.
  jac_*       the same thing analytically: singular values of the local Jacobian.
              aniso = σmax/σmin − 1 (shear), scale_cv = spread of √(σmax·σmin) across the
              window (non-uniform scaling).
  micro_*     how far micro-registration moved things relative to the rigid stage. Not a
              distortion measure at all — a measure of how much work the warp is doing.
  simres_*    how far the warp departs from the best-fitting LOCAL similarity inside the
              window. A field that a similarity already explains is one the §3.5 budget
              could have handled.

GROUND TRUTH for scoring: realized TRE at ANHIR expert landmarks, which no registration
sees, via `_predict_local_affine` — the same truth arm_gt uses.

A candidate EARNS the name certificate only if it correlates with realized TRE and can
separate windows that are cell-scale accurate from windows that are not. If distance
distortion does not, then route 3 as originally conceived is dead, and §23.7 item 3 has
to be withdrawn rather than published.

Run:  .venv/bin/python -m validation.nonrigid_bench.route3 [--windows 30]
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

from validation.nonrigid_bench import arms as A          # noqa: E402
from validation.nonrigid_bench import warpfield as wf    # noqa: E402
from validation.valis_bench import common as C           # noqa: E402

OUT = os.path.join(_HERE, "results")
MAX_NEAR_FRAC = 0.06
BAND_UM = (10.0, 20.0)      # the contact band the cell-scale claim is made in


def _fit_similarity(P, Q):
    """Least-squares similarity P->Q; returns per-point residual norms."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    mp, mq = P.mean(0), Q.mean(0)
    P0, Q0 = P - mp, Q - mq
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    var = (P0 ** 2).sum()
    s = (S * [1.0, d]).sum() / var if var > 0 else 1.0
    pred = (s * (R @ P0.T)).T + mq
    return np.linalg.norm(pred - Q, axis=1)


def window_features(pw, wbb, rng, n_pts=900):
    """Everything an operator could compute with no ground truth."""
    dens = pw._dens_m
    P = dens.draw(n_pts, rng, bbox=wbb)
    if len(P) < 200:
        return None
    Pr, Pn = pw.rigid(P), pw.nonrigid(P)
    ok = np.isfinite(Pr).all(1) & np.isfinite(Pn).all(1)
    P, Pr, Pn = P[ok], Pr[ok], Pn[ok]
    if len(P) < 200:
        return None
    px = pw.px_um
    f = {}

    # ── micro-registration magnitude ──────────────────────────────────────────
    dmag = np.linalg.norm(Pn - Pr, axis=1) * px
    f["micro_median_um"] = float(np.median(dmag))
    f["micro_p90_um"] = float(np.percentile(dmag, 90))
    # spatial VARIATION of the displacement, not its size: a constant offset is a
    # translation and harms nothing, a varying one is what bends geometry
    f["micro_sd_um"] = float(np.linalg.norm((Pn - Pr) - (Pn - Pr).mean(0), axis=1).std() * px)

    # ── distance distortion in the analysis band (route 3 as conceived) ───────
    from scipy.spatial import cKDTree
    prs = np.array(sorted(cKDTree(Pn).query_pairs(BAND_UM[1] / px)), dtype=int)
    if len(prs) < 100:
        return None
    if len(prs) > 60000:
        prs = prs[rng.permutation(len(prs))[:60000]]
    i, j = prs[:, 0], prs[:, 1]
    dn = np.linalg.norm(Pn[i] - Pn[j], axis=1) * px
    dr = np.linalg.norm(Pr[i] - Pr[j], axis=1) * px
    sel = (dr >= BAND_UM[0]) & (dr < BAND_UM[1])
    if sel.sum() < 50:
        return None
    rel = np.abs(dn[sel] - dr[sel]) / np.maximum(dr[sel], 1e-9)
    f["distort_median"] = float(np.median(rel))
    f["distort_p90"] = float(np.percentile(rel, 90))
    f["distort_max"] = float(rel.max())

    # ── local Jacobian: shear and non-uniform scale ───────────────────────────
    h = max(2.0, 0.02 * (wbb[2] - wbb[0]))
    Q = P[rng.permutation(len(P))[:300]]
    Jx = (pw.nonrigid(Q + [h, 0]) - pw.nonrigid(Q - [h, 0])) / (2 * h)
    Jy = (pw.nonrigid(Q + [0, h]) - pw.nonrigid(Q - [0, h])) / (2 * h)
    J = np.stack([Jx, Jy], axis=-1)              # N x 2 x 2
    good = np.isfinite(J).all(axis=(1, 2))
    if good.sum() < 50:
        return None
    sv = np.linalg.svd(J[good], compute_uv=False)
    smax, smin = sv[:, 0], np.maximum(sv[:, 1], 1e-9)
    f["jac_aniso_median"] = float(np.median(smax / smin - 1.0))
    f["jac_aniso_p90"] = float(np.percentile(smax / smin - 1.0, 90))
    g = np.sqrt(smax * smin)
    f["jac_scale_cv"] = float(np.std(g) / max(np.mean(g), 1e-9))

    # ── departure from the best local similarity ──────────────────────────────
    res = _fit_similarity(P, Pn) * px
    f["simres_median_um"] = float(np.median(res))
    f["simres_p90_um"] = float(np.percentile(res, 90))
    return f


def run(n_windows, seed0):
    pairs = [p for p in wf.load_all() if p.px_um and p.px_um <= A.MAX_PX_UM]
    print(f"[route3] {len(pairs)} pairs x {n_windows} windows", flush=True)
    rows = []
    t0 = time.time()
    for pi, pw in enumerate(pairs, 1):
        p = None
        for q in C.get_pairs():
            if q["pair_id"].replace("/", "_").replace(" ", "_") == pw.pair_id:
                p = q
                break
        if p is None:
            continue
        lm_ref = np.asarray(p["fixed_lm"], float)
        lm_mov = np.asarray(p["moving_lm"], float)
        if len(lm_ref) < 8:
            continue
        mi = A._find_image(pw, "moving")
        if not mi:
            continue
        dens = A.Density(mi)
        if not dens.ok:
            continue
        pw._dens_m = dens
        half = 0.5 * A.WINDOW_UM / pw.px_um
        max_near = MAX_NEAR_FRAC * float(np.hypot(*pw.moving_wh))

        for w in range(n_windows):
            rng = np.random.default_rng(seed0 + 131 * w + hash(pw.pair_id) % 9973)
            c = lm_mov[rng.integers(0, len(lm_mov))]
            wbb = (c[0] - half, c[1] - half, c[0] + half, c[1] + half)
            f = window_features(pw, wbb, rng)
            if f is None:
                continue
            # ground truth: realized error at expert landmarks
            Pg = dens.draw(300, rng, bbox=wbb)
            Gt = C._predict_local_affine(Pg, lm_mov, lm_ref, k=6, max_near_px=max_near)
            keep = np.isfinite(Gt).all(axis=1)
            Pg, Gt = Pg[keep], Gt[keep]
            if len(Pg) < 60:
                continue
            Rg, Ng = pw.rigid(Pg), pw.nonrigid(Pg)
            ok = np.isfinite(Rg).all(1) & np.isfinite(Ng).all(1)
            if ok.sum() < 60:
                continue
            tre_r = float(np.median(np.linalg.norm(Rg[ok] - Gt[ok], axis=1)) * pw.px_um)
            tre_n = float(np.median(np.linalg.norm(Ng[ok] - Gt[ok], axis=1)) * pw.px_um)
            rows.append({"pair_id": pw.pair_id, "set": pw.set, "win": w,
                         "px_um": pw.px_um, "tre_rigid_um": tre_r,
                         "tre_nonrigid_um": tre_n, **f})
        print(f"  [{pi}/{len(pairs)}] {pw.pair_id[:38]} -> {len(rows)} rows "
              f"({time.time()-t0:.0f}s)", flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(rows, open(os.path.join(OUT, "route3.json"), "w",
                              encoding="utf-8"), indent=1)
    return rows


def analyse(rows):
    from scipy.stats import spearmanr
    feats = [k for k in rows[0] if k.startswith(("distort_", "jac_", "micro_", "simres_"))]
    y = np.array([r["tre_nonrigid_um"] for r in rows])
    print(f"\n{'='*76}\nDoes any field-derived quantity predict the REALIZED error?  "
          f"(n={len(rows)})\n{'='*76}")
    print(f"realized non-rigid TRE: median {np.median(y):.1f} µm, "
          f"IQR {np.percentile(y,25):.1f}-{np.percentile(y,75):.1f}")
    print(f"\n{'feature':22s} {'Spearman r':>11s} {'p':>10s} {'AUC(TRE<20µm)':>14s}")
    good = y < 20.0
    out = []
    for f in sorted(feats):
        x = np.array([r[f] for r in rows], float)
        m = np.isfinite(x)
        if m.sum() < 30:
            continue
        r, pv = spearmanr(x[m], y[m])
        # AUC of -x as a score for "this window is cell-scale accurate"
        auc = np.nan
        if good[m].sum() >= 5 and (~good[m]).sum() >= 5:
            xp, xn = x[m][good[m]], x[m][~good[m]]
            auc = float((xp[:, None] < xn[None, :]).mean()
                        + 0.5 * (xp[:, None] == xn[None, :]).mean())
        out.append((f, r, pv, auc))
        print(f"{f:22s} {r:11.3f} {pv:10.2e} {auc:14.3f}")
    print(f"\ncell-scale-accurate windows (TRE < 20 µm): {good.sum()}/{len(y)}")
    print("AUC 0.5 = the quantity carries no information about whether this window is "
          "safe to analyse.\nA usable certificate needs AUC well above 0.5 AND a "
          "monotone relation to realized error.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    path = os.path.join(OUT, "route3.json")
    if a.analyse_only and os.path.exists(path):
        rows = json.load(open(path, encoding="utf-8"))
    else:
        rows = run(a.windows, a.seed)
    if rows:
        analyse(rows)


if __name__ == "__main__":
    main()
