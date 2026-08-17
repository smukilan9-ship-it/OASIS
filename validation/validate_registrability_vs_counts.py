#!/usr/bin/env python
"""
validate_registrability_vs_counts.py — do the pairs that REGISTER have the CELLS?

Three pairs hinted at something worth checking across the whole cohort: the one that
registered at 4.8 µm had 8 positive cells on one side, while the ones carrying hundreds of
cells would not register at all. If that holds over all 36 pairs it reframes the problem,
because it means no amount of registration work reaches the pairs that could answer a
spatial question.

Also measured, per pair: whether the residual shrinks when the transform is given more
freedom. A residual flat across similarity, affine and quadratic is NOT tissue deformation —
deformation is exactly what extra degrees of freedom absorb. It is wrong correspondences,
and no transform class, non-rigid included, repairs those. That distinction decides whether
a better registrar could help at all.

Correspondences come from LoFTR on the raw pair. FLE is deliberately not measured here (it
re-runs the whole matcher five times under noise); this asks how well a transform can fit,
not whether the fit certifies, so the fit residual is the right statistic and it is the best
case for each model.

Run:  OASIS_PAIR_MANIFEST=... python validation/validate_registrability_vs_counts.py
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST = os.path.expanduser(os.environ.get("OASIS_PAIR_MANIFEST",
                                             "~/oasis_pair_manifest.json"))
IMG_ROOT = os.path.expanduser(os.environ.get("OASIS_10X_ANALYSED", "~/Desktop/10x analyzed"))
GATE_UM = 10.0                      # the certification gate on held-out error
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "registrability_vs_counts_results.json")


def find_image(stem):
    for marker in ("cd8", "tim3"):
        p = os.path.join(IMG_ROOT, marker, stem, stem + ".tif")
        if os.path.exists(p):
            return p
    return None


def fit_residuals(rp, mp, px):
    """Median fit residual in µm for three transform classes of increasing freedom."""
    from oasis.spatial.serial_registration import _fit_similarity_robust, _apply_affine
    out = {}
    M = _fit_similarity_robust(mp, rp)
    out["similarity"] = (float(np.median(np.linalg.norm(_apply_affine(mp, M) - rp, axis=1)) * px)
                         if M is not None else None)
    A = np.hstack([mp, np.ones((len(mp), 1))])
    out["affine"] = float(np.median(np.linalg.norm(
        A @ np.linalg.lstsq(A, rp, rcond=None)[0] - rp, axis=1)) * px)
    x, y = mp[:, 0], mp[:, 1]
    Q = np.column_stack([x, y, np.ones_like(x), x * x, y * y, x * y])
    out["quadratic"] = float(np.median(np.linalg.norm(
        Q @ np.linalg.lstsq(Q, rp, rcond=None)[0] - rp, axis=1)) * px)
    return out


def main():
    if not os.path.exists(MANIFEST):
        print(f"No pair manifest at {MANIFEST}. Set OASIS_PAIR_MANIFEST.")
        return 2
    from oasis.spatial import loftr_matcher as lm
    from oasis.spatial.serial_registration import _load_rgb_thumbnail

    pairs = json.load(open(MANIFEST, encoding="utf-8"))["all_pairs"]
    print("=" * 104)
    print("Registrability against cell counts, all matched pairs")
    print("=" * 104)
    print("'flat' means the residual does not shrink with more transform freedom — wrong")
    print("correspondences rather than deformation, which no registrar repairs.\n")
    print(f"{'pair':<24}{'CD8+':>6}{'TIM3+':>7}{'min':>6}{'n_corr':>8}"
          f"{'similarity':>12}{'affine':>9}{'quad':>8}  verdict")
    print("-" * 104)

    rows, t0 = [], time.time()
    for r in pairs:
        sa, sb = r["file_a"][:-4], r["file_b"][:-4]
        pa, pb = find_image(sa), find_image(sb)
        if not pa or not pb:
            print(f"{r['base']:<24}  images not found"); continue
        R, _ = _load_rgb_thumbnail(pa, max_side=1920)
        M, _ = _load_rgb_thumbnail(pb, max_side=1920)
        px = float(r["px"])
        c = lm.loftr_correspondences(R, M, px)
        n = int(c.get("n") or 0)
        row = {**{k: r[k] for k in ("base", "region", "pos_a", "pos_b", "px", "passes")},
               "n_corr": n}
        if n >= 6:
            res = fit_residuals(np.asarray(c["ref_points"], float),
                                np.asarray(c["mov_points"], float), px)
            row.update(res)
            sim = res["similarity"]
            best = min(v for v in res.values() if v is not None)
            # "flat" = the most flexible model buys less than 25 % over the similarity
            row["flat"] = bool(sim is not None and best > 0.75 * sim)
            row["registers"] = bool(sim is not None and sim <= GATE_UM)
            v = ("registers" if row["registers"]
                 else ("flat, correspondences" if row["flat"] else "deformation"))
            print(f"{r['base']:<24}{r['pos_a']:>6}{r['pos_b']:>7}"
                  f"{min(r['pos_a'], r['pos_b']):>6}{n:>8}"
                  f"{sim:>11.1f}µ{res['affine']:>8.1f}µ{res['quadratic']:>7.1f}µ  {v}", flush=True)
        else:
            row.update(similarity=None, registers=False, flat=None)
            print(f"{r['base']:<24}{r['pos_a']:>6}{r['pos_b']:>7}"
                  f"{min(r['pos_a'], r['pos_b']):>6}{n:>8}"
                  f"{'—':>12}{'—':>9}{'—':>8}  too few correspondences", flush=True)
        rows.append(row)

    print("-" * 104)
    ok = [x for x in rows if x.get("registers")]
    good = [x for x in rows if x.get("similarity") is not None]
    print(f"{len(ok)} of {len(rows)} pairs register within {GATE_UM:.0f} µm.")
    if ok:
        print("  they carry (min marker count): "
              + ", ".join(f"{x['base']} {min(x['pos_a'], x['pos_b'])}" for x in ok))
    if len(good) >= 4:
        sim = np.array([x["similarity"] for x in good], float)
        mn = np.array([min(x["pos_a"], x["pos_b"]) for x in good], float)
        # Spearman without scipy: correlate the ranks
        rs = np.corrcoef(np.argsort(np.argsort(sim)), np.argsort(np.argsort(mn)))[0, 1]
        print(f"  rank correlation, residual vs the rarer marker's count: {rs:+.3f}  "
              f"(n={len(good)}; positive means more cells goes with worse registration)")
    flat = [x for x in rows if x.get("flat")]
    print(f"  {len(flat)} of {len(good)} fitted pairs are FLAT across transform classes — "
          f"their residual is correspondence error, not deformation.")
    both = [x for x in rows if x.get("registers") and x.get("passes")]
    print(f"\n  pairs that BOTH register and carry >50 of each marker: {len(both)}"
          + (" — " + ", ".join(x["base"] for x in both) if both else ""))
    json.dump({"gate_um": GATE_UM, "rows": rows},
              open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print(f"\nWrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
