"""
validate_fw_calibration_extended.py — raise n on the FW bound's only external check.

WHY THIS EXISTS. `validate_fw_anhir_calibration.py` is the sole annotator-independent
calibration of the §3.5 cell-error budget, and it runs on THREE pairs. The whole
architecture trades registration accuracy for that bound (§23.5), so a reviewer who reads
n=3 has a fair objection. This raises n without pretending the added pairs are as good as
the original three.

THE CEILING, and why it is real. Only four CIMA/ANHIR sets carry two annotators, and
`lung-lobes_3` has JB on just one image of the set, so no pair can be formed from it.
Three undirected pairs is all the two-annotator data there is. That is a property of the
dataset, not of the harness.

TWO ARMS, and the second is credentialed by the first.

  GOLD   two annotators, exactly as validate_fw_anhir_calibration.py. FLE is measured
         inter-observer; the held-out landmarks were localised by a DIFFERENT person, so
         their errors are independent of the fit's. Extended here from 3 undirected to 6
         DIRECTED pairs: A->B and B->A are different fits, different design matrices and
         different held-out residuals, so both are legitimate evaluations of the bound.

  PROXY  one annotator, split-half. Fit the similarity on a random half of the landmarks;
         predict and measure realized error at the held-out half. Runs on ANY pair, so it
         scales to the whole of ANHIR — at the cost of two biases that pull OPPOSITE ways:

           - the held-out points were clicked by the SAME annotator, so systematic bias is
             shared and realized error may be UNDER-stated  → ratio biased LOW (unsafe);
           - FLE is not measurable without a second pass, so it is supplied, and a
             too-small FLE shrinks PREDICTED error → ratio biased HIGH (safe).

         Which dominates is an empirical question, and it is answerable: run BOTH arms on
         the six gold pairs and compare. That comparison is the whole point of this file.
         If the proxy tracks the gold there, the proxy's wider n means something. If it
         does not, it is reported as failed and n stays at 6.

Run:  .venv/bin/python -m validation.validate_fw_calibration_extended [--anhir 24]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from oasis.spatial import serial_registration as sr   # noqa: E402
from validation.valis_bench import common as C        # noqa: E402

TISSUE_PX = {"lung-lesion": (0.174, 0.50), "lung-lobes": (1.274, 1.00),
             "mammary-gland": (2.294, 1.00)}
IMAGE_WH = {"lung-lesion_3": (8920, 6610), "mammary-gland_1": (10000, 8000),
            "mammary-gland_2": (10000, 8000)}
GOLD_SETS = [
    ("lung-lesion_3", "29-041-Izd2-w35-He-les3.csv", "29-041-Izd2-w35-proSPC-4-les3.csv"),
    ("mammary-gland_1", "s1_37-HE_A4926-4L.csv", "s1_40-PR_A4926-4L.csv"),
    ("mammary-gland_2", "s2_63-HE_A4926-4L.csv", "s2_68-ER-A4962-4L.csv"),
]
N_SPLITS = 12          # random half-splits averaged per proxy pair
MIN_LM = 40


def _roots():
    for p in (os.path.join(_HERE, "public_landmarks", "annotations"),
              os.path.expanduser("~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations")):
        if os.path.isdir(p):
            yield p


def load_xy(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]
    xi = hdr.index("x") if "x" in hdr else -2
    yi = hdr.index("y") if "y" in hdr else -1
    out = []
    for r in rows[1:]:
        try:
            out.append([float(r[xi]), float(r[yi])])
        except (ValueError, IndexError):
            pass
    return np.array(out, float)


def find(tissue, user, base):
    for root in _roots():
        hits = glob.glob(os.path.join(root, tissue, f"user-{user}_scale-*", base))
        if hits:
            return hits[0]
    return None


def px_for(tissue):
    for pref, (native, scale) in TISSUE_PX.items():
        if tissue.startswith(pref):
            return native / scale
    raise KeyError(tissue)


# ───────────────────────────── the shared calibration core ─────────────────────────────

def _calibrate(fit_ref, fit_mov, eval_ref, eval_mov, px, fle, wh=None):
    """Fit on (fit_ref, fit_mov); predict and measure realized error at the eval set.

    Identical arithmetic in both arms — only WHERE the eval points come from differs.
    """
    M = sr._fit_similarity_robust(fit_mov, fit_ref)
    dec = sr.deformation_from_landmarks(fit_ref, fit_mov, M, px, fle, method="robust")
    deform = max(dec["deformation_p90_um"] or 0.0, dec["deformation_rms_um"] or 0.0)
    deform_ub = max(dec["deformation_p90_ub_um"] or 0.0, dec["deformation_rms_ub_um"] or 0.0)
    tre = sr.transform_prediction_error(fit_ref, fle * np.sqrt(2.0), eval_ref)
    if tre is None:
        return None
    ann = 2.0 * fle ** 2                    # the eval landmark's own clicking noise, both images
    pred = np.sqrt(tre ** 2 + deform ** 2 + ann)
    pred_ub = np.sqrt(tre ** 2 + deform_ub ** 2 + ann)
    real = np.linalg.norm(sr._apply_affine(eval_mov, M) - eval_ref, axis=1) * px
    return {"ratio": float(np.percentile(real, 90) / np.percentile(pred, 90)),
            "cov": float(np.mean(real <= pred_ub)),
            "real_p90": float(np.percentile(real, 90)),
            "pred_p90": float(np.percentile(pred, 90)),
            "n_fit": int(len(fit_ref)), "n_eval": int(len(eval_ref))}


def gold_pair(tissue, fixed, moving, flip):
    """Two-annotator calibration. `flip` swaps which image is the fixed frame."""
    if flip:
        fixed, moving = moving, fixed
    paths = {(u, k): find(tissue, u, b) for u in ("PS", "JB")
             for k, b in (("ref", fixed), ("mov", moving))}
    if any(v is None for v in paths.values()):
        return None
    P = {k: load_xy(v) for k, v in paths.items()}
    n = min(len(v) for v in P.values())
    ps_r, ps_m = P[("PS", "ref")][:n], P[("PS", "mov")][:n]
    jb_r, jb_m = P[("JB", "ref")][:n], P[("JB", "mov")][:n]
    px = px_for(tissue)

    f_ref = sr.fle_from_repeat(ps_r, jb_r, px)
    f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    if f_ref["fle_um"] is None or f_mov["fle_um"] is None:
        return None
    keep = np.array(f_ref["concordant"]) & np.array(f_mov["concordant"])
    ps_r, ps_m, jb_r, jb_m = ps_r[keep], ps_m[keep], jb_r[keep], jb_m[keep]
    if len(ps_r) < 8:
        return None
    f_ref = sr.fle_from_repeat(ps_r, jb_r, px)
    f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    fle = float(np.sqrt(np.mean([f_ref["fle_um"] ** 2, f_mov["fle_um"] ** 2])))

    out = _calibrate(ps_r, ps_m, jb_r, jb_m, px, fle, IMAGE_WH.get(tissue))
    if out is None:
        return None
    out.update(arm="gold", pair=f"{tissue}{' (rev)' if flip else ''}", tissue=tissue,
               fle_um=fle, px_um=px)
    # the SAME pair scored by the proxy design, using the SAME measured FLE, so the two
    # arms differ in nothing except where the held-out points come from
    pr = proxy_from_points(ps_r, ps_m, px, fle)
    out["proxy_ratio"] = pr["ratio"] if pr else None
    out["proxy_cov"] = pr["cov"] if pr else None
    return out


def proxy_from_points(ref, mov, px, fle, n_splits=N_SPLITS, seed=0):
    """Split-half calibration averaged over random splits of ONE annotator's landmarks."""
    rng = np.random.default_rng(seed)
    n = len(ref)
    if n < MIN_LM:
        return None
    rs, cs = [], []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        h = n // 2
        fi, ei = idx[:h], idx[h:]
        r = _calibrate(ref[fi], mov[fi], ref[ei], mov[ei], px, fle)
        if r:
            rs.append(r["ratio"]); cs.append(r["cov"])
    if not rs:
        return None
    return {"ratio": float(np.median(rs)), "cov": float(np.median(cs)),
            "ratio_iqr": [float(np.percentile(rs, 25)), float(np.percentile(rs, 75))],
            "n_splits": len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anhir", type=int, default=24, help="ANHIR pairs for the proxy arm")
    ap.add_argument("--fle-um", type=float, default=None,
                    help="FLE for the proxy arm on ANHIR (default: gold median)")
    a = ap.parse_args()

    print("=" * 80)
    print("FW bound calibration — raising n, and saying what each pair is worth")
    print("=" * 80)

    # ── ARM 1: GOLD ──────────────────────────────────────────────────────────
    print("\nARM 1 — GOLD (two annotators, held-out annotator independent)\n")
    print(f"  {'pair':26s} {'FLE µm':>7s} {'real p90':>9s} {'pred p90':>9s} "
          f"{'ratio':>6s} {'cov':>5s} {'proxy ratio':>12s}")
    gold = []
    for tis, f, m in GOLD_SETS:
        for flip in (False, True):
            r = gold_pair(tis, f, m, flip)
            if not r:
                continue
            gold.append(r)
            pr = f"{r['proxy_ratio']:.2f}" if r["proxy_ratio"] else "n/a"
            print(f"  {r['pair'][:26]:26s} {r['fle_um']:7.1f} {r['real_p90']:9.1f} "
                  f"{r['pred_p90']:9.1f} {r['ratio']:6.2f} {100*r['cov']:4.0f}% {pr:>12s}")
    if not gold:
        print("  no gold pairs found — is the CIMA annotation tree present?")
        return 1
    gr = np.array([r["ratio"] for r in gold])
    gc = np.array([r["cov"] for r in gold])
    print(f"\n  n = {len(gold)} directed pairs   ratio median {np.median(gr):.2f} "
          f"(min {gr.min():.2f}, max {gr.max():.2f})   coverage median {100*np.median(gc):.0f}%")

    # ── the credential ───────────────────────────────────────────────────────
    both = [(r["ratio"], r["proxy_ratio"]) for r in gold if r["proxy_ratio"]]
    print("\nARM 2 CREDENTIAL — does the split-half proxy reproduce the gold answer?\n")
    if len(both) >= 3:
        g = np.array([b[0] for b in both]); p = np.array([b[1] for b in both])
        d = p - g
        print(f"  paired on {len(both)} gold pairs, SAME measured FLE, only the held-out "
              f"set differs")
        print(f"  gold ratio  median {np.median(g):.2f}   proxy ratio median {np.median(p):.2f}")
        print(f"  proxy − gold: median {np.median(d):+.2f}, "
              f"range {d.min():+.2f} to {d.max():+.2f}")
        biased = abs(np.median(d)) > 0.15
        print(f"  => the proxy {'DOES NOT track' if biased else 'tracks'} the gold design"
              f"{' — its wider n is not usable' if biased else ''}")
    else:
        biased = True
        print("  too few paired observations to credential the proxy")

    # ── ARM 2: PROXY across ANHIR ────────────────────────────────────────────
    fle_default = a.fle_um if a.fle_um else float(np.median([r["fle_um"] for r in gold]))
    print(f"\nARM 2 — PROXY (split-half, single annotator) on ANHIR, "
          f"FLE = {fle_default:.1f} µm\n")
    pairs = C.stratified_pairs(C.get_pairs(), max(1, a.anhir // 8))
    rows = []
    print(f"  {'pair':34s} {'µm/px':>6s} {'n_lm':>5s} {'ratio':>6s} {'cov':>5s}")
    for p in pairs:
        ref = np.asarray(p["fixed_lm"], float)
        mov = np.asarray(p["moving_lm"], float)
        px = C.px_um_for(p["set"], p.get("img_scale_pc", 25))
        if px is None or len(ref) < MIN_LM:
            continue
        r = proxy_from_points(ref, mov, px, fle_default)
        if not r:
            continue
        r.update(arm="proxy", pair=p["pair_id"], tissue=C.tissue_of(p["set"]), px_um=px,
                 n_lm=int(len(ref)))
        rows.append(r)
        print(f"  {p['pair_id'][:34]:34s} {px:6.2f} {len(ref):5d} "
              f"{r['ratio']:6.2f} {100*r['cov']:4.0f}%")
    if rows:
        pr = np.array([r["ratio"] for r in rows])
        pc = np.array([r["cov"] for r in rows])
        print(f"\n  n = {len(rows)} pairs over "
              f"{len(set(r['tissue'] for r in rows))} tissue types")
        print(f"  ratio median {np.median(pr):.2f} (IQR {np.percentile(pr,25):.2f}–"
              f"{np.percentile(pr,75):.2f}, max {pr.max():.2f})")
        print(f"  coverage median {100*np.median(pc):.0f}%  "
              f"({int((pc>=0.85).sum())}/{len(pc)} pairs at >=85%)")

    print("\n" + "=" * 80)
    print(f"REPORTABLE n:  gold {len(gold)} directed pairs"
          + (f"  +  proxy {len(rows)} pairs (credentialed)" if rows and not biased
             else "   (proxy NOT usable)"))
    print("=" * 80)
    json.dump({"gold": gold, "proxy": rows, "proxy_credentialed": bool(not biased),
               "fle_proxy_um": fle_default},
              open(os.path.join(_HERE, "fw_calibration_extended.json"), "w",
                   encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
