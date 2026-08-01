"""Is the scale-consistency filter's tolerance too tight — measured against EXPERT landmarks?

THE OBSERVATION. On real 270 µm ROIs of LL477_CD8_x10_1 <-> Tim3_x10_1, every region the app
reported as NO_MATCHES had plenty of matches:

    283 raw -> 224 cycle-consistent -> 46 scale-consistent -> 31 locally smooth -> 5 in ROI

Six failing regions, all the same shape: the scale filter removes 85-93 %. Running the same
crops with the filter off (single scale) takes them from 5 usable matches to ~78.

THE SUSPECTED CAUSE, which is a units mismatch rather than a bad idea. `tol_px = tol_um /
pixel_size_um` is an absolute distance; the coarse pass matches on a grid of stride 8/scale =
16 full-resolution pixels. At the working scales certification actually uses, tol_px is 2-5 px
against that 16 px grid, so the filter asks the coarse pass to agree to a third of its own
cell. It cannot, so it culls almost everything — including correct matches.

WHY A RESIDUAL CANNOT ANSWER THIS, and why this file exists. The filter SELECTS the points
its own residual is measured on. Loosening it admits more points and the residual will move
for reasons that have nothing to do with whether the transform got better. The only honest
question is whether the resulting TRANSFORM lands expert-annotated anatomy closer to where the
expert put it. ANHIR training pairs answer exactly that: LoFTR never sees the landmarks, so
the whole expert set is held out from every arm.

    A  shipped          scales (0.75, 0.5), absolute tol only      <- what ships today
    B  filter off       scales (0.75,)                             <- upper bound on n
    C  stride-aware 0.5 tol = max(tol_px, 0.5 * coarse_stride)
    D  stride-aware 1.0 tol = max(tol_px, 1.0 * coarse_stride)

Fit a similarity from each arm's correspondences, apply it to the expert SOURCE landmarks,
measure the distance to the expert TARGET landmarks. Lower is better; the paired per-pair
delta against arm A is the result.

WHAT WOULD JUSTIFY A CHANGE. Relaxing a filter is the OVER-CERTIFYING direction, so the bar is
not "no worse". It is: materially more correspondences AND expert-landmark TRE that does not
degrade. If TRE degrades, the filter is doing real work at its current tightness and the
NO_MATCHES regions are honest refusals. Either answer is useful; only one of them is a change.

Reported in PIXELS at the working scale, following the sibling harness: a paired A/B on
identical images needs no µm conversion, and asserting an unverified per-tissue µm/px would be
exactly the unchecked bookkeeping this kind of test exists to avoid.

Run:  .venv/bin/python validation/validate_scale_filter_anhir.py [--limit N]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr          # noqa: E402
from oasis.spatial import loftr_matcher as lm                # noqa: E402

ROOT = os.path.expanduser("~/oasis_validation_datasets/ANHIR_medium/images")
WORK_MAX = 1024          # long side LoFTR runs at; every arm identical
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scale_filter_anhir_results.json")

# (tag, label, kwargs) — A is the control every delta is measured against.
ARMS = [
    ("A", "shipped (absolute tol)",  dict()),
    ("B", "scale filter OFF",        dict(scales=(0.75,))),
    ("C", "stride-aware 0.5x",       dict(scale_tol_stride=0.5)),
    ("D", "stride-aware 1.0x",       dict(scale_tol_stride=1.0)),
]


def load_landmarks(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    hdr, out = rows[0], []
    xi = hdr.index("X") if "X" in hdr else 1
    yi = hdr.index("Y") if "Y" in hdr else 2
    for r in rows[1:]:
        try:
            out.append((float(r[xi]), float(r[yi])))
        except (ValueError, IndexError):
            continue
    return np.asarray(out, float)


def load_image(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = min(1.0, WORK_MAX / float(max(W, H)))
    if s < 1.0:
        im = im.resize((max(int(W * s), 8), max(int(H * s), 8)), Image.LANCZOS)
    return np.asarray(im), s


def tre(M, src_lm, dst_lm):
    """Realized error at the expert landmarks under transform M (source -> target)."""
    return np.linalg.norm(sr._apply_affine(src_lm, M) - dst_lm, axis=1)


def run(limit, seed_shuffle=0):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "dataset_medium.csv")))
            if r["status"] == "training"]
    rng = np.random.default_rng(seed_shuffle)
    rng.shuffle(rows)                     # avoid testing only COAD_*, which sorts first
    out = []
    hdr = f"{'pair':<40}" + "".join(f"{t + '·n':>7}{t + '·med':>8}" for t, _, _ in ARMS)
    print(hdr)
    for r in rows[:limit]:
        sp = os.path.join(ROOT, r["Source image"])
        tp = os.path.join(ROOT, r["Target image"])
        slp = os.path.join(ROOT, r["Source landmarks"])
        tlp = os.path.join(ROOT, r["Target landmarks"])
        if not all(os.path.exists(p) for p in (sp, tp, slp, tlp)):
            continue
        try:
            mov_rgb, s_mov = load_image(sp)
            ref_rgb, s_ref = load_image(tp)
            src_lm = load_landmarks(slp) * s_mov          # into working pixels
            dst_lm = load_landmarks(tlp) * s_ref
        except Exception as e:
            print(f"{r['Source image'][:38]:<40}  LOAD ERROR {e}")
            continue
        n_lm = min(len(src_lm), len(dst_lm))
        if n_lm < 8:
            continue
        src_lm, dst_lm = src_lm[:n_lm], dst_lm[:n_lm]

        res, stride = {}, None
        for tag, _label, kw in ARMS:
            c = lm.loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um=1.0, **kw)
            stride = c.get("coarse_stride_px", stride)
            if not c["ok"]:
                res[tag] = None
                continue
            rp = np.asarray(c["ref_points"], float)      # ref (target) pixels
            mp = np.asarray(c["mov_points"], float)      # mov (source) pixels
            M = sr._fit_similarity_robust(mp, rp)        # source -> target
            e = tre(M, src_lm, dst_lm)
            res[tag] = {"n": int(c["n"]), "n_after_scale": c.get("n_after_scale"),
                        "tol_scale_px": c.get("tol_scale_px"),
                        "med": float(np.median(e)), "p90": float(np.percentile(e, 90))}
        lm.clear_loftr_caches()
        if res.get("A") is None:
            print(f"{r['Source image'].split('/')[0][:38]:<40}  control arm found nothing "
                  f"— skipped (it cannot anchor a paired delta)")
            continue
        name = f"{r['Source image'].split('/')[0]} {os.path.basename(sp)}→{os.path.basename(tp)}"
        cells = "".join(
            (f"{res[t]['n']:>7}{res[t]['med']:>8.1f}" if res.get(t) else f"{'--':>7}{'--':>8}")
            for t, _, _ in ARMS)
        print(f"{name[:38]:<40}{cells}")
        out.append({"pair": name, "n_landmarks": int(n_lm),
                    "coarse_stride_px": stride,
                    **{t: res.get(t) for t, _, _ in ARMS}})
    return out


def summarise(out):
    if not out:
        print("\nno pairs completed")
        return {}
    print(f"\n=== {len(out)} ANHIR training pairs, expert landmarks held out from every arm ===")
    stride = next((o["coarse_stride_px"] for o in out if o.get("coarse_stride_px")), None)
    print(f"coarse grid stride: {stride} px   shipped tolerance: "
          f"{next((o['A']['tol_scale_px'] for o in out if o.get('A')), None)} px"
          f"   -> the filter asks the coarse pass to agree to "
          f"{(next((o['A']['tol_scale_px'] for o in out if o.get('A')), 0) or 0) / (stride or 1):.2f}"
          f" of its own cell\n")

    base = [o for o in out if o.get("A")]
    summary = {"n_pairs": len(out), "coarse_stride_px": stride, "arms": {}}
    print(f"{'arm':<24}{'pairs':>6}{'n (med)':>9}{'med TRE':>9}{'p90 TRE':>9}"
          f"{'Δmed vs A':>11}{'better':>8}{'worse':>7}{'p':>9}")
    for tag, label, _kw in ARMS:
        paired = [o for o in base if o.get(tag)]
        if not paired:
            print(f"{label:<24}{0:>6}   (no arm completed)")
            continue
        n_med = float(np.median([o[tag]["n"] for o in paired]))
        med = float(np.median([o[tag]["med"] for o in paired]))
        p90 = float(np.median([o[tag]["p90"] for o in paired]))
        d = np.array([o[tag]["med"] - o["A"]["med"] for o in paired])
        better, worse = int((d < 0).sum()), int((d > 0).sum())
        pval = None
        if tag != "A":
            try:
                from scipy.stats import wilcoxon
                if len(d) >= 6 and np.any(d != 0):
                    pval = float(wilcoxon(d).pvalue)
            except Exception:
                pass
        print(f"{label:<24}{len(paired):>6}{n_med:>9.0f}{med:>9.2f}{p90:>9.2f}"
              f"{np.median(d):>+11.2f}{better:>8}{worse:>7}"
              f"{('' if pval is None else format(pval, '.3g')):>9}")
        summary["arms"][tag] = {
            "label": label, "pairs": len(paired), "n_median": n_med,
            "tre_median_px": round(med, 3), "tre_p90_px": round(p90, 3),
            "delta_median_vs_A": round(float(np.median(d)), 3),
            "better": better, "worse": worse, "wilcoxon_p": pval}

    _verdict(summary)
    return summary


def _verdict(s):
    a = s["arms"].get("A")
    if not a:
        return
    print("\n" + "-" * 78)
    print("READING IT")
    print("-" * 78)
    print("  The bar for changing the shipped filter is NOT 'no worse'. Relaxing a filter is")
    print("  the over-certifying direction, so it needs materially more correspondences AND")
    print("  expert-landmark TRE that does not degrade.\n")
    for tag in ("B", "C", "D"):
        v = s["arms"].get(tag)
        if not v:
            continue
        gain = v["n_median"] / max(a["n_median"], 1e-9)
        d = v["delta_median_vs_A"]
        p = v["wilcoxon_p"]
        degraded = d > 0 and p is not None and p < 0.05
        improved = d < 0 and p is not None and p < 0.05
        call = ("DEGRADES expert TRE — the filter is doing real work at its current tightness"
                if degraded else
                "IMPROVES expert TRE" if improved else
                "no significant change in expert TRE")
        print(f"  {v['label']:<22} {gain:5.2f}x correspondences,  Δmed {d:+.2f} px  -> {call}")
    print("\n  A change is justified only for an arm with a large n gain and no degradation.")
    print("  If every relaxed arm degrades TRE, the NO_MATCHES regions are honest refusals")
    print("  and the answer is a different region size, not a looser filter.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    rows = run(args.limit)
    s = summarise(rows)
    with open(args.out, "w") as f:
        json.dump({"pairs": rows, "summary": s}, f, indent=2)
    print(f"\nWrote {args.out}")
