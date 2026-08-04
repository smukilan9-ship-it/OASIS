"""
distortion.py — the mechanism, measured directly and with no statistics in the way.

A cross-type spatial statistic consumes INTER-POINT DISTANCES. A rigid/similarity map
preserves every one of them (up to a global scale that the statistic knows about). A
non-rigid map does not. So before asking what the test concludes, ask the prior
question: how much does micro-registration change the distance between two cells that
were d µm apart?

For each registered pair we take real tissue points, carry them through both
transforms, and compare the two distance matrices at the radii the statistic actually
integrates over. The headline is the distortion at CONTACT scale (10-20 µm), because
that is the band the cell-scale claim is made in.

Reported per pair and pooled:
  d_rigid            the distance under the distance-preserving map
  d_nonrigid         the same point pair under micro-registration
  |Δd| / d           relative distortion
  sign of Δd         whether the warp systematically CONTRACTS (brings cells together,
                     which reads as attraction) or expands

Run:  .venv/bin/python -m validation.nonrigid_bench.distortion
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.nonrigid_bench import arms as A          # noqa: E402
from validation.nonrigid_bench import warpfield as wf    # noqa: E402

BANDS_UM = [(0, 10), (10, 20), (20, 50), (50, 100)]
N_PTS = 4000
MAX_PX_UM = 2.5


def main():
    pairs = [p for p in wf.load_all() if p.px_um and p.px_um <= MAX_PX_UM]
    print(f"{len(pairs)} pairs\n")
    pooled = {b: [] for b in BANDS_UM}
    pooled_signed = {b: [] for b in BANDS_UM}
    rows = []
    for pw in pairs:
        img = A._find_image(pw, "moving")
        if not img:
            continue
        dens = A.Density(img)
        if not dens.ok:
            continue
        # sample DENSELY inside ROI-scale windows, else no point pair is ever within
        # 20 µm of another and the contact band cannot be measured at all
        rng = np.random.default_rng(0)
        half = 0.5 * A.WINDOW_UM / pw.px_um
        P = []
        for _ in range(8):
            c = dens.draw(1, rng)
            if not len(c):
                continue
            cx, cy = c[0]
            P.append(dens.draw(N_PTS // 8, rng,
                               bbox=(cx - half, cy - half, cx + half, cy + half)))
        if not P:
            continue
        P = np.vstack(P)
        Pr, Pn = pw.rigid(P), pw.nonrigid(P)
        ok = np.isfinite(Pr).all(1) & np.isfinite(Pn).all(1)
        Pr, Pn = Pr[ok], Pn[ok]
        if len(Pr) < 500:
            continue
        from scipy.spatial import cKDTree
        prs = np.array(sorted(cKDTree(Pr).query_pairs(100.0 / pw.px_um)), dtype=int)
        if len(prs) < 500:
            continue
        if len(prs) > 400_000:
            prs = prs[rng.permutation(len(prs))[:400_000]]
        i, j = prs[:, 0], prs[:, 1]
        dr = np.linalg.norm(Pr[i] - Pr[j], axis=1) * pw.px_um
        dn = np.linalg.norm(Pn[i] - Pn[j], axis=1) * pw.px_um
        rec = {"pair_id": pw.pair_id, "set": pw.set, "px_um": pw.px_um}
        for b in BANDS_UM:
            sel = (dr >= b[0]) & (dr < b[1])
            if sel.sum() < 200:
                continue
            delta = dn[sel] - dr[sel]
            rel = np.abs(delta) / np.maximum(dr[sel], 1e-9)
            pooled[b].append(rel)
            pooled_signed[b].append(delta / np.maximum(dr[sel], 1e-9))
            rec[f"{b[0]}-{b[1]}"] = {
                "n": int(sel.sum()),
                "median_abs_rel": float(np.median(rel)),
                "median_abs_um": float(np.median(np.abs(delta))),
                "frac_contracted": float((delta < 0).mean()),
            }
        rows.append(rec)
        c = rec.get("10-20")
        if c:
            print(f"{pw.pair_id[:44]:44s} contact-band |Δd| median "
                  f"{c['median_abs_um']:6.2f} µm = {100*c['median_abs_rel']:6.1f}% "
                  f"of the distance   contracted {100*c['frac_contracted']:.0f}%")

    print("\nPOOLED across pairs")
    print(f"  {'band (µm)':>12s} {'median |Δd|/d':>15s} {'p90 |Δd|/d':>12s} {'contracted':>11s}")
    summary = {}
    for b in BANDS_UM:
        if not pooled[b]:
            continue
        v = np.concatenate(pooled[b])
        s = np.concatenate(pooled_signed[b])
        summary[f"{b[0]}-{b[1]}"] = {
            "median_rel": float(np.median(v)), "p90_rel": float(np.percentile(v, 90)),
            "frac_contracted": float((s < 0).mean()), "n": int(v.size)}
        print(f"  {b[0]:5d}-{b[1]:<6d} {100*np.median(v):14.1f}% "
              f"{100*np.percentile(v,90):11.1f}% {100*(s<0).mean():10.1f}%")
    out = os.path.join(_HERE, "results", "distortion.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"per_pair": rows, "pooled": summary}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
