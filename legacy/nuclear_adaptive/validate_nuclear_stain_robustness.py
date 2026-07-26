"""
validate_nuclear_stain_robustness.py — does the adaptive threshold hold as staining drifts?

THE CLAIM to test. On clean DeepLIIF Ki67 the GMM-adaptive threshold and a fixed 0.20 OD
cutoff tie on F1. The case for making adaptive the DEFAULT is that it SELF-CORRECTS for
per-image / batch staining variation while a fixed absolute cutoff drifts. This harness
perturbs each image's DAB signal — the way real stain-batch variation does — and measures
F1 for both, on the SAME IF-derived ground-truth labels (the truth does not move when the
chromogen intensity does).

Perturbation model on the per-cell DAB signal x:   x' = gain·x + offset
  • gain   (multiplicative)  darker/lighter DAB staining — the dominant real-world variation
  • offset (additive)        counterstain / background bleed into the DAB channel

A pure gain is *exactly* what per-image adaptive thresholding is scale-equivariant to, so
the GMM is expected to be near-flat there (that is the point); offset and random batch
variation are the harder, honest tests.

Usage:  python validation/validate_nuclear_stain_robustness.py [n_images]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from validation.datasets.resolve import dataset_dir
from validation.validate_nuclear_classifier import gather, _confusion, _prfk
from oasis.quant.nuclear_classify import classify_nuclear

FIXED = 0.20
ASHMAN = 1.25


def _score(per_image, gain, offset, jitter_gain=0.0, jitter_off=0.0, seed=0):
    """Score fixed vs gmm after perturbing each image's DAB. jitter_* > 0 draws a random
    PER-IMAGE gain/offset (realistic batch variation) on top of the systematic shift."""
    rng = np.random.default_rng(seed)
    Ff = dict(tp=0, fp=0, fn=0, tn=0)
    Fg = dict(tp=0, fp=0, fn=0, tn=0)
    n_abst = 0
    for im in per_image:
        g = gain * (np.exp(rng.normal(0, jitter_gain)) if jitter_gain else 1.0)
        c = offset + (rng.normal(0, jitter_off) if jitter_off else 0.0)
        allp = g * im["all_dab"] + c
        mp = g * im["m_dab"] + c
        gt = im["m_gt"] == 1
        # fixed absolute cutoff (does NOT know staining drifted)
        for k, v in zip(("tp", "fp", "fn", "tn"), _confusion(mp > FIXED, gt)):
            Ff[k] += v
        # adaptive GMM (re-estimates the threshold from the drifted distribution)
        vals = allp[np.isfinite(allp)]
        dec = classify_nuclear(vals, fixed_threshold=FIXED, ashman_min=ASHMAN,
                               allow_fixed_fallback=True)
        thr = dec["threshold"] if dec["threshold"] is not None else FIXED
        if dec["abstain"]:
            n_abst += 1
        for k, v in zip(("tp", "fp", "fn", "tn"), _confusion(mp > thr, gt)):
            Fg[k] += v
    rf, rg = _prfk(**Ff), _prfk(**Fg)
    return rf["f1"], rg["f1"], n_abst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200)
    a = ap.parse_args()
    root = os.path.join(str(dataset_dir("deepliif")), "_generated_outputs", "pipeline_validation")
    gt_all = json.load(open(os.path.join(root, "ground_truth.json")))
    per_image = gather(root, "raw_instanseg", gt_all, a.n, want_macenko=False)
    N = len(per_image)
    print(f"Nuclear stain-robustness — DeepLIIF Ki67, {N} images "
          f"(fixed@{FIXED}, gmm ashman={ASHMAN})\n")

    print("A) SYSTEMATIC GAIN (offset 0) — multiplicative stain intensity drift")
    print(f"   {'gain':>5} {'F1 fixed':>9} {'F1 gmm':>8} {'gmm abstain':>12}")
    for g in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
        ff, fg, na = _score(per_image, g, 0.0)
        print(f"   {g:>5.2f} {ff:>9.3f} {fg:>8.3f} {na:>10}/{N}")

    print("\nB) SYSTEMATIC OFFSET (gain 1) — additive background/counterstain bleed")
    print(f"   {'offset':>6} {'F1 fixed':>9} {'F1 gmm':>8} {'gmm abstain':>12}")
    for c in (-0.05, 0.0, 0.05, 0.10, 0.15):
        ff, fg, na = _score(per_image, 1.0, c)
        print(f"   {c:>6.2f} {ff:>9.3f} {fg:>8.3f} {na:>10}/{N}")

    print("\nC) RANDOM PER-IMAGE BATCH VARIATION (mean gain 1, offset 0; rising spread)")
    print(f"   {'jitter σ':>8} {'F1 fixed':>9} {'F1 gmm':>8} {'gmm abstain':>12}")
    for s in (0.0, 0.1, 0.2, 0.35, 0.5):
        ff, fg, na = _score(per_image, 1.0, 0.0, jitter_gain=s, jitter_off=s * 0.15, seed=1)
        print(f"   {s:>8.2f} {ff:>9.3f} {fg:>8.3f} {na:>10}/{N}")

    print("\nRead: F1 fixed should fall away from gain/offset 0 while F1 gmm stays flat — "
          "that gap is the case for adaptive-as-default.")


if __name__ == "__main__":
    main()
