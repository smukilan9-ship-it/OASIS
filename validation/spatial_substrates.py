#!/usr/bin/env python
"""
Point-pattern substrates for the spatial-statistics validations — real tissue, not blobs.

WHY THIS EXISTS. Every spatial validation in this repo has been run on a Gaussian-mixture
"tissue": a dozen isotropic compartments plus a diffuse background. Real tissue is not that.
It has sharp compartment boundaries, vessels, tumour nests inside stroma, and structure at
several scales at once. That matters more than it looks, because the PRIMARY null
(cross_k_inhom_reweighted_test) ESTIMATES the intensity surface from the data it is given.
A smooth unimodal-ish substrate is the easiest possible case for a kernel intensity
estimate, so a null calibrated only there is calibrated on the easy case.

So the same experiments run on both, and a conclusion that holds on one substrate and not
the other is not a conclusion.

REAL SUBSTRATES available on this machine:
  ll477_cd8    4,796 cells over 1442 x 1082 um   OASIS's own InstanSeg output on the
  ll477_tim3   5,882 cells over 1163 x  872 um   actual cohort — the tissue it ships for
  keren_p13    7,665 cells over 1008 x 1008 um   real TNBC multiplex (Keren et al.), an
  keren_p16    8,212 cells                       INDEPENDENT tissue type and modality, so
  keren_p32    5,158 cells                       agreement across the two means something

HOW AN ASSOCIATION IS IMPOSED — by THINNING, never by inventing coordinates.
    A is a random sample of real cells. B is then sampled from the REMAINING real cells with
    a weight that is raised only for cells whose distance to the nearest A point falls in a
    target annulus. Every coordinate in both patterns is therefore a real segmented cell,
    carrying real tissue architecture; the only thing constructed is WHICH real cells were
    selected. Generating recruited points from a Gaussian around A would have put synthetic
    coordinates into a "real tissue" test and quietly given back the blob substrate.

    An annulus, not a Gaussian, because the truth has to be CONFINED to one band for band
    attribution to be measurable at all — a Gaussian at 8 um leaks mass into 20-50 um by
    construction and makes any leakage result uninterpretable.

CO-ORDINATES. Everything here is in MICRONS, and callers pass pixel_size_um=1.0 so the
statistic's "pixel" is one micron. That keeps the real substrates free of any rescaling.
"""
from __future__ import annotations

import csv
import os

import numpy as np

# Detection tables from real runs. Neither cohort is in the repository — LL477 is
# unpublished lab data and the Keren TNBC renders are derived from a set with its own terms
# — so the two roots are named by environment variable. A substrate whose root is unset is
# skipped by the caller rather than faked.
_SPATIAL_ROOT = os.path.expanduser(
    os.environ.get("OASIS_SPATIAL_RESULTS", "~/ihc_spatial_results"))
_KEREN_ROOT = os.path.expanduser(
    os.environ.get("OASIS_KEREN_RENDERS", "~/OASIS_keren_tnbc_validation/rendered_ui_inputs"))

REAL_SOURCES = {
    "ll477_cd8": os.path.join(_SPATIAL_ROOT, "LL477_CD8_x10_3__roi0",
                              "LL477_CD8_x10_3.tif - LL477_CD8_x10_3.tif #1_detections.csv"),
    "ll477_tim3": os.path.join(_SPATIAL_ROOT, "LL477_CD8_x10_3__roi0",
                               "LL477_Tim3_10X_3.tif - LL477_Tim3_10X_3.tif #1_detections.csv"),
    "keren_p13": os.path.join(_KEREN_ROOT, "external_scaffold",
                              "p13_external_keren_scaffold_detections.csv"),
    "keren_p16": os.path.join(_KEREN_ROOT, "external_scaffold",
                              "p16_external_keren_scaffold_detections.csv"),
    "keren_p32": os.path.join(_KEREN_ROOT, "external_scaffold",
                              "p32_external_keren_scaffold_detections.csv"),
}

# Synthetic substrate — kept so every result can be reported on BOTH, which is the only way
# to tell "this is a property of the statistic" from "this is a property of my blobs".
SYN_N_COMPARTMENTS = 12
SYN_COMPARTMENT_SD_UM = 180.0
SYN_DIFFUSE_FRAC = 0.35
SYN_SIDE_UM = 1000.0


def available(require_real=False):
    """Substrate names usable on this machine. Real ones need their CSV present."""
    names = [k for k, p in REAL_SOURCES.items() if os.path.exists(p)]
    return names if require_real else names + ["synthetic"]


def _read_centroids(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    xy, cls = [], []
    for r in rows:
        try:
            xy.append((float(r["Centroid X µm"]), float(r["Centroid Y µm"])))
        except (KeyError, TypeError, ValueError):
            continue
        cls.append((r.get("Classification") or r.get("Class") or "").strip())
    return np.asarray(xy, float), cls


def load_substrate(name, seed=0):
    """(points_um (N,2), window (w_um, h_um)). `synthetic` regenerates from `seed`."""
    if name == "synthetic":
        rng = np.random.default_rng(seed)
        centres = rng.uniform(0, SYN_SIDE_UM, (SYN_N_COMPARTMENTS, 2))
        n = 6000
        n_diff = int(SYN_DIFFUSE_FRAC * n)
        pts = np.vstack([
            rng.uniform(0, SYN_SIDE_UM, (n_diff, 2)),
            centres[rng.integers(0, SYN_N_COMPARTMENTS, n - n_diff)]
            + rng.normal(0, SYN_COMPARTMENT_SD_UM, (n - n_diff, 2))])
        pts = pts[(pts > 0).all(1) & (pts < SYN_SIDE_UM).all(1)]
        return pts, (SYN_SIDE_UM, SYN_SIDE_UM)

    path = REAL_SOURCES.get(name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"substrate {name!r} not available at {path!r}")
    xy, _ = _read_centroids(path)
    # Shift to a (0,0) origin so the window is a plain bounding box, as
    # spatial_stats.bounding_box_area assumes.
    xy = xy - xy.min(axis=0)
    return xy, (float(xy[:, 0].max()), float(xy[:, 1].max()))


def solve_boost(p, target_enrichment):
    """Weight boost that makes a boosted annulus `target_enrichment` times its baseline share.

    With weight (1+b) on a fraction p of candidates, the selected share of that fraction is
    (1+b)/(1 + b*p) times baseline, so

        b = (E - 1) / (1 - E*p)

    The denominator states the hard limit: enrichment cannot exceed 1/p, because a set that
    already holds a fraction p of the candidates cannot be over-represented beyond filling
    the sample. A FLAT boost across two annuli therefore does NOT give them equal
    enrichment — the larger annulus soaks up the sample and both come out diluted, which is
    how the "both scales" truth silently became an unfair test (contact enrichment fell
    3.9x -> 1.8x). Solving per draw from the observed p removes that.
    """
    p = float(np.clip(p, 1e-6, 0.999999))
    E = float(target_enrichment)
    if E * p >= 1.0:
        return None                      # unachievable; caller must widen or lower E
    return (E - 1.0) / (1.0 - E * p)


def impose_band_association(pts, annulus_um, n_a=300, n_b=500, boost=6.0,
                            target_enrichment=None, rng=None):
    """Split real cells into A and B with a known excess CONFINED to `annulus_um`.

    Returns (A, B). `boost` multiplies the selection weight of candidate B cells whose
    nearest-A distance lies inside the annulus; boost=0 gives an independent pair, which is
    what the size check needs. Both patterns are subsets of `pts`, so both carry the real
    architecture and neither contains an invented coordinate.

    `annulus_um` may be one (lo, hi) tuple or a list of them — a list gives a truth that
    genuinely lives at BOTH scales, which is the only way to check that a decomposition can
    report two findings when there really are two.
    """
    from scipy.spatial import cKDTree
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    n_a = min(n_a, n // 3)
    idx = rng.permutation(n)
    a_idx = idx[:n_a]
    rest = idx[n_a:]
    A = pts[a_idx]

    d = cKDTree(A).query(pts[rest])[0]
    annuli = [annulus_um] if isinstance(annulus_um[0], (int, float)) else list(annulus_um)
    w = np.ones(len(rest), float)
    if target_enrichment is not None:
        # Solve the boost from the OBSERVED occupancy of all boosted annuli together, so
        # every truth carries the same per-band enrichment and the attribution matrix
        # compares like with like.
        inside = np.zeros(len(rest), bool)
        for lo, hi in annuli:
            inside |= (d >= lo) & (d < hi)
        b = solve_boost(inside.mean(), target_enrichment)
        if b is None:
            raise ValueError(
                f"enrichment {target_enrichment} unachievable: the boosted annuli already "
                f"hold {inside.mean():.3f} of candidates (max {1/max(inside.mean(),1e-9):.1f}x)")
        w[inside] += b
    elif boost > 0:
        for lo, hi in annuli:
            w[(d >= lo) & (d < hi)] += boost
    w /= w.sum()

    n_b = min(n_b, len(rest))
    b_sel = rng.choice(len(rest), size=n_b, replace=False, p=w)
    return A, pts[rest[b_sel]]


def describe(name, seed=0):
    pts, (w, h) = load_substrate(name, seed)
    return {"substrate": name, "n_cells": int(len(pts)),
            "window_um": [round(w, 1), round(h, 1)],
            "density_per_mm2": round(len(pts) / max(w * h, 1e-9) * 1e6, 1)}


if __name__ == "__main__":
    for nm in available():
        try:
            print(describe(nm))
        except Exception as e:                                   # noqa: BLE001
            print(nm, "->", e)
