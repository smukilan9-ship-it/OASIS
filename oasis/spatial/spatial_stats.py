"""
spatial_stats.py
Cross-type spatial association statistics for registered serial sections.

Implements the population-level statistic that replaces mutual-nearest-neighbour
cell matching in the Spatial Association pipeline:

  • cross-type Ripley's K function  K_ab(r)
  • cross-type pair correlation function  g_ab(r)
  • the association curve  L_ab(r) - r
  • a Monte-Carlo null model (complete spatial randomness for population B,
    randomized *within the tissue mask*)
  • a tissue-mask estimator that bounds both the area normalization and the
    null randomization

Unlike MNN matching, none of these statistics make a single-cell pairing claim.
They measure whether two cell *populations* are spatially associated across a
registered pair of sections, relative to spatial randomness — the established
approach for cross-type spatial point-pattern analysis. Single-cell
co-expression cannot be inferred from serial sections; it requires multiplex
imaging on the same physical section.

All point inputs are in CD8 (reference) image-space pixels. Radii are passed in
pixels; outputs are converted to microns using pixel_size_um.
"""

import numpy as np


# Biologically relevant scale for the global significance call. Cross-type
# T-cell association is expected at short range (a few cell diameters up to a
# small microregion); beyond this the curve mostly reflects tissue architecture.
_RELEVANT_SCALE_UM = 50.0

# Radius band for the global DCLF test (microns). Below ~one cell diameter
# (~10 µm) hard-core exclusion dominates — two centroids cannot coincide, so the
# L−r curve is forced negative there for reasons that have nothing to do with
# biological association. Above ~50 µm the curve mostly reflects tissue
# architecture rather than cell–cell association. Restricting the test to this
# band keeps it focused on the biologically meaningful range.
_DCLF_RMIN_UM = 10.0
_DCLF_RMAX_UM = 50.0

# Default KDE bandwidth (µm) for the inhomogeneous null's intensity estimate.
# Chosen = the top of the DCLF test band (50 µm) on purpose: it PRESERVES first-
# order intensity variation at scales ≥ the interaction scale we are testing
# (tissue-compartment structure), while NOT preserving structure finer than the
# band — so genuine cell–cell interaction inside the 10–50 µm band can still be
# detected. A data-adaptive rule (Scott's/Silverman's) was rejected as the default
# because it is driven by the global spread of the points and over-smooths a
# MULTIMODAL intensity (many small tissue compartments) toward uniform, collapsing
# the inhomogeneous null back toward homogeneous CSR. The 0.5×/1×/2× sensitivity
# sweep reports robustness to this choice.
_KDE_BANDWIDTH_UM = 50.0

# Default bandwidth (µm) for the intensity surfaces of the PRIMARY production test
# (cross_k_inhom_reweighted_test). It must sit ABOVE the 10–50 µm interaction band
# so the estimated λ captures the tissue ARCHITECTURE (the shared compartments)
# without absorbing the within-band cell–cell interaction being tested. The exact
# value is chosen by the 3-regime calibration (validate_primary_null_calibration.py
# / validate_reweighted_null.py), not assumed — see ihc.md §15.
_REWEIGHT_BANDWIDTH_UM = 75.0

# Dense-tissue fallback null. This is deliberately NOT a smaller KDE bandwidth
# on the marker positives: validation rejected that as anti-conservative. The
# accepted candidate conditions B* on marker-independent total-cell morphology
# support (all detected nuclei in the reference section), with a very small
# jitter so the null preserves dense tissue packing while randomizing marker
# identity/location relative to A.
_DENSE_MORPHOLOGY_JITTER_UM = 2.0
_DENSE_DCLF_RMIN_UM = 10.0
_DENSE_DCLF_RMAX_UM = 30.0

# ──────────────────────────────────────────────────────────────────────────────
# Registration error → smallest interpretable radius
# ──────────────────────────────────────────────────────────────────────────────
# Residual registration error ε displaces each B point by ~ε in a direction
# uncorrelated with the biology, blurring the observed cross-K toward the null.
# validation/validate_radius_floor.py measures what that does to the pipeline's OWN
# DCLF test, and the result is not what the historical ≤5 µm gate assumes:
#
#   1. SIZE IS PRESERVED. On independent A/B, the false-positive rate stays at the
#      nominal ~5% for every ε up to 20 µm, using the ordinary 10 µm band. Registration
#      error does NOT invalidate the test. (Points displaced out of the analysis window
#      must be dropped, which run_spatial_association already does; without that the
#      density bookkeeping breaks and size inflates — an artefact, not a real effect.)
#   2. POWER DEGRADES GRACEFULLY. On a weak true association, detection falls ~0.44 →
#      0.34 as ε goes 0 → 20 µm. Error costs sensitivity, never validity.
#   3. RAISING THE BAND FLOOR DOES NOT HELP. Power at ε = 12 µm is 0.42 from a 10 µm
#      band and 0.38 from a 3ε band. Clipping the DCLF band discards radii that still
#      carry signal and buys nothing, so THE BAND IS NOT CLIPPED.
#
# Both (1) and (2) hold because the transform is LANDMARK-driven and therefore blind to
# the stained cells: the error field is uncorrelated with where the cells are. An
# INTENSITY-driven non-rigid warp would not have this property — it optimises on a
# signal correlated with cell density and could pull A-rich tissue onto B-rich tissue,
# manufacturing the association under test. Do not use one here.
#
# What the floor below IS for: an INTERPRETATION boundary, not a gate. A pair aligned to
# within ε cannot resolve inter-cell distances of order ε, so reporting "no enrichment at
# 10 µm" for a pair with ε = 12 µm would be misleading — that is an unmeasurable radius,
# not a biological absence. The floor marks where the curve stops being readable. The
# historical ≤5 µm certification gate is this same rule frozen at the default 10 µm band
# start (10 ≈ 2 × 5), mis-encoded as permission to run at all.
_RADIUS_FLOOR_FACTOR = 3.0


def registration_radius_floor(tre_um, factor: float = _RADIUS_FLOOR_FACTOR):
    """Smallest inter-cell distance (µm) that a pair registered to within `tre_um` can
    resolve. Radii below it are reported as not interpretable rather than as null.

    This is a reporting boundary, NOT a validity gate: the DCLF test remains correctly
    sized below it (see module notes). Returns None when TRE is unknown, so callers must
    fail closed rather than assume any radius is readable.
    """
    if tre_um is None:
        return None
    tre = float(tre_um)
    if not np.isfinite(tre) or tre < 0:
        return None
    return round(float(factor) * tre, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-type Ripley's K / g(r) / L(r)
# ──────────────────────────────────────────────────────────────────────────────

def _pair_counts(points_a: np.ndarray, points_b: np.ndarray,
                 radii_px: np.ndarray) -> np.ndarray:
    """
    Cumulative number of cross-type pairs (a_i, b_j) with distance <= r, for
    every r in radii_px. Uses cKDTree.count_neighbors which evaluates all radii
    in a single pass.
    """
    from scipy.spatial import cKDTree
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    # count_neighbors returns cumulative pair counts at each radius (float array)
    return np.asarray(
        tree_a.count_neighbors(tree_b, radii_px), dtype=np.float64)


def _k_from_counts(counts: np.ndarray, area_px: float,
                   n_a: int, n_b: int) -> np.ndarray:
    """K_ab(r) = (area / (N*M)) * (number of pairs within r), in px^2."""
    if n_a == 0 or n_b == 0:
        return np.zeros_like(counts)
    return (area_px / (n_a * n_b)) * counts


def cross_k_function(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii: np.ndarray,
    area: float,
    pixel_size_um: float,
) -> dict:
    """
    Cross-type Ripley's K function and its derived curves.

        K_ab(r) = (area / (N*M)) * Σ_{i,j} 1[dist(a_i, b_j) <= r]

    Also returns the cross-type pair correlation function

        g_ab(r) = (1 / (2*pi*r)) * dK/dr          (dimensionless; CSR → g = 1)

    and the association curve

        L_ab(r) - r,    L_ab(r) = sqrt(K_ab(r) / pi)

    Under spatial independence L_ab(r) - r = 0; positive ⇒ association
    (attraction / co-clustering), negative ⇒ segregation.

    Args:
        points_a:       (N,2) reference (CD8+) centroids, CD8 image-space pixels
        points_b:       (M,2) registered (TIM-3+) centroids, CD8 image-space pixels
        radii:          (R,) distances at which to evaluate K, in pixels
        area:           tissue area in pixels^2
        pixel_size_um:  µm per pixel of the reference image

    Returns:
        {
          "radii_um":   [...],   # r in microns
          "K_observed": [...],   # K_ab(r) in µm^2
          "g_observed": [...],   # g_ab(r), dimensionless
          "L_minus_r":  [...],   # (L_ab(r) - r) in microns
        }
    """
    radii_px = np.asarray(radii, dtype=np.float64)
    s   = float(pixel_size_um)
    n_a = int(len(points_a))
    n_b = int(len(points_b))

    if n_a == 0 or n_b == 0:
        zeros = np.zeros_like(radii_px)
        return {
            "radii_um":   (radii_px * s).tolist(),
            "K_observed": zeros.tolist(),
            "g_observed": zeros.tolist(),
            "L_minus_r":  zeros.tolist(),
        }

    counts = _pair_counts(points_a, points_b, radii_px)
    k_px   = _k_from_counts(counts, float(area), n_a, n_b)        # px^2

    # L - r in pixels, then converted to microns
    l_px   = np.sqrt(np.clip(k_px, 0.0, None) / np.pi)            # px
    lmr_px = l_px - radii_px

    g = _pcf_from_k(k_px, radii_px)
    # Emit None (not NaN) for undefined entries → valid JSON for the UI
    g_out = [None if not np.isfinite(v) else float(v) for v in g]

    return {
        "radii_um":   (radii_px * s).tolist(),
        "K_observed": (k_px * s * s).tolist(),                    # µm^2
        "g_observed": g_out,
        "L_minus_r":  (lmr_px * s).tolist(),                      # µm
    }


def _pcf_from_k(k_px: np.ndarray, radii_px: np.ndarray) -> np.ndarray:
    """
    Pair correlation function g(r) = (1/(2*pi*r)) dK/dr, estimated by finite
    differences on K(r). Dimensionless and scale-invariant, so it does not
    matter whether K and r are in pixels or microns as long as they are
    consistent. g(0) is undefined (1/r) → reported as NaN.
    """
    g = np.full_like(k_px, np.nan)
    if len(radii_px) < 2:
        return g
    dk_dr = np.gradient(k_px, radii_px)
    nz    = radii_px > 0
    g[nz] = dk_dr[nz] / (2.0 * np.pi * radii_px[nz])
    return g


# ──────────────────────────────────────────────────────────────────────────────
# Monte-Carlo null model
# ──────────────────────────────────────────────────────────────────────────────

def _sample_in_polygon(polygon, n: int, rng, bbox) -> np.ndarray:
    """
    Draw n uniform random points inside a shapely polygon by bounding-box
    rejection sampling (vectorized via shapely.contains_xy).
    """
    import shapely
    minx, miny, maxx, maxy = bbox
    out = np.empty((0, 2), dtype=np.float64)
    # Cap the number of batches so a pathological polygon can't loop forever;
    # any shortfall is topped up from the bbox (negligible for sane masks).
    for _ in range(64):
        if len(out) >= n:
            break
        need = (n - len(out)) * 2 + 16
        xs = rng.uniform(minx, maxx, need)
        ys = rng.uniform(miny, maxy, need)
        inside = shapely.contains_xy(polygon, xs, ys)
        if np.any(inside):
            out = np.vstack([out, np.column_stack([xs[inside], ys[inside]])])
    if len(out) < n:
        xs = rng.uniform(minx, maxx, n - len(out))
        ys = rng.uniform(miny, maxy, n - len(out))
        out = np.vstack([out, np.column_stack([xs, ys])])
    return out[:n]


def cross_k_null(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii: np.ndarray,
    area: float,
    pixel_size_um: float,
    n_perm: int = 1000,
    seed: int = 0,
    tissue_polygon=None,
    dclf_rmin_um: float = _DCLF_RMIN_UM,
    dclf_rmax_um: float = _DCLF_RMAX_UM,
) -> dict:
    """
    Monte-Carlo null model for the cross-type K function.

    For each of n_perm iterations: hold population A fixed, reposition population
    B under complete spatial randomness (uniform within the tissue mask if
    `tissue_polygon` is given, otherwise within the bounding box of all points —
    a conservative/biased fallback), and recompute K_ab(r).

    From the null curves we report, at every r: the mean, the 95% envelope
    (2.5 / 97.5 percentiles) in both K and L-r space, and a one-sided per-r
    p-value (fraction of null K >= observed K, +1 corrected). These per-radius
    quantities drive the plot's envelope but are NOT used for the overall yes/no
    call (ORing significance across ~50 radii inflates the family-wise error).

    The single global call is a DCLF (Diggle–Cressie–Loosmore–Ford) rank
    envelope test over the whole L−r curve within [dclf_rmin_um, dclf_rmax_um] —
    one honest p-value with no multiple-comparison inflation (see
    `_global_dclf_summary`).

    The seed is fixed (default 0) for reproducible significance numbers, matching
    the existing null-model convention.

    Returns a dict containing the observed curves, the null envelope, per-r
    p-values, and the global summary (see keys below).
    """
    radii_px = np.asarray(radii, dtype=np.float64)
    s   = float(pixel_size_um)
    n_a = int(len(points_a))
    n_b = int(len(points_b))
    n_r = len(radii_px)

    observed = cross_k_function(points_a, points_b, radii_px, area, pixel_size_um)

    # Degenerate input → return observed curves with empty/neutral null.
    if n_a == 0 or n_b == 0 or n_r == 0:
        zeros = np.zeros(n_r)
        return {
            **observed,
            "null_mean_K":  zeros.tolist(),
            "null_lower_K": zeros.tolist(),
            "null_upper_K": zeros.tolist(),
            "null_lower_L": zeros.tolist(),
            "null_upper_L": zeros.tolist(),
            "p_values":     np.ones(n_r).tolist(),
            "n_perm":       int(n_perm),
            "tissue_area_um2": float(area) * s * s,
            "global":       _empty_global(dclf_rmin_um, dclf_rmax_um),
        }

    # Observed K in pixels (recomputed cheaply; the null is compared in px space)
    obs_counts = _pair_counts(points_a, points_b, radii_px)
    obs_k_px   = _k_from_counts(obs_counts, float(area), n_a, n_b)

    pts_a = np.asarray(points_a, dtype=np.float64)
    allpts = np.vstack([pts_a, np.asarray(points_b, dtype=np.float64)])
    if tissue_polygon is not None:
        bbox = tissue_polygon.bounds
    else:
        print("  Spatial null: no tissue mask — randomizing within bounding box "
              "(results are conservative/biased without a mask)")
        mn, mx = allpts.min(axis=0), allpts.max(axis=0)
        bbox = (mn[0], mn[1], mx[0], mx[1])

    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)

    rng     = np.random.default_rng(int(seed))
    null_k  = np.empty((n_perm, n_r), dtype=np.float64)   # px^2
    norm    = float(area) / (n_a * n_b)
    for kperm in range(n_perm):
        if tissue_polygon is not None:
            b_perm = _sample_in_polygon(tissue_polygon, n_b, rng, bbox)
        else:
            b_perm = np.column_stack([
                rng.uniform(bbox[0], bbox[2], n_b),
                rng.uniform(bbox[1], bbox[3], n_b),
            ])
        counts = tree_a.count_neighbors(cKDTree(b_perm), radii_px)
        null_k[kperm] = norm * np.asarray(counts, dtype=np.float64)

    # Per-r summaries in pixel space
    null_mean_px  = null_k.mean(axis=0)
    null_lo_px    = np.percentile(null_k, 2.5,  axis=0)
    null_hi_px    = np.percentile(null_k, 97.5, axis=0)
    # One-sided per-r p-value with +1 correction
    ge       = (null_k >= obs_k_px[None, :]).sum(axis=0)
    p_values = (ge + 1.0) / (n_perm + 1.0)

    # L-r envelope: transform each null K curve, then take percentiles in L-r space
    null_l_px   = np.sqrt(np.clip(null_k, 0.0, None) / np.pi) - radii_px[None, :]
    null_l_lo   = np.percentile(null_l_px, 2.5,  axis=0)
    null_l_hi   = np.percentile(null_l_px, 97.5, axis=0)

    obs_lmr_px  = np.asarray(observed["L_minus_r"], dtype=np.float64) / s   # back to px

    global_summary = _global_dclf_summary(
        radii_px, obs_lmr_px, null_l_px, p_values, s,
        dclf_rmin_um, dclf_rmax_um)

    return {
        **observed,
        "null_mean_K":  (null_mean_px * s * s).tolist(),   # µm^2
        "null_lower_K": (null_lo_px   * s * s).tolist(),
        "null_upper_K": (null_hi_px   * s * s).tolist(),
        "null_lower_L": (null_l_lo * s).tolist(),           # µm
        "null_upper_L": (null_l_hi * s).tolist(),
        "p_values":     p_values.tolist(),
        "n_perm":       int(n_perm),
        "tissue_area_um2": float(area) * s * s,
        "global":       global_summary,
    }


def _empty_global(rmin_um=_DCLF_RMIN_UM, rmax_um=_DCLF_RMAX_UM) -> dict:
    return {
        "significant":          False,
        "global_p_dclf":        None,
        "global_p_association": None,
        "global_p_segregation": None,
        "direction":            "none",
        "peak_r_um":            None,
        "peak_L_minus_r":       None,
        "peak_p_value":         None,
        "dclf_rmin_um":         float(rmin_um),
        "dclf_rmax_um":         float(rmax_um),
        "relevant_scale_um":    _RELEVANT_SCALE_UM,
    }


def _global_dclf_summary(radii_px, obs_lmr, null_lmr, p_values, s,
                         rmin_um=_DCLF_RMIN_UM, rmax_um=_DCLF_RMAX_UM) -> dict:
    """
    Global envelope test (Diggle–Cressie–Loosmore–Ford) on the L−r curve.

    The per-radius envelope test asks "is the curve outside the band at *this*
    radius?" at ~50 radii; ORing those decisions inflates the family-wise false
    positive rate far above 0.05. The DCLF test instead reduces the entire curve
    to a single deviation statistic and ranks it against the null, giving ONE
    honest p-value with no multiple-comparison inflation:

        u = Σ_r ( L(r) − Lbar(r) )²        ( ∝ ∫ (L−Lbar)² dr; the constant
                                             radius spacing dr only rescales u
                                             and cancels in the rank, so the
                                             discrete sum is exact )

    u is computed identically for the observed curve and every simulated null
    curve, all relative to the mean of the pooled set (observed + nulls) so the
    observed is exactly exchangeable with the nulls under H0 — which makes the
    two-sided p-value uniform under the null (the property validated in
    validation/validate_dclf.py). The rank gives, with the standard +1
    correction:

        global_p_dclf = (1 + #{ u_null ≥ u_obs }) / (1 + n_perm)

    Directional one-sided variants use only the positive deviations (observed
    ABOVE the mean → association/attraction) or only the negative deviations
    (observed BELOW the mean → segregation/repulsion).

    The test is restricted to the band [rmin_um, rmax_um] (see _DCLF_RMIN_UM /
    _DCLF_RMAX_UM): below one cell diameter hard-core exclusion forces L−r
    negative for non-biological reasons, and above ~50 µm the curve reflects
    tissue architecture rather than cell–cell association.

    `obs_lmr` / `null_lmr` are L−r in pixels (obs: (R,), null: (n_perm, R)).
    Per-radius `p_values` are kept only to report the peak radius's local p.
    """
    radii_um = radii_px * s
    band = (radii_um >= rmin_um) & (radii_um <= rmax_um)
    if not np.any(band):                      # band falls outside evaluated radii
        band = np.ones_like(radii_um, dtype=bool)

    n_perm = int(null_lmr.shape[0])
    obs_b  = np.asarray(obs_lmr, dtype=np.float64)[band]
    null_b = np.asarray(null_lmr, dtype=np.float64)[:, band]

    # Pool observed with nulls → exact exchangeability under H0
    allc   = np.vstack([obs_b[None, :], null_b])          # (n_perm+1, nb)
    mean_c = allc.mean(axis=0)
    dev    = allc - mean_c[None, :]

    u_two = np.sum(dev ** 2, axis=1)                       # two-sided
    u_pos = np.sum(np.clip(dev,  0.0, None) ** 2, axis=1)  # association (above)
    u_neg = np.sum(np.clip(-dev, 0.0, None) ** 2, axis=1)  # segregation (below)

    def _rank_p(u):
        # u[0] is observed; rank among the n_perm nulls, +1 corrected
        return (1.0 + float(np.sum(u[1:] >= u[0]))) / (1.0 + n_perm)

    p_dclf  = _rank_p(u_two)
    p_assoc = _rank_p(u_pos)
    p_seg   = _rank_p(u_neg)

    significant = p_dclf < 0.05
    direction   = "none"
    if significant:
        direction = "association" if p_assoc <= p_seg else "segregation"

    # Peak (most positive in-band L−r) — purely descriptive, for plot annotation
    cand = np.where(band)[0]
    peak = cand[int(np.argmax(np.asarray(obs_lmr)[cand]))]

    return {
        "significant":          bool(significant),
        "global_p_dclf":        round(float(p_dclf), 5),
        "global_p_association": round(float(p_assoc), 5),
        "global_p_segregation": round(float(p_seg), 5),
        "direction":            direction,
        "peak_r_um":            round(float(radii_um[peak]), 2),
        "peak_L_minus_r":       round(float(obs_lmr[peak] * s), 3),
        "peak_p_value":         round(float(p_values[peak]), 5),
        "dclf_rmin_um":         float(rmin_um),
        "dclf_rmax_um":         float(rmax_um),
        "relevant_scale_um":    _RELEVANT_SCALE_UM,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Structure-preserving null models
#
# The original null (cross_k_null, above) repositions population B uniformly at
# random inside the tissue mask — *homogeneous* Complete Spatial Randomness (CSR).
# CSR is the weakest possible null: if CD8⁺ and TIM-3⁺ cells *independently* both
# prefer the same tissue compartment (say inflamed stroma), homogeneous CSR will
# scatter the null B across the whole mask, the observed (compartment-concentrated)
# pattern will sit far above that null, and the test will report "association"
# even though the two populations are conditionally independent given tissue
# architecture. That is a first-order (intensity) effect masquerading as a
# second-order (interaction) effect.
#
# To separate "real cross-type interaction" from "shared tissue preference" we add
# two structure-preserving nulls and report all three together (cross_k_all_nulls).
#
# IMPORTANT — why NOT random labeling. The textbook null for a single multiplex
# image is to randomly permute the type labels among the *pooled* points. We
# CANNOT use it here: CD8 and TIM-3 come from two DIFFERENT physical serial
# sections, so a "label" is inseparable from which section the cell was segmented
# in. Swapping a CD8 label onto a TIM-3 position (or vice-versa) invents a cell
# that was never observed and destroys the per-section intensity that is the whole
# point of the comparison. The valid analogues are CSR-type nulls that reposition
# ONE population while holding the other fixed — which is what all three nulls do.
# ──────────────────────────────────────────────────────────────────────────────

def _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, null_k, s, n_perm,
                         dclf_rmin_um, dclf_rmax_um):
    """
    Turn a matrix of null K curves into the same summary the per-radius envelope +
    DCLF reporting uses (identical maths to cross_k_null's tail). `null_k` is
    (n_perm, n_r) in px²; `obs_k_px` / `obs_lmr_px` are the observed curves in px.
    Returns a dict of µm-converted envelope, per-r p-values and the DCLF global.
    """
    null_mean_px = null_k.mean(axis=0)
    null_lo_px   = np.percentile(null_k, 2.5,  axis=0)
    null_hi_px   = np.percentile(null_k, 97.5, axis=0)
    ge       = (null_k >= obs_k_px[None, :]).sum(axis=0)
    p_values = (ge + 1.0) / (n_perm + 1.0)

    null_l_px = np.sqrt(np.clip(null_k, 0.0, None) / np.pi) - radii_px[None, :]
    null_l_lo = np.percentile(null_l_px, 2.5,  axis=0)
    null_l_hi = np.percentile(null_l_px, 97.5, axis=0)

    global_summary = _global_dclf_summary(
        radii_px, obs_lmr_px, null_l_px, p_values, s, dclf_rmin_um, dclf_rmax_um)

    return {
        "null_mean_K":  (null_mean_px * s * s).tolist(),
        "null_lower_K": (null_lo_px   * s * s).tolist(),
        "null_upper_K": (null_hi_px   * s * s).tolist(),
        "null_lower_L": (null_l_lo * s).tolist(),
        "null_upper_L": (null_l_hi * s).tolist(),
        "p_values":     p_values.tolist(),
        "global":       global_summary,
        "n_perm":       int(n_perm),
    }


def _build_kde_sampler(points_b, tissue_polygon, bbox, bandwidth_px, max_cells=160):
    """
    Build a grid estimate of population B's intensity surface λ_B(x) by binning B's
    positions and Gaussian-smoothing at `bandwidth_px`, then masking to the tissue
    window. Returns a sampler dict; draws from it reproduce B's own spatial
    preference (an inhomogeneous Poisson surface) without copying B's exact points.
    """
    import shapely
    from scipy.ndimage import gaussian_filter

    bx0, by0, bx1, by1 = bbox
    W = max(bx1 - bx0, 1e-6)
    H = max(by1 - by0, 1e-6)
    nx = max(int(np.ceil(W / max(bandwidth_px / 3.0, W / max_cells))), 1)
    ny = max(int(np.ceil(H / max(bandwidth_px / 3.0, H / max_cells))), 1)

    ex = np.linspace(bx0, bx1, nx + 1)
    ey = np.linspace(by0, by1, ny + 1)
    counts, _, _ = np.histogram2d(points_b[:, 0], points_b[:, 1], bins=[ex, ey])

    cell_w, cell_h = W / nx, H / ny
    dens = gaussian_filter(counts, sigma=(bandwidth_px / cell_w,
                                          bandwidth_px / cell_h), mode="constant")

    cx = (ex[:-1] + ex[1:]) / 2.0
    cy = (ey[:-1] + ey[1:]) / 2.0
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    cxr, cyr = CX.ravel(), CY.ravel()

    if tissue_polygon is not None:
        inside = shapely.contains_xy(tissue_polygon, cxr, cyr)
        dens = dens.ravel() * inside
    else:
        dens = dens.ravel()

    total = dens.sum()
    if total <= 0:                      # degenerate → fall back to uniform-in-window
        if tissue_polygon is not None:
            dens = shapely.contains_xy(tissue_polygon, cxr, cyr).astype(float)
        else:
            dens = np.ones_like(cxr)
        total = max(dens.sum(), 1.0)

    return {"probs": dens / total, "cxr": cxr, "cyr": cyr,
            "cell_w": cell_w, "cell_h": cell_h}


def _sample_from_kde(kde, n, rng):
    """Draw n points from a KDE sampler: pick grid cells ∝ intensity, jitter within."""
    idx = rng.choice(len(kde["probs"]), size=n, p=kde["probs"])
    x = kde["cxr"][idx] + (rng.random(n) - 0.5) * kde["cell_w"]
    y = kde["cyr"][idx] + (rng.random(n) - 0.5) * kde["cell_h"]
    return np.column_stack([x, y])


def _null_k_homogeneous(tree_a, radii_px, norm, n_b, tissue_polygon, bbox,
                        n_perm, rng):
    """Homogeneous CSR: B uniform in the window (rejection-sampled in the mask)."""
    from scipy.spatial import cKDTree
    n_r = len(radii_px)
    null_k = np.empty((n_perm, n_r), dtype=np.float64)
    for k in range(n_perm):
        if tissue_polygon is not None:
            b = _sample_in_polygon(tissue_polygon, n_b, rng, bbox)
        else:
            b = np.column_stack([rng.uniform(bbox[0], bbox[2], n_b),
                                 rng.uniform(bbox[1], bbox[3], n_b)])
        counts = tree_a.count_neighbors(cKDTree(b), radii_px)
        null_k[k] = norm * np.asarray(counts, dtype=np.float64)
    return null_k


def _null_k_inhomogeneous(tree_a, radii_px, norm, n_b, kde, n_perm, rng):
    """Inhomogeneous: B drawn from its own estimated intensity surface (preserves
    B's tissue preference; tests cross-structure beyond first-order intensity)."""
    from scipy.spatial import cKDTree
    n_r = len(radii_px)
    null_k = np.empty((n_perm, n_r), dtype=np.float64)
    for k in range(n_perm):
        b = _sample_from_kde(kde, n_b, rng)
        counts = tree_a.count_neighbors(cKDTree(b), radii_px)
        null_k[k] = norm * np.asarray(counts, dtype=np.float64)
    return null_k


def _null_k_toroidal(tree_a, radii_px, norm, points_b, bbox, n_perm, rng):
    """
    Toroidal shift: B's ENTIRE observed pattern (its real clustering) is rigidly
    translated by a random vector with wrap-around on the window's bounding box,
    randomizing only B's position relative to A. Preserves B's second-order
    structure exactly; assumes the pattern is stationary over the rectangular
    window (the standard toroidal-shift assumption — wrapping is meaningful).
    """
    from scipy.spatial import cKDTree
    bx0, by0, bx1, by1 = bbox
    W = max(bx1 - bx0, 1e-9)
    H = max(by1 - by0, 1e-9)
    b0 = np.asarray(points_b, dtype=np.float64) - np.array([bx0, by0])
    n_r = len(radii_px)
    null_k = np.empty((n_perm, n_r), dtype=np.float64)
    for k in range(n_perm):
        dx = rng.uniform(0.0, W)
        dy = rng.uniform(0.0, H)
        b = np.column_stack([np.mod(b0[:, 0] + dx, W) + bx0,
                             np.mod(b0[:, 1] + dy, H) + by0])
        counts = tree_a.count_neighbors(cKDTree(b), radii_px)
        null_k[k] = norm * np.asarray(counts, dtype=np.float64)
    return null_k


# ──────────────────────────────────────────────────────────────────────────────
# PRIMARY production test — intensity-reweighted inhomogeneous cross-K
#
# (See ihc.md §15 for the full diagnosis.) The resampling Kinhom and the toroidal
# shift were both anti-conservative under shared tissue preference. The fix is the
# literature-standard inhomogeneous cross-K (Baddeley–Møller–Waagepetersen):
#
#   K_AB^inhom(r) = (1/|W|) Σ_i Σ_j  1[ d(a_i,b_j) ≤ r ] / ( λ_A(a_i) · λ_B(b_j) )
#
# Reweighting each pair by 1/(λ_A·λ_B) makes E[K_AB^inhom(r)] = π r² under
# INDEPENDENCE *regardless of the shared first-order intensity* — the architectural
# response (margins/vessels/stroma both cell types follow) is cancelled
# analytically, so what remains is interaction BEYOND shared preference. The null
# distribution is a parametric bootstrap that holds A fixed (serial-section
# constraint: randomize one population), draws B* from λ̂_B, and CRUCIALLY
# RE-ESTIMATES the intensity from each B* — so the observed and the simulated curves
# are treated symmetrically and the plug-in (double-dipping) bias that made the old
# Kinhom anti-conservative is removed.
# ──────────────────────────────────────────────────────────────────────────────

def _build_intensity_grid(points, tissue_polygon, bbox, bandwidth_px, max_cells=160):
    """
    Grid intensity estimate λ̂(x) (points per px²) via binning + Gaussian smoothing
    at `bandwidth_px`, masked to the tissue window and normalized so its integral
    over the window equals the point count. Returns a dict that supports both point
    evaluation (`_lambda_at_points`) and sampling (`_draw_n_from_grid`) off the SAME
    surface, so the reweighting and the bootstrap draws are mutually consistent.
    """
    import shapely
    from scipy.ndimage import gaussian_filter

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    bx0, by0, bx1, by1 = bbox
    W = max(bx1 - bx0, 1e-6)
    H = max(by1 - by0, 1e-6)
    nx = max(int(np.ceil(W / max(bandwidth_px / 3.0, W / max_cells))), 1)
    ny = max(int(np.ceil(H / max(bandwidth_px / 3.0, H / max_cells))), 1)

    ex = np.linspace(bx0, bx1, nx + 1)
    ey = np.linspace(by0, by1, ny + 1)
    if len(pts):
        counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[ex, ey])
    else:
        counts = np.zeros((nx, ny))
    cell_w, cell_h = W / nx, H / ny
    cell_area = cell_w * cell_h
    smoothed = gaussian_filter(counts, sigma=(bandwidth_px / cell_w,
                                              bandwidth_px / cell_h), mode="constant")

    cx = (ex[:-1] + ex[1:]) / 2.0
    cy = (ey[:-1] + ey[1:]) / 2.0
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    if tissue_polygon is not None:
        inside = shapely.contains_xy(
            tissue_polygon, CX.ravel(), CY.ravel()).reshape(nx, ny)
    else:
        inside = np.ones((nx, ny), dtype=bool)
    smoothed = smoothed * inside

    n_fit = max(len(pts), 1)
    tot = float(smoothed.sum())
    if tot <= 0:                       # degenerate → uniform within window
        smoothed = inside.astype(float)
        tot = max(float(smoothed.sum()), 1.0)
    # Normalize so ∫ λ over the window = n_fit (λ in points per px²)
    lam = smoothed * (n_fit / tot) / cell_area
    win_area = max(float(inside.sum()) * cell_area, cell_area)
    lam_floor = 0.02 * (n_fit / win_area)     # floor to keep 1/λ finite
    probs = (smoothed / smoothed.sum()).ravel()

    return {"lam": lam, "lam_floor": lam_floor, "probs": probs,
            "bx0": bx0, "by0": by0, "cell_w": cell_w, "cell_h": cell_h,
            "nx": nx, "ny": ny, "cxr": CX.ravel(), "cyr": CY.ravel()}


def _lambda_at_points(grid, pts):
    """Evaluate the grid intensity λ̂ at arbitrary points (nearest cell), floored."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ix = np.clip(((pts[:, 0] - grid["bx0"]) / grid["cell_w"]).astype(int),
                 0, grid["nx"] - 1)
    iy = np.clip(((pts[:, 1] - grid["by0"]) / grid["cell_h"]).astype(int),
                 0, grid["ny"] - 1)
    return np.maximum(grid["lam"][ix, iy], grid["lam_floor"])


def _draw_n_from_grid(grid, n, rng):
    """Draw n points from the grid intensity (cells ∝ λ, jittered within cell)."""
    idx = rng.choice(len(grid["probs"]), size=n, p=grid["probs"])
    x = grid["cxr"][idx] + (rng.random(n) - 0.5) * grid["cell_w"]
    y = grid["cyr"][idx] + (rng.random(n) - 0.5) * grid["cell_h"]
    return np.column_stack([x, y])


def _loo_kernel_intensity(points, bandwidth_px, win_area):
    """
    Leave-one-out kernel intensity estimate at each point:
        λ̂(x_i) = Σ_{j≠i} K_h(x_i − x_j),   K_h = isotropic Gaussian, ∫K_h = 1.
    Excluding the point's own kernel (LOO) removes the plug-in self-attraction
    bias that otherwise makes the reweighted test mildly anti-conservative
    (spatstat uses leaveoneout=TRUE for exactly this reason). Units: points per
    px². Floored to keep 1/λ finite. O(N·neighbours) via a 4h cutoff.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    floor = 0.02 * (max(n, 1) / max(win_area, 1.0))
    if n <= 1:
        return np.full(n, max(n / max(win_area, 1.0), floor))
    h = float(bandwidth_px)
    inv2h2 = 1.0 / (2.0 * h * h)
    norm = 1.0 / (2.0 * np.pi * h * h)
    lam = np.zeros(n)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(4.0 * h, output_type="ndarray")   # i<j, excludes self
    if len(pairs):
        d2 = np.sum((pts[pairs[:, 0]] - pts[pairs[:, 1]]) ** 2, axis=1)
        kv = np.exp(-d2 * inv2h2) * norm
        np.add.at(lam, pairs[:, 0], kv)
        np.add.at(lam, pairs[:, 1], kv)
    return np.maximum(lam, floor)


def _cross_k_inhom_weighted(points_a, points_b, inv_lam_a, inv_lam_b,
                            radii_px, area_px):
    """
    Inhomogeneous cross-K estimator (uncorrected, reweighted):
        K(r) = (1/|W|) Σ_{i,j} 1[d_ij ≤ r] / (λ_A(a_i) λ_B(b_j))
    `inv_lam_a` = 1/λ_A at A's points, `inv_lam_b` = 1/λ_B at B's points. Returns the
    K curve (same length as radii_px) in px². No edge correction — the bootstrap
    null uses the identical estimator/window so the boundary bias cancels in the
    significance test (same argument as the homogeneous estimator).
    """
    from scipy.spatial import cKDTree
    n_r = len(radii_px)
    if len(points_a) == 0 or len(points_b) == 0:
        return np.zeros(n_r)
    ta = cKDTree(points_a)
    tb = cKDTree(points_b)
    rmax = float(np.max(radii_px))
    sdm = ta.sparse_distance_matrix(tb, rmax, output_type="coo_matrix")
    d = sdm.data
    if d.size == 0:
        return np.zeros(n_r)
    w = inv_lam_a[sdm.row] * inv_lam_b[sdm.col]
    order = np.argsort(d, kind="mergesort")
    d_s = d[order]
    cumw = np.concatenate([[0.0], np.cumsum(w[order])])
    idx = np.searchsorted(d_s, radii_px, side="right")
    return cumw[idx] / float(area_px)


def cross_k_inhom_reweighted_test(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii: np.ndarray,
    area: float,
    pixel_size_um: float,
    n_perm: int = 1000,
    seed: int = 0,
    tissue_polygon=None,
    bandwidth_um: float = _REWEIGHT_BANDWIDTH_UM,
    dclf_rmin_um: float = _DCLF_RMIN_UM,
    dclf_rmax_um: float = _DCLF_RMAX_UM,
) -> dict:
    """
    PRIMARY production test for cross-type association beyond shared tissue
    preference (see module note above and ihc.md §15).

    Returns the same summary shape as the other nulls (radii_um, K_observed,
    L_minus_r, null envelope, per-r p-values, global DCLF, n_perm) plus
    bandwidth_um / label / description / method, so existing consumers work
    unchanged. K and L−r here are the INHOMOGENEOUS (reweighted) versions.
    """
    radii_px = np.asarray(radii, dtype=np.float64)
    s = float(pixel_size_um)
    A = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    B = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    n_a, n_b, n_r = len(A), len(B), len(radii_px)

    meta = {
        "bandwidth_um": round(bandwidth_um, 3),
        "label": "Inhomogeneous reweighted cross-K",
        "description": "Intensity-reweighted inhomogeneous cross-K (Baddeley–Møller–"
                       "Waagepetersen): pairs weighted by 1/(λ_A·λ_B) so the shared "
                       "architectural intensity cancels (null mean π r²); null = "
                       "parametric bootstrap of B from λ̂_B with per-simulation "
                       "intensity re-estimation (removes plug-in bias).",
        "method": "reweighted_kinhom",
    }
    if n_a == 0 or n_b == 0 or n_r == 0:
        zeros = np.zeros(n_r)
        return {
            "radii_um": (radii_px * s).tolist(),
            "K_observed": zeros.tolist(), "g_observed": [None] * n_r,
            "L_minus_r": zeros.tolist(),
            "null_mean_K": zeros.tolist(), "null_lower_K": zeros.tolist(),
            "null_upper_K": zeros.tolist(), "null_lower_L": zeros.tolist(),
            "null_upper_L": zeros.tolist(), "p_values": np.ones(n_r).tolist(),
            "global": _empty_global(dclf_rmin_um, dclf_rmax_um),
            "n_perm": int(n_perm), **meta,
        }

    if tissue_polygon is not None:
        bbox = tissue_polygon.bounds
    else:
        allp = np.vstack([A, B])
        mn, mx = allp.min(axis=0), allp.max(axis=0)
        bbox = (mn[0], mn[1], mx[0], mx[1])

    h_px = bandwidth_um / s
    win_area = float(area)
    # Grid surface ONLY for sampling B* in the bootstrap; the intensity used for
    # REWEIGHTING is the leave-one-out kernel estimate (removes self-attraction bias).
    gB = _build_intensity_grid(B, tissue_polygon, bbox, h_px)
    inv_a = 1.0 / _loo_kernel_intensity(A, h_px, win_area)   # A held fixed throughout
    inv_b = 1.0 / _loo_kernel_intensity(B, h_px, win_area)

    k_obs = _cross_k_inhom_weighted(A, B, inv_a, inv_b, radii_px, area)   # px²
    l_obs = np.sqrt(np.clip(k_obs, 0.0, None) / np.pi) - radii_px

    rng = np.random.default_rng(int(seed))
    null_k = np.empty((n_perm, n_r), dtype=np.float64)
    for k in range(n_perm):
        b_star = _draw_n_from_grid(gB, n_b, rng)
        inv_bs = 1.0 / _loo_kernel_intensity(b_star, h_px, win_area)  # RE-ESTIMATE (LOO)
        null_k[k] = _cross_k_inhom_weighted(A, b_star, inv_a, inv_bs, radii_px, area)

    summ = _null_summary_from_k(radii_px, k_obs, l_obs, null_k, s,
                                n_perm, dclf_rmin_um, dclf_rmax_um)
    g_out = _pcf_from_k(k_obs, radii_px)
    return {
        "radii_um": (radii_px * s).tolist(),
        "K_observed": (k_obs * s * s).tolist(),
        "g_observed": [None if not np.isfinite(v) else float(v) for v in g_out],
        "L_minus_r": (l_obs * s).tolist(),
        **summ, **meta,
    }


def _draw_n_from_support_jitter(support_points, n, rng, sigma_px, tissue_polygon=None):
    """
    Draw n points from marker-independent morphology support by choosing support
    nuclei with replacement and adding isotropic Gaussian jitter. If a tissue
    polygon/window is provided, reject jittered points outside it.
    """
    pts = np.asarray(support_points, dtype=np.float64).reshape(-1, 2)
    n = int(n)
    if n <= 0 or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if tissue_polygon is None:
        anchors = pts[rng.integers(0, len(pts), size=n)]
        return anchors + rng.normal(0.0, float(sigma_px), size=anchors.shape)

    import shapely
    out = []
    batch = max(n * 4, 512)
    for _ in range(128):
        if len(out) >= n:
            break
        anchors = pts[rng.integers(0, len(pts), size=batch)]
        cand = anchors + rng.normal(0.0, float(sigma_px), size=anchors.shape)
        keep = shapely.contains_xy(tissue_polygon, cand[:, 0], cand[:, 1])
        if np.any(keep):
            out.extend(cand[keep].tolist())
    if len(out) < n:
        # Extremely thin/fragmented windows can reject most jittered draws. Fall
        # back to unjittered support points already inside the window so the null
        # remains conditioned on morphology rather than bbox-uniform noise.
        extra = pts[rng.integers(0, len(pts), size=n - len(out))]
        out.extend(extra.tolist())
    return np.asarray(out[:n], dtype=np.float64)


def cross_k_dense_morphology_test(
    points_a: np.ndarray,
    points_b: np.ndarray,
    morphology_support: np.ndarray,
    radii: np.ndarray,
    area: float,
    pixel_size_um: float,
    n_perm: int = 1000,
    seed: int = 0,
    tissue_polygon=None,
    jitter_um: float = _DENSE_MORPHOLOGY_JITTER_UM,
    dclf_rmin_um: float = _DENSE_DCLF_RMIN_UM,
    dclf_rmax_um: float = _DENSE_DCLF_RMAX_UM,
) -> dict:
    """
    Dense-tissue primary candidate for fields where the 75 µm reweighted null is
    not size-controlled.

    Null hypothesis:
        B is independently assigned over the marker-independent all-cell
        morphology field in the certified analysis window.

    Implementation:
        Hold A fixed. Draw each B* from all detected reference-section nuclei
        inside the analysis window, plus 2 µm Gaussian jitter, then recompute the
        same unweighted cross-K and DCLF statistic. This preserves dense tissue
        architecture without using the A/B marker positives to define the null.
    """
    from scipy.spatial import cKDTree

    radii_px = np.asarray(radii, dtype=np.float64)
    s = float(pixel_size_um)
    A = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    B = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if morphology_support is None:
        support = np.empty((0, 2), dtype=np.float64)
    else:
        support = np.asarray(morphology_support, dtype=np.float64).reshape(-1, 2)
    n_a, n_b, n_r = len(A), len(B), len(radii_px)

    meta = {
        "jitter_um": round(float(jitter_um), 3),
        "dclf_band_um": [float(dclf_rmin_um), float(dclf_rmax_um)],
        "support_n": int(len(support)),
        "label": "Dense morphology-conditioned cross-K",
        "description": "Dense-tissue null: B* sampled from marker-independent "
                       "all-cell morphology support in the certified analysis "
                       "window plus 2 µm jitter; validated as the dense fallback "
                       "candidate on public CODEX architecture and rendered "
                       "image-derived morphology.",
        "method": "dense_morphology_support_jitter",
        "validation_ids": [
            "public_codex_dense_null",
            "dense_null_image_morphology",
            "dense_null_real_ll477",
        ],
    }
    observed = cross_k_function(A, B, radii_px, area, pixel_size_um)
    if n_a == 0 or n_b == 0 or n_r == 0 or len(support) == 0:
        zeros = np.zeros(n_r)
        return {
            "radii_um": (radii_px * s).tolist(),
            "K_observed": zeros.tolist(), "g_observed": [None] * n_r,
            "L_minus_r": zeros.tolist(),
            "null_mean_K": zeros.tolist(), "null_lower_K": zeros.tolist(),
            "null_upper_K": zeros.tolist(), "null_lower_L": zeros.tolist(),
            "null_upper_L": zeros.tolist(), "p_values": np.ones(n_r).tolist(),
            "global": _empty_global(dclf_rmin_um, dclf_rmax_um),
            "n_perm": int(n_perm), **meta,
        }

    obs_counts = _pair_counts(A, B, radii_px)
    obs_k_px = _k_from_counts(obs_counts, float(area), n_a, n_b)
    obs_lmr_px = np.sqrt(np.clip(obs_k_px, 0.0, None) / np.pi) - radii_px

    rng = np.random.default_rng(int(seed))
    tree_a = cKDTree(A)
    norm = float(area) / (n_a * n_b)
    sigma_px = float(jitter_um) / max(s, 1e-9)
    null_k = np.empty((n_perm, n_r), dtype=np.float64)
    for k in range(n_perm):
        b_star = _draw_n_from_support_jitter(
            support, n_b, rng, sigma_px, tissue_polygon=tissue_polygon)
        null_counts = tree_a.count_neighbors(cKDTree(b_star), radii_px)
        null_k[k] = norm * np.asarray(null_counts, dtype=np.float64)

    summ = _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, null_k, s,
                                n_perm, dclf_rmin_um, dclf_rmax_um)
    return {**observed, **summ, **meta}


# Human-readable descriptions of each null (surfaced in JSON / UI / docs).
_NULL_META = {
    "homogeneous": {
        "label": "Homogeneous CSR",
        "description": "Population B placed uniformly at random in the tissue "
                       "window. Weakest null — cannot tell real interaction from a "
                       "shared tissue-compartment preference. Baseline only.",
    },
    "inhomogeneous": {
        "label": "Inhomogeneous (Kinhom)",
        "description": "Population B resampled from its OWN estimated intensity "
                       "surface (Gaussian KDE), preserving its tissue preference. "
                       "Tests for cross-type structure BEYOND shared first-order "
                       "intensity — the literature standard.",
    },
    "toroidal": {
        "label": "Toroidal shift",
        "description": "Population B's entire real pattern is rigidly shifted "
                       "(wrap-around), preserving its exact clustering and only "
                       "randomizing its position relative to A. Robust to KDE "
                       "bandwidth choice; assumes stationarity over the window.",
    },
    "dense_morphology": {
        "label": "Dense morphology-conditioned cross-K",
        "description": "Dense-tissue primary: B* sampled from marker-independent "
                       "all-cell morphology support in the certified analysis "
                       "window plus 2 µm jitter. Used only when the 75 µm "
                       "reweighted primary is not size-controlled and dense-mode "
                       "support/count gates pass.",
    },
}


def _assess_robustness(summaries: dict, primary: str = None) -> dict:
    """
    Reduce the per-null DCLF calls to a single verdict, GATED ON THE CALIBRATED
    PRIMARY (the intensity-reweighted inhomogeneous cross-K), with homogeneous CSR
    kept only as a diagnostic baseline. (See ihc.md §15: the old resampling-Kinhom
    and toroidal nulls were anti-conservative under shared tissue preference and no
    longer gate anything.) Verdicts:
      • "robust"   — significant under the CALIBRATED primary → association beyond
                     the shared architectural response (size-controlled, see §15).
      • "csr_only" — significant under the weak homogeneous-CSR baseline but NOT the
                     calibrated primary → CRITICAL: the signal is shared tissue
                     preference, not interaction; reported honestly, never hidden.
      • "none"     — nothing significant.
    """
    sig = {n: bool(s.get("global", {}).get("significant")) for n, s in summaries.items()}
    dire = {n: s.get("global", {}).get("direction", "none") for n, s in summaries.items()}
    gp  = {n: s.get("global", {}).get("global_p_dclf") for n, s in summaries.items()}

    if primary is None:
        primary = ("reweighted" if "reweighted" in summaries
                   else "dense_morphology" if "dense_morphology" in summaries
                   else "inhomogeneous" if "inhomogeneous" in summaries
                   else "homogeneous" if "homogeneous" in summaries
                   else None)
    prim_sig = sig.get(primary, False)
    prim_dir = dire.get(primary, "none")
    homog_sig = sig.get("homogeneous", False)
    homog_dir = dire.get("homogeneous", "none")

    if prim_sig:
        direction = prim_dir
        verdict = "robust"
        if primary == "dense_morphology":
            summary = (f"Significant {direction} under the dense morphology-"
                       f"conditioned cross-K (all-cell support + 2 µm jitter, "
                       f"10–30 µm DCLF band) — association beyond dense tissue "
                       f"morphology.")
        else:
            summary = (f"Significant {direction} under the calibrated reweighted "
                       f"inhomogeneous cross-K (the production primary, size-"
                       f"controlled under shared tissue preference) — association "
                       f"beyond the shared architectural response.")
    elif homog_sig:
        direction = homog_dir
        verdict = "csr_only"
        summary = (f"CRITICAL: significant {homog_dir} ONLY under the weak homogeneous-"
                   f"CSR baseline; it VANISHES under the calibrated primary null — "
                   f"consistent with NO association beyond shared tissue preference.")
    else:
        direction = "none"
        verdict = "none"
        if primary == "dense_morphology":
            summary = "No significant cross-type association under the dense morphology-conditioned primary."
        else:
            summary = "No significant cross-type association under the calibrated primary."

    return {
        "verdict": verdict,
        "direction": direction,
        "summary": summary,
        "primary_null": primary,
        "per_null_significant": sig,
        "per_null_direction": dire,
        "per_null_global_p": gp,
    }


def cross_k_all_nulls(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii: np.ndarray,
    area: float,
    pixel_size_um: float,
    n_perm: int = 1000,
    seed: int = 0,
    tissue_polygon=None,
    dclf_rmin_um: float = _DCLF_RMIN_UM,
    dclf_rmax_um: float = _DCLF_RMAX_UM,
    kde_bandwidth_um: float = None,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
    reweight_bandwidth_um: float = _REWEIGHT_BANDWIDTH_UM,
    nulls=("reweighted", "homogeneous"),
    morphology_support=None,
    dense_jitter_um: float = _DENSE_MORPHOLOGY_JITTER_UM,
) -> dict:
    """
    Cross-type spatial association under the CALIBRATED PRIMARY null + a baseline.

    Default `nulls`:
      • "reweighted"  — PRIMARY. Intensity-reweighted inhomogeneous cross-K with
        per-simulation intensity re-estimation (cross_k_inhom_reweighted_test). It
        is size-controlled under shared tissue preference AND uniform independence
        AND retains power (validated in validate_reweighted_null.py / ihc.md §15).
        Its curves drive the top-level result and the robustness verdict.
      • "homogeneous" — DIAGNOSTIC baseline only. Significant-under-CSR-but-not-the-
        primary ⇒ the "csr_only" verdict that flags shared-preference artifacts.

    The retired "inhomogeneous" (resampling) and "toroidal" nulls are still
    COMPUTABLE on request (they back the calibration/diagnostic scripts and document
    why they were dropped — both anti-conservative under shared preference), but are
    no longer in the default set and never gate the verdict.

    Top-level `global` / `null_*` / `L_minus_r` mirror the PRIMARY (reweighted) so
    existing consumers (overlay/UI/QC) work unchanged with the calibrated null.
    """
    radii_px = np.asarray(radii, dtype=np.float64)
    s   = float(pixel_size_um)
    n_a = int(len(points_a))
    n_b = int(len(points_b))
    n_r = len(radii_px)

    observed = cross_k_function(points_a, points_b, radii_px, area, pixel_size_um)

    if n_a == 0 or n_b == 0 or n_r == 0:
        empty = {**observed, "n_perm": int(n_perm),
                 "tissue_area_um2": float(area) * s * s,
                 "nulls": {}, "robustness": _assess_robustness({}),
                 "null_lower_L": observed["L_minus_r"], "null_upper_L": observed["L_minus_r"],
                 "null_lower_K": observed["K_observed"], "null_upper_K": observed["K_observed"],
                 "null_mean_K": observed["K_observed"],
                 "p_values": np.ones(n_r).tolist(),
                 "global": _empty_global(dclf_rmin_um, dclf_rmax_um)}
        return empty

    obs_counts = _pair_counts(points_a, points_b, radii_px)
    obs_k_px   = _k_from_counts(obs_counts, float(area), n_a, n_b)
    obs_lmr_px = np.sqrt(np.clip(obs_k_px, 0.0, None) / np.pi) - radii_px

    pts_a  = np.asarray(points_a, dtype=np.float64)
    pts_b  = np.asarray(points_b, dtype=np.float64)
    allpts = np.vstack([pts_a, pts_b])
    if tissue_polygon is not None:
        bbox = tissue_polygon.bounds
    else:
        mn, mx = allpts.min(axis=0), allpts.max(axis=0)
        bbox = (mn[0], mn[1], mx[0], mx[1])

    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    norm   = float(area) / (n_a * n_b)
    rng    = np.random.default_rng(int(seed))

    out_nulls = {}

    # ── PRIMARY: calibrated reweighted inhomogeneous cross-K ───────────────────
    if "reweighted" in nulls:
        out_nulls["reweighted"] = cross_k_inhom_reweighted_test(
            points_a, points_b, radii_px, area, pixel_size_um,
            n_perm=n_perm, seed=int(seed), tissue_polygon=tissue_polygon,
            bandwidth_um=reweight_bandwidth_um,
            dclf_rmin_um=dclf_rmin_um, dclf_rmax_um=dclf_rmax_um)

    if "dense_morphology" in nulls:
        out_nulls["dense_morphology"] = cross_k_dense_morphology_test(
            points_a, points_b, morphology_support, radii_px, area, pixel_size_um,
            n_perm=n_perm, seed=int(seed), tissue_polygon=tissue_polygon,
            jitter_um=dense_jitter_um,
            dclf_rmin_um=dclf_rmin_um, dclf_rmax_um=dclf_rmax_um)

    if "homogeneous" in nulls:
        nk = _null_k_homogeneous(tree_a, radii_px, norm, n_b,
                                 tissue_polygon, bbox, n_perm, rng)
        summ = _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, nk, s,
                                    n_perm, dclf_rmin_um, dclf_rmax_um)
        out_nulls["homogeneous"] = {**summ, **_NULL_META["homogeneous"]}

    if "inhomogeneous" in nulls:
        h_px = (kde_bandwidth_um or _KDE_BANDWIDTH_UM) / s
        # main (1×)
        kde  = _build_kde_sampler(pts_b, tissue_polygon, bbox, h_px)
        nk   = _null_k_inhomogeneous(tree_a, radii_px, norm, n_b, kde, n_perm, rng)
        summ = _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, nk, s,
                                    n_perm, dclf_rmin_um, dclf_rmax_um)
        # bandwidth sensitivity sweep (report the global call at each)
        sensitivity = {}
        for mult in bandwidth_multipliers:
            kde_m = _build_kde_sampler(pts_b, tissue_polygon, bbox, h_px * mult)
            nk_m  = _null_k_inhomogeneous(tree_a, radii_px, norm, n_b, kde_m, n_perm, rng)
            summ_m = _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, nk_m, s,
                                          n_perm, dclf_rmin_um, dclf_rmax_um)
            sensitivity[f"{mult:g}x"] = {
                "bandwidth_um": round(h_px * mult * s, 3),
                "global": summ_m["global"],
            }
        out_nulls["inhomogeneous"] = {
            **summ, **_NULL_META["inhomogeneous"],
            "bandwidth_um": round(h_px * s, 3),
            "bandwidth_method": "fixed_um",
            "bandwidth_sensitivity": sensitivity,
        }

    if "toroidal" in nulls:
        nk = _null_k_toroidal(tree_a, radii_px, norm, pts_b, bbox, n_perm, rng)
        summ = _null_summary_from_k(radii_px, obs_k_px, obs_lmr_px, nk, s,
                                    n_perm, dclf_rmin_um, dclf_rmax_um)
        out_nulls["toroidal"] = {**summ, **_NULL_META["toroidal"]}

    # Primary = the calibrated reweighted test if present, else (for diagnostic
    # callers that request only the old nulls) inhomogeneous, else homogeneous.
    primary_name = ("reweighted" if "reweighted" in out_nulls
                    else "dense_morphology" if "dense_morphology" in out_nulls
                    else "inhomogeneous" if "inhomogeneous" in out_nulls
                    else "homogeneous" if "homogeneous" in out_nulls
                    else next(iter(out_nulls), None))
    primary = out_nulls.get(primary_name, {})
    robustness = _assess_robustness(out_nulls, primary=primary_name)

    # Top-level OBSERVED curves mirror the primary. For the reweighted primary these
    # are the INHOMOGENEOUS (reweighted) K / L−r, so the plotted curve and its
    # envelope are on the same scale; for a diagnostic-only call they fall back to
    # the unweighted observed.
    if primary_name in ("reweighted", "dense_morphology") and primary:
        top_obs = {k: primary[k] for k in
                   ("radii_um", "K_observed", "g_observed", "L_minus_r")}
    else:
        top_obs = observed

    return {
        **top_obs,
        "n_perm":          int(n_perm),
        "tissue_area_um2": float(area) * s * s,
        "nulls":           out_nulls,
        "primary_null":    primary_name,
        "robustness":      robustness,
        # ── top-level mirrors the PRIMARY (calibrated) null ──
        "null_mean_K":  primary.get("null_mean_K"),
        "null_lower_K": primary.get("null_lower_K"),
        "null_upper_K": primary.get("null_upper_K"),
        "null_lower_L": primary.get("null_lower_L"),
        "null_upper_L": primary.get("null_upper_L"),
        "p_values":     primary.get("p_values"),
        "global":       primary.get("global", _empty_global(dclf_rmin_um, dclf_rmax_um)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tissue mask
# ──────────────────────────────────────────────────────────────────────────────

def estimate_tissue_mask(image_path: str, pixel_size_um: float):
    """
    Estimate the tissue region of an IHC image for K-function normalization and
    null randomization.

    A downsampled grayscale thumbnail is Otsu-thresholded (stained tissue is
    darker than the bright background), morphologically closed to fill gaps, and
    the largest connected component(s) are kept as tissue. The area is reported
    in full-resolution pixels; a simplified shapely polygon of the tissue
    boundary (in full-resolution coordinates) is returned for bounding the null.

    Args:
        image_path:     path to the reference (CD8) image
        pixel_size_um:  µm per pixel (currently informational; area is in pixels)

    Returns:
        (area_pixels, polygon_or_None). On any failure returns (None, None) so
        the caller can fall back to a bounding-box area with a logged warning.
    """
    try:
        import cv2
        from oasis.common.registration import _load_thumbnail
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        gray, scale = _load_thumbnail(image_path, max_side=2000, mode="L")
        if gray is None or scale <= 0:
            return None, None
        gray = np.ascontiguousarray(gray.astype(np.uint8))

        # Otsu split; tissue (darker) becomes foreground via INV
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Close gaps inside tissue, then open to drop speckle
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lbl <= 1:
            return None, None

        # Component areas (skip background label 0); keep those >= 10% of the
        # largest so multi-fragment tissue sections are preserved.
        areas      = stats[1:, cv2.CC_STAT_AREA]
        largest    = int(areas.max())
        keep_labels = [i + 1 for i, a in enumerate(areas) if a >= 0.10 * largest]

        keep_mask = np.isin(labels, keep_labels).astype(np.uint8) * 255
        area_thumb = int(np.count_nonzero(keep_mask))
        if area_thumb == 0:
            return None, None
        # Thumbnail → full-resolution pixel area
        area_pixels = float(area_thumb) / (scale * scale)

        # Boundary polygon(s) in full-resolution coordinates
        contours, _ = cv2.findContours(
            keep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if len(c) < 3:
                continue
            ring = (c.reshape(-1, 2).astype(np.float64) / scale)
            try:
                poly = Polygon(ring)
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
            except Exception:
                continue
        polygon = None
        if polys:
            merged = unary_union(polys) if len(polys) > 1 else polys[0]
            # Simplify to keep rejection sampling fast (tolerance in px)
            polygon = merged.simplify(2.0 / scale, preserve_topology=True)
            if polygon.is_empty:
                polygon = merged

        return area_pixels, polygon
    except Exception as e:
        print(f"  Tissue mask estimation failed: {e}")
        return None, None


def bounding_box_area(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Fallback tissue area: bounding box of all points, in pixels^2."""
    if len(points_a) == 0 and len(points_b) == 0:
        return 0.0
    pts = np.vstack([
        np.asarray(points_a, dtype=np.float64).reshape(-1, 2),
        np.asarray(points_b, dtype=np.float64).reshape(-1, 2),
    ])
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return float((mx[0] - mn[0]) * (mx[1] - mn[1]))


# ──────────────────────────────────────────────────────────────────────────────
# A∩B intersection tissue window
#
# The single-image (A-only) mask treats regions present in just one section as
# valid analysis area. But cross-SECTION analysis can only be informed where BOTH
# sections actually have tissue: a fold, tear, or missing-tissue region in section
# B contributes no B cells there, so any "segregation" measured against A in that
# region is an artifact of the missing section, not biology. We therefore restrict
# the analysis window to A_tissue ∩ B_tissue (B's mask registered into A space).
# ──────────────────────────────────────────────────────────────────────────────

def estimate_tissue_polygon(image_path: str, pixel_size_um: float,
                            max_side: int = 2000):
    """
    Like estimate_tissue_mask, but returns a shapely (Multi)Polygon that PRESERVES
    internal holes/lumens/background (do NOT fill holes — empty lumens are not
    tissue and must not count as analysis area). Returns (area_px, polygon) in
    full-resolution coordinates, or (None, None) on failure.
    """
    try:
        import cv2
        from oasis.common.registration import _load_thumbnail
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        gray, scale = _load_thumbnail(image_path, max_side=max_side, mode="L")
        if gray is None or scale <= 0:
            return None, None
        gray = np.ascontiguousarray(gray.astype(np.uint8))

        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lbl <= 1:
            return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max())
        keep_labels = [i + 1 for i, a in enumerate(areas) if a >= 0.10 * largest]
        keep_mask = np.isin(labels, keep_labels).astype(np.uint8) * 255
        if np.count_nonzero(keep_mask) == 0:
            return None, None

        # RETR_CCOMP → 2-level hierarchy: outer boundaries + their holes. Build
        # shells minus holes so internal lumens/background are preserved.
        contours, hierarchy = cv2.findContours(
            keep_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None, None
        shells, holes = [], []
        for cnt, h in zip(contours, hierarchy[0]):
            ring = cnt.reshape(-1, 2).astype(np.float64) / scale
            if len(ring) < 3:
                continue
            try:
                poly = Polygon(ring)
            except Exception:
                continue
            if not poly.is_valid or poly.area <= 0:
                poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            (holes if h[3] != -1 else shells).append(poly)

        if not shells:
            return None, None
        geom = unary_union(shells)
        if holes:
            geom = geom.difference(unary_union(holes))
        if geom.is_empty:
            return None, None
        geom = geom.simplify(2.0 / scale, preserve_topology=True)
        if geom.is_empty:
            geom = unary_union(shells)
        return float(geom.area), geom
    except Exception as e:
        print(f"  Tissue polygon estimation failed: {e}")
        return None, None


def transform_polygon(polygon, reg_result):
    """Map a shapely (Multi)Polygon through a registration transform (moving →
    reference space), applying it to the exterior and every interior ring so holes
    survive. Uses the same transform_centroids path the cell points use."""
    from shapely.geometry import Polygon, MultiPolygon
    from oasis.common.registration import transform_centroids

    def _tx(coords):
        arr = np.asarray(coords, dtype=np.float64)[:, :2]
        return transform_centroids(arr.astype(np.float32), reg_result)

    def _tx_poly(poly):
        ext = _tx(list(poly.exterior.coords))
        ints = [_tx(list(r.coords)) for r in poly.interiors]
        out = Polygon(ext, ints)
        return out if out.is_valid else out.buffer(0)

    try:
        if polygon.geom_type == "Polygon":
            return _tx_poly(polygon)
        if polygon.geom_type == "MultiPolygon":
            return MultiPolygon([_tx_poly(p) for p in polygon.geoms])
    except Exception as e:
        print(f"  Polygon transform failed: {e}")
    return polygon


def intersection_window(poly_a, poly_b_in_a):
    """
    Intersect A's tissue polygon with B's registered tissue polygon. Returns
    (window, area_px, overlap_iou, frac_of_a) or (None, …) if empty/degenerate.
    `overlap_iou` = |A∩B| / |A∪B|; `frac_of_a` = |A∩B| / |A|.
    """
    from shapely.ops import unary_union
    try:
        inter = poly_a.intersection(poly_b_in_a)
    except Exception:
        inter = poly_a.buffer(0).intersection(poly_b_in_a.buffer(0))
    if inter.is_empty:
        return None, 0.0, 0.0, 0.0
    if inter.geom_type not in ("Polygon", "MultiPolygon"):
        parts = [g for g in getattr(inter, "geoms", [])
                 if g.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return None, 0.0, 0.0, 0.0
        inter = unary_union(parts)
    area  = float(inter.area)
    union = float(poly_a.union(poly_b_in_a).area)
    iou   = area / union if union > 0 else 0.0
    frac  = area / poly_a.area if poly_a.area > 0 else 0.0
    return inter, area, iou, frac


def filter_points_in_polygon(points: np.ndarray, polygon):
    """Keep only points inside `polygon` (holes respected). Returns
    (inside_points, n_excluded)."""
    import shapely
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0 or polygon is None:
        return pts, 0
    inside = shapely.contains_xy(polygon, pts[:, 0], pts[:, 1])
    return pts[inside], int((~inside).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Cohort-level multiple-comparison correction (REPORTING ONLY)
#
# Each pair already gets one honest DCLF p-value (the per-radius multiplicity is
# handled inside the DCLF test). But running N independent pairs and then quoting
# "the smallest p across pairs" is a multiplicity trap: with 8 pairs you expect a
# p≈0.05 by chance even if nothing is real. Any COHORT-LEVEL statement ("the
# cohort shows association") must therefore use a multiplicity-controlled p, not
# the raw minimum. This function does NOT change any per-pair statistic — it only
# adjusts a set of already-computed per-pair p-values for reporting.
# ──────────────────────────────────────────────────────────────────────────────

def cohort_multiple_comparison_correction(per_pair_pvalues, method: str = "bh",
                                          alpha: float = 0.05) -> dict:
    """
    Multiple-comparison correction across the per-pair global DCLF p-values.

    Args:
        per_pair_pvalues: iterable of per-pair p-values. Entries that are None /
                          non-finite (e.g. QC-invalid pairs with no usable p) are
                          dropped before correction and counted separately.
        method:           "bh" (Benjamini-Hochberg FDR; default) or
                          "bonferroni" (family-wise).
        alpha:            significance level for the post-correction tally.

    Returns a dict:
        {
          "method", "alpha", "n_tested",        # n_tested = finite p-values used
          "n_dropped",                          # None / non-finite inputs dropped
          "raw_pvalues",                        # the finite inputs, original order
          "adjusted_pvalues",                   # same order, q-values (BH) / scaled
          "n_significant_raw",                  # # raw p <= alpha (the trap count)
          "n_significant_adjusted",             # # adjusted p <= alpha (defensible)
          "min_raw_p", "min_adjusted_p",
          "note",                               # plain-language guidance
        }

    BH q-values are computed in the standard way: sort ascending, q_(i) =
    p_(i) * n / i, then enforce monotonic non-decreasing q from the largest rank
    downward (cumulative minimum), and clip to 1.0. Bonferroni multiplies each p
    by n (clipped to 1.0).
    """
    raw = []
    n_dropped = 0
    for p in (per_pair_pvalues or []):
        try:
            pf = float(p)
        except (TypeError, ValueError):
            n_dropped += 1
            continue
        if not np.isfinite(pf):
            n_dropped += 1
            continue
        raw.append(pf)

    n = len(raw)
    base = {
        "method":                 method,
        "alpha":                  float(alpha),
        "n_tested":               n,
        "n_dropped":              int(n_dropped),
        "raw_pvalues":            [round(p, 6) for p in raw],
        "adjusted_pvalues":       [],
        "n_significant_raw":      0,
        "n_significant_adjusted": 0,
        "min_raw_p":              round(min(raw), 6) if raw else None,
        "min_adjusted_p":         None,
        "note": ("Cohort-level claims must use the adjusted p-values, not the raw "
                 "minimum (quoting the smallest of several p-values inflates the "
                 "false-positive rate). Per-pair results remain descriptive."),
    }
    if n == 0:
        return base

    raw_arr = np.asarray(raw, dtype=np.float64)
    if method == "bonferroni":
        adj = np.clip(raw_arr * n, 0.0, 1.0)
    else:  # Benjamini-Hochberg FDR
        order = np.argsort(raw_arr, kind="mergesort")
        ranks = np.arange(1, n + 1)
        q_sorted = raw_arr[order] * n / ranks
        # enforce monotonicity from the largest p downward, then clip
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0.0, 1.0)
        adj = np.empty(n, dtype=np.float64)
        adj[order] = q_sorted

    base["adjusted_pvalues"]       = [round(float(q), 6) for q in adj]
    base["n_significant_raw"]      = int(np.sum(raw_arr <= alpha))
    base["n_significant_adjusted"] = int(np.sum(adj <= alpha))
    base["min_adjusted_p"]         = round(float(adj.min()), 6)
    return base


# ── Architecture-scale estimator + validity gate (audit A6 / ihc.md §15.5) ──────
# The reweighted primary null (bandwidth _REWEIGHT_BANDWIDTH_UM) ASSUMES the tissue
# architecture varies on a scale COARSER than the reweighting bandwidth, so the
# intensity reweighting removes shared compartment preference without absorbing the
# 10–50 µm cell-scale interaction. When the real architecture scale is inside the
# interaction band, that separation fails and the test becomes anti-conservative.
# This estimator MEASURES the architecture scale per pattern so the assumption can be
# checked at runtime instead of merely disclosed. Its operating characteristics are
# calibrated by validation/validate_architecture_scale.py.

_ARCH_PILOT_BANDWIDTH_UM = 15.0   # small pilot kernel: NOT circular with the 75 µm bw
# Minimum architecture scale for a size-controlled reweighted test. Calibrated by
# validation/validate_architecture_scale.py, which shows the test stays
# anti-conservative until ℓ̂ is comfortably ABOVE the bandwidth (size control only
# once ℓ̂ ≳ 2× bandwidth); deliberately conservative. Re-derive at paper-grade sims.
_ARCH_MIN_SCALE_FACTOR = 2.0


def estimate_architecture_scale(points, pixel_size_um, tissue_polygon=None,
                                pilot_bandwidth_um=_ARCH_PILOT_BANDWIDTH_UM,
                                bbox=None):
    """
    Characteristic architecture length ℓ̂ (µm) of a point pattern: the e-folding range
    (autocorrelation = 1/e) of the pattern's intensity field, estimated at a small
    PILOT bandwidth well below the reweighting bandwidth so the estimate is not
    circular with the 75 µm reweight kernel.

    Method: bin points to a grid (cell ≈ pilot/2), Gaussian-smooth at pilot_bandwidth,
    FFT autocorrelation, radially average, return the lag where the normalized
    autocorrelation first falls to 1/e. Cannot resolve architecture finer than the
    pilot; saturates at ~field/3 for architecture coarser than the window. Returns
    None if there are too few points to estimate.
    """
    from scipy.ndimage import gaussian_filter
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 30:
        return None
    s = float(pixel_size_um)
    if bbox is None:
        if tissue_polygon is not None:
            bbox = tissue_polygon.bounds
        else:
            mn, mx = pts.min(axis=0), pts.max(axis=0)
            bbox = (mn[0], mn[1], mx[0], mx[1])
    x0, y0, x1, y1 = bbox
    W = max(x1 - x0, 1.0); H = max(y1 - y0, 1.0)
    pilot_px = max(pilot_bandwidth_um / s, 1.0)
    cell = max(pilot_px / 2.0, 1.0)
    nx = int(min(max(round(W / cell), 8), 1024))
    ny = int(min(max(round(H / cell), 8), 1024))
    ix = np.clip(((pts[:, 0] - x0) / W * nx).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / H * ny).astype(int), 0, ny - 1)
    grid = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(grid, (iy, ix), 1.0)
    bx_um = (W / nx) * s; by_um = (H / ny) * s      # bin size in µm per axis
    f = gaussian_filter(grid, sigma=(pilot_bandwidth_um / by_um,
                                     pilot_bandwidth_um / bx_um), mode="nearest")
    f = f - f.mean()
    if not np.any(f):
        return None
    F = np.fft.rfft2(f)
    ac = np.fft.irfft2(np.abs(F) ** 2, s=f.shape)
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    if peak <= 0:
        return None
    ac = ac / peak
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices(ac.shape)
    r_um = np.sqrt(((xx - cx) * bx_um) ** 2 + ((yy - cy) * by_um) ** 2)
    rmax = min(W * s, H * s) / 3.0                  # don't trust beyond a third of field
    nb = 40
    edges = np.linspace(0.0, rmax, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nb, np.nan)
    for i in range(nb):
        m = (r_um >= edges[i]) & (r_um < edges[i + 1])
        if m.any():
            prof[i] = ac[m].mean()
    thr = 1.0 / np.e
    below = np.where(prof <= thr)[0]
    if len(below) == 0:
        return float(rmax)                           # coarser than resolvable → large
    i = int(below[0])
    if i == 0:
        return float(centers[0])
    p0, p1 = prof[i - 1], prof[i]
    c0, c1 = centers[i - 1], centers[i]
    if not np.isfinite(p0) or p0 == p1:
        return float(c1)
    return float(max(c0 + (thr - p0) * (c1 - c0) / (p1 - p0), 0.0))


def architecture_scale_verdict(scale_um, bandwidth_um=_REWEIGHT_BANDWIDTH_UM):
    """
    Classify whether the reweighted null's coarse-architecture assumption holds, using
    the empirically calibrated envelope from validate_architecture_scale.py (size
    control only once ℓ̂ ≳ _ARCH_MIN_SCALE_FACTOR × bandwidth):
      ok         ℓ̂ ≥ 2·bandwidth     — size-controlled regime; 'robust' is trustworthy
      caution    bandwidth ≤ ℓ̂ < 2·bandwidth — boundary; treat 'robust' with care
      dense_tissue ℓ̂ < bandwidth     — architecture near/inside the interaction band;
                                        use dense-mode handling or fail closed
    Returns {"scale_um", "status", "ok", "bandwidth_um", "min_ok_scale_um"};
    status="unknown" only when the caller cannot estimate the scale.
    """
    min_ok = _ARCH_MIN_SCALE_FACTOR * bandwidth_um
    if scale_um is None:
        return {"scale_um": None, "status": "unknown", "ok": None,
                "bandwidth_um": bandwidth_um, "min_ok_scale_um": min_ok}
    if scale_um >= min_ok:
        status = "ok"
    elif scale_um >= bandwidth_um:
        status = "caution"
    else:
        status = "dense_tissue"
    return {"scale_um": round(float(scale_um), 1), "status": status,
            "ok": status == "ok", "bandwidth_um": bandwidth_um,
            "min_ok_scale_um": min_ok}
