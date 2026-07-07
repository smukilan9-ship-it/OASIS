"""
validate_real_data.py — run spatial_stats cross-K on REAL CODEX cells, under the
exact conditions the production Spatial Association pipeline uses.

Dataset: Schurch et al. 2020 Cell, CRC CODEX single-cell table
(Mendeley 10.17632/mpjzbtfgfr.1, CC BY 4.0) — CRC_clusters_neighborhoods_markers.csv
(258,385 cells; 140 TMA spots; each spot an independent ~1920x1440 px tissue core).
We use the authors' own validated cell-type labels (ClusterName) and per-cell X/Y.

Three cross-type controls, each a known spatial relationship, run per qualifying
spot and aggregated:
  1. CD8+ vs CD4+ T cells   — known POSITIVE (T cells co-infiltrate) -> association
  2. CD8+ vs Tregs (FOXP3+) — informational (TIM-3 unavailable; immune marker)
  3. CD8+ vs tumor cells    — known NEGATIVE (compartment segregation)

────────────────────────────────────────────────────────────────────────────────
CONDITIONS — matched to the production pipeline (spatial.run_spatial_association)
────────────────────────────────────────────────────────────────────────────────
• PIXEL SIZE. CODEX X:X / Y:Y are in pixels. The Schurch CRC CODEX images were
  acquired on a Keyence microscope with a 20x objective; the published nominal
  resolution is ~0.377 µm/px. We use PIXEL_SIZE_UM = 0.3775 µm/px (stated, not
  derived from the table — the coordinate table carries no calibration field).
  This makes the pipeline's DCLF band genuinely 10–50 µm  (≈ 26.5–132.5 px),
  NOT 10–50 raw pixels. The earlier run set pixel_size=1.0, which silently
  collapsed the band to 10–50 px (≈ 3.8–18.9 µm) — squarely inside the cell-
  diameter hard-core zone (see the legacy artifact section at the bottom).

• RADII. radii_um = arange(0, MAX_RADIUS_UM + RADIUS_STEP_UM, RADIUS_STEP_UM),
  radii_px = radii_um / PIXEL_SIZE_UM — identical to spatial.run_spatial_association
  with its defaults (max_radius_um=100, radius_step_um=2).

• NULL / TEST. cross_k_null with n_perm=1000, seed=0, and the DCLF global call on
  the [10, 50] µm band — the production defaults. The per-spot verdict is the
  pipeline's own res["global"] (direction-resolved DCLF), not a per-radius OR.

────────────────────────────────────────────────────────────────────────────────
DELIBERATE, DOCUMENTED DIFFERENCE FROM THE PIPELINE — the tissue mask
────────────────────────────────────────────────────────────────────────────────
The production pipeline derives the tissue mask from the CD8 *image* via Otsu
thresholding (estimate_tissue_mask). CODEX gives cell COORDINATES directly: there
is no brightfield image to Otsu-threshold here. We therefore bound the area and
the CSR null with the CONVEX HULL of all cells in the spot. This is a conscious
substitution, not the pipeline's image front end:

  - What we are validating is the STATISTIC (cross-type Ripley's K / L−r + the
    Monte-Carlo CSR null + the DCLF global call) on REAL point patterns whose
    spatial relationship is biologically known. The Otsu step is an image-
    processing stage that produces a mask polygon; once a polygon exists, the
    statistic treats it identically regardless of how it was obtained.
  - The convex hull is a defensible tissue boundary for a densely-sampled TMA
    core (cells tile the core), and crucially it is the SAME window for the
    observed pattern and every null replicate, so the edge/area normalization
    cancels in significance exactly as it does with an Otsu polygon.
  - It is NOT claimed to reproduce the Otsu mask numerically; it is the
    coordinate-data analogue of "a tissue polygon bounding the null".

No pipeline code is modified.
"""

import os, sys, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_stats import cross_k_null   # noqa: E402

CSV = os.path.join(os.path.dirname(__file__), "CRC_clusters_neighborhoods_markers.csv")
OUT = os.path.join(os.path.dirname(__file__), "real_data_validation_output.txt")

CD8   = {"CD8+ T cells"}
CD4   = {"CD4+ T cells CD45RO+", "CD4+ T cells", "CD4+ T cells GATA3+"}
TREG  = {"Tregs"}
TUMOR = {"tumor cells"}

# Published nominal CODEX resolution for the Schurch CRC dataset (Keyence 20x).
# Stated value, not recoverable from the coordinate table — see module docstring.
PIXEL_SIZE_UM = 0.3775

# Pipeline-default evaluation radii (µm), converted to px exactly as spatial.py does.
MAX_RADIUS_UM  = 100.0
RADIUS_STEP_UM = 2.0
RADII_UM = np.arange(0.0, MAX_RADIUS_UM + RADIUS_STEP_UM, RADIUS_STEP_UM)
RADII_PX = RADII_UM / PIXEL_SIZE_UM

N_PERM = 1000          # pipeline default (spatial.N_PERMUTATIONS)
SEED   = 0             # pipeline default (spatial._NULL_SEED)
MIN_N  = 30            # a spot qualifies when it has >= MIN_N of EACH population
# No spot cap: every qualifying spot is run and reported (true N).

# µm radii at which to summarize the mean observed L−r curve (all inside the band).
LMR_REPORT_UM = [15.0, 30.0, 50.0]

_log_lines = []
def log(s=""):
    print(s)
    _log_lines.append(s)


def hull_mask(all_xy):
    from shapely.geometry import MultiPoint
    hull = MultiPoint([tuple(p) for p in all_xy]).convex_hull
    return hull, float(hull.area)


def qualifying_spots(grouped, set_b):
    """All (patient, spot) keys with >= MIN_N CD8+ and >= MIN_N of set_b."""
    keys = []
    for key, gdf in grouped:
        a = gdf["ClusterName"].isin(CD8).sum()
        b = gdf["ClusterName"].isin(set_b).sum()
        if a >= MIN_N and b >= MIN_N:
            keys.append(key)
    return keys


def run_comparison(grouped, set_b, name, expect, pixel_size_um=PIXEL_SIZE_UM,
                   band_label="10–50 µm"):
    """
    Run cross_k_null on every qualifying spot and aggregate the pipeline's own
    direction-resolved global DCLF verdict. `pixel_size_um` is parameterized so
    the same code path can reproduce the legacy (buggy) pixel-size condition.
    """
    keys = qualifying_spots(grouped, set_b)
    radii_px = RADII_UM / pixel_size_um

    # Per-spot global DCLF outcomes (the pipeline's actual significance call).
    n_assoc = n_seg = n_sig = 0          # two-sided significant + its direction
    p_assoc_lt = p_seg_lt = 0            # one-sided p < 0.05 (each direction)
    p_dclf_all = []
    lmr_curves = []                      # observed L−r (µm) per spot, for the mean curve
    # Per-radius band diagnostic (kept only to expose the hard-core artifact):
    perr_seg = 0                         # spots dipping below null lower env in band

    for key in keys:
        gdf = grouped.get_group(key)
        all_xy = gdf[["X:X", "Y:Y"]].to_numpy(float)
        A = gdf[gdf["ClusterName"].isin(CD8)][["X:X", "Y:Y"]].to_numpy(float)
        B = gdf[gdf["ClusterName"].isin(set_b)][["X:X", "Y:Y"]].to_numpy(float)
        hull, area = hull_mask(all_xy)
        res = cross_k_null(A, B, radii_px, area, pixel_size_um,
                           n_perm=N_PERM, seed=SEED, tissue_polygon=hull)
        g = res["global"]
        p_dclf_all.append(g["global_p_dclf"])
        if g["global_p_association"] < 0.05:
            p_assoc_lt += 1
        if g["global_p_segregation"] < 0.05:
            p_seg_lt += 1
        if g["significant"]:
            n_sig += 1
            if g["direction"] == "association":
                n_assoc += 1
            elif g["direction"] == "segregation":
                n_seg += 1

        lmr = np.array(res["L_minus_r"])          # µm
        lmr_curves.append(lmr)
        lo  = np.array(res["null_lower_L"])        # µm
        radii_um_eff = radii_px * pixel_size_um
        band = (radii_um_eff >= g["dclf_rmin_um"]) & (radii_um_eff <= g["dclf_rmax_um"])
        if np.any((lmr < lo) & band):
            perr_seg += 1

    n = len(keys)
    mean_curve = np.mean(np.vstack(lmr_curves), axis=0)
    radii_um_eff = radii_px * pixel_size_um
    lmr_at = {}
    for r in LMR_REPORT_UM:
        j = int(np.argmin(np.abs(radii_um_eff - r)))
        lmr_at[r] = mean_curve[j]

    log(f"\n{'='*74}\n{name}\n  expect: {expect}\n{'='*74}")
    log(f"  pixel size            : {pixel_size_um:.4f} µm/px")
    log(f"  DCLF band             : {band_label}  (@ {pixel_size_um:.4f} µm/px)")
    log(f"  qualifying spots (N)  : {n}   (>= {MIN_N} CD8+ and >= {MIN_N} of target)")
    log(f"  mean observed L−r (µm): " +
        "   ".join(f"r={int(r)}µm -> {lmr_at[r]:+6.2f}" for r in LMR_REPORT_UM))
    log(f"  global DCLF p (median): {np.median(p_dclf_all):.4f}")
    log(f"  GLOBAL DCLF verdict (pipeline's actual call):")
    log(f"     significant (two-sided p<0.05) : {n_sig}/{n} ({100*n_sig/n:.0f}%)")
    log(f"        ├─ direction = ASSOCIATION   : {n_assoc}/{n} ({100*n_assoc/n:.0f}%)")
    log(f"        └─ direction = SEGREGATION   : {n_seg}/{n} ({100*n_seg/n:.0f}%)")
    log(f"  one-sided p<0.05  association      : {p_assoc_lt}/{n} ({100*p_assoc_lt/n:.0f}%)")
    log(f"  one-sided p<0.05  segregation      : {p_seg_lt}/{n} ({100*p_seg_lt/n:.0f}%)")
    log(f"  [diag] per-radius dip below null env in band: "
        f"{perr_seg}/{n} ({100*perr_seg/n:.0f}%)  (NOT the pipeline call)")
    return dict(name=name, n=n, n_assoc=n_assoc, n_seg=n_seg, n_sig=n_sig,
                p_assoc_lt=p_assoc_lt, p_seg_lt=p_seg_lt,
                perr_seg=perr_seg, lmr_at=lmr_at)


def reproduce_original_buggy_metric(grouped):
    """
    Faithful reproduction of the PREVIOUS validation script's exact reported
    metric, to ground the numbers the earlier report asserted. The old script:
      • set pixel_size_um = 1.0  (so 'µm' radii were really CODEX pixels)
      • radii = arange(0, 168, 8) px, band = radii <= 130 px (includes r→0)
      • capped at the 35 spots with the most cells
      • counted SEGREGATION as: observed L−r dips below the null lower envelope
        at ANY single radius in the band (a per-radius OR), and POSITIVE as:
        observed L−r above the upper envelope with per-radius p<0.05 at any radius.
    Because the band starts at r=0 px, the sub-cell-diameter hard-core zone
    (≈ ≤26 CODEX px) is inside it, so the per-radius OR flags 'segregation' on a
    biologically POSITIVE control. The two counts are reported TOGETHER here to
    show the mixed-sign behavior the earlier one-line summary omitted.
    """
    RADII = np.arange(0.0, 168.0, 8.0)
    BAND  = RADII <= 130.0
    avail = []
    for key, gdf in grouped:
        a = gdf["ClusterName"].isin(CD8).sum()
        b = gdf["ClusterName"].isin(CD4).sum()
        if a >= MIN_N and b >= MIN_N:
            avail.append((min(a, b), key))
    avail.sort(reverse=True)
    spots = [k for _, k in avail[:35]]
    neg = pos = 0
    for key in spots:
        gdf = grouped.get_group(key)
        all_xy = gdf[["X:X", "Y:Y"]].to_numpy(float)
        A = gdf[gdf["ClusterName"].isin(CD8)][["X:X", "Y:Y"]].to_numpy(float)
        B = gdf[gdf["ClusterName"].isin(CD4)][["X:X", "Y:Y"]].to_numpy(float)
        hull, area = hull_mask(all_xy)
        res = cross_k_null(A, B, RADII, area, 1.0, n_perm=149, seed=SEED,
                           tissue_polygon=hull)
        lmr = np.array(res["L_minus_r"]); lo = np.array(res["null_lower_L"])
        hi  = np.array(res["null_upper_L"]); pv = np.array(res["p_values"])
        if np.any((lmr < lo) & BAND):
            neg += 1
        if np.any((lmr > hi) & (pv < 0.05) & BAND):
            pos += 1
    n = len(spots)
    log(f"  reproduced OLD per-radius metric (35 spots, px=1.0, band r<=130px):")
    log(f"     'positive association' flagged : {pos}/{n} ({100*pos/n:.0f}%)  "
        f"<- the figure the old report headlined")
    log(f"     'segregation' flagged (SAME spots, omitted before): "
        f"{neg}/{n} ({100*neg/n:.0f}%)")
    log(f"     => the old metric was mixed-sign: it called BOTH on a positive "
        f"control, because r=0–26 px hard-core leaked into the band.")
    return dict(pos=pos, neg=neg, n=n)


if __name__ == "__main__":
    log("Loading real CODEX single-cell table…")
    df = pd.read_csv(CSV, usecols=["patients", "spots", "ClusterName", "X:X", "Y:Y"],
                     low_memory=False)
    grouped = df.groupby(["patients", "spots"])
    log(f"  {len(df):,} cells, {grouped.ngroups} spots")
    log(f"  pixel size {PIXEL_SIZE_UM} µm/px (published nominal CODEX 20x); "
        f"radii 0–{MAX_RADIUS_UM:.0f} µm step {RADIUS_STEP_UM:.0f} µm; "
        f"n_perm={N_PERM}; DCLF band 10–50 µm")

    r1 = run_comparison(grouped, CD4,   "TEST 1  CD8+  vs  CD4+ T cells",
                        "POSITIVE association (co-infiltrating T cells)")
    r2 = run_comparison(grouped, TREG,  "TEST 2  CD8+  vs  Tregs (FOXP3+)",
                        "POSITIVE-ish (co-infiltrating immune; informational)")
    r3 = run_comparison(grouped, TUMOR, "TEST 3  CD8+  vs  tumor cells",
                        "SEGREGATION (immune/tumor compartments separate)")

    # ── Legacy artifact reproduction ───────────────────────────────────────────
    # Re-run the known-POSITIVE CD8–CD4 control with the previous run's broken
    # pixel size (1.0). With s=1.0 the DCLF 10–50 µm band lands at 10–50 PIXELS
    # (≈ 3.8–18.9 µm), inside the cell-diameter hard-core zone where two centroids
    # physically cannot coincide and L−r is driven negative for non-biological
    # reasons. This shows WHY the earlier run produced spurious "segregation" on a
    # control that is biologically positive.
    log(f"\n\n{'#'*74}\n# LEGACY ARTIFACT REPRODUCTION — buggy pixel_size_um = 1.0\n"
        f"# (DCLF 10–50 µm band collapses to 10–50 px ≈ 3.8–18.9 µm = hard-core zone)\n"
        f"{'#'*74}")
    rL = run_comparison(grouped, CD4,
                        "LEGACY  CD8+ vs CD4+  (buggy px=1.0; same statistic)",
                        "should still be POSITIVE — any segregation here is artifact",
                        pixel_size_um=1.0, band_label="10–50 px")
    log("")
    rOld = reproduce_original_buggy_metric(grouped)

    # ── Verdict ────────────────────────────────────────────────────────────────
    log(f"\n{'='*74}\nVERDICT (corrected conditions, pixel size {PIXEL_SIZE_UM} µm/px)\n{'='*74}")
    ok1 = r1["lmr_at"][15.0] > 0 and r1["n_assoc"] >= r1["n_seg"]
    ok3 = r3["lmr_at"][15.0] < 0 and r3["n_seg"] >= r3["n_assoc"]
    log(f"  TEST 1 CD8–CD4 (expect association):  "
        f"{'PASS' if ok1 else 'CHECK'}  "
        f"assoc {r1['n_assoc']}/{r1['n']}, seg {r1['n_seg']}/{r1['n']}, "
        f"mean L−r@15µm={r1['lmr_at'][15.0]:+.2f}")
    log(f"  TEST 2 CD8–Treg (informational):      "
        f"assoc {r2['n_assoc']}/{r2['n']}, seg {r2['n_seg']}/{r2['n']}, "
        f"mean L−r@15µm={r2['lmr_at'][15.0]:+.2f}")
    log(f"  TEST 3 CD8–tumor (expect segregation):"
        f"{'PASS' if ok3 else ' CHECK'}  "
        f"seg {r3['n_seg']}/{r3['n']}, assoc {r3['n_assoc']}/{r3['n']}, "
        f"mean L−r@15µm={r3['lmr_at'][15.0]:+.2f}")
    log(f"\n  HARD-CORE ARTIFACT CHECK (CD8–CD4 positive control):")
    log(f"     corrected band 10–50 µm : segregation called in "
        f"{r1['n_seg']}/{r1['n']} spots  (DCLF direction)")
    log(f"     legacy band  10–50 px   : segregation called in "
        f"{rL['n_seg']}/{rL['n']} spots  (DCLF direction)")
    log(f"     -> artifact {'GONE' if r1['n_seg'] <= rL['n_seg'] and r1['n_assoc'] > r1['n_seg'] else 'PERSISTS'} "
        f"after the pixel-size fix")
    log(f"\n  Statistic reproduces known biology: "
        f"{'YES' if (ok1 and ok3) else 'PARTIAL — see per-test numbers above'}")

    with open(OUT, "w") as f:
        f.write("\n".join(_log_lines) + "\n")
    log(f"\n  (full output saved to {os.path.relpath(OUT)})")
