"""
validate_null_models.py — prove the three null models behave correctly, and in
particular that the structure-preserving nulls fix the homogeneous-CSR bias.

Constructed point patterns (no images), all in a square window, run through
spatial_stats.cross_k_all_nulls:

  1. SHARED PREFERENCE (decisive): A and B each cluster INDEPENDENTLY in the same
     sub-region (shared tissue preference, NO cross-attraction). Expected:
       • homogeneous CSR  -> FALSELY reports association
       • inhomogeneous K  -> correctly NOT significant
       • toroidal shift   -> correctly NOT significant
     This is the test that proves the structure-preserving nulls remove the
     first-order (shared-preference) artifact.

  2. GENUINE ATTRACTION: B sits tightly next to A (real cross-interaction).
     Expected: ALL THREE nulls report association.

  3. SEGREGATION: B actively avoids A's neighbourhood. Expected: ALL THREE nulls
     report segregation.

Exits non-zero if any expectation fails.
"""

import os, sys
import numpy as np
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_stats import cross_k_all_nulls   # noqa: E402

PIX   = 1.0                       # 1 µm/px → DCLF band 10–50 µm == 10–50 px
RADII = np.arange(0.0, 100.0, 4.0)
NPERM = 199
SEED  = 0
WIN   = 1000.0


def _dir(summary):
    g = summary.get("global", {})
    return (g.get("significant"), g.get("direction"), g.get("global_p_dclf"))


def run(name, A, B, window, expect):
    # Explicitly request the (now-retired) three nulls — this script is the record
    # of WHY they were dropped (homogeneous + the two structure-preserving nulls).
    res = cross_k_all_nulls(A, B, RADII, window.area, PIX,
                            n_perm=NPERM, seed=SEED, tissue_polygon=window,
                            nulls=("homogeneous", "inhomogeneous", "toroidal"))
    nulls = res["nulls"]
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    rows = {}
    for nm in ("homogeneous", "inhomogeneous", "toroidal"):
        sig, direction, p = _dir(nulls[nm])
        rows[nm] = (sig, direction)
        extra = ""
        if nm == "inhomogeneous":
            extra = f"  [bw={nulls[nm]['bandwidth_um']}µm]"
        print(f"  {nm:14s}: significant={str(bool(sig)):5s} "
              f"direction={direction:11s} p={p}{extra}")
    print(f"  robustness verdict: {res['robustness']['verdict']}")
    print(f"    {res['robustness']['summary']}")
    # bandwidth sensitivity for the inhomogeneous null
    sens = nulls["inhomogeneous"].get("bandwidth_sensitivity", {})
    sline = "  inhom bandwidth sensitivity: " + "  ".join(
        f"{k}({v['bandwidth_um']}µm)->{'sig-'+str(v['global'].get('direction')) if v['global'].get('significant') else 'n.s.'}"
        for k, v in sens.items())
    print(sline)

    ok = True
    for nm, (want_sig, want_dir) in expect.items():
        got_sig, got_dir = rows[nm]
        cond = (bool(got_sig) == want_sig) and (not want_sig or got_dir == want_dir)
        ok = ok and cond
        if not cond:
            print(f"  ✗ {nm}: expected sig={want_sig} dir={want_dir}, "
                  f"got sig={bool(got_sig)} dir={got_dir}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    rng = np.random.default_rng(1)
    results = []

    # ── 1. SHARED PREFERENCE — A and B independently concentrate in the SAME set
    # of tissue compartments (e.g. inflamed-stroma islands distributed through the
    # tissue), with NO cross-attraction. We model the compartments as a periodic
    # grid of Gaussian hotspots over a window whose size is an exact multiple of the
    # grid period — i.e. an (approximately) STATIONARY inhomogeneity, which is what
    # the toroidal shift assumes and what distributed tissue compartments look like.
    #   • homogeneous CSR  spreads B over the whole window, ignoring the islands ->
    #     observed (both in islands) sits far above the null -> FALSE association.
    #   • inhomogeneous K  resamples B from its island intensity -> matches observed.
    #   • toroidal shift   re-aligns B's islands onto A's islands for a good fraction
    #     of shifts (periodic field) -> wide envelope -> observed not extreme.
    # The compartments are COARSER than the 10–50 µm test band (σ = 70 µm > 50 µm):
    # real tissue compartments (stroma regions) vary at scales larger than the cell–
    # cell interaction scale, so a bandwidth inside the band can preserve the
    # compartment intensity without copying any within-band interaction.
    WIN1 = 960.0                          # 3 × 320 grid period
    centers = np.array([(cx, cy)
                        for cx in (160, 480, 800)
                        for cy in (160, 480, 800)], dtype=float)
    def _from_field(n):
        idx = rng.integers(0, len(centers), n)
        return np.clip(centers[idx] + rng.normal(0, 70.0, (n, 2)), 1, WIN1 - 1)
    A1 = _from_field(300)
    B1 = _from_field(300)                  # independent draw from the SAME field
    win1 = box(0, 0, WIN1, WIN1)
    results.append(run(
        "1. SHARED PREFERENCE (same distributed compartments, no cross-attraction)",
        A1, B1, win1,
        expect={"homogeneous":  (True,  "association"),
                "inhomogeneous": (False, None),
                "toroidal":      (False, None)}))

    win = box(0, 0, WIN, WIN)

    # ── 2. GENUINE ATTRACTION — B tightly next to A across the whole window.
    A2 = rng.uniform(0, WIN, (220, 2))
    jitter = rng.normal(0, 7.0, A2.shape)
    B2 = np.clip(A2 + jitter, 1, WIN - 1)
    results.append(run(
        "2. GENUINE CROSS-ATTRACTION (B beside A)",
        A2, B2, win,
        expect={"homogeneous":   (True, "association"),
                "inhomogeneous": (True, "association"),
                "toroidal":      (True, "association")}))

    # ── 3. SEGREGATION — B uniform but excluded from a 35 px halo around any A.
    A3 = rng.uniform(0, WIN, (200, 2))
    from scipy.spatial import cKDTree
    tA = cKDTree(A3)
    keep = []
    trial = rng.uniform(0, WIN, (6000, 2))
    d, _ = tA.query(trial, k=1)
    for p in trial[d > 35.0]:
        keep.append(p)
        if len(keep) >= 220:
            break
    B3 = np.array(keep)
    results.append(run(
        "3. SEGREGATION (B avoids A's neighbourhood)",
        A3, B3, win,
        expect={"homogeneous":   (True, "segregation"),
                "inhomogeneous": (True, "segregation"),
                "toroidal":      (True, "segregation")}))

    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    if all(results):
        print("  ALL SCENARIOS PASS.")
        print("  Decisive result: under shared tissue preference, homogeneous CSR "
              "false-positives\n  while inhomogeneous K and toroidal shift correctly "
              "report no association.")
        return 0
    print(f"  FAILED: {sum(1 for r in results if not r)}/{len(results)} scenarios.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
