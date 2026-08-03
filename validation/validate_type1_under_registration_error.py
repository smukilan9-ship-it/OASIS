"""Does registration error INVENT an association? Type I error vs eps, on all 140 CODEX spots.

WHY THIS EXISTS. Every existing sweep that injects registration error imposes a real
association first and measures POWER — validate_detectable_effect, validate_radius_floor,
validate_radius_floor_calibration, validate_radius_floor_localisation. All of them answer
"would we MISS a real effect?" (Type II). None answers "would we FIND one that is not
there?" (Type I).

That asymmetry matters more than it looks. A power loss makes the tool report "no
association" — conservative, and the MDE curve already tells a reader how weak an effect
would have been missed. A size inflation makes it report an association that does not
exist, which is a wrong published finding and the exact failure a fail-closed gate is sold
to prevent. The claim currently carrying that load is a one-line assertion in
validate_detectable_effect.py — "error cannot INVENT a finding (validated separately)" —
whose only support is tests/test_degradation.py, at ONE CODEX field and ONE error level of
1.89 um.

THE ARGUMENT FOR IT IS NOT AIRTIGHT, which is why measuring is worth the compute. "Smearing
moves the observed curve toward the null" holds for iid jitter. It is not obvious for the
two error models that actually occur:

  * a real misregistration is SYSTEMATIC (a residual rotation/translation), not jitter. Both
    markers track tissue architecture, so rotating B relative to A can pair B's structure
    against a DIFFERENT part of A's rather than blurring it;
  * _displace_smooth is a CORRELATED field by construction, which moves neighbouring points
    together and can preserve structure instead of destroying it.

THE SUBSTRATE. Schurch et al. 2020 CRC CODEX: 258,385 cells, 140 spots, 29 phenotyped cell
types, all on ONE physical section — so the cross-type truth is observable without any
registration. Coordinates are PIXELS at the published nominal 0.3775 um/px (the same
constant validate_real_data_production.py uses); passing 1.0 instead, as the keystone test
does, makes a "5 px" shift 1.89 um rather than 5 um and leaves the 10-50 um DCLF band only
partly covered by a 0-100 px radius grid. Both are fixed here.

THREE ARMS.
  permuted  cell-type labels shuffled within the spot: null BY CONSTRUCTION, preserving the
            marginal counts and the tissue's own cell layout. This is the clean Type I
            substrate — no circularity, because the truth does not come from the statistic.
  real/H0   real pairs the statistic calls not-associated at eps=0. Type I on real structure.
  real/H1   real pairs it calls associated at eps=0. Type II at 140x the scale of the
            existing two-substrate MDE sweep, as a cross-check on it.

Run:  python validation/validate_type1_under_registration_error.py
      python validation/validate_type1_under_registration_error.py --quick
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import cross_k_all_nulls, _BAND_STATISTIC   # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.3775        # published nominal CODEX resolution (validate_real_data_production.py)
RADII_PX = np.arange(0.0, 161.0, 2.0)   # 0-60.4 um: covers the whole 10-50 um DCLF band
MIN_CELLS = 40                # per type, per spot — below this the pair is not measurable
EPS_UM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
MODELS = ("translation", "rotation", "smooth")
ALPHA = 0.05

# matched to validate_radius_floor_localisation so "smooth" is the SAME error model
CORR_LENGTH_UM = 200.0
NUGGET_FRAC = 0.021

PAIRS = (
    ("CD8+ T cells", "CD4+ T cells CD45RO+"),
    ("CD8+ T cells", "tumor cells"),
    ("CD8+ T cells", "CD68+CD163+ macrophages"),
    ("tumor cells", "vasculature"),
    ("stroma", "smooth muscle"),
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSONL = os.path.join(HERE, "type1_under_registration_error_records.jsonl")
OUT_JSON = os.path.join(HERE, "type1_under_registration_error_results.json")


# ── error models ─────────────────────────────────────────────────────────────────────────
def _translate(pts, eps_um, rng, bounds):
    """Pure offset of magnitude eps in a random direction — the simplest real misregistration."""
    th = rng.uniform(0, 2 * np.pi)
    d = (eps_um / PIXEL_SIZE_UM) * np.array([np.cos(th), np.sin(th)])
    return pts + d


def _rotate(pts, eps_um, rng, bounds):
    """Rotation about the field centre, calibrated so the MEDIAN induced displacement is eps.

    A point at radius r from the centre moves 2*r*sin(theta/2), so fixing the median
    displacement fixes theta against the spot's own geometry rather than against an
    arbitrary angle — which is what makes the eps axis comparable across the three models.
    """
    c = np.array([(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0])
    r_med = float(np.median(np.linalg.norm(pts - c, axis=1)))
    if r_med <= 0:
        return pts
    want = eps_um / PIXEL_SIZE_UM
    s = np.clip(want / (2.0 * r_med), -1.0, 1.0)
    th = 2.0 * np.arcsin(s) * (1.0 if rng.random() < 0.5 else -1.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ (pts - c).T).T + c


def _smooth(pts, eps_um, rng, bounds):
    """Spatially-correlated warp, same construction as validate_radius_floor_localisation
    (_displace_smooth) but generalised off that module's fixed 2000 px square so it can run
    on a CODEX spot of any size. Coarse iid field -> smoothed to CORR_LENGTH_UM -> bilinear
    sample -> NUGGET_FRAC uncorrelated -> scaled so RMS displacement MAGNITUDE equals eps.
    """
    g = 64
    x0, y0, x1, y1 = bounds
    side_px = max(x1 - x0, y1 - y0, 1.0)
    sigma_cells = (CORR_LENGTH_UM / PIXEL_SIZE_UM) / (side_px / g)
    field = rng.normal(0, 1, (g, g, 2))
    for c in range(2):
        field[..., c] = gaussian_filter(field[..., c], max(sigma_cells, 1e-6), mode="wrap")

    uv = np.clip((pts - np.array([x0, y0])) / side_px, 0, 1 - 1e-9) * (g - 1)
    i0, j0 = np.floor(uv[:, 1]).astype(int), np.floor(uv[:, 0]).astype(int)
    fi, fj = uv[:, 1] - i0, uv[:, 0] - j0
    i1, j1 = np.minimum(i0 + 1, g - 1), np.minimum(j0 + 1, g - 1)
    d = (field[i0, j0] * ((1 - fi) * (1 - fj))[:, None] +
         field[i0, j1] * ((1 - fi) * fj)[:, None] +
         field[i1, j0] * (fi * (1 - fj))[:, None] +
         field[i1, j1] * (fi * fj)[:, None])
    sd = d.std() if d.std() > 0 else 1.0
    d = d + np.sqrt(NUGGET_FRAC / max(1.0 - NUGGET_FRAC, 1e-9)) * rng.normal(0, sd, d.shape)
    rms = float(np.sqrt((d ** 2).sum(1).mean()))
    if rms > 0:
        d *= (eps_um / PIXEL_SIZE_UM) / rms
    return pts + d


INJECT = {"translation": _translate, "rotation": _rotate, "smooth": _smooth}


# ── data ─────────────────────────────────────────────────────────────────────────────────
def load_spots(path):
    """{spot: {"xy": Nx2 px, "types": np.array[str], "bounds": (x0,y0,x1,y1)}}."""
    by = collections.defaultdict(lambda: {"x": [], "y": [], "t": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            d = by[r["spots"]]
            d["x"].append(float(r["X:X"]))
            d["y"].append(float(r["Y:Y"]))
            d["t"].append(r["ClusterName"])
    out = {}
    for s, d in by.items():
        xy = np.column_stack([d["x"], d["y"]])
        out[s] = {"xy": xy, "types": np.asarray(d["t"]),
                  "bounds": (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())}
    return out


def _area_px2(bounds):
    return float((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))


# ── one measurement ──────────────────────────────────────────────────────────────────────
def one(job):
    (spot, ta, tb, arm, model, eps, seed, n_perm,
     xy, types, bounds, xy2, types2, bounds2) = job
    rng = np.random.default_rng(seed)

    if arm == "crossspot":
        # A and B are the REAL cell types, each keeping its own clustered structure, but
        # taken from DIFFERENT spots — so there is no true relationship to find while the
        # marginals stay realistic. Label permutation cannot do this: it replaces each type
        # with a random subset of all cells, which destroys the clustering too, and so tests
        # calibration on a simpler pattern than any real one.
        A = xy[types == ta]
        B = xy2[types2 == tb]
        # common window: both bounding boxes moved to the origin, intersected
        A = A - np.array([bounds[0], bounds[1]])
        B = B - np.array([bounds2[0], bounds2[1]])
        w = min(bounds[2] - bounds[0], bounds2[2] - bounds2[0])
        h = min(bounds[3] - bounds[1], bounds2[3] - bounds2[1])
        A = A[(A[:, 0] <= w) & (A[:, 1] <= h)]
        B = B[(B[:, 0] <= w) & (B[:, 1] <= h)]
        bounds = (0.0, 0.0, w, h)
    else:
        lab = types
        if arm == "permuted":
            lab = rng.permutation(types)      # destroys type-type association, keeps layout
        A = xy[lab == ta]
        B = xy[lab == tb]

    if len(A) < MIN_CELLS or len(B) < MIN_CELLS:
        return None
    if eps > 0:
        B = INJECT[model](B, eps, np.random.default_rng(seed + 77_000), bounds)
    try:
        r = cross_k_all_nulls(A, B, RADII_PX, _area_px2(bounds), PIXEL_SIZE_UM,
                              n_perm=n_perm, seed=seed)
    except Exception as e:                    # a spot that cannot be measured is not evidence
        return {"spot": spot, "a": ta, "b": tb, "arm": arm, "model": model,
                "eps_um": eps, "error": f"{type(e).__name__}: {e}"}
    band = r["nulls"]["reweighted"][_BAND_STATISTIC]
    rob = r["robustness"]
    return {"spot": spot, "a": ta, "b": tb, "arm": arm, "model": model, "eps_um": eps,
            "n_a": int(len(A)), "n_b": int(len(B)),
            "verdict": rob["verdict"],
            "direction": rob.get("direction"),
            "p_primary": (rob.get("per_null_global_p") or {}).get("reweighted"),
            "p_coloc": band["colocalization"]["global_p_dclf"],
            "coloc_sig": bool(band["colocalization"]["significant"]),
            "coloc_dir": band["colocalization"]["direction"],
            "p_coinf": band["coinfiltration"]["global_p_dclf"]}


def build_jobs(spots, n_perm, quick):
    jobs = []
    names = sorted(spots)
    if quick:
        names = names[:12]
    ARM_SEED = {"real": 0, "permuted": 7, "crossspot": 13}
    for si, s in enumerate(names):
        d = spots[s]
        d2 = spots[names[(si + 1) % len(names)]]        # partner spot for the crossspot arm
        cnt, cnt2 = collections.Counter(d["types"]), collections.Counter(d2["types"])
        for pi, (ta, tb) in enumerate(PAIRS):
            for arm in ("real", "permuted", "crossspot"):
                have_b = cnt2[tb] if arm == "crossspot" else cnt[tb]
                if cnt[ta] < MIN_CELLS or have_b < MIN_CELLS:
                    continue
                base = 1_000_000 + si * 977 + pi * 31 + ARM_SEED[arm]
                payload = (d["xy"], d["types"], d["bounds"],
                           d2["xy"], d2["types"], d2["bounds"])
                # eps=0 is identical across models — measure it once, label it "none"
                jobs.append((s, ta, tb, arm, "none", 0.0, base, n_perm) + payload)
                for model in MODELS:
                    for eps in EPS_UM:
                        if eps == 0.0:
                            continue
                        jobs.append((s, ta, tb, arm, model, eps, base, n_perm) + payload)
    return jobs


# ── analysis ─────────────────────────────────────────────────────────────────────────────
def summarise(recs):
    """Type I / Type II rates vs eps, with Wilson 95% intervals."""
    def wilson(k, n, z=1.96):
        if n == 0:
            return (None, None)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (round(max(c - h, 0.0), 4), round(min(c + h, 1.0), 4))

    ok = [r for r in recs if "verdict" in r]
    # truth at eps=0, per (spot, pair, arm)
    truth = {(r["spot"], r["a"], r["b"], r["arm"]): r["verdict"]
             for r in ok if r["eps_um"] == 0.0}

    out = {"n_records": len(recs), "n_ok": len(ok), "n_errors": len(recs) - len(ok),
           "alpha": ALPHA, "pixel_size_um": PIXEL_SIZE_UM, "arms": {}}

    for arm_key, sel in (
            ("permuted_constructed_null",
             lambda r: r["arm"] == "permuted"),
            ("real_null_at_eps0",
             lambda r: r["arm"] == "real"
             and truth.get((r["spot"], r["a"], r["b"], "real")) not in (None, "robust")),
            ("real_associated_at_eps0",
             lambda r: r["arm"] == "real"
             and truth.get((r["spot"], r["a"], r["b"], "real")) == "robust")):
        arm = {}
        for model in ("none",) + MODELS:
            for eps in EPS_UM:
                if (model == "none") != (eps == 0.0):
                    continue
                sub = [r for r in ok if sel(r) and r["model"] == model and r["eps_um"] == eps]
                if not sub:
                    continue
                n = len(sub)
                rob = sum(1 for r in sub if r["verdict"] == "robust")
                sig = sum(1 for r in sub
                          if r["p_coloc"] is not None and r["p_coloc"] <= ALPHA)
                agg = sum(1 for r in sub if r["verdict"] == "robust"
                          and r.get("direction") == "aggregation")
                key = f"{model}@{eps:g}"
                arm[key] = {
                    "n": n,
                    "frac_robust": round(rob / n, 4),
                    "frac_robust_ci95": wilson(rob, n),
                    "frac_robust_aggregation": round(agg / n, 4),
                    "frac_p_coloc_sig": round(sig / n, 4),
                    "frac_p_coloc_sig_ci95": wilson(sig, n),
                }
        out["arms"][arm_key] = arm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="12 spots, fewer permutations")
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    path = a.csv
    if path is None:
        from validation.datasets import resolve as R
        path = R.resolve("codex_crc")
    if path is None or not os.path.exists(str(path)):
        print("dataset 'codex_crc' not found", file=sys.stderr)
        return 2

    n_perm = 99 if a.quick else a.n_perm
    print(f"loading {path} ...", flush=True)
    spots = load_spots(str(path))
    jobs = build_jobs(spots, n_perm, a.quick)
    print(f"{len(spots)} spots · {len(PAIRS)} pairs · {len(jobs)} runs · "
          f"n_perm={n_perm} · {a.workers} workers", flush=True)

    t0 = time.time()
    recs = []
    with open(OUT_JSONL, "w") as fh, mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            if r is None:
                continue
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
            if i % 250 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el/60:.1f} min  "
                      f"eta {(el/i)*(len(jobs)-i)/60:.1f} min", flush=True)

    res = summarise(recs)
    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    res["n_perm"] = n_perm
    res["pairs"] = [list(p) for p in PAIRS]
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nwrote {OUT_JSON}  ({res['elapsed_min']} min, "
          f"{res['n_ok']} usable / {res['n_records']} records)")
    for arm, tab in res["arms"].items():
        print(f"\n{arm}")
        print(f"  {'condition':>18}  {'n':>5}  {'frac robust':>12}  {'p_coloc<=.05':>13}")
        for k, v in tab.items():
            print(f"  {k:>18}  {v['n']:>5}  {v['frac_robust']:>12.3f}  "
                  f"{v['frac_p_coloc_sig']:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
