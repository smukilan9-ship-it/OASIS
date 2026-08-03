#!/usr/bin/env python
"""
run_full_cohort_overnight.py — certify and analyse every CD8/TIM-3 pair in the cohort.

WHY THIS EXISTS. 75 complete pairs sit under ~/Desktop/Region of interest across LL477-LL481
at 4X/10X/20X. Driving each through the GUI costs ~4 minutes, so the whole cohort is a
five-hour click session; this runs the SAME backend calls the GUI makes
(`API.certify_spatial_auto`, then `run_pipeline --mode spatial`) so the compute happens
unattended while the interface work continues in parallel.

It is deliberately not a shortcut around the UI. The GUI is still driven by hand for
bug-finding; this exists so the cohort-wide RESULTS are ready regardless of how far the
click-through gets.

DESIGN NOTES, all of them learned the hard way in this repo:
  - RESUMABLE. Every pair's certificate is written to disk the moment it is produced, and a
    re-run skips what is already there. A crash at pair 60 does not cost the first 59.
  - FAULT-TOLERANT PER PAIR. One bad image must not end the run, so every pair is wrapped and
    its traceback recorded as a result rather than raised.
  - NOTHING IS INVENTED ON FAILURE. A pair that fails to certify is recorded as uncertified
    and is NOT analysed. `require_landmark_certification` stays on, so the pipeline itself
    also refuses it — the fail-closed gate is the point of the tool and is not relaxed to make
    a batch look complete.
  - PIXEL SIZE COMES FROM MAGNIFICATION, which is in the path. 0.7519 µm/px at 10X is the
    measured value for this scanner (research/ihc.md § 12); 20X and 4X scale from it.

Run:  .venv/bin/python validation/run_full_cohort_overnight.py            (all pairs)
      .venv/bin/python validation/run_full_cohort_overnight.py --limit 5  (smoke test)
      .venv/bin/python validation/run_full_cohort_overnight.py --certify-only
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.expanduser("~/Desktop/Region of interest")
OUT = os.path.expanduser("~/Documents/OASIS/results/cohort_overnight")
CERT_DIR = os.path.join(OUT, "certificates")
PX_10X = 0.7519                      # measured for this scanner at 10X
PX = {"4X": PX_10X * 10 / 4, "10X": PX_10X, "20X": PX_10X / 2}


def discover():
    """Every directory holding one CD8 image and one TIM-3 image."""
    pairs = []
    for d, _sub, files in os.walk(ROOT):
        tifs = [f for f in files
                if f.lower().endswith((".tif", ".tiff")) and "scale" not in f.lower()]
        cd8 = sorted(f for f in tifs if re.search(r"cd8", f, re.I))
        tim = sorted(f for f in tifs if re.search(r"tim.?3", f, re.I))
        if cd8 and tim:
            mag = "20X" if "20X" in d else ("4X" if "4X" in d else "10X")
            pairs.append({"sample_id": os.path.basename(d), "mag": mag,
                          "path_a": os.path.join(d, cd8[0]),
                          "path_b": os.path.join(d, tim[0]),
                          "pixel_size_um": PX[mag]})
    pairs.sort(key=lambda p: (p["mag"], p["sample_id"]))
    return pairs


def certify_all(pairs, log):
    from oasis.webui.api import API
    api = API()
    os.makedirs(CERT_DIR, exist_ok=True)
    done = 0
    for i, p in enumerate(pairs, 1):
        cache = os.path.join(CERT_DIR, f"{p['sample_id']}__{p['mag']}.json")
        if os.path.exists(cache):                      # resumable
            p["certification"] = json.load(open(cache))
            done += 1
            continue
        t0 = time.time()
        try:
            r = api.certify_spatial_auto({
                "ref_path": p["path_a"], "mov_path": p["path_b"],
                "pixel_size_um": p["pixel_size_um"], "region_um": None, "max_regions": 9})
        except Exception as e:
            r = {"status": "error", "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-1500:]}
        regs = r.get("regions") or []
        # MIRROR THE SHAPE THE GUI SENDS (index.html, spatialCertifications[...]).
        # The pipeline only uses the certified transform when the certificate carries a
        # top-level `matrix`; a first cohort run omitted it, so run_pipeline silently fell
        # back to SimpleITK on all 36 certified pairs and reported registration_method=
        # "simpleitk" while still stamping is_certified=true. The pipeline now refuses that
        # outright, and this builds the certificate properly so it never arises.
        first = regs[0] if regs else {}
        cert = {
            "is_certified": bool(r.get("status") == "ok" and regs),
            "mode": r.get("mode"),
            "verdict": first.get("verdict") or r.get("verdict"),
            "status": (f"{len(regs)} ROIs certified" if len(regs) > 1
                       else first.get("verdict") or r.get("verdict")),
            "n_regions": len(regs),
            "matrix": first.get("local_matrix"),
            "roi_polygon": first.get("roi_polygon"),
            "cell_error_um": first.get("cell_error_um") or r.get("cell_error_um"),
            "tre_median_um": first.get("cell_error_um") or r.get("cell_error_um"),
            "min_interpretable_radius_um": first.get("min_interpretable_radius_um"),
            "n": first.get("n_correspondences"),
            "method": "user_roi_loftr_local",
            "roi_certifications": regs,
            "reason": r.get("error") or r.get("reason"),
            "regions": regs, "seconds": round(time.time() - t0, 1),
        }
        json.dump(cert, open(cache, "w"), indent=1)
        p["certification"] = cert
        done += 1
        log(f"[{i}/{len(pairs)}] {p['mag']:>3} {p['sample_id']:<26} "
            f"{'CERT ' + str(cert['n_regions']) + ' region(s)' if cert['is_certified'] else 'NOT CERTIFIED'}"
            f"  {cert.get('verdict') or cert.get('reason') or ''}  ({cert['seconds']}s)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--certify-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    pairs = discover()
    if a.limit:
        pairs = pairs[:a.limit]
    log("=" * 84)
    log(f"COHORT RUN — {len(pairs)} pairs   started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 84)

    t0 = time.time()
    certify_all(pairs, log)
    ok = [p for p in pairs if p["certification"]["is_certified"]]
    log(f"\nCERTIFIED {len(ok)}/{len(pairs)}   ({time.time() - t0:.0f}s)")
    json.dump({"pairs": [{k: v for k, v in p.items() if k != 'certification'} |
                         {"certification": {kk: vv for kk, vv in p["certification"].items()
                                            if kk != "regions"}} for p in pairs]},
              open(os.path.join(OUT, "certification_summary.json"), "w"), indent=1)
    if a.certify_only:
        return 0

    # ── analysis: one config, every certified pair, the gate left ON ────────────────────
    import yaml
    cfg = {
        "mode": "spatial",
        "output_dir": os.path.join(OUT, "spatial"),
        "dab_threshold": 0.15,
        "max_distance_um": 10.0,
        "enable_registration": True,
        "require_landmark_certification": True,     # uncertified pairs stay refused
        "pixel_size_um": PX_10X,
        "spatial_pairs": [{
            "sample_id": f"{p['sample_id']}__{p['mag']}",
            "stain_a": "CD8", "stain_b": "TIM-3",
            "path_a": p["path_a"], "path_b": p["path_b"],
            "pixel_size_um": p["pixel_size_um"],
            "certification": p["certification"],
        } for p in pairs],
    }
    cpath = os.path.join(OUT, "cohort_config.yaml")
    yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
    log(f"\nwrote {cpath}\nrunning the spatial pipeline…\n")

    import subprocess
    r = subprocess.run([sys.executable, "run_pipeline.py", "--config", cpath, "--mode", "spatial"],
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       capture_output=True, text=True)
    log(r.stdout[-8000:])
    if r.stderr.strip():
        log("STDERR:\n" + r.stderr[-4000:])
    log(f"\nDONE in {time.time() - t0:.0f}s — outputs in {cfg['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
