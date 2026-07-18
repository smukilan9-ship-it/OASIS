"""
run_correspondence.py — DIRECT validation of LoFTR correspondences (main .venv):

    .venv/bin/python -m validation.valis_bench.run_correspondence

Answers "does LoFTR identify correspondences PROPERLY?" — not via downstream rTRE, but by
checking each individual match against the independent expert-landmark geometry. For every
pair it runs LoFTR, then (common.correspondence_quality) predicts each match's true moving
location from the LOCAL expert-landmark affine and measures LoFTR's deviation in µm. Real
ANHIR pixel sizes are used so the error is in physical units. Fast: LoFTR only, no fitting.

Writes correspondence_results.json + prints a summary.
"""
import os
import json
import time
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")
from validation.valis_bench import common as C
from oasis.spatial.loftr_matcher import loftr_correspondences

WORK_MAX = 2000
NOMINAL_FALLBACK_PX = {5.0: 4.0, 10.0: 2.0, 25.0: 0.8, 50.0: 0.4, 100.0: 0.2}


def _ds(img, f):
    if f >= 1.0:
        return img
    return cv2.resize(img, (max(int(img.shape[1] * f), 8), max(int(img.shape[0] * f), 8)),
                      interpolation=cv2.INTER_AREA)


def run(tol_um=10.0):
    pairs = C.get_pairs()
    print(f"[corr] {len(pairs)} pairs; tol={tol_um}µm")
    results = []
    for i, p in enumerate(pairs, 1):
        t0 = time.time()
        fixed = C.load_image(p["fixed_img"]); moving = C.load_image(p["moving_img"])
        long_side = max(fixed.shape[0], fixed.shape[1], moving.shape[0], moving.shape[1])
        f = min(1.0, float(WORK_MAX) / float(long_side))
        px_full = C.px_um_for(p["set"], p["img_scale_pc"]) or NOMINAL_FALLBACK_PX.get(p["img_scale_pc"], 4.0)
        px = px_full / f                              # working µm/px after downsample
        fx, mv = _ds(fixed, f), _ds(moving, f)
        # landmarks into WORK coords so the local-affine GT and LoFTR points share a frame
        lm_ref = np.asarray(p["fixed_lm"], float) * f
        lm_mov = np.asarray(p["moving_lm"], float) * f
        rec = {"pair_id": p["pair_id"], "set": p["set"], "px_um": px_full, "work_f": round(f, 4),
               "moving_stain": p["moving_stain"], "fixed_stain": p["fixed_stain"]}
        try:
            c = loftr_correspondences(fx, mv, pixel_size_um=px, tol_um=4.0)
            rec["funnel"] = {"raw": c.get("n_raw"), "cycle": c.get("n_after_cycle"),
                             "scale": c.get("n_after_scale"), "final": c.get("n")}
            if c.get("ok"):
                q = C.correspondence_quality(c["ref_points"], c["mov_points"],
                                             lm_ref, lm_mov, px, tol_um=tol_um)
                rec["corr"] = q
            else:
                rec["corr"] = None
                rec["msg"] = c.get("msg")
        except Exception as e:
            rec["corr"] = None; rec["error"] = str(e)[:200]
        rec["secs"] = round(time.time() - t0, 1)
        results.append(rec)
        q = rec.get("corr")
        if q and q.get("median_um") is not None:
            print(f"  [{i}/{len(pairs)}] {p['moving_stain'][-10:]}->{p['fixed_stain'][-10:]} "
                  f"matches={q['n_matches']} eval={q['n_eval']} "
                  f"median_err={q['median_um']:.2f}µm inliers={q['inlier_rate']:.0%} {rec['secs']}s")
        else:
            print(f"  [{i}/{len(pairs)}] {p['pair_id'][-24:]} — no usable matches ({rec.get('msg','')})")
        if i % 15 == 0:                          # checkpoint
            with open(os.path.join(C.OUT_DIR, "correspondence_results.json"), "w") as fck:
                json.dump({"summary": {"partial": True, "done": len(results)},
                           "results": results}, fck, indent=1)

    # aggregate
    med = [r["corr"]["median_um"] for r in results if r.get("corr") and r["corr"]["median_um"] is not None]
    inl = [r["corr"]["inlier_rate"] for r in results if r.get("corr") and r["corr"]["inlier_rate"] is not None]
    summary = {"n_pairs": len(results), "n_with_matches": len(med),
               "median_corr_err_um": float(np.median(med)) if med else None,
               "mean_inlier_rate": float(np.mean(inl)) if inl else None, "tol_um": tol_um}
    out = os.path.join(C.OUT_DIR, "correspondence_results.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(f"\n[corr] SUMMARY: {summary['n_with_matches']}/{summary['n_pairs']} pairs matched; "
          f"median correspondence error {summary['median_corr_err_um']}µm; "
          f"mean inlier rate @{tol_um}µm = "
          f"{summary['mean_inlier_rate']:.0%}" if summary['mean_inlier_rate'] is not None else "n/a")
    print(f"[corr] wrote {out}")


if __name__ == "__main__":
    run()
