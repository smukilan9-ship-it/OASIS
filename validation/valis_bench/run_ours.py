"""
run_ours.py — OASIS side of the VALIS benchmark. RUN IN THE MAIN PROJECT .venv.

    .venv/bin/python -m validation.valis_bench.run_ours

For every usable ANHIR/CIMA pair (common.get_pairs), registers moving→fixed using ONLY
the images — never the expert landmarks — and scores rTRE on the held-out expert
landmarks. Two OASIS registrations per pair:

  ours_loftr  : LoFTR cycle+scale-consistent correspondences → robust SIMILARITY fit.
                This is the path the user asked to measure ("our loftr correspondence").
  ours_auto   : register_similarity (structural hematoxylin MI/NGF, fully automated) —
                the no-correspondence production fallback, for reference.

It ALSO exercises OUR GATE (landmark_register_and_verify) on the LoFTR correspondences
and records its verdict + measured held-out TRE. The gate NEVER sees the expert
landmarks, so tabulating gate-verdict against the independent expert rTRE is a genuine,
non-circular calibration test of the gate (done in compare.py).

Big-image handling: ANHIR at 25pc reaches 16k+ px for COAD/breast/gastric — too large for
whole-image LoFTR. We downsample both images to a working long-side (WORK_MAX) for the
GLOBAL similarity fit, then warp the FULL-RES landmarks through it (scale in → fit → scale
out) so rTRE is measured at full resolution. This mirrors VALIS, which also downsamples for
its rigid step. Real µm/px (from the ANHIR tissue table) drive the gate + structural σ.

Writes ours_results.json next to this file.
"""
import os
import json
import time
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")

from validation.valis_bench import common as C
from oasis.spatial import serial_registration as sr
from oasis.spatial.loftr_matcher import loftr_correspondences

WORK_MAX = 2000          # working long-side for the global fit
NOMINAL_FALLBACK_PX = {5.0: 4.0, 10.0: 2.0, 25.0: 0.8, 50.0: 0.4, 100.0: 0.2}


def _px_um(p):
    return C.px_um_for(p["set"], p["img_scale_pc"]) or NOMINAL_FALLBACK_PX.get(p["img_scale_pc"], 4.0)


def _downsample(img, f):
    if f >= 1.0:
        return img
    return cv2.resize(img, (max(int(img.shape[1] * f), 8), max(int(img.shape[0] * f), 8)),
                      interpolation=cv2.INTER_AREA)


def _score_via(M, mov_lm, fixed_lm, wh, f):
    """Warp FULL-res moving landmarks through a transform fitted in DOWNSAMPLED coords."""
    warped = C.apply_affine_2x3(np.asarray(mov_lm, float) * f, M) / f
    return C.rtre(warped, fixed_lm, wh)


def run():
    pairs = C.get_pairs()
    _strat = os.environ.get("ANHIR_STRATIFY")
    if _strat:
        pairs = C.stratified_pairs(pairs, int(_strat))
    print(f"[ours] {len(pairs)} pairs")
    results = []
    for i, p in enumerate(pairs, 1):
        t0 = time.time()
        fixed = C.load_image(p["fixed_img"]); moving = C.load_image(p["moving_img"])
        H, W = fixed.shape[:2]; wh = (W, H)
        long_side = max(H, W, moving.shape[0], moving.shape[1])
        f = min(1.0, float(WORK_MAX) / float(long_side))
        px_full = _px_um(p)
        px_work = px_full / f                       # a downsampled pixel covers more µm
        fx, mv = _downsample(fixed, f), _downsample(moving, f)
        mov_lm = np.asarray(p["moving_lm"], float); fixed_lm = np.asarray(p["fixed_lm"], float)
        rec = {"pair_id": p["pair_id"], "set": p["set"], "status": p.get("status"),
               "moving_stain": p["moving_stain"], "fixed_stain": p["fixed_stain"],
               "n_landmarks": int(len(fixed_lm)), "fixed_wh": wh, "px_um": px_full,
               "work_f": round(f, 4), "initial": C.initial_rtre(mov_lm, fixed_lm, wh)}

        # ── ours_loftr: LoFTR correspondences → similarity (+ gate) ──────────────
        try:
            c = loftr_correspondences(fx, mv, pixel_size_um=px_work, tol_um=4.0)
            rec["loftr_n"] = int(c.get("n") or 0); rec["loftr_msg"] = c.get("msg", "")
            if c.get("ok"):
                rp = np.asarray(c["ref_points"], float); mp = np.asarray(c["mov_points"], float)
                M = sr._fit_similarity_robust(mp, rp)     # mov→ref in WORK coords
                rec["ours_loftr"] = _score_via(M, mov_lm, fixed_lm, wh, f)
                try:
                    gate = sr.landmark_register_and_verify(
                        rp, mp, float(px_work), image_wh=(mv.shape[1], mv.shape[0]),
                        landmarks_are_model_selected=False)
                    rec["gate"] = {"verdict": gate.get("verdict"),
                                   "loo_tre_um": gate.get("loo_tre_um") or gate.get("tre_um")
                                   or gate.get("held_out_tre_um"),
                                   "n": gate.get("n") or int(len(rp))}
                except Exception as e:
                    rec["gate"] = {"verdict": "GATE_ERROR", "msg": str(e)[:200]}
            else:
                rec["ours_loftr"] = None
                rec["gate"] = {"verdict": "NO_MATCHES", "msg": c.get("msg", "")}
        except Exception as e:
            rec["ours_loftr"] = None; rec["loftr_error"] = str(e)[:200]

        # ── ours_auto: fully-automated structural register_similarity ────────────
        try:
            out = sr.register_similarity(fx, mv, pixel_size_um=px_work)
            M = np.asarray(out.get("matrix"), float)
            rec["ours_auto"] = _score_via(M, mov_lm, fixed_lm, wh, f)
            rec["ours_auto_scale"] = out.get("scale")
        except Exception as e:
            rec["ours_auto"] = None; rec["ours_auto_error"] = str(e)[:200]

        rec["secs"] = round(time.time() - t0, 1)
        results.append(rec)
        il = rec.get("ours_loftr")
        msg = (f"loftr={il['median']:.4f}" if il else f"loftr=FAIL({rec.get('loftr_n',0)})")
        print(f"  [{i}/{len(pairs)}] {p['set']} {p['moving_stain']}->{p['fixed_stain']} "
              f"init={rec['initial']['median']:.4f} {msg} n={rec.get('loftr_n',0)} "
              f"gate={rec.get('gate',{}).get('verdict')} {rec['secs']}s")

        if i % 20 == 0:                              # checkpoint periodically
            with open(os.path.join(C.OUT_DIR, "ours_results.json"), "w") as fjson:
                json.dump({"method": "oasis", "n_pairs": len(results), "results": results}, fjson, indent=1)

    with open(os.path.join(C.OUT_DIR, "ours_results.json"), "w") as fjson:
        json.dump({"method": "oasis", "n_pairs": len(results), "results": results}, fjson, indent=1)
    print(f"[ours] wrote ours_results.json ({len(results)} pairs)")


if __name__ == "__main__":
    run()
