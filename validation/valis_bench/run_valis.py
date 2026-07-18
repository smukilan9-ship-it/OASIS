"""
run_valis.py — VALIS side of the benchmark. RUN IN THE ISOLATED ~/valis_runtime venv:

    DYLD_LIBRARY_PATH=/opt/homebrew/lib \
      ~/valis_runtime/venv/bin/python -m validation.valis_bench.run_valis

(run from the project root so `validation.valis_bench.common` imports; the isolated
venv only needs numpy+PIL for common.py, which it has.)

For every usable ANHIR/CIMA pair it registers moving→fixed with VALIS and scores rTRE
on the SAME held-out expert landmarks, via the SAME common.rtre — so the number is
directly comparable to run_ours.py. Two transforms from ONE registration:

  valis_rigid    : warp_xy_from_to(..., non_rigid=False) — rigid only. This is the
                   distance-preserving transform that is apples-to-apples with OASIS's
                   similarity (and cross-K-safe).
  valis_nonrigid : warp_xy_from_to(..., non_rigid=True) — rigid + micro-registration
                   (B-spline/flow). This is the warp OASIS FORBIDS for spatial-association
                   stats; reported only as an accuracy upper bound / reference.

ANTI-CIRCULARITY: VALIS never receives the expert landmarks. It registers from the
image pixels; the landmarks are used only by common.rtre for scoring.

Writes valis_results.json next to this file.
"""
import os
import sys
import json
import time
import shutil
import tempfile
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# make the project root importable for validation.valis_bench.common
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.valis_bench import common as C  # noqa: E402


def _warp(slide, to_slide, xy, non_rigid):
    """Warp Nx2 points from `slide` original space to `to_slide` original space.

    Tries the direct slide→slide API, then falls back to warp to registered space.
    """
    xy = np.asarray(xy, float)
    try:
        return np.asarray(slide.warp_xy_from_to(xy, to_slide, non_rigid=non_rigid), float)
    except TypeError:
        # older/newer signature without non_rigid kw
        return np.asarray(slide.warp_xy_from_to(xy, to_slide), float)


def run():
    from valis import registration

    pairs = C.get_pairs()
    _strat = os.environ.get("ANHIR_STRATIFY")
    if _strat:
        pairs = C.stratified_pairs(pairs, int(_strat))
    print(f"[valis] {len(pairs)} directed pairs")
    results = []
    work = tempfile.mkdtemp(prefix="valis_bench_")
    try:
        for i, p in enumerate(pairs, 1):
            t0 = time.time()
            fixed_lm = np.asarray(p["fixed_lm"], float)
            mov_lm = np.asarray(p["moving_lm"], float)
            # VALIS reads an image size from disk; get fixed wh from the file
            with Image.open(p["fixed_img"]) as im:
                W, H = im.size
            wh = (W, H)
            rec = {"pair_id": p["pair_id"], "set": p["set"], "annotator": p["annotator"],
                   "moving_stain": p["moving_stain"], "fixed_stain": p["fixed_stain"],
                   "n_landmarks": int(len(fixed_lm)), "fixed_wh": wh,
                   "initial": C.initial_rtre(mov_lm, fixed_lm, wh)}

            src = os.path.join(work, f"pair_{i}")
            dst = os.path.join(work, f"pair_{i}_out")
            os.makedirs(src, exist_ok=True)
            f_name = os.path.basename(p["fixed_img"])
            m_name = os.path.basename(p["moving_img"])
            shutil.copy(p["fixed_img"], os.path.join(src, f_name))
            shutil.copy(p["moving_img"], os.path.join(src, m_name))

            try:
                registrar = registration.Valis(
                    src, dst, reference_img_f=f_name, imgs_ordered=False,
                    crop="reference")
                registrar.register()
                mov_slide = registrar.get_slide(m_name)
                fix_slide = registrar.get_slide(f_name)
                for tag, nr in (("valis_rigid", False), ("valis_nonrigid", True)):
                    try:
                        warped = _warp(mov_slide, fix_slide, mov_lm, nr)
                        rec[tag] = C.rtre(warped, fixed_lm, wh)
                    except Exception as e:
                        rec[tag] = None
                        rec[tag + "_error"] = str(e)[:200]
            except Exception as e:
                rec["valis_error"] = str(e)[:300]
                rec["valis_rigid"] = rec["valis_nonrigid"] = None
            finally:
                shutil.rmtree(dst, ignore_errors=True)
                shutil.rmtree(src, ignore_errors=True)

            rec["secs"] = round(time.time() - t0, 1)
            results.append(rec)
            if i % 10 == 0:                       # checkpoint
                with open(os.path.join(C.OUT_DIR, "valis_results.json"), "w") as fck:
                    json.dump({"method": "valis", "n_pairs": len(results), "results": results},
                              fck, indent=1)
            rr = rec.get("valis_rigid"); nn = rec.get("valis_nonrigid")
            print(f"  [{i}/{len(pairs)}] {p['moving_stain'][-10:]}->{p['fixed_stain'][-10:]} "
                  f"init={rec['initial']['median']:.4f} "
                  f"rigid={rr['median']:.4f} " if rr else
                  f"  [{i}/{len(pairs)}] {p['pair_id']} rigid=FAIL ",
                  f"nonrigid={nn['median']:.4f} " if nn else "nonrigid=FAIL ",
                  f"{rec['secs']}s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            registration.kill_jvm()
        except Exception:
            pass

    out_path = os.path.join(C.OUT_DIR, "valis_results.json")
    with open(out_path, "w") as f:
        json.dump({"method": "valis", "n_pairs": len(results), "results": results}, f, indent=1)
    print(f"[valis] wrote {out_path}")


if __name__ == "__main__":
    run()
