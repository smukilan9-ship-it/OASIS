"""
export_warps.py — STEP 1. Run VALIS ONCE per ANHIR pair and export BOTH transforms
as dense warped grids, so every downstream experiment can reapply the real warps to
arbitrary point sets without re-running the registrar.

RUN IN THE ISOLATED VALIS VENV (numpy + PIL only):

    DYLD_LIBRARY_PATH=/opt/homebrew/lib \
      ~/valis_runtime/venv/bin/python -m validation.nonrigid_bench.export_warps

Why a grid rather than warping the experiment's points directly: the point patterns
are regenerated hundreds of times per arm (replicates, seeds, three arms). VALIS costs
~30 s per pair. Exporting the DISPLACEMENT FIELD once and interpolating it is the same
map, evaluated cheaply. Interpolation fidelity is checked in check_warps.py against a
held-out random point set warped by VALIS itself.

Both transforms come from ONE registration, so rigid and non-rigid differ ONLY in
whether the micro-registration (B-spline / flow) stage is applied:

  rigid    : warp_xy_from_to(..., non_rigid=False)  — distance-preserving
  nonrigid : warp_xy_from_to(..., non_rigid=True)   — rigid + micro-registration

Writes warps/<pair_id>.npz + warps/index.json.
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
Image.MAX_IMAGE_PIXELS = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.valis_bench import common as C  # noqa: E402

OUT = os.path.join(_HERE, "warps")
GRID_N = 80          # 80x80 = 6400 grid points per pair
PER_TISSUE = int(os.environ.get("NRB_PER_TISSUE", "3"))
HOLDOUT_N = 400      # random points warped by VALIS directly, to audit interpolation


def _warp(slide, to_slide, xy, non_rigid):
    xy = np.asarray(xy, float)
    return np.asarray(slide.warp_xy_from_to(xy, to_slide, non_rigid=non_rigid), float)


def run():
    from valis import registration

    os.makedirs(OUT, exist_ok=True)
    pairs = C.stratified_pairs(C.get_pairs(), PER_TISSUE)
    print(f"[export] {len(pairs)} pairs ({PER_TISSUE}/tissue)", flush=True)

    index = []
    work = tempfile.mkdtemp(prefix="nrb_")
    rng = np.random.default_rng(0)
    try:
        for i, p in enumerate(pairs, 1):
            t0 = time.time()
            pid = p["pair_id"].replace("/", "_").replace(" ", "_")
            out_npz = os.path.join(OUT, f"{pid}.npz")
            if os.path.exists(out_npz):
                print(f"  [{i}/{len(pairs)}] {pid} cached", flush=True)
                index.append(json.load(open(out_npz + ".json", encoding="utf-8")))
                continue

            with Image.open(p["moving_img"]) as im:
                MW, MH = im.size
            with Image.open(p["fixed_img"]) as im:
                FW, FH = im.size

            # grid over the moving image, inset by half a cell so no point sits on the edge
            gx = np.linspace(MW / (2 * GRID_N), MW - MW / (2 * GRID_N), GRID_N)
            gy = np.linspace(MH / (2 * GRID_N), MH - MH / (2 * GRID_N), GRID_N)
            GX, GY = np.meshgrid(gx, gy)
            grid = np.column_stack([GX.ravel(), GY.ravel()])
            hold = np.column_stack([rng.uniform(0, MW, HOLDOUT_N),
                                    rng.uniform(0, MH, HOLDOUT_N)])
            probe = np.vstack([grid, hold])

            src = os.path.join(work, f"p{i}")
            dst = os.path.join(work, f"p{i}_out")
            os.makedirs(src, exist_ok=True)
            f_name = os.path.basename(p["fixed_img"])
            m_name = os.path.basename(p["moving_img"])
            shutil.copy(p["fixed_img"], os.path.join(src, f_name))
            shutil.copy(p["moving_img"], os.path.join(src, m_name))

            rec = {"pair_id": pid, "set": p["set"],
                   "moving_stain": p["moving_stain"], "fixed_stain": p["fixed_stain"],
                   "moving_wh": [MW, MH], "fixed_wh": [FW, FH],
                   "px_um": C.px_um_for(p["set"], p.get("img_scale_pc", 25)),
                   "grid_n": GRID_N, "holdout_n": HOLDOUT_N}
            try:
                registrar = registration.Valis(src, dst, reference_img_f=f_name,
                                               imgs_ordered=False, crop="reference")
                registrar.register()
                mov = registrar.get_slide(m_name)
                fix = registrar.get_slide(f_name)
                wr = _warp(mov, fix, probe, False)
                wn = _warp(mov, fix, probe, True)
                np.savez_compressed(
                    out_npz,
                    src=probe[:len(grid)], rigid=wr[:len(grid)], nonrigid=wn[:len(grid)],
                    hold_src=probe[len(grid):], hold_rigid=wr[len(grid):],
                    hold_nonrigid=wn[len(grid):])
                d = np.linalg.norm(wn[:len(grid)] - wr[:len(grid)], axis=1)
                d = d[np.isfinite(d)]
                rec["ok"] = True
                rec["residual_px"] = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                                      "rms": float(np.sqrt(np.mean(d ** 2))), "max": float(d.max())}
                print(f"  [{i}/{len(pairs)}] {pid} nonrigid-vs-rigid median "
                      f"{rec['residual_px']['median']:.1f}px p90 {rec['residual_px']['p90']:.1f}px "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                rec["ok"] = False
                rec["error"] = str(e)[:300]
                print(f"  [{i}/{len(pairs)}] {pid} FAILED {rec['error'][:90]}", flush=True)
            finally:
                shutil.rmtree(dst, ignore_errors=True)
                shutil.rmtree(src, ignore_errors=True)

            rec["secs"] = round(time.time() - t0, 1)
            if rec.get("ok"):
                json.dump(rec, open(out_npz + ".json", "w", encoding="utf-8"), indent=1)
            index.append(rec)
            json.dump(index, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            registration.kill_jvm()
        except Exception:
            pass

    json.dump(index, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1)
    ok = sum(1 for r in index if r.get("ok"))
    print(f"[export] {ok}/{len(index)} pairs exported -> {OUT}")


if __name__ == "__main__":
    run()
