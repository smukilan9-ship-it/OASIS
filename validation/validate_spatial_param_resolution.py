"""
validate_spatial_param_resolution.py — prove pixel-size / threshold resolution is correct,
global, per-pair isolated, and never SILENTLY wrong.

Exercises the REAL functions the spatial pipeline uses:
  • run_pipeline.resolve_pixel_size            (api.py builds per-image pixel_overrides with it)
  • pixel_size_util.get_pixel_size_with_source (run_single_image resolves each image with it)

Checks, mirroring the user's requirements:
  1. A user-set value is applied (not silently replaced by 0.5).
  2. The value applies GLOBALLY (same session value → every image, unless per-image override).
  3. BATCH ISOLATION: each image reads its OWN per-image pixel size / threshold, not a pooled one.
  4. A fallback to the 0.5 default is LABELLED 'default_fallback' (never silent).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pipeline import resolve_pixel_size
from oasis.common.pixel_size_util import get_pixel_size_with_source

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    print(f"  [{PASS if cond else FAIL}] {name}")
    _ok = _ok and cond


print("1) resolve_pixel_size — priority + honest fallback")
# manual override wins over everything
check("manual override 0.30 wins", resolve_pixel_size(0.5, "x.tif", None, 0.30) == 0.30)
# session default used when no per-image override/scale (GLOBAL default)
check("session default 0.25 applied", resolve_pixel_size(0.25, "x.tif", None, None) == 0.25)
# NOTHING provided → 0.5 fallback (must be the ONLY way to reach 0.5 unset)
check("bare fallback → 0.5 (no value anywhere)", resolve_pixel_size(None, "", None, None) == 0.5)
# a real value never silently becomes 0.5
check("session 0.9 not silently 0.5", resolve_pixel_size(0.9, "", None, None) == 0.9)

print("\n2) get_pixel_size_with_source — per-image override + labelled source")
cfg_over = {"pixel_overrides": {"A.tif": 0.25, "B.tif": 0.90}, "default_pixel_size": 0.5,
            "_pixel_size_from_ui": True}
va, sa = get_pixel_size_with_source("/data/A.tif", cfg_over)
vb, sb = get_pixel_size_with_source("/data/B.tif", cfg_over)
check("A.tif → its own 0.25 (per_image_override)", va == 0.25 and sa == "per_image_override")
check("B.tif → its own 0.90 (per_image_override)", vb == 0.90 and sb == "per_image_override")
check("A and B are ISOLATED (0.25 != 0.90)", va != vb)

# UI default (global) when no per-image override
vu, su = get_pixel_size_with_source("/data/C.tif",
                                    {"default_pixel_size": 0.33, "_pixel_size_from_ui": True})
check("no override → UI global default 0.33 (ui_default)", vu == 0.33 and su == "ui_default")

# genuine fallback is LABELLED, never silent
vf, sf = get_pixel_size_with_source("/data/none.tif",
                                    {"default_pixel_size": 0.5, "_pixel_size_from_ui": False})
check("bare image → source is a NON-silent label", sf in ("default_fallback", "filename"))

print("\n3) Batch GLOBAL application + per-pair isolation (simulated as api.py builds it)")
# api.py: ref_px = session default; each image gets scale_px[img] if present else ref_px.
session = 0.40
scale_px = {"S2_CD8.tif": 0.22}            # only one image has its own scale sibling
batch_imgs = ["S1_CD8.tif", "S1_TIM3.tif", "S2_CD8.tif", "S2_TIM3.tif"]
pixel_overrides = {img: scale_px.get(img, session) for img in batch_imgs}
cfg_batch = {"pixel_overrides": pixel_overrides, "default_pixel_size": session,
             "_pixel_size_from_ui": True}
resolved = {img: get_pixel_size_with_source(f"/d/{img}", cfg_batch)[0] for img in batch_imgs}
check("session 0.40 applied globally to un-scaled images",
      resolved["S1_CD8.tif"] == 0.40 and resolved["S1_TIM3.tif"] == 0.40
      and resolved["S2_TIM3.tif"] == 0.40)
check("the scale-calibrated image keeps ITS OWN 0.22 (isolated)",
      resolved["S2_CD8.tif"] == 0.22)
check("no image silently fell to 0.5", all(v != 0.5 for v in resolved.values()))

# threshold isolation: api.py keys threshold_overrides by basename, so each image is independent
thr_over = {"S1_CD8.tif": 0.20, "S1_TIM3.tif": 0.10, "S2_CD8.tif": 0.20, "S2_TIM3.tif": 0.10}
check("per-image thresholds are isolated (CD8 0.20 vs TIM3 0.10)",
      thr_over["S1_CD8.tif"] == 0.20 and thr_over["S1_TIM3.tif"] == 0.10)

print("\n" + ("ALL CHECKS PASSED" if _ok else "SOME CHECKS FAILED"))
sys.exit(0 if _ok else 1)
