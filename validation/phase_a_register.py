"""
phase_a_register.py — compute + FREEZE the registration transform for each pair
(MI-selected similarity on the structural channel) and persist to
phase_a_qc/registration.json, regenerating the green/magenta + checkerboard
overlays. The manual-landmark TRE (gold standard) then verifies these transforms.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial.serial_registration import (detect_scale_bar_px, register_similarity,    # noqa
                                 _load_rgb_thumbnail, save_qc_overlays, lumen_tre,
                                 tissue_mask)

DATA = "/Users/mukilan/Desktop/052526"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_a_qc")
LIVER_UM = 0.7519

PAIRS = [
    ("Tumor_1", "Tumor/LL477_CD8_x10_1.tif", "Tumor/LL477_Tim3_x10_1.tif",
     "Tumor/LL477_CD8_x10_1_scale.tif", "Tumor/LL477_Tim3_x10_1_Scale.tif"),
    ("Tumor_2", "Tumor/LL477_CD8_x10_2.tif", "Tumor/LL477_Tim3_x10_2.tif",
     "Tumor/LL477_CD8_x10_2_scale.tif", "Tumor/LL477_Tim3_x10_2_scale.tif"),
    ("Tumor_3", "Tumor/LL477_CD8_x10_3.tif", "Tumor/LL477_Tim3_10X_3.tif",
     "Tumor/LL477_CD8_x10_3_scale.tif", "Tumor/LL477_Tim3_10X_3_scale.tif"),
    ("Liver_1", "Liver/LL477_Liver_CD8_10X_1.tif", "Liver/LL477_Liver_Tim3_10X_1.tif", None, None),
    ("Liver_2", "Liver/LL477_Liver_CD8_10X_2.tif", "Liver/LL477_Liver_Tim3_10X_2.tif", None, None),
    ("Liver_3", "Liver/LL477_Liver_CD8_10X_3.tif", "Liver/LL477_Liver_Tim3_10X_3.tif", None, None),
    ("Liver_4", "Liver/LL477_Liver_CD8_10X_4.tif", "Liver/LL477_Liver_Tim3_10X_4.tif", None, None),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    reg_db = {}
    for sid, ref_rel, mov_rel, ref_scale, mov_scale in PAIRS:
        ref_path = os.path.join(DATA, ref_rel); mov_path = os.path.join(DATA, mov_rel)
        if ref_scale:
            rb = detect_scale_bar_px(os.path.join(DATA, ref_scale))["bar_px"]
            mb = detect_scale_bar_px(os.path.join(DATA, mov_scale))["bar_px"]
            px = round(100.0 / rb, 4); src = "scale_bar"
        else:
            rb = mb = None; px = LIVER_UM; src = "manual_reference(0.7519)"

        ref_rgb, _ = _load_rgb_thumbnail(ref_path, max_side=1920)
        mov_rgb, _ = _load_rgb_thumbnail(mov_path, max_side=1920)
        reg = register_similarity(ref_rgb, mov_rgb, px)
        lum = lumen_tre(tissue_mask(ref_rgb, px), tissue_mask(mov_rgb, px),
                        reg["matrix"], px)
        ov, cb = save_qc_overlays(ref_rgb, mov_rgb, reg["matrix"],
                                  os.path.join(OUT, sid), recs=reg["recs"], lumens=lum)
        reg_db[sid] = {
            "ref_path": ref_path, "mov_path": mov_path,
            "pixel_size_um": px, "pixel_size_source": src,
            "ref_bar_px": rb, "mov_bar_px": mb,
            "matrix": reg["matrix"].tolist(), "method": reg["method"],
            "est_scale": reg["est_scale"], "mi_value": reg["mi_value"],
            "struct_ncc": reg["struct_ncc"], "struct_dice": reg["struct_dice"],
            "diag_patchflow_median_um": reg["flow"]["median_um"],
            "diag_lumen_tre_median_um": lum["median_um"], "diag_lumen_n": lum["n_corr"],
            "overlay": ov, "checkerboard": cb,
        }
        print(f"{sid}: px={px}({src}) method={reg['method']} MI={reg['mi_value']} "
              f"est_scale={reg['est_scale']} tx,ty="
              f"{reg['matrix'][0,2]:.1f},{reg['matrix'][1,2]:.1f}")

    with open(os.path.join(OUT, "registration.json"), "w") as f:
        json.dump(reg_db, f, indent=2)
    print(f"\nWrote {OUT}/registration.json + overlays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
