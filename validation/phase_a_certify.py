"""
phase_a_certify.py — Phase A driver: register, QC, auto-TRE, and CERTIFY each of
the 7 CD8/TIM-3 serial-section pairs (read-only on the images). Produces the
Gate-A per-pair table (markdown + JSON) and per-pair QC overlays.

Scope (do not drift): 3 tumor + 4 liver pairs. Tumor #4 is EXCLUDED (its TIM-3 is
a byte-identical copy of Liver TIM-3 #1). Tumor pixel size is self-calibrated from
the burned-in 100 µm bar in the *_scale.tif images; liver uses 0.7519 µm/px as a
DOCUMENTED manual absolute reference (the weaker calibration link), with within-
pair relative scale handled by the similarity transform.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serial_registration import detect_scale_bar_px, certify_pair  # noqa: E402

DATA = "/Users/mukilan/Desktop/052526"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_a_qc")
LIVER_MANUAL_UM = 0.7519   # manual absolute reference (Q1); flagged as weaker link

# (sample_id, ref CD8, mov TIM3, ref_scale_img | None, mov_scale_img | None)
PAIRS = [
    ("Tumor_1", "Tumor/LL477_CD8_x10_1.tif", "Tumor/LL477_Tim3_x10_1.tif",
     "Tumor/LL477_CD8_x10_1_scale.tif", "Tumor/LL477_Tim3_x10_1_Scale.tif"),
    ("Tumor_2", "Tumor/LL477_CD8_x10_2.tif", "Tumor/LL477_Tim3_x10_2.tif",
     "Tumor/LL477_CD8_x10_2_scale.tif", "Tumor/LL477_Tim3_x10_2_scale.tif"),
    ("Tumor_3", "Tumor/LL477_CD8_x10_3.tif", "Tumor/LL477_Tim3_10X_3.tif",
     "Tumor/LL477_CD8_x10_3_scale.tif", "Tumor/LL477_Tim3_10X_3_scale.tif"),
    ("Liver_1", "Liver/LL477_Liver_CD8_10X_1.tif", "Liver/LL477_Liver_Tim3_10X_1.tif",
     None, None),
    ("Liver_2", "Liver/LL477_Liver_CD8_10X_2.tif", "Liver/LL477_Liver_Tim3_10X_2.tif",
     None, None),
    ("Liver_3", "Liver/LL477_Liver_CD8_10X_3.tif", "Liver/LL477_Liver_Tim3_10X_3.tif",
     None, None),
    ("Liver_4", "Liver/LL477_Liver_CD8_10X_4.tif", "Liver/LL477_Liver_Tim3_10X_4.tif",
     None, None),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sid, ref_rel, mov_rel, ref_scale, mov_scale in PAIRS:
        print(f"\n{'='*72}\n{sid}\n{'='*72}")
        ref_path = os.path.join(DATA, ref_rel)
        mov_path = os.path.join(DATA, mov_rel)

        ref_bar = mov_bar = None
        if ref_scale and mov_scale:
            rb = detect_scale_bar_px(os.path.join(DATA, ref_scale))
            mb = detect_scale_bar_px(os.path.join(DATA, mov_scale))
            ref_bar, mov_bar = rb["bar_px"], mb["bar_px"]
            px = rb["pixel_size_um"]      # common-frame px = reference (CD8) bar value
            src = "scale_bar"
            print(f"  CD8  bar={ref_bar}px -> {rb['pixel_size_um']} µm/px")
            print(f"  TIM3 bar={mov_bar}px -> {mb['pixel_size_um']} µm/px")
        else:
            px = LIVER_MANUAL_UM
            src = "manual_reference(0.7519)"
            print(f"  no scale bar -> manual {px} µm/px (weaker absolute link)")

        row = certify_pair(sid, ref_path, mov_path, px, OUT,
                           ref_bar_px=ref_bar, mov_bar_px=mov_bar,
                           pixel_size_source=src)
        rows.append(row)
        print(f"  method={row.get('method')}  est_scale={row.get('est_scale')}  "
              f"NCC={row.get('struct_ncc')}  Dice={row.get('struct_dice')}")
        print(f"  patches={row.get('n_patches')}  TRE med={row.get('tre_median_um')}µm  "
              f"p90={row.get('tre_p90_um')}  worst-region={row.get('region_max_um')}  "
              f"raw-max={row.get('tre_max_um')}")
        print(f"  lumen TRE med={row.get('lumen_tre_median_um')}µm "
              f"(n={row.get('lumen_n_corr')})")
        print(f"  scale_xcheck: est {row.get('est_scale')} vs bar "
              f"{row.get('bar_scale_expected')} ok={row.get('scale_xcheck_ok')}")
        print(f"  -> {row.get('status')}: {row.get('reason')}")

    # ── Gate-A table ──────────────────────────────────────────────────────────
    hdr = ["pair", "px_um", "px_src", "method", "est_scale", "xcheck",
           "NCC", "Dice", "TRE_med", "TRE_p90", "region_max", "raw_max",
           "lumenTRE(n)", "status"]
    def fmt(r):
        return [r["sample_id"], r.get("pixel_size_um"), r.get("pixel_size_source"),
                r.get("method"), r.get("est_scale"), r.get("scale_xcheck_ok"),
                r.get("struct_ncc"), r.get("struct_dice"),
                r.get("tre_median_um"), r.get("tre_p90_um"),
                r.get("region_max_um"), r.get("tre_max_um"),
                f"{r.get('lumen_tre_median_um')}({r.get('lumen_n_corr')})",
                r.get("status")]
    md = ["| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in fmt(r)) + " |")
    table = "\n".join(md)
    print(f"\n{'='*72}\nGATE-A TABLE\n{'='*72}\n{table}")

    with open(os.path.join(OUT, "gate_a_table.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT, "gate_a_results.json"), "w") as f:
        json.dump([{k: v for k, v in r.items()
                    if k not in ("ref_matched", "mapped_matched")} for r in rows],
                  f, indent=2)
    print(f"\nWrote: {OUT}/gate_a_table.md, gate_a_results.json, per-pair overlays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
