#!/usr/bin/env python
"""
Render HyReCo whole-slide TIFFs into images OASIS can actually work on.

THE MISMATCH. A HyReCo slide is 95,601 x 218,145 px at 0.243 µm/px — 23.2 x 53.0 mm of
tissue. Everything OASIS has been developed and validated against is a 10x microscope FIELD:
LL477 is 1920 x 1440 at 0.7519 µm/px, or 1.44 x 1.08 mm. The slide is roughly a thousand
times the area of a field. Handing a WSI to the spatial pipeline unchanged would either blow
up or, worse, quietly thumbnail it to ~12 µm/px and certify a registration at a resolution
where a cell is a fifth of a pixel.

So there are TWO renderings, and they answer different questions. Getting this wrong is how
you produce a number that looks like a validation and is not one.

  --mode section   The whole section at a working resolution (default long side 4096, giving
                   ~13 µm/px). This is the ONLY rendering the expert landmarks can score,
                   because they are 11-19 points spread across the entire section — case 611's
                   fourteen span 8 x 14.6 mm, a mean spacing of about 3 mm. It validates
                   registration and TRE at the section scale, which is what ANHIR does too.

  --mode fields    10x-equivalent tiles at the pyramid level closest to a target µm/px
                   (default 0.7519, matching LL477 exactly). This is what the quant and
                   spatial pipelines are built for — cells are resolved, DAB is measurable,
                   and a certification means what it means on LL477.

WHAT THE LANDMARKS CAN AND CANNOT SETTLE, stated plainly because it bounds what this dataset
is worth. A 1.4 mm field contains ZERO OR ONE landmark, so `fields` output cannot be scored
against experts directly — the landmarks validate the section-scale transform, not OASIS's
450 µm ROI certification. Interpolating fourteen points down to a 450 µm neighbourhood is
extrapolation, not ground truth, and treating it as truth would manufacture exactly the kind
of confident wrong answer this repo keeps finding. What `fields` IS for: measuring LoFTR's
cross-stain blunder rate on properly prepared serial sections, which is the open question
from validate_matchers_on_cohort.py, and giving the quant pipeline real CD8 WSI-derived data.

Run:
    python validation/datasets/hyreco_render.py --mode section --cases 611 679
    python validation/datasets/hyreco_render.py --mode fields --cases 611 --n-fields 6
"""
import argparse
import csv
import os

import numpy as np

SRC = "/Volumes/Expansion/oasis_datasets/HyReCo/HyReCo"
OUT = "/Volumes/Expansion/oasis_datasets/HyReCo/_rendered"
STAINS = ("CD8", "HE", "CD45", "KI67", "PHH3")
TARGET_UM_PX = 0.7519          # LL477's 10x field, so a certification is comparable
FIELD_WH = (1920, 1440)        # LL477's frame


def load_landmarks_mm(path):
    out = []
    for r in csv.reader(open(path)):
        if len(r) >= 2:
            try:
                out.append((float(r[0]), float(r[1])))
            except ValueError:
                pass
    return np.asarray(out, float)


def open_slide(path):
    import openslide
    s = openslide.OpenSlide(path)
    mpp = float(s.properties[openslide.PROPERTY_NAME_MPP_X])
    return s, mpp


def render_section(case, stain, long_side, outdir):
    """Whole section at a workable resolution, plus its landmarks in the NEW pixel frame.

    The landmark CSV is in millimetres from the slide origin; verified against the real files
    by converting every point and confirming each lands on tissue rather than glass — a sign
    or origin error produces TRE numbers that look entirely plausible and are wrong.
    """
    import cv2
    src = f"{SRC}/{stain}/{case}.tif"
    if not os.path.exists(src):
        return None
    s, mpp = open_slide(src)
    W, H = s.dimensions
    scale = long_side / float(max(W, H))
    lev = s.get_best_level_for_downsample(1.0 / scale)
    lw, lh = s.level_dimensions[lev]
    img = np.asarray(s.read_region((0, 0), lev, (lw, lh)).convert("RGB"))
    out_w, out_h = max(int(W * scale), 8), max(int(H * scale), 8)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    px_um = mpp * (W / float(out_w))

    lm = load_landmarks_mm(f"{SRC}/{stain}/{case}.csv")
    lm_px = lm * 1000.0 / px_um                      # mm -> µm -> rendered px

    os.makedirs(outdir, exist_ok=True)
    stem = f"{case}_{stain}"
    cv2.imwrite(f"{outdir}/{stem}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.savetxt(f"{outdir}/{stem}_landmarks_px.csv", lm_px, delimiter=",",
               header="x_px,y_px", comments="")
    s.close()
    return {"file": f"{stem}.png", "wh": (out_w, out_h), "um_px": round(px_um, 4),
            "n_landmarks": len(lm),
            "landmarks_inside": int(((lm_px[:, 0] >= 0) & (lm_px[:, 0] < out_w) &
                                     (lm_px[:, 1] >= 0) & (lm_px[:, 1] < out_h)).sum())}


def render_fields(case, stain, n_fields, outdir, target_um=TARGET_UM_PX):
    """10x-equivalent fields centred on the landmarks — the regions an expert already judged
    worth annotating, which is a better sampling of the section than an arbitrary grid."""
    import cv2
    src = f"{SRC}/{stain}/{case}.tif"
    if not os.path.exists(src):
        return []
    s, mpp = open_slide(src)
    lev = s.get_best_level_for_downsample(target_um / mpp)
    lev_um = mpp * s.level_downsamples[lev]
    # read at the pyramid level nearest the target, then resample the rest of the way so every
    # field lands at exactly target_um and is directly comparable with an LL477 frame
    need_w = int(round(FIELD_WH[0] * target_um / lev_um))
    need_h = int(round(FIELD_WH[1] * target_um / lev_um))
    ds = s.level_downsamples[lev]

    lm_px0 = load_landmarks_mm(f"{SRC}/{stain}/{case}.csv") * 1000.0 / mpp   # level-0 px
    os.makedirs(outdir, exist_ok=True)
    made = []
    for i, (cx, cy) in enumerate(lm_px0[:n_fields]):
        x0 = int(cx - need_w * ds / 2.0)
        y0 = int(cy - need_h * ds / 2.0)
        tile = np.asarray(s.read_region((x0, y0), lev, (need_w, need_h)).convert("RGB"))
        tile = cv2.resize(tile, FIELD_WH, interpolation=cv2.INTER_AREA)
        name = f"{case}_{stain}_lm{i:02d}.png"
        cv2.imwrite(f"{outdir}/{name}", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        made.append({"file": name, "landmark_index": i, "level0_xy": (x0, y0),
                     "um_px": target_um, "tissue_frac": round(float((tile.mean(2) < 225).mean()), 3)})
    s.close()
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("section", "fields"), default="section")
    ap.add_argument("--cases", nargs="+", default=["611", "679"])
    ap.add_argument("--stains", nargs="+", default=list(STAINS))
    ap.add_argument("--long-side", type=int, default=4096)
    ap.add_argument("--n-fields", type=int, default=6)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    for case in a.cases:
        outdir = os.path.join(a.out, a.mode, case)
        print(f"\n=== case {case} -> {outdir}")
        for stain in a.stains:
            if not os.path.exists(f"{SRC}/{stain}/{case}.tif"):
                print(f"  {stain:<6} (slide not downloaded)")
                continue
            if a.mode == "section":
                r = render_section(case, stain, a.long_side, outdir)
                print(f"  {stain:<6} {r['wh'][0]}x{r['wh'][1]} @ {r['um_px']} µm/px   "
                      f"landmarks {r['landmarks_inside']}/{r['n_landmarks']} inside")
            else:
                fs = render_fields(case, stain, a.n_fields, outdir)
                tf = np.mean([f["tissue_frac"] for f in fs]) if fs else 0
                print(f"  {stain:<6} {len(fs)} fields of {FIELD_WH[0]}x{FIELD_WH[1]} "
                      f"@ {TARGET_UM_PX} µm/px   mean tissue {tf:.0%}")


if __name__ == "__main__":
    main()
