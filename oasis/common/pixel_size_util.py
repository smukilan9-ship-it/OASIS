"""
pixel_size_util.py
Determines correct pixel size in microns for an image.

Priority order:
1. Per-image override from config (pixel_overrides dict)
2. TIFF/OME metadata
3. Config default_pixel_size (set by user in UI)
4. Filename parsing (x10, 20x, etc.)
5. Interactive prompt
6. Default fallback (0.5 um/px)
"""

import os
import re
from pathlib import Path

PIXEL_SIZE_MAP = {
    4:   2.50,
    10:  1.00,
    20:  0.50,
    40:  0.25,
    60:  0.165,
    100: 0.10,
}


def is_scaled_image(filename: str) -> bool:
    """
    Single source of truth for identifying "scale" images — images that exist
    only to carry a burned-in scale bar for pixel-size calibration and must
    NEVER be used for analysis.

    True when the filename stem (case-insensitive) ends with '_scale'.
        "LL477_CD8_x10_scale.png" → True
        "LL477_CD8_x10.png"       → False
    """
    return Path(filename).stem.lower().endswith("_scale")


# How many microns the burned-in bar represents. Constant for a given microscope/export
# preset, so it is a setting rather than an argument every caller has to thread through.
DEFAULT_SCALE_BAR_UM = 100.0


def _bar_candidates(crop, image_width):
    """Components in `crop` that are shaped like a drawn scale bar.

    Shape, not darkness ranking. The previous version Otsu-thresholded the strip, opened it
    with a horizontal kernel, accepted anything up to 25% of the crop HEIGHT as "line-like"
    (54 px on a 1440 px image) and then took the WIDEST survivor. On real slides a tissue
    edge is wider than the bar, so it won, silently:

        measured on the LL477 cohort, true bar 133 px in all six scale images
            CD8_x10_1  156 px (+17%)   Tim3_10X_3  165 px (+24%)
            CD8_x10_2  139 px  (+5%)   Tim3_x10_1  174 px (+31%)

    A 31% error in um/px scales EVERY distance the spatial statistic reports -- the 10-20 um
    colocalization band, the 20-50 um co-infiltration band, the 75 um architecture bandwidth,
    and the certified cell error. So the test is now what a drawn bar actually looks like:
    near-black, solid, thin in absolute terms, much wider than tall, and the same length on
    every one of its rows.
    """
    import cv2
    import numpy as np

    ch, cw = crop.shape
    mask = (crop < 100).astype(np.uint8)          # drawn black, not merely "darker than Otsu"
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if bw < 40 or bw > 0.40 * image_width:
            continue                              # too short to measure, or not a bar
        if bh < 2 or bh > 0.10 * ch:
            continue                              # a bar is thin in ABSOLUTE terms
        if area / float(bw * bh) < 0.80:
            continue                              # solid rectangle, not a texture
        if bw / float(bh) < 6:
            continue                              # wide and thin
        rows = (labels[y:y + bh, x:x + bw] == i).sum(axis=1)
        if rows.std() / max(rows.mean(), 1.0) > 0.08:
            continue                              # same length on every row
        out.append({"length_px": bw, "thickness_px": bh, "x": x, "y": y,
                    "fill": round(area / float(bw * bh), 3)})
    out.sort(key=lambda c: -c["length_px"])
    return out


def _detect_scale_bar(image_path: str, bar_um: float = None):
    """Measure the burned-in scale bar. Returns (pixel_size_um, bar_length_px).

    FAILS CLOSED. If no component in the bottom strip is shaped like a bar, this returns
    (None, None) and the caller falls back to a value the user typed. That is the right
    trade: a missing number is visible and gets corrected, whereas a wrong one is invisible
    and rescales the whole analysis.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  Scale bar: opencv-python / numpy not available")
        return None, None

    bar_um = float(bar_um if bar_um else DEFAULT_SCALE_BAR_UM)
    arr = None
    try:
        from PIL import Image
        arr = np.array(Image.open(image_path).convert("RGB"))
    except Exception:
        bgr = cv2.imread(image_path)
        if bgr is not None:
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if arr is None:
        print(f"  Scale bar: could not load {os.path.basename(image_path)}")
        return None, None

    h, w = arr.shape[:2]
    crop = cv2.cvtColor(arr[int(h * 0.85):, :], cv2.COLOR_RGB2GRAY)
    cands = _bar_candidates(crop, w)
    if not cands:
        print(f"  Scale bar: no bar-shaped segment found in "
              f"{os.path.basename(image_path)} — enter the pixel size by hand")
        return None, None

    bar_len = cands[0]["length_px"]
    pixel_size = bar_um / bar_len
    extra = ""
    if len(cands) > 1:
        # More than one bar-shaped object is not fatal, but it is worth seeing: the widest
        # is taken and the runners-up are named so a wrong pick is noticeable.
        extra = (" · also matched " +
                 ", ".join(str(c["length_px"]) + "px" for c in cands[1:4]))
    print(f"  Scale bar: {bar_len}px = {bar_um:g} µm → {pixel_size:.4f} µm/px{extra}")
    return pixel_size, bar_len


def from_tiff_metadata(image_path: str):
    try:
        import tifffile
        with tifffile.TiffFile(image_path) as tif:
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels = root.find('.//ome:Pixels', ns)
                if pixels is not None:
                    px = pixels.get('PhysicalSizeX')
                    unit = pixels.get('PhysicalSizeXUnit', 'µm')
                    if px:
                        val = float(px)
                        if 'nm' in unit.lower():
                            val /= 1000
                        print(f"  Pixel size from OME metadata: {val} µm/px")
                        return val

            if tif.is_svs:
                desc = tif.pages[0].description
                match = re.search(r'MPP\s*=\s*([\d.]+)', desc)
                if match:
                    val = float(match.group(1))
                    print(f"  Pixel size from SVS metadata: {val} µm/px")
                    return val

            page = tif.pages[0]
            x_res = page.tags.get('XResolution')
            res_unit = page.tags.get('ResolutionUnit')
            if x_res and res_unit:
                try:
                    tag_val = x_res.value if hasattr(x_res, 'value') else x_res
                    if isinstance(tag_val, tuple):
                        num, den = tag_val
                    else:
                        num, den = float(tag_val), 1
                    res = num / den
                    unit_val = res_unit.value if hasattr(res_unit, 'value') else res_unit
                    if unit_val == 3 and res > 0:
                        val = 10000 / res
                        if 0.01 < val < 100:
                            print(f"  Pixel size from TIFF resolution: {val:.4f} µm/px")
                            return val
                except Exception:
                    pass
    except Exception:
        pass
    return None


def from_filename(image_path: str):
    filename = os.path.basename(image_path).lower()
    patterns = [
        r'[\-_\s](\d+)x[\-_\s\.]',
        r'[\-_\s]x(\d+)[\-_\s\.]',
        r'^(\d+)x[\-_\s\.]',
        r'[\-_\s](\d+)x$',
        r'[\-_]x(\d+)[\-_]',
        r'(\d+)x(?=\D|$)',
        r'x(\d+)(?=\D|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            mag = int(match.group(1))
            if mag in PIXEL_SIZE_MAP:
                pixel_size = PIXEL_SIZE_MAP[mag]
                print(f"  Pixel size from filename ({mag}x): {pixel_size} µm/px")
                return pixel_size
    return None


def prompt_user(image_name: str) -> float:
    print(f"\n  Could not auto-detect magnification for: {image_name}")
    while True:
        try:
            mag = int(input("  Enter magnification (e.g. 10, 20, 40): ").strip())
            if mag in PIXEL_SIZE_MAP:
                return PIXEL_SIZE_MAP[mag]
            else:
                return float(input("  Pixel size (µm/px): ").strip())
        except (ValueError, KeyboardInterrupt):
            print("  Using default 0.5 µm/px")
            return 0.5


def get_pixel_size_with_source(image_path: str, cfg: dict,
                               interactive: bool = False):
    """
    Resolve pixel size AND report where the value came from.

    Returns (pixel_size_um, source) where source is one of:
        "per_image_override" | "tiff_metadata" | "ui_default" |
        "filename" | "interactive" | "default_fallback"

    Priority (identical to the old get_pixel_size, which is now a thin wrapper
    around this so the quantification pipeline's resolved value is unchanged):
    1. Per-image override (pixel_overrides in config)
    2. TIFF/OME metadata
    3. Config default_pixel_size (set by user in UI — beats filename parsing)
    4. Filename parsing
    5. Interactive prompt
    6. Default fallback

    The `source` lets the spatial pipeline detect a silent fall-through to the
    hardcoded default ("default_fallback") and warn / record provenance, instead
    of mis-scaling the Ripley's K / DCLF band without anyone noticing.
    """
    default = cfg.get("default_pixel_size", 0.5)
    image_name = os.path.basename(image_path)
    print(f"  Detecting pixel size for: {image_name}")

    # 1. Per-image override from experiment overrides
    overrides = cfg.get("pixel_overrides", {})
    if image_name in overrides:
        val = float(overrides[image_name])
        print(f"  Pixel size from per-image override: {val} µm/px")
        return val, "per_image_override"

    # 2. TIFF/OME metadata
    val = from_tiff_metadata(image_path)
    if val:
        return val, "tiff_metadata"

    # 3. User-configured default (set in UI experiment page)
    # Only skip this if it's the fallback 0.5 AND magnification is auto
    # If user explicitly set a value in UI, use it before filename parsing
    if cfg.get("_pixel_size_from_ui", False):
        print(f"  Pixel size from UI config: {default} µm/px")
        return default, "ui_default"

    # 4. Filename parsing
    val = from_filename(image_path)
    if val:
        return val, "filename"

    # 5. Interactive prompt
    if interactive:
        return prompt_user(image_name), "interactive"

    # 6. Default fallback
    print(f"  Pixel size: using default {default} µm/px")
    return default, "default_fallback"


def get_pixel_size(image_path: str, cfg: dict, interactive: bool = False) -> float:
    """
    Get pixel size using the priority chain (see get_pixel_size_with_source).

    Thin wrapper returning only the value, so existing callers (the
    quantification pipeline) are byte-for-byte unchanged.
    """
    return get_pixel_size_with_source(image_path, cfg, interactive=interactive)[0]