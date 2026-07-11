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


def _detect_scale_bar(image_path: str):
    """
    Detect the burned-in scale bar in the bottom strip of an image and measure
    its length in pixels.

    Returns (pixel_size_um, bar_length_px) on a confident detection, else
    (None, None). The public wrapper extract_pixel_size_from_scale_bar() returns
    only the pixel size; the API uses this richer form to surface the bar length
    for user confidence display.

    Assumes the bar represents 100 µm (constant across this dataset).
    Confidence: longest horizontal segment must be > 50 px and < 40% of the
    image width.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  Scale bar: opencv-python / numpy not available")
        return None, None

    # Load with PIL, fall back to cv2
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
    crop = arr[int(h * 0.85):, :]                      # bottom 15%
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Otsu — bar is dark on light background, so invert to make the bar white
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Isolate horizontal segments: open with a horizontal kernel (erodes away
    # vertical/short structures like text, keeps long horizontal runs).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    max_h_for_line = max(0.25 * crop.shape[0], 10)

    # Collect line-like candidate segments (short enough to be a bar), longest first
    candidates = []
    for i in range(1, n):                              # skip background label 0
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if ch <= max_h_for_line:                       # line-like: wide & short
            candidates.append((cw, ch))
    candidates.sort(key=lambda c: c[0], reverse=True)
    bar_len = candidates[0][0] if candidates else 0

    if bar_len <= 50 or bar_len >= 0.40 * w:
        print(f"  Scale bar: no confident detection "
              f"(longest segment {bar_len}px, image width {w}px)")
        return None, None

    pixel_size = 100.0 / bar_len
    print(f"  Scale bar: {bar_len}px → {pixel_size:.4f} µm/px (assuming 100 µm bar)")
    return pixel_size, bar_len


def extract_pixel_size_from_scale_bar(image_path: str):
    """
    Extract pixel size (µm/px) from a burned-in 100 µm scale bar.
    Returns a float, or None if no scale bar is confidently detected.
    """
    return _detect_scale_bar(image_path)[0]


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