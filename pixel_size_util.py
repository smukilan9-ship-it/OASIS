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

PIXEL_SIZE_MAP = {
    4:   2.50,
    10:  1.00,
    20:  0.50,
    40:  0.25,
    60:  0.165,
    100: 0.10,
}


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


def get_pixel_size(image_path: str, cfg: dict, interactive: bool = False) -> float:
    """
    Get pixel size using priority chain.

    Priority:
    1. Per-image override (pixel_overrides in config)
    2. TIFF/OME metadata
    3. Config default_pixel_size (set by user in UI — beats filename parsing)
    4. Filename parsing
    5. Interactive prompt
    6. Default fallback
    """
    default = cfg.get("default_pixel_size", 0.5)
    image_name = os.path.basename(image_path)
    print(f"  Detecting pixel size for: {image_name}")

    # 1. Per-image override from experiment overrides
    overrides = cfg.get("pixel_overrides", {})
    if image_name in overrides:
        val = float(overrides[image_name])
        print(f"  Pixel size from per-image override: {val} µm/px")
        return val

    # 2. TIFF/OME metadata
    val = from_tiff_metadata(image_path)
    if val:
        return val

    # 3. User-configured default (set in UI experiment page)
    # Only skip this if it's the fallback 0.5 AND magnification is auto
    # If user explicitly set a value in UI, use it before filename parsing
    if cfg.get("_pixel_size_from_ui", False):
        print(f"  Pixel size from UI config: {default} µm/px")
        return default

    # 4. Filename parsing
    val = from_filename(image_path)
    if val:
        return val

    # 5. Interactive prompt
    if interactive:
        return prompt_user(image_name)

    # 6. Default fallback
    print(f"  Pixel size: using default {default} µm/px")
    return default