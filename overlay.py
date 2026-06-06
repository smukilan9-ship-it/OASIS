"""
overlay.py
Generates cell boundary overlays from QuPath GeoJSON + original image.
Uses actual cell polygon boundaries exported from QuPath.
"""

import os
import json
import numpy as np
from pathlib import Path


def generate_overlay(
    image_path: str,
    geojson_path: str,
    output_path: str,
    pixel_size_um: float = 0.5,
    downsample: float = 1.0,
    pos_color: tuple = (255, 50, 50),    # bright red — high contrast on H-DAB
    neg_color: tuple = (50, 205, 50),    # lime green — high contrast on blue/purple nuclei
    line_thickness: int = 2,             # thicker for visibility at downsample
    show_negative: bool = True,
) -> str:
    """
    Draw cell boundary overlays on original image using QuPath GeoJSON.

    Parameters
    ----------
    image_path      : path to original image
    geojson_path    : path to QuPath GeoJSON export
    output_path     : where to save overlay PNG
    pixel_size_um   : microns per pixel (for coordinate conversion)
    downsample      : output image downsample factor (2 = half size)
    pos_color       : RGB color for positive cells
    neg_color       : RGB color for negative cells
    line_thickness  : outline thickness in pixels
    show_negative   : draw negative cell outlines too

    Returns
    -------
    output_path on success, None on failure
    """
    try:
        import cv2
    except ImportError:
        print("  Installing opencv-python...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
        import cv2

    from PIL import Image

    if not os.path.exists(image_path):
        print(f"  ERROR: Image not found: {image_path}")
        return None

    if not os.path.exists(geojson_path):
        print(f"  ERROR: GeoJSON not found: {geojson_path}")
        return None

    # Load original image
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)

    # Downsample if needed
    if downsample != 1.0:
        new_w = int(img.shape[1] / downsample)
        new_h = int(img.shape[0] / downsample)
        img = cv2.resize(img, (new_w, new_h))

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load GeoJSON
    with open(geojson_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        print("  WARNING: No features in GeoJSON")
        return None

    pos_count = 0
    neg_count = 0
    skipped = 0

    for feature in features:
        props = feature.get("properties", {})
        classification = props.get("classification", {})

        if isinstance(classification, dict):
            cls_name = classification.get("name", "Negative")
        else:
            cls_name = str(classification) if classification else "Negative"

        is_positive = "positive" in cls_name.lower()

        if not is_positive and not show_negative:
            continue

        color_rgb = pos_color if is_positive else neg_color
        # OpenCV uses BGR
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        geometry = feature.get("geometry", {})
        geo_type = geometry.get("type", "")
        coords = geometry.get("coordinates", [])

        if not coords:
            skipped += 1
            continue

        try:
            if geo_type == "Polygon":
                # coords = [[[x,y], [x,y], ...]]
                for ring in coords:
                    # QuPath exports in pixel coordinates
                    pts = np.array([[int(x / downsample), int(y / downsample)]
                                    for x, y in ring], dtype=np.int32)
                    cv2.polylines(img_bgr, [pts], isClosed=True,
                                  color=color_bgr, thickness=line_thickness)

            elif geo_type == "MultiPolygon":
                for polygon in coords:
                    for ring in polygon:
                        pts = np.array([[int(x / downsample), int(y / downsample)]
                                        for x, y in ring], dtype=np.int32)
                        cv2.polylines(img_bgr, [pts], isClosed=True,
                                      color=color_bgr, thickness=line_thickness)

            if is_positive:
                pos_count += 1
            else:
                neg_count += 1

        except Exception as e:
            skipped += 1
            continue

    print(f"  Overlay: {pos_count} positive + {neg_count} negative cells drawn ({skipped} skipped)")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"  Overlay saved: {output_path}")
    return output_path


def generate_overlays_for_batch(
    batch_metrics: list,
    input_dir: str,
    output_dir: str,
    pixel_size_um: float = 0.5,
    downsample: float = 1.0,
):
    """
    Generate overlays for all images in a batch.
    Looks for GeoJSON files matching each image.
    """
    results = []

    for metrics in batch_metrics:
        # Image_Name is like "LL477_CD8_x10_1.tif - LL477_CD8_x10_1.tif #1"
        # We want just the filename without extension for searching
        raw_name = metrics["Image_Name"].split(" - ")[0]  # "LL477_CD8_x10_1.tif"
        image_name = os.path.splitext(raw_name)[0]  # "LL477_CD8_x10_1"

        # Try original extension first, then common ones
        extensions = [os.path.splitext(raw_name)[1]] + [".tif", ".tiff", ".png", ".jpg"]
        img_path = None
        for ext in extensions:
            candidate = os.path.join(input_dir, image_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  WARNING: Could not find image for {image_name}")
            continue

        import glob
        # QuPath adds full image name + #1 to filename, use glob to find it
        matches = glob.glob(os.path.join(output_dir, f"{image_name}*_detections.geojson"))
        if matches:
            geojson_path = matches[0]
        else:
            print(f"  WARNING: No GeoJSON found for {image_name}")
            continue

        overlay_path = os.path.join(output_dir, image_name + "_overlay.png")

        result = generate_overlay(
            image_path=img_path,
            geojson_path=geojson_path,
            output_path=overlay_path,
            pixel_size_um=metrics.get("Pixel_Size_um", pixel_size_um),
            downsample=downsample,
        )
        if result:
            results.append(result)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python overlay.py <image_path> <geojson_path> <output_path>")
        sys.exit(1)
    generate_overlay(sys.argv[1], sys.argv[2], sys.argv[3])