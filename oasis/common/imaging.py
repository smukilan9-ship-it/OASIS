"""Two things the UI and the pipeline must agree on exactly: what counts as a scale
image, and what "normalized" means.

Both were duplicated before. The scale-image test lived only in the webui's batch
matcher, so the *matcher* knew a `*_scale.tif` was a photograph of a ruler while the
*pipeline* did not -- it globbed the folder and segmented the scale bars as if they were
tissue, producing rows in results.csv for images containing no cells. The white-point
estimate was written out twice, in the preview and in the pipeline, and both copies
carried the same bug (below), so a slide could fail to normalize while the preview
claimed it had.
"""
from pathlib import Path

import numpy as np

# Filename convention for a scale-bar photograph. It is a convention rather than a
# measurement because the alternative -- deciding from the pixels whether an image is a
# ruler or a tissue section -- fails silently in both directions, and a researcher can
# always rename a file.
SCALE_TOKEN = "scale"


def is_scale_image(name) -> bool:
    """True if `name` is a scale-bar photograph rather than an analysis image."""
    return SCALE_TOKEN in Path(str(name)).stem.lower()


def split_scale_images(names):
    """(analysis, scale) -- the same list partitioned, order preserved."""
    names = list(names or [])
    return ([n for n in names if not is_scale_image(n)],
            [n for n in names if is_scale_image(n)])


def estimate_white_point(rgb) -> np.ndarray:
    """Per-channel background white point of one image, as float RGB in 0-255.

    The slide's own background is taken to be white, so each channel is scaled to send
    it to 255. The estimate reads the 99th percentile of the brightest fifth of pixels:
    bright enough to be background, robust enough not to be set by a single blown-out
    pixel. Clipped to >=200 so a section that fills the whole frame -- with no true
    background to measure -- cannot have its tissue scaled up as if it were background.

    THE BUG THIS FIXES: the brightest fifth used to be selected with a strict `>`
    against the 80th percentile. On a slide whose background is saturated white over
    more than a fifth of the frame -- which is most 10X fields of a small biopsy -- the
    80th percentile IS 255, nothing is strictly greater, and the selection came back
    empty. `np.percentile` of an empty array then raised `index -1 is out of bounds`.
    In the preview that surfaced as "could not render"; in the pipeline it was caught,
    logged as one line among hundreds, and the image was segmented WITHOUT the
    normalization the operator had asked for. Four of the fourteen images in the two
    test folders hit it. `>=` keeps the saturated pixels, which is the correct answer:
    a white background means the white point is 255 and the correction is the identity.
    """
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if flat.size == 0:
        return np.array([255.0, 255.0, 255.0])
    lum = flat.mean(1)
    bright = flat[lum >= np.percentile(lum, 80)]
    if bright.size == 0:                       # single-valued image
        bright = flat
    return np.clip(np.percentile(bright, 99, axis=0), 200, 255)


def white_balance(rgb):
    """Scale each channel so the image's own background maps to white.

    Corrects tone and illumination, which vary slide to slide. It does NOT rescale DAB
    relative to hematoxylin -- the same linear factor is applied to every pixel of a
    channel -- which is why normalization cannot by itself move a positivity count.
    Returns (uint8 image, white point).
    """
    arr = np.asarray(rgb)
    wp = estimate_white_point(arr)
    out = np.clip(arr.astype(np.float64) * (255.0 / wp.reshape(1, 1, 3)),
                  0, 255).astype(np.uint8)
    return out, wp
