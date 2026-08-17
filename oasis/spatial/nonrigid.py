"""Non-rigid registration via DeeperHistReg (RegWSI).

WHY THIS REGISTRAR. research/ihc.md § 23.8: RegWSI (Wodziński et al.) won ACROBAT 2023 at
0.2 % average rTRE with 0.9898 robustness, has cell-level accuracy on HyReCo re-stained, and
ships open as DeeperHistReg. VALIS is the most *deployed* registrar, not the most accurate;
it is the benchmark baseline in § 7.1 for that reason and is not what to adopt from.

WHAT THIS IS NOT. The warp is never certified. § 23.10 established that no function of a
displacement field predicts whether that field put cells in the right place — a transform
cannot certify itself. This module only produces the alignment. Certification stays with
`landmark_register_and_verify`, which fits a similarity to correspondences carrying an
independently measured localisation error, and it must be run on correspondences the warp
never saw. Re-deriving matches from the warped image and certifying those is circular: the
warp has already made the images agree, so a similarity fits by construction and the
certificate passes on everything.

Also from § 23.7: non-rigid helps below ~21 µm of displacement and destroys above ~55 µm, and
a third of replicates are wrong even at its best. It is an alignment tool to be checked, not
a fix to be trusted.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

# DeeperHistReg's defaults are built for whole slides: it resamples to 20 % and works at a
# 4096 px initial resolution. A 1920 x 1440 field would be thrown away by that, so the
# resampling is disabled and the working resolution set to the frame itself.
_FIELD_RESOLUTION = 2048


def available():
    """Is the registrar importable? Kept cheap so callers can offer the option or hide it."""
    try:
        import deeperhistreg  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        # MPS is deliberately not offered: DeeperHistReg's optimiser hits operators that
        # silently fall back to CPU on Metal, which is slower than asking for CPU outright.
        return "cpu"
    except Exception:
        return "cpu"


def _set_device(node, dev):
    """Rewrite every device/cuda key anywhere in the parameter tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                _set_device(v, dev)
            elif k == "cuda":
                node[k] = dev.startswith("cuda")
            elif k == "device" or (isinstance(v, str) and v.startswith("cuda")):
                node[k] = dev
    elif isinstance(node, list):
        for v in node:
            _set_device(v, dev)


def warp_image(ref_path, mov_path, out_dir=None, fast=False, device=None, case_name="pair"):
    """Register `mov` onto `ref` non-rigidly.

    Returns a dict with the warped image as an array plus the paths DeeperHistReg wrote, or
    {"error": ...}. The moving image is the one that moves; the reference is untouched, which
    matches the convention everywhere else in the pipeline (CD8 is the reference).
    """
    if not available():
        return {"error": "deeperhistreg is not installed"}
    import deeperhistreg
    from deeperhistreg import configs

    tmp = out_dir or tempfile.mkdtemp(prefix="oasis_nonrigid_")
    os.makedirs(tmp, exist_ok=True)
    params = (configs.default_initial_nonrigid_fast() if fast
              else configs.default_initial_nonrigid())
    dev = device or _device()
    # Set device EVERYWHERE it appears rather than at the keys this version happens to use:
    # the presets carry it at four nested paths, and missing one fails deep inside the run
    # with "Torch not compiled with CUDA enabled" and no output.
    _set_device(params, dev)
    params["echo"] = False
    params["save_final_images"] = True
    params["save_final_displacement_field"] = True

    # Full resolution in, full resolution out: these are single fields, not slides.
    params["loading_params"]["source_resample_ratio"] = 1.0
    params["loading_params"]["target_resample_ratio"] = 1.0
    params["preprocessing_params"]["initial_resolution"] = _FIELD_RESOLUTION

    cfg = {
        "registration_parameters": params,
        "source_path": str(mov_path),           # the image that moves
        "target_path": str(ref_path),           # the fixed reference
        "output_path": str(tmp),
        "case_name": case_name,
        "temporary_path": os.path.join(tmp, "work"),
        "copy_target": False,
        "save_displacement_field": True,
        # Keep the intermediate results: the warped image is copied out of them, and when a
        # run goes wrong they are the only record of how far it got.
        "delete_temporary_results": False,
    }

    # run_registration catches its own exceptions and only prints them, so a failure here
    # returns normally with nothing written. The outputs are the only reliable signal.
    try:
        deeperhistreg.run_registration(**cfg)
    except Exception as e:                                          # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "output_dir": tmp}

    warped = _find(tmp, ("warped_source",), (".tiff", ".tif", ".png", ".jpg"))
    field = _find(tmp, ("displacement_field",), (".mha", ".npy", ".tiff"))
    out = {"output_dir": tmp, "warped_path": warped, "displacement_field_path": field,
           "device": dev}
    if warped:
        try:
            from oasis.quant.cell_expansion import _load_rgb_full
            out["warped"] = _load_rgb_full(warped)[..., :3]
        except Exception as e:                                      # noqa: BLE001
            out["warp_load_error"] = str(e)
    else:
        out["error"] = "registration produced no warped image"
    return out


def _find(root, name_parts, exts):
    """DeeperHistReg's output layout varies by version; locate by name and extension."""
    hits = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            low = n.lower()
            if any(p in low for p in name_parts) and low.endswith(tuple(exts)):
                hits.append(os.path.join(dirpath, n))
    if not hits:
        return None
    return max(hits, key=os.path.getsize)


def warp_points(displacement_field_path, points_xy):
    """Push points through a saved displacement field.

    This is the piece certification needs: correspondences taken from the UNWARPED pair are
    carried through the warp and their residual measured against the reference. Those points
    were never available to the registrar, so the residual is a held-out error rather than a
    restatement of the fit.
    """
    if not displacement_field_path or not os.path.exists(displacement_field_path):
        return None
    try:
        if displacement_field_path.lower().endswith(".mha"):
            import SimpleITK as sitk           # DeeperHistReg writes .mha, not .npy
            field = sitk.GetArrayFromImage(sitk.ReadImage(displacement_field_path))
        else:
            field = np.load(displacement_field_path)
    except Exception:                                               # noqa: BLE001
        return None
    if field.ndim != 3 or field.shape[0] not in (2,) and field.shape[-1] not in (2,):
        return None
    if field.shape[0] == 2:                       # (2, H, W) -> (H, W, 2)
        field = np.transpose(field, (1, 2, 0))
    h, w = field.shape[:2]
    pts = np.asarray(points_xy, float)
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return pts + field[yi, xi]


def cleanup(result):
    """Remove a temporary output directory created by `warp_image`."""
    d = (result or {}).get("output_dir")
    if d and os.path.isdir(d) and os.path.basename(d).startswith("oasis_nonrigid_"):
        shutil.rmtree(d, ignore_errors=True)
