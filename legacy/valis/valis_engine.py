"""
valis_engine.py — MAIN-venv bridge to VALIS-rigid as a registration engine.

ROLE (see docs/valis_certification_research.md): VALIS is a cross-modal REGISTRATION engine,
NOT a self-certifying gate. The API uses its global rigid transform as the PROVISIONAL for the
validated Fitzpatrick-West gate (LoFTR-in-ROI / manual landmarks); the structural residual below
is only a fail-closed GROSS-error QC. A single global rigid cannot reach cell scale on deforming
serial sections, and the structural residual texture-locks on near-identical hematoxylin, so it
must never be presented as a cell-scale certificate.

VALIS registers (in the isolated ~/valis_runtime venv, via valis_worker.py subprocess) and
returns a RIGID moving->reference similarity in original pixels. OASIS then CERTIFIES that
transform in-house with the STAIN-ROBUST structural patch-residual TRE (serial_registration's
patch_residual_flow + lumen_tre) — coordinate-clean and fail-closed, and (unlike LoFTR) it works
on cross-modal H&E<->IHC where LoFTR returns 0 matches.

The gate/ROI framework is unchanged: this is an alternative way to OBTAIN a transform to certify;
`certify_local_roi` calls it only when LoFTR fails. Only the rigid (distance-preserving) transform
is ever used — `assert_distance_preserving` is enforced here, main-side.
"""
import os
import sys
import json
import subprocess
import numpy as np

VALIS_PY = os.path.expanduser("~/valis_runtime/venv/bin/python")
_WORKER = "oasis.spatial.valis_worker"
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# structural-certification thresholds mirror the landmark gate (CERTIFICATION_GATES)
_CERT_UM = 5.0        # median residual <= this -> certified
_DEFORMED_UM = 15.0   # worst-region residual <= this -> analysable (radius-limited)


def valis_available():
    """True if the isolated VALIS runtime is present and importable."""
    if not os.path.exists(VALIS_PY):
        return False
    try:
        r = subprocess.run([VALIS_PY, "-c", "import valis"],
                           env={**os.environ, "DYLD_LIBRARY_PATH": "/opt/homebrew/lib"},
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _valis_transform(ref_path, mov_path, timeout=600):
    """Subprocess to the isolated worker -> rigid moving->reference matrix (original px)."""
    env = {**os.environ, "DYLD_LIBRARY_PATH": "/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")}
    try:
        r = subprocess.run([VALIS_PY, "-m", _WORKER], cwd=_PROJECT, env=env,
                           input=json.dumps({"ref_path": ref_path, "mov_path": mov_path}),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"VALIS worker timed out ({timeout}s)"}
    if r.returncode != 0:
        return {"ok": False, "error": f"VALIS worker exit {r.returncode}: {r.stderr[-400:]}"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"bad worker output: {e}; stderr {r.stderr[-300:]}"}


def _tissue_any(rgb):
    """Stain-agnostic tissue mask: anything that is NOT near-white background. Robust across
    H&E and IHC (unlike a hematoxylin threshold, whose extent differs between stains and would
    make cross-modal overlap collapse)."""
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    return (((g < 220) | (sat > 20))).astype(np.uint8)


_STRUCT_PX_FLOOR = 2.0   # µm/px: never MEASURE structural residual finer than this (see below)


def _structural_certify(ref_rgb, mov_rgb, M_work, px_work, roi_mask=None,
                        px_floor=_STRUCT_PX_FLOOR):
    """Fail-closed structural certification of a moving->reference transform (work coords).
    Measures residual on the low-frequency structural (hematoxylin) channel, which is shared
    across serial sections regardless of stain -> works cross-modal. Overlap uses a
    stain-agnostic tissue mask so cross-modal H&E<->IHC still overlaps.

    roi_mask : optional bool/uint8 array in the REFERENCE work frame. When given, the residual
    is measured ONLY inside it (per-ROI certification against a global whole-image transform),
    and the overlap fraction is reported relative to the ROI area.

    px_floor : the residual is never measured finer than this. cv2.phaseCorrelate on the near-
    identical hematoxylin texture of two serial sections locks onto local texture at fine
    resolution and UNDER-reports real anatomical deformation ("explains away" a true 24µm offset
    as ~0µm), which would over-certify. Flooring the measurement resolution keeps the residual a
    faithful, resolution-stable estimate of displacement (validated on ANHIR lung-lesion: with
    the floor the structural residual tracks the held-out landmark TRE; without it collapses)."""
    import cv2
    from oasis.spatial import serial_registration as sr

    A = np.asarray(M_work, float)[:2].copy()
    if px_work < px_floor:                      # downscale so the measurement can't texture-lock
        f = px_work / px_floor
        def _ds(im):
            return cv2.resize(im, (max(int(im.shape[1] * f), 8), max(int(im.shape[0] * f), 8)),
                              interpolation=cv2.INTER_AREA)
        ref_rgb, mov_rgb = _ds(ref_rgb), _ds(mov_rgb)
        A[:, 2] = A[:, 2] * f                   # similarity: translation scales, rotation/scale not
        if roi_mask is not None:
            roi_mask = cv2.resize((np.asarray(roi_mask) > 0).astype(np.uint8),
                                  (ref_rgb.shape[1], ref_rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        px_work = px_floor

    Hr, Wr = ref_rgb.shape[:2]
    ref_struct = sr.structural_channel(ref_rgb, px_work)
    mov_struct = sr.structural_channel(mov_rgb, px_work)
    ref_mask = sr.tissue_mask(ref_rgb, px_work)          # for lumen cross-check (hema-based)
    mov_mask = sr.tissue_mask(mov_rgb, px_work)
    ref_any = _tissue_any(ref_rgb); mov_any = _tissue_any(mov_rgb)   # for overlap (stain-agnostic)
    warped_struct = cv2.warpAffine(mov_struct, A, (Wr, Hr))
    warped_any = cv2.warpAffine(mov_any, A, (Wr, Hr), flags=cv2.INTER_NEAREST)
    overlap = ((ref_any > 0) & (warped_any > 0)).astype(np.float32)
    if roi_mask is not None:
        rm = (np.asarray(roi_mask) > 0).astype(np.float32)
        overlap = overlap * rm
        roi_area = float(rm.sum())
        ov_frac = float(overlap.sum() / roi_area) if roi_area > 0 else 0.0
    else:
        ov_frac = float(overlap.mean())

    recs = sr.patch_residual_flow(ref_struct, warped_struct, overlap, px_work)
    stats = sr.flow_stats(recs, ref_struct.shape)
    # independent cross-check: lumen-centroid TRE (anatomical, resolution-stable)
    lumen = None
    try:
        lt = sr.lumen_tre(ref_mask, mov_mask, A, px_work)
        lumen = lt.get("median_um") if isinstance(lt, dict) else None
    except Exception:
        pass

    med = stats.get("median_um"); reg_max = stats.get("region_max_um")
    # Take the MORE CONSERVATIVE of the texture residual and the anatomical lumen TRE so a
    # gross lumen disagreement can veto a texture pass (fail-closed).
    eff_med = med
    eff_reg = reg_max
    if lumen is not None and med is not None:
        eff_med = max(med, lumen)
        eff_reg = max(reg_max if reg_max is not None else 0.0, lumen)
    if med is None or ov_frac < 0.10:
        verdict, reason = "NOT_CERTIFIABLE", f"insufficient structural overlap ({ov_frac:.0%})"
    elif eff_med <= _CERT_UM and eff_reg <= _DEFORMED_UM:
        verdict, reason = "STRUCTURALLY_CERTIFIED", "structural residual within tolerance"
    elif eff_reg <= _DEFORMED_UM:
        verdict, reason = "RADIUS_LIMITED", "median above tolerance but no region deformed"
    else:
        verdict, reason = "NOT_CERTIFIED", "structural residual exceeds deformation limit"
    return {"verdict": verdict, "reason": reason, "median_um": med,
            "region_max_um": reg_max, "n_patches": stats.get("n"),
            "lumen_tre_um": lumen, "overlap_frac": round(ov_frac, 3)}


def valis_register_and_certify(ref_path, mov_path, pixel_size_um, work_max_side=1600):
    """Register moving->reference with VALIS-rigid and certify structurally.

    Returns dict: ok, verdict, matrix (moving->reference, ORIGINAL px, distance-preserving),
    median_um/region_max_um/lumen_tre_um, n_matches, source, plus error on failure.
    """
    from oasis.common.registration import _load_rgb_thumbnail
    from oasis.spatial import serial_registration as sr

    vt = _valis_transform(ref_path, mov_path)
    if not vt.get("ok"):
        return {"ok": False, "source": "valis_rigid", "error": vt.get("error", "VALIS failed")}
    M = np.asarray(vt["matrix"], float)                     # moving->reference, ORIGINAL px
    # enforce the invariant main-side: never let a non-similarity through
    try:
        sr.assert_distance_preserving(M, "valis_rigid")
    except Exception as e:
        return {"ok": False, "source": "valis_rigid", "error": f"non-similarity from VALIS: {e}"}

    ref_rgb, s_r = _load_rgb_thumbnail(os.path.expanduser(ref_path), max_side=work_max_side)
    mov_rgb, s_m = _load_rgb_thumbnail(os.path.expanduser(mov_path), max_side=work_max_side)
    if ref_rgb is None or mov_rgb is None:
        return {"ok": False, "source": "valis_rigid", "error": "could not load images for certification"}
    # scale the ORIGINAL-px transform into the reference working frame
    A = M[:2, :2] * (s_r / max(s_m, 1e-9))
    t = M[:2, 2] * s_r
    M_work = np.hstack([A, t.reshape(2, 1)])
    px_work = float(pixel_size_um) / max(s_r, 1e-9)

    cert = _structural_certify(ref_rgb, mov_rgb, M_work, px_work)
    cert.update({"ok": cert["verdict"] in ("STRUCTURALLY_CERTIFIED", "RADIUS_LIMITED"),
                 "source": "valis_rigid", "matrix": M.tolist(),
                 "n_matches": vt.get("n_matches"), "valis_rigid_rtre": vt.get("valis_rigid_rtre"),
                 "secs": vt.get("secs")})
    return cert


def register_crops_and_certify(crop_ref_rgb, crop_mov_rgb, pixel_size_um):
    """ROI path: register two RGB crop ARRAYS with VALIS-rigid and certify structurally.
    Writes the crops to temp files for the isolated worker; returns the transform in
    CROP-LOCAL coords (crop_mov -> crop_ref) plus the structural verdict. The caller
    (certify_local_roi) maps the crop-local matrix back to full-image coords.
    """
    import tempfile
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="valis_roi_")
    try:
        rp = os.path.join(tmp, "ref.png"); mp = os.path.join(tmp, "mov.png")
        Image.fromarray(np.asarray(crop_ref_rgb).astype(np.uint8)).save(rp)
        Image.fromarray(np.asarray(crop_mov_rgb).astype(np.uint8)).save(mp)
        return valis_register_and_certify(rp, mp, pixel_size_um,
                                          work_max_side=max(crop_ref_rgb.shape[:2]))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def valis_whole_matrix(ref_path, mov_path, timeout=900):
    """Register the WHOLE fixed<-moving with VALIS-rigid and return the distance-preserving
    moving->reference similarity in ORIGINAL full-res pixels (+ n_matches). VALIS is a
    whole-slide registrar: one global rigid transform is the correct unit — feeding it small
    ROI crops (located by a possibly-bad cross-modal provisional) mis-registers. Returns
    {ok, matrix, n_matches} or {ok: False, error}."""
    from oasis.spatial import serial_registration as sr
    vt = _valis_transform(os.path.expanduser(ref_path), os.path.expanduser(mov_path),
                          timeout=timeout)
    if not vt.get("ok"):
        return {"ok": False, "error": vt.get("error", "VALIS failed")}
    M = np.asarray(vt["matrix"], float)                    # moving->reference, ORIGINAL px
    try:
        sr.assert_distance_preserving(M, "valis_rigid")    # never let a non-similarity through
    except Exception as e:
        return {"ok": False, "error": f"non-similarity from VALIS: {e}"}
    return {"ok": True, "matrix": M, "n_matches": vt.get("n_matches"),
            "valis_rigid_rtre": vt.get("valis_rigid_rtre"), "secs": vt.get("secs")}


def certify_roi_with_matrix(ref_rgb, mov_rgb, M_work, px_work, roi_poly_work=None):
    """Certify ONE region against an already-computed (global) moving->reference transform,
    given in the REFERENCE work frame. Measures the stain-robust structural residual inside
    the ROI polygon (work-frame px); if roi_poly_work is None the whole overlap is used.
    Returns the same verdict dict shape as the whole-image path."""
    roi_mask = None
    if roi_poly_work is not None:
        import cv2
        Hr, Wr = ref_rgb.shape[:2]
        roi_mask = np.zeros((Hr, Wr), np.uint8)
        poly = np.round(np.asarray(roi_poly_work, float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(roi_mask, [poly], 1)
    return _structural_certify(ref_rgb, mov_rgb, M_work, px_work, roi_mask=roi_mask)


if __name__ == "__main__":
    # smoke: python -m oasis.spatial.valis_engine <ref> <mov> <px_um>
    print("valis_available:", valis_available())
    if len(sys.argv) >= 4:
        out = valis_register_and_certify(sys.argv[1], sys.argv[2], float(sys.argv[3]))
        out.pop("matrix", None)
        print(json.dumps(out, indent=1))
