"""
validate_e2e_knownwarp_deepliif.py — Validation B of the end-to-end suite.

REAL DAB PIXELS + FULL PIPELINE, known-warp reconstruction (ihc.md §10).

The real-DAB cell-scale end-to-end ground truth we want (two DIFFERENT chromogenic
markers on corresponding sections with a known association) does not exist. This
validation bounds one side of that gap: it proves that REAL chromogenic DAB pixels
flow correctly through the WHOLE pipeline (InstanSeg segmentation → registration
reconstruction → cross-type Ripley's K), using a geometric ground truth we CAN
construct — a known affine warp.

Design (per DeepLIIF tile; the leftmost 512×512 panel is the real IHC/DAB image):
  section 1  = the real IHC panel (reference)
  section 2  = the same panel under a KNOWN affine warp (rotation + translation)
  → segment BOTH with the real pipeline → register section2→section1 →
    (i) reconstruction TRE = residual of  T(M(grid)) vs grid   (must be small)
    (ii) cross-K of ref cells vs T-aligned moving cells → same population → ASSOCIATED
    (iii) NECESSITY control: cross-K of ref vs UN-aligned moving cells → must differ
          (misregistration breaks the answer, so registration is doing real work)

Proves: real DAB pixels segment + register + feed the statistic correctly at cell
scale. Does NOT prove: two different markers (association here is same-marker/trivial)
— that is Validation A (render CODEX cross-marker truth into brightfield). Together
with A and the CODEX degradation keystone, the three bound the untestable real case.

Reference: DeepLIIF (Ghahremani et al. 2022). Needs QuPath + InstanSeg. Long-running.
  python validation/validate_e2e_knownwarp_deepliif.py [--tiles N] [--nperm M]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatial_stats as ss
from validation.datasets import resolve as R

PIXEL_SIZE_UM = 0.25                       # DeepLIIF is 40× (~0.25 µm/px)
PANEL = 512                                # IHC panel is the leftmost 512×512
RADII = np.arange(0.0, 101.0, 4.0)
WARP = (12.0, 55.0, -40.0)                 # known (angle_deg, tx_px, ty_px); large
                                            # enough that UN-registered breaks the verdict
                                            # (displacement ≫ the 10–50 µm band) yet the
                                            # same-image content still registers cleanly
TRE_TOL_UM = 5.0


def _setup():
    import yaml
    p = os.path.expanduser("~/.ihc_analyzer/setup.yaml")
    s = yaml.safe_load(open(p)) if os.path.exists(p) else {}
    return {
        "qupath": os.path.expanduser(s.get(
            "qupath_binary", "/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64")),
        "model": os.path.expanduser(s.get(
            "instanseg_model", "~/QuPath/v0.7/instanseg/downloaded/brightfield_nuclei-0.1.1")),
        "device": s.get("device", "mps"),
    }


def _ihc_panel(tile_path):
    """Leftmost 512×512 panel = the real IHC/DAB brightfield image."""
    from PIL import Image
    im = np.asarray(Image.open(tile_path).convert("RGB"))
    return im[:, :PANEL, :]


def _warp(img, angle, tx, ty):
    import cv2
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))
    return warped, M


def _apply(M, pts):
    if len(pts) == 0:
        return pts
    return (np.asarray(M, float) @ np.c_[pts, np.ones(len(pts))].T).T


def _segment(img, workdir, setup):
    """Run the REAL pipeline (QuPath/InstanSeg) on an RGB array → Nx2 cell centroids."""
    import yaml
    from PIL import Image
    os.makedirs(workdir, exist_ok=True)
    ind = os.path.join(workdir, "in"); os.makedirs(ind, exist_ok=True)
    name = "tile.png"
    Image.fromarray(img).save(os.path.join(ind, name))
    cfg = {"mode": "automated", "stain_type": "hdab", "input_dir": ind,
           "output_dir": workdir, "dashboard_dir": os.path.join(workdir, "_dash"),
           "qupath_binary": setup["qupath"], "instanseg_model": setup["model"],
           "device": setup["device"], "instanseg_threads": 4,
           "default_pixel_size": PIXEL_SIZE_UM, "dab_threshold": 0.15,
           "export_geojson": True, "generate_overlays": False,
           "image_extensions": ["*.png"]}
    cfgp = os.path.join(workdir, "cfg.yaml"); yaml.safe_dump(cfg, open(cfgp, "w"))
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "run_pipeline.py"), "--config", cfgp, "--mode", "quant"],
        capture_output=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=900)
    geo = glob.glob(os.path.join(workdir, "*_detections.geojson"))
    pts = []
    if geo:
        for ft in json.load(open(geo[0])).get("features", []):
            g = ft.get("geometry", {}) or {}; c = g.get("coordinates") or []
            ring = c[0] if g.get("type") == "Polygon" and c else None
            if ring:
                pts.append(np.asarray(ring, float).mean(0))
    return np.asarray(pts) if pts else np.zeros((0, 2))


def _verdict(a, b, n_perm):
    if len(a) < 12 or len(b) < 12:
        return "insufficient_cells"
    o = ss.cross_k_all_nulls(a, b, RADII, float(PANEL * PANEL), PIXEL_SIZE_UM,
                             n_perm=n_perm, seed=0)
    return o["robustness"]["verdict"]


def run(n_tiles, n_perm):
    import serial_registration as sr
    setup = _setup()
    tiles = sorted(glob.glob(os.path.join(
        R.resolve("deepliif"), "DeepLIIF_Testing_Set", "*.png")))[:n_tiles]
    work = tempfile.mkdtemp(prefix="oasis_e2e_b_")
    grid = np.array([[x, y] for x in np.linspace(40, PANEL - 40, 5)
                     for y in np.linspace(40, PANEL - 40, 5)], float)
    rows = []
    for i, t in enumerate(tiles):
        ref = _ihc_panel(t)
        mov, M = _warp(ref, *WARP)
        a = _segment(ref, os.path.join(work, f"{i}_ref"), setup)
        bm = _segment(mov, os.path.join(work, f"{i}_mov"), setup)
        if len(a) < 12 or len(bm) < 12:
            rows.append({"tile": os.path.basename(t), "skip": "too few cells"}); continue
        reg = sr.register_similarity(ref, mov, PIXEL_SIZE_UM)
        T = np.asarray(reg["matrix"], float)
        tre = float(np.median(np.linalg.norm(_apply(T, _apply(M, grid)) - grid, axis=1)) * PIXEL_SIZE_UM)
        v_reg = _verdict(a, _apply(T, bm), n_perm)      # registered → same cells → associated
        v_noreg = _verdict(a, bm, n_perm)               # necessity control
        rows.append({"tile": os.path.basename(t), "n_ref": len(a), "n_mov": len(bm),
                     "recon_tre_um": round(tre, 2), "struct_dice": round(float(reg.get("struct_dice", 0)), 3),
                     "verdict_registered": v_reg, "verdict_noreg": v_noreg,
                     "reg_necessary": v_reg != v_noreg})
        r = rows[-1]
        print(f"  {r['tile']:<34} TRE={r['recon_tre_um']:>5}µm  reg={v_reg:<10} "
              f"noreg={v_noreg:<10} necessary={r['reg_necessary']}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--nperm", type=int, default=199)
    args = ap.parse_args(argv)

    print(f"\nEnd-to-end B — real-DAB known-warp reconstruction "
          f"(DeepLIIF, warp={WARP}, {args.tiles} tiles)\n")
    rows = run(args.tiles, args.nperm)
    scored = [r for r in rows if "recon_tre_um" in r]
    if not scored:
        print("\n  RESULT: FAIL (no tile produced enough cells)"); return 1

    tre_ok = float(np.median([r["recon_tre_um"] for r in scored]))
    n_assoc = sum(1 for r in scored if r["verdict_registered"] in ("robust", "csr_only"))
    n_nec = sum(1 for r in scored if r["reg_necessary"])
    metrics = {"n_tiles": len(scored), "median_recon_tre_um": round(tre_ok, 2),
               "tre_tol_um": TRE_TOL_UM,
               "n_registered_associated": n_assoc,
               "n_registration_necessary": n_nec, "table": rows}
    print("\n##METRICS## " + json.dumps(metrics))

    ok = (tre_ok <= TRE_TOL_UM and n_assoc >= max(1, len(scored) // 2)
          and n_nec >= max(1, len(scored) // 2))
    print(f"\n  median reconstruction TRE={tre_ok:.2f}µm (≤{TRE_TOL_UM}) · "
          f"registered-associated {n_assoc}/{len(scored)} · "
          f"registration-necessary {n_nec}/{len(scored)}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
