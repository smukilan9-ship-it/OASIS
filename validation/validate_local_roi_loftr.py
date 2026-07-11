"""
End-to-end test of certify_local_roi: draw a small ROI, let LoFTR match INSIDE it, fit a
local rigid transform, and run the ordinary Fitzpatrick-West gate windowed to that ROI.
Ground-truthed by held-out annotator JB inside the same ROI.

WHAT MUST BE TRUE for the draw-your-own-ROI feature to be honest:
  - on good tissue (lung-lesion), a small ROI should certify AND its realized error at the
    independent JB landmarks inside the ROI should actually be small (the pass is earned);
  - on bad tissue (mammary), it should still REFUSE (FLE floor too high / deformation rough),
    i.e. the feature does not become a way to manufacture a pass.

Run:  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python validation/validate_local_roi_loftr.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import serial_registration as sr          # noqa: E402
import loftr_matcher as lm                # noqa: E402

NATIVE_PX_100 = {"lung-lesion": 0.174, "lung-lobes": 1.274, "mammary-gland": 2.294}
LANDMARK_SCALE_PC = {"lung-lesion": 50, "lung-lobes": 100, "mammary-gland": 100}
IMG_ROOT = "/Volumes/Expansion/registration/anhir_cima"
# (tissue, fixed, moving, [ROI radii in mm to sweep]) — small lung ROIs where LoFTR's
# density matters and hand-landmarks run out; one big mammary ROI as the refuse-check.
PAIRS = [
    ("lung-lesion_3", "29-041-Izd2-w35-He-les3", "29-041-Izd2-w35-proSPC-4-les3",
     [0.60, 0.40, 0.30, 0.20]),
    ("mammary-gland_2", "s2_63-HE_A4926-4L", "s2_68-ER-A4962-4L", [4.0]),
]


def _prefix(t):
    for p in NATIVE_PX_100:
        if t.startswith(p):
            return p
    raise KeyError(t)


def load_xy(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]
    xi = hdr.index("x") if "x" in hdr else -2
    yi = hdr.index("y") if "y" in hdr else -1
    out = []
    for r in rows[1:]:
        try:
            out.append([float(r[xi]), float(r[yi])])
        except (ValueError, IndexError):
            pass
    return np.array(out, float)


def find_lm(tissue, user, base):
    for root in (os.path.join(HERE, "public_landmarks", "annotations"),
                 os.path.expanduser("~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations")):
        hits = glob.glob(os.path.join(root, tissue, f"user-{user}_scale-*", base + ".csv"))
        if hits:
            return hits[0]
    return None


def find_image(tissue, base):
    for ext in ("jpg", "png"):
        hits = glob.glob(os.path.join(IMG_ROOT, tissue, "scale-50pc", base + "." + ext))
        if hits:
            return hits[0]
    return None


def load_rgb(path):
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def one_pair(tissue, fixed, moving, radii_mm):
    import cv2
    from matplotlib.path import Path as MplPath
    print(f"\n{'-'*74}\n{tissue}")
    ir, im = find_image(tissue, fixed), find_image(tissue, moving)
    if not ir or not im:
        print("  images missing -- skipped"); return
    lp = {(u, k): find_lm(tissue, u, b) for u in ("PS", "JB")
          for k, b in (("ref", fixed), ("mov", moving))}
    if any(v is None for v in lp.values()):
        print("  annotations missing -- skipped"); return
    P = {k: load_xy(v) for k, v in lp.items()}
    n = min(len(v) for v in P.values())
    f = 50 / LANDMARK_SCALE_PC[_prefix(tissue)]                 # -> 50pc image pixels
    ps_r, ps_m = P[("PS", "ref")][:n] * f, P[("PS", "mov")][:n] * f
    jb_r, jb_m = P[("JB", "ref")][:n] * f, P[("JB", "mov")][:n] * f
    px = NATIVE_PX_100[_prefix(tissue)] * 2.0                   # 50pc um/px
    rgb_r, rgb_m = load_rgb(ir), load_rgb(im)
    M_global = sr._fit_similarity_robust(ps_m, ps_r)            # provisional (mov->ref)

    print(f"  {'ROI radius':>10}  {'verdict':>17}  {'source':>16}  {'n_corr':>6}  "
          f"{'FLE':>5}  {'JB realized p90 (local | global)':>32}")
    for radius_mm in radii_mm:
        R = radius_mm * 1000.0 / px                             # px
        # a circular-ish ROI a user might draw, centred on the densest neighbourhood
        cnt = np.array([np.sum(np.linalg.norm(ps_r - c, axis=1) <= R) for c in ps_r])
        center = ps_r[int(np.argmax(cnt))]
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        roi = np.c_[center[0] + R * np.cos(theta), center[1] + R * np.sin(theta)]
        cert = lm.certify_local_roi(rgb_r, rgb_m, roi, px, provisional_matrix=M_global,
                                    fallback_ref_lm=ps_r, fallback_mov_lm=ps_m)
        jb_in = MplPath(roi).contains_points(jb_r)
        loc = glob_ = float('nan')
        if cert.get("local_matrix") is not None and jb_in.sum() >= 4:
            Ml = np.asarray(cert["local_matrix"], float)
            loc = float(np.percentile(np.linalg.norm(
                sr._apply_affine(jb_m[jb_in], Ml) - jb_r[jb_in], axis=1) * px, 90))
            glob_ = float(np.percentile(np.linalg.norm(
                sr._apply_affine(jb_m[jb_in], M_global) - jb_r[jb_in], axis=1) * px, 90))
        fle = cert.get("fle_um_loftr")
        print(f"  {radius_mm:8.2f}mm  {str(cert.get('verdict')):>17}  {str(cert.get('source')):>16}  "
              f"{str(cert.get('n_correspondences')):>6}  {(f'{fle:.2f}' if fle else '  -- '):>5}  "
              f"{loc:12.1f} | {glob_:8.1f} um  (n={int(jb_in.sum())})")


def main():
    print("=" * 74)
    print("certify_local_roi end-to-end: LoFTR-in-ROI + local rigid fit + FW gate")
    print("=" * 74)
    if not os.path.isdir(IMG_ROOT):
        print("image root missing"); return 2
    for p in PAIRS:
        one_pair(*p)
    print("\n" + "=" * 74)
    print("Expect: lung-lesion ROI local-fit error << global; mammary still refuses/high.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
