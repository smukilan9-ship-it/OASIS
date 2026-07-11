"""
Positive demo of certify_local_roi on the real LL477 CD8/TIM-3 pair: auto-register, then
certify a drawn ROI via LoFTR-in-ROI + local rigid fit + the ordinary FW gate. Renders the
winning ROI on both sections.

This is the demonstration ANHIR could not give (ANHIR tissue is too deformed to certify
anywhere). LL477 is low-deformation (it locally certified at 2.85 um over 67% of the field),
so a drawn ROI should come back CERTIFIED / LOCALLY_CERTIFIED from the NEW LoFTR path.

Run:  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python validation/demo_ll477_local_roi.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import serial_registration as sr        # noqa: E402
import loftr_matcher as lm              # noqa: E402

REF = "/Users/mukilan/Downloads/052526/Tumor/LL477_CD8_x10_3.tif"
MOV = "/Users/mukilan/Downloads/052526/Tumor/LL477_Tim3_10X_3.tif"
PX = 0.7519
OUT = "/private/tmp/claude-501/-Users-mukilan-PycharmProjects-ihc-original-copy/c603cc1f-14ab-4c98-b466-8a30a5538d4b/scratchpad/ll477_local_roi.png"
RANK = {"CERTIFIED": 0, "LOCALLY_CERTIFIED": 1, "RADIUS_LIMITED": 2}


def load_rgb(p):
    import cv2
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def map_to_mov(poly, M):
    A = M[:2, :2]; t = M[:2, 2]
    return (poly - t) @ np.linalg.inv(A).T


def circle(cx, cy, r, k=48):
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]


def main():
    import cv2
    ref, mov = load_rgb(REF), load_rgb(MOV)
    H, W = ref.shape[:2]
    print(f"field {W}x{H}px = {W*PX/1000:.2f}x{H*PX/1000:.2f} mm at {PX} um/px")

    reg = sr.register_similarity(ref, mov, PX)
    M = np.asarray(reg["matrix"], float)
    print(f"provisional register: method={reg['method']} ncc={reg['struct_ncc']} "
          f"dice={reg['struct_dice']}")

    # candidate ROIs a user might draw: centre + 4 offsets, radius ~350 um
    r = 350.0 / PX
    cxs = [W / 2, W / 2 - r, W / 2 + r, W / 2, W / 2]
    cys = [H / 2, H / 2, H / 2, H / 2 - r * 0.8, H / 2 + r * 0.8]
    results = []
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        if cx - r < 0 or cy - r < 0 or cx + r > W or cy + r > H:
            r_i = min(cx, cy, W - cx, H - cy) * 0.95
        else:
            r_i = r
        roi = circle(cx, cy, r_i)
        cert = lm.certify_local_roi(ref, mov, roi, PX, provisional_matrix=M)
        v = cert.get("verdict")
        cell = cert.get("cell_error_p90_um") or cert.get("tre_p90_um") or cert.get("tre_median_um")
        print(f"  ROI{i} c=({cx:.0f},{cy:.0f}) r={r_i*PX:.0f}um  verdict={v:<18} "
              f"source={cert.get('source')} n={cert.get('n_correspondences')} "
              f"cell~{cell}")
        results.append((cert, roi))

    # rank: certified kinds first, then by cell error / tre
    def key(cr):
        c = cr[0]
        v = c.get("verdict")
        cell = c.get("cell_error_p90_um") or c.get("tre_p90_um") or c.get("tre_median_um") or 1e9
        return (RANK.get(v, 9), float(cell))
    results.sort(key=key)
    best, roi = results[0]
    print(f"\nBEST: verdict={best.get('verdict')}  source={best.get('source')}  "
          f"n={best.get('n_correspondences')}  reason={str(best.get('reason'))[:120]}")

    # render ROI on both sections
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roi_mov = map_to_mov(roi, M)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].imshow(ref); ax[0].set_title(f"CD8 (fixed) — drawn ROI\n{best.get('verdict')}")
    ax[0].add_patch(plt.Polygon(roi, fill=False, edgecolor="lime", lw=2.5))
    ax[1].imshow(mov); ax[1].set_title("TIM-3 (moving) — same ROI mapped\nlocal LoFTR fit")
    ax[1].add_patch(plt.Polygon(roi_mov, fill=False, edgecolor="lime", lw=2.5))
    for a in ax:
        a.axis("off")
    v = best.get("verdict")
    cell = best.get("cell_error_p90_um") or best.get("tre_p90_um") or best.get("tre_median_um")
    fig.suptitle(f"LL477 CD8/TIM-3 — certify_local_roi via LoFTR-in-ROI  |  "
                 f"{v}  (cell-error ~{cell} um, {best.get('source')})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"\nsaved overlay -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
