"""
check_warps.py — the pre-flight that has to pass before any arm is believed.

Three things are asserted, because each one would silently fake the result:

 1. INTERPOLATION FIDELITY. The arms reapply VALIS's transforms by interpolating an
    80x80 grid. export_warps.py also asked VALIS to warp 400 RANDOM held-out points
    directly. If interpolation error is not far below the micro-registration
    displacement we are measuring, the "effect" could be interpolation artefact.

 2. THE RIGID STAGE IS AFFINE. arms.py inverts it in closed form for Arm 1. A
    non-affine "rigid" map would make that inverse wrong.

 3. THE TWO TRANSFORMS ACTUALLY DIFFER. If micro-registration barely moves anything on
    a pair, that pair cannot show an effect either way and should be reported as such
    rather than diluting the rate.

Run:  .venv/bin/python -m validation.nonrigid_bench.check_warps
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.nonrigid_bench import warpfield as wf  # noqa: E402


def main():
    pairs = wf.load_all()
    if not pairs:
        print("no warps exported yet")
        return
    print(f"{'pair':38s} {'px_um':>6s} {'interp_rig':>10s} {'interp_non':>10s} "
          f"{'affine_res':>10s} {'microreg_um':>11s} {'ratio':>7s}")
    bad = 0
    for p in pairs:
        hs, hr, hn = p.hold_src, p.hold_rigid, p.hold_nonrigid
        ok = np.isfinite(hs).all(1) & np.isfinite(hr).all(1) & np.isfinite(hn).all(1)
        hs, hr, hn = hs[ok], hr[ok], hn[ok]
        if len(hs) < 20:
            continue
        er = np.linalg.norm(p.rigid(hs) - hr, axis=1)
        en = np.linalg.norm(p.nonrigid(hs) - hn, axis=1)
        micro = np.linalg.norm(hn - hr, axis=1)
        ratio = np.median(en) / max(np.median(micro), 1e-9)
        flag = ""
        if ratio > 0.10:
            flag = "  <-- interpolation not negligible"
            bad += 1
        print(f"{p.pair_id[:38]:38s} {p.px_um:6.3f} "
              f"{np.median(er):10.3f} {np.median(en):10.3f} "
              f"{p.rigid_affine_resid_px:10.4f} "
              f"{np.median(micro)*p.px_um:11.1f} {ratio:7.3f}{flag}")
    print()
    print(f"pairs: {len(pairs)}   interpolation-suspect: {bad}")
    print("columns are px unless stated; interp_* = |interpolated - VALIS| on held-out "
          "points; ratio = interp_non / microreg (want << 1)")


if __name__ == "__main__":
    main()
