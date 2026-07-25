"""
valis_worker.py — runs INSIDE the isolated ~/valis_runtime venv (Python 3.11, valis-wsi).
NEVER imported by the main pipeline (valis-wsi cannot import on the repo's Python 3.14).
Invoked as a subprocess by oasis.spatial.valis_engine.

Contract (stdin/stdout JSON):
  in : {"ref_path": <fixed image>, "mov_path": <moving image>}
  out: {"ok": bool, "matrix": [[..],[..]] (moving->reference SIMILARITY, ORIGINAL px),
        "n_matches": int, "valis_rigid_rtre": float|null, "secs": float, "error": str?}

Only the RIGID transform is ever returned (non-rigid is never used, per the cross-K
distance-preserving invariant). The transform is obtained the coordinate-clean way that the
spike verified: warp probe points moving->reference with warp_xy_from_to(non_rigid=False)
and solve the 2x3 similarity — NOT by touching VALIS's internal feature-frame keypoints.
"""
import sys
import os
import json
import time
import tempfile
import shutil
import warnings

warnings.filterwarnings("ignore")


def _run(ref_path, mov_path):
    import numpy as np
    import cv2
    from PIL import Image
    from valis import registration

    t0 = time.time()
    work = tempfile.mkdtemp(prefix="valis_worker_")
    try:
        src = os.path.join(work, "in"); os.makedirs(src)
        fn = os.path.basename(ref_path); mn = os.path.basename(mov_path)
        # guarantee distinct filenames so VALIS keys them apart
        if fn == mn:
            mn = "mov_" + mn
        shutil.copy(ref_path, os.path.join(src, fn)); shutil.copy(mov_path, os.path.join(src, mn))
        reg = registration.Valis(src, os.path.join(work, "out"), reference_img_f=fn,
                                 imgs_ordered=False, crop="reference")
        # FULL registration (rigid-only flag is buggy in 1.2.0); we USE only the rigid part.
        rr = reg.register()
        rigid_registrar = rr[0] if isinstance(rr, (tuple, list)) else rr
        n_matches = 0
        try:
            for o in rigid_registrar.img_obj_list:
                for _k, mi in getattr(o, "match_dict", {}).items():
                    kp = getattr(mi, "matched_kp1_xy", None)
                    n_matches = max(n_matches, len(kp) if kp is not None else 0)
        except Exception:
            pass
        # VALIS's own rigid error (self-consistency; reported for provenance only)
        rtre = None
        try:
            import pandas as pd  # noqa: F401
            edf = rr[2] if isinstance(rr, (tuple, list)) and len(rr) > 2 else None
            if edf is not None and "rigid_rTRE" in edf.columns:
                v = edf["rigid_rTRE"].dropna()
                rtre = float(v.median()) if len(v) else None
        except Exception:
            pass

        # rigid transform in ORIGINAL moving->reference px: warp probe points, solve similarity
        W, H = Image.open(mov_path).size
        probe = np.array([[0, 0], [W, 0], [0, H], [W, H], [W/2, H/2],
                          [W/4, H/4], [3*W/4, 3*H/4]], dtype=np.float64)
        ms = reg.get_slide(mn); fs = reg.get_slide(fn)
        warped = np.asarray(ms.warp_xy_from_to(probe, fs, non_rigid=False), float)
        M, _inl = cv2.estimateAffinePartial2D(probe.astype(np.float32), warped.astype(np.float32))
        if M is None:
            return {"ok": False, "error": "could not solve rigid transform", "secs": round(time.time()-t0, 1)}
        try:
            registration.kill_jvm()
        except Exception:
            pass
        return {"ok": True, "matrix": M.tolist(), "n_matches": int(n_matches),
                "valis_rigid_rtre": rtre, "secs": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    try:
        payload = json.load(sys.stdin)
        out = _run(os.path.expanduser(payload["ref_path"]), os.path.expanduser(payload["mov_path"]))
    except Exception as e:
        import traceback
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
