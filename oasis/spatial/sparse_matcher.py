"""
DISK + LightGlue correspondences — the PRIMARY matcher, with LoFTR behind it.

WHY THIS REPLACED LoFTR AS THE DEFAULT. `loftr_matcher` was chosen because detector-based
matching failed on this data — its docstring records "SIFT with mutual-NN + Lowe returns zero
matches at any sane ratio". That verdict was fair on a 1999 detector paired with
nearest-neighbour matching, and it does not carry over to a learned detector paired with a
learned matcher.

Measured (validation/validate_matcher_blunders_anhir.py, 20 ANHIR training pairs, expert
landmarks held out from every arm, every matcher handed the same hematoxylin+CLAHE input):

    arm              n med   blunder%   max/med   TRE med   TRE p90
    loftr (shipped)    505       0.9%      8.12      4.63     11.23
    loftr raw          740       5.1%     41.18      8.48     16.95
    DISK + LightGlue   888       0.0%      4.84      4.18     10.96

Paired over the 19 pairs both completed: LoFTR's mean blunder fraction is 2.10 % against
DISK's 0.34 %, lower on 11 of 19 and higher on 3, Wilcoxon p = 0.011. Expert-landmark TRE is
indistinguishable (p = 0.39), and DISK returns 1.76x the correspondences.

Blunders are what actually hurt: the certification gate reads a p90, so a handful of
confidently-wrong matches sets the reported cell error. On a real LL477 region, 39
correspondences with a 6.55 µm median residual produced a 58 µm cell error because ~4 of them
were gross.

WHY LoFTR IS STILL HERE, AND MUST STAY. DISK is a DETECTOR, and on cross-modal pairs it cannot
fire on corresponding structures. On 4 of those 19 pairs it returned under 40 matches where
LoFTR still found 47-230:

    breast_1 HER2->HE   LoFTR  47   DISK 13
    lung-lesion_3       LoFTR 230   DISK  8
    COAD_01             LoFTR  56   DISK 10
    mice-kidney_1 PAS   LoFTR  64   DISK 20

That is precisely the regime detector-free matching exists for. So this module dispatches:
DISK first, LoFTR when DISK comes back thin. Neither is `legacy` — they cover different
failures, and the fallback is not a formality.

AND ON THE ACTUAL TARGET TISSUE, DISK RETURNS NOTHING. Measured on LL477 CD8 <-> TIM-3, the
cohort every disputed number in research/registration.md comes from:

    full 1920 px   DISK keypoints 2048   LightGlue matches   2
    800 px crop    DISK keypoints 2048   LightGlue matches   0

DISK detects 2048 keypoints and LightGlue matches essentially none of them across the pair.
The tissue is homogeneous spindle-cell proliferation whose only distinctive features are
scattered vacuoles; a detector fires on texture that does not correspond between two sections
4 um apart. LoFTR, attending over the whole image, gets 140 on the same ROI.

So the ANHIR result does NOT transfer to H-DAB serial sections, and "make DISK primary" is
wrong for this cohort even though it is right on ANHIR. The dispatcher is what makes both
true at once: DISK where it wins, LoFTR where it is the only thing that works. Which arm ran
is recorded on every certification as `matcher`, so no verdict is unattributable.

WHAT IS AND IS NOT FILTERED. LoFTR needs three model-free filters because its raw output
carries 5.1 % blunders. LightGlue already does mutual, confidence-weighted assignment and its
raw output measured 0.0 % median blunders, so cycle and scale consistency are not reproduced
here — they would cost two extra forward passes to re-derive something the matcher already
did. LOCAL SMOOTHNESS is still applied: it asks a question no matcher asks itself (is the
displacement field continuous here?), it is matcher-agnostic, and it is cheap.
"""
import numpy as np

from oasis.spatial.loftr_matcher import _local_smoothness, _arr_key

_MODELS = {}
# Keyed by image content like the LoFTR caches, for the same reason: a re-probed crop or a
# re-run pair costs nothing. Bounded so a long batch cannot grow memory without limit.
_FEAT_CACHE = {}
_FEAT_CACHE_MAX = 64

# Below this many correspondences the pair is handed to LoFTR. Set at 40 because that is where
# DISK's failures sat in the benchmark (8-20) and its successes did not (503-1266); it is not
# a tuned threshold, it separates two clearly--separated populations.
DISK_MIN_MATCHES = 40


def clear_sparse_caches():
    _FEAT_CACHE.clear()


def _device():
    """CUDA when present, else CPU. MPS is skipped for the same measured reason as LoFTR —
    the attention ops fall back and it buys nothing."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _models():
    if "disk" not in _MODELS:
        import kornia.feature as KF
        dev = _device()
        _MODELS["disk"] = KF.DISK.from_pretrained("depth").eval().to(dev)
        _MODELS["lg"] = KF.LightGlueMatcher("disk").eval().to(dev)
        _MODELS["dev"] = dev
    return _MODELS["disk"], _MODELS["lg"], _MODELS["dev"]


def _prep(rgb):
    """Hematoxylin, CLAHE-equalised — byte-for-byte what loftr_matcher._prep feeds LoFTR, so
    a matcher comparison is a matcher comparison and not a preprocessing one."""
    import cv2
    from oasis.common.registration import extract_hematoxylin
    h = extract_hematoxylin(rgb).astype(np.float32)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(h)


def _features(gray, max_keypoints):
    import torch
    ck = (_arr_key(gray), int(max_keypoints))
    hit = _FEAT_CACHE.get(ck)
    if hit is not None:
        return hit
    disk, _lg, dev = _models()
    t = torch.from_numpy(gray).float()[None, None].to(dev) / 255.0
    with torch.inference_mode():
        f = disk(t.repeat(1, 3, 1, 1), int(max_keypoints), pad_if_not_divisible=True)[0]
    out = (f.keypoints.detach().cpu(), f.descriptors.detach().cpu())
    if len(_FEAT_CACHE) >= _FEAT_CACHE_MAX:
        _FEAT_CACHE.clear()
    _FEAT_CACHE[ck] = out
    return out


def sparse_correspondences(ref_rgb, mov_rgb, pixel_size_um, tol_um=4.0,
                           max_keypoints=2048, local_k=8, **_ignored):
    """DISK keypoints matched by LightGlue. Same return contract as loftr_correspondences,
    so `certify_local_roi` cannot tell which matcher produced a set."""
    import torch
    import kornia.feature as KF

    out = {"ref_points": [], "mov_points": [], "confidence": [], "n": 0,
           "n_raw": 0, "n_after_cycle": 0, "n_after_scale": 0, "n_after_local": 0,
           "local_drop_frac": 0.0, "tol_um": float(tol_um), "weights": "disk+lightglue",
           "matcher": "disk_lightglue", "ok": False, "msg": ""}

    ref_g, mov_g = _prep(ref_rgb), _prep(mov_rgb)
    ka, da = _features(ref_g, max_keypoints)
    kb, db = _features(mov_g, max_keypoints)
    if len(ka) < 6 or len(kb) < 6:
        out["msg"] = f"DISK found {len(ka)}/{len(kb)} keypoints"
        return out

    _disk, lg, dev = _models()
    with torch.inference_mode():
        _d, idxs = lg(da.to(dev), db.to(dev),
                      KF.laf_from_center_scale_ori(ka[None]).to(dev),
                      KF.laf_from_center_scale_ori(kb[None]).to(dev),
                      hw1=torch.tensor(ref_g.shape), hw2=torch.tensor(mov_g.shape))
    if idxs is None or len(idxs) < 6:
        out["msg"] = f"LightGlue returned {0 if idxs is None else len(idxs)} matches"
        return out
    i = idxs.detach().cpu().numpy()
    p = ka.numpy()[i[:, 0]].astype(float)
    q = kb.numpy()[i[:, 1]].astype(float)
    # LightGlue's assignment is already mutual and confidence-gated, so the cycle and scale
    # stages LoFTR needs have no work to do here; recorded as pass-through rather than
    # silently absent, so the funnel a caller prints still has four honest numbers.
    out.update(n_raw=int(len(p)), n_after_cycle=int(len(p)), n_after_scale=int(len(p)))

    tol_px = float(tol_um) / float(pixel_size_um)
    if local_k and len(p) > int(local_k) + 1:
        keep = _local_smoothness(p, q, tol_px, k=int(local_k))
        n_before = len(p)
        p, q = p[keep], q[keep]
        out["local_drop_frac"] = round(1.0 - len(p) / float(n_before), 4)
    out["n_after_local"] = int(len(p))
    if len(p) < 6:
        out["msg"] = f"only {len(p)} matches survive local smoothness"
        return out

    out.update(ref_points=p.tolist(), mov_points=q.tolist(), n=int(len(p)),
               confidence=[1.0] * len(p), ok=True,
               msg=(f"DISK+LightGlue {out['n_raw']} matches → {len(p)} locally smooth "
                    f"(dropped {out['local_drop_frac']:.0%}, tol {tol_um} µm); "
                    f"no transform used"))
    return out


def correspondences(ref_rgb, mov_rgb, pixel_size_um, matcher="auto", **kw):
    """The dispatcher every certification path goes through.

    "auto" runs DISK first and falls back to LoFTR when it comes back thin, because the two
    fail on different pairs (see the module docstring). "disk" and "loftr" force one, which is
    what the A/B harness uses.
    """
    from oasis.spatial.loftr_matcher import loftr_correspondences

    if matcher == "loftr":
        c = loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um, **kw)
        c.setdefault("matcher", "loftr")
        return c

    lo_kw = {k: v for k, v in kw.items()
             if k in ("weights", "scales", "conf_floor", "noise", "rng",
                      "scale_tol_stride", "tol_um", "local_k")}
    sp_kw = {k: v for k, v in kw.items()
             if k in ("tol_um", "local_k", "max_keypoints")}

    c = sparse_correspondences(ref_rgb, mov_rgb, pixel_size_um, **sp_kw)
    if matcher == "disk":
        return c
    if c["ok"] and c["n"] >= DISK_MIN_MATCHES:
        return c
    # Thin or failed: this is the cross-modal case a detector cannot serve.
    fb = loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um, **lo_kw)
    fb["matcher"] = "loftr_fallback"
    fb["msg"] = (f"DISK returned {c['n']} (<{DISK_MIN_MATCHES}), fell back to LoFTR — "
                 + (fb.get("msg") or ""))
    fb["disk_n"] = int(c["n"])
    return fb
