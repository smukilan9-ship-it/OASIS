#!/usr/bin/env python
"""
Which matcher actually works on LL477 H-DAB serial sections — the target cohort.

WHY NOT ANHIR. validate_matcher_blunders_anhir.py found DISK+LightGlue markedly cleaner than
LoFTR (0.34 % blunders against 2.10 %, p = 0.011). It does not transfer. On LL477, DISK
detects 2048 keypoints and LightGlue matches TWO of them at full resolution and ZERO at the
800 px crop certification uses, where LoFTR returns 140. ANHIR is largely well-textured
cross-stain whole sections; LL477 is homogeneous spindle-cell proliferation whose only
distinctive features are scattered vacuoles. A benchmark on the wrong tissue answered the
wrong question, so this one runs on the cohort every disputed number comes from.

TWO TESTS, because they fail differently.

  A. SYNTHETIC WARP — exact ground truth on this tissue.
     Warp one real section by a transform we choose and match it against itself. Every
     correspondence's true partner is known analytically, so accuracy and blunder rate are
     measured, not inferred. This isolates "can this matcher handle spindle-cell H-DAB
     texture at all" from "can it bridge two different stains".

  B. REAL PAIR — the actual problem.
     CD8 <-> TIM-3, two physical sections, no ground truth. Only coverage and a blunder proxy
     (max/median residual against a robust similarity) are available, but this is the test
     that killed DISK, so it is the one that decides.

A matcher has to pass BOTH. Sailing through A and dying in B is exactly DISK's failure: it
localises beautifully on texture it can detect and cannot bridge the stain difference.

Run:  .venv/bin/python validation/validate_matchers_on_cohort.py
      SSL_CERT_FILE=$(python -m certifi) may be needed for first-time weight downloads.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr            # noqa: E402
from oasis.spatial import loftr_matcher as lm                  # noqa: E402
from oasis.spatial import sparse_matcher as sm                 # noqa: E402

CD8 = os.path.expanduser("~/Desktop/cd8_input")
TIM3 = os.path.expanduser("~/Desktop/tim3 input")
PAIRS = [("LL477_CD8_x10_1.tif", "LL477_Tim3_x10_1.tif", "spindle-1"),
         ("LL477_CD8_x10_2.tif", "LL477_Tim3_x10_2.tif", "spindle-2"),
         ("LL477_CD8_x10_3.tif", "LL477_Tim3_10X_3.tif", "liver-3")]
PX = 0.7519
WORK = 800                 # what certify_local_roi actually runs at
GROSS_UM = 15.0            # a correspondence this far out is a blunder, not noise
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "matchers_on_cohort_results.json")

_MODELS = {}


def _gray(rgb):
    return sm._prep(rgb)


# ── matcher arms: each returns (ref_pts, mov_pts) in pixels of the given arrays ──────
def _loftr(a, b, px):
    c = lm.loftr_correspondences(a, b, pixel_size_um=px)
    if not c["ok"]:
        return None, None
    return np.asarray(c["ref_points"], float), np.asarray(c["mov_points"], float)


def _lightglue_pair(a, b, kind):
    import torch, kornia.feature as KF
    ga, gb = _gray(a), _gray(b)
    ta = torch.from_numpy(ga).float()[None, None] / 255.0
    tb = torch.from_numpy(gb).float()[None, None] / 255.0
    with torch.inference_mode():
        if kind == "disk":
            if "disk" not in _MODELS:
                _MODELS["disk"] = KF.DISK.from_pretrained("depth").eval()
            d = _MODELS["disk"]
            fa = d(ta.repeat(1, 3, 1, 1), 2048, pad_if_not_divisible=True)[0]
            fb = d(tb.repeat(1, 3, 1, 1), 2048, pad_if_not_divisible=True)[0]
            ka, kb, da, db = fa.keypoints, fb.keypoints, fa.descriptors, fb.descriptors
            la, lb = KF.laf_from_center_scale_ori(ka[None]), KF.laf_from_center_scale_ori(kb[None])
            lgname = "disk"
        elif kind == "dedode":
            if "dedode" not in _MODELS:
                _MODELS["dedode"] = KF.DeDoDe.from_pretrained(
                    detector_weights="L-upright", descriptor_weights="B-upright")
            d = _MODELS["dedode"]
            ka, _s, da = d(ta.repeat(1, 3, 1, 1))
            kb, _s, db = d(tb.repeat(1, 3, 1, 1))
            da, db = da[0], db[0]
            la, lb = KF.laf_from_center_scale_ori(ka), KF.laf_from_center_scale_ori(kb)
            ka, kb = ka[0], kb[0]
            lgname = "dedodeb"
        else:                                        # sift
            if "sift" not in _MODELS:
                _MODELS["sift"] = KF.SIFTFeature(2048).eval()
            d = _MODELS["sift"]
            la, _r, da = d(ta)
            lb, _r, db = d(tb)
            ka, kb = KF.get_laf_center(la)[0], KF.get_laf_center(lb)[0]
            da, db = da[0], db[0]
            lgname = "sift"
        key = "lg_" + lgname
        if key not in _MODELS:
            _MODELS[key] = KF.LightGlueMatcher(lgname).eval()
        _d, idx = _MODELS[key](da, db, la, lb,
                               hw1=torch.tensor(ga.shape), hw2=torch.tensor(gb.shape))
    if idx is None or len(idx) < 6:
        return None, None
    i = idx.detach().cpu().numpy()
    return (ka.detach().cpu().numpy()[i[:, 0]].astype(float),
            kb.detach().cpu().numpy()[i[:, 1]].astype(float))


def _keynet(a, b, px):
    import torch, kornia.feature as KF
    ga, gb = _gray(a), _gray(b)
    ta = torch.from_numpy(ga).float()[None, None] / 255.0
    tb = torch.from_numpy(gb).float()[None, None] / 255.0
    if "keynet" not in _MODELS:
        _MODELS["keynet"] = KF.KeyNetAffNetHardNet(2048).eval()
    with torch.inference_mode():
        la, _r, da = _MODELS["keynet"](ta)
        lb, _r, db = _MODELS["keynet"](tb)
        _d, idx = KF.DescriptorMatcher("smnn", 0.95)(da[0], db[0])
    if idx is None or len(idx) < 6:
        return None, None
    i = idx.detach().cpu().numpy()
    ka, kb = KF.get_laf_center(la)[0].numpy(), KF.get_laf_center(lb)[0].numpy()
    return ka[i[:, 0]].astype(float), kb[i[:, 1]].astype(float)


ARMS = [
    ("loftr",           lambda a, b, px: _loftr(a, b, px)),
    ("disk+lightglue",  lambda a, b, px: _lightglue_pair(a, b, "disk")),
    ("dedode+lightglue", lambda a, b, px: _lightglue_pair(a, b, "dedode")),
    ("sift+lightglue",  lambda a, b, px: _lightglue_pair(a, b, "sift")),
    ("keynet+hardnet",  lambda a, b, px: _keynet(a, b, px)),
]


def _resize(im, work):
    import cv2
    r = min(1.0, float(work) / max(im.shape[:2]))
    if r >= 1.0:
        return im, 1.0
    return cv2.resize(im, (int(im.shape[1] * r), int(im.shape[0] * r)),
                      interpolation=cv2.INTER_AREA), r


# ── A. synthetic warp: exact truth on this tissue ────────────────────────────────────
def test_synthetic(rgb, px_work, warp="rotation", amount=3.0):
    import cv2
    H, W = rgb.shape[:2]
    c = ((W - 1) / 2.0, (H - 1) / 2.0)
    th = np.deg2rad(amount)
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.zeros((2, 3)); M[:, :2] = A; M[:, 2] = np.array(c) - A @ np.array(c)
    mov = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    out = {}
    for tag, fn in ARMS:
        t0 = time.time()
        try:
            p, q = fn(rgb, mov, px_work)
        except Exception as e:
            out[tag] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
            continue
        if p is None or len(p) < 6:
            out[tag] = {"n": 0}
            continue
        truth = p @ M[:, :2].T + M[:, 2]
        err = np.linalg.norm(q - truth, axis=1) * px_work
        out[tag] = {"n": int(len(p)), "seconds": round(time.time() - t0, 1),
                    "err_med_um": round(float(np.median(err)), 3),
                    "err_p90_um": round(float(np.percentile(err, 90)), 3),
                    "gross_frac": round(float((err > GROSS_UM).mean()), 4)}
    return out


# ── B. real pair: the test that decides ──────────────────────────────────────────────
def test_real(ref, mov, px_work):
    out = {}
    for tag, fn in ARMS:
        t0 = time.time()
        try:
            p, q = fn(ref, mov, px_work)
        except Exception as e:
            out[tag] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
            continue
        if p is None or len(p) < 6:
            out[tag] = {"n": 0, "seconds": round(time.time() - t0, 1)}
            continue
        M = sr._fit_similarity_robust(q, p)
        r = np.linalg.norm(sr._apply_affine(q, M) - p, axis=1) * px_work
        med = float(np.median(r))
        out[tag] = {"n": int(len(p)), "seconds": round(time.time() - t0, 1),
                    "resid_med_um": round(med, 3),
                    "max_over_med": round(float(r.max() / max(med, 1e-9)), 2),
                    "gross_frac": round(float((r > GROSS_UM).mean()), 4)}
    return out


def main():
    from oasis.common.registration import _load_rgb_thumbnail
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()

    results = {}
    for cd8, tim3, tag in PAIRS:
        rp, mp = os.path.join(CD8, cd8), os.path.join(TIM3, tim3)
        if not (os.path.exists(rp) and os.path.exists(mp)):
            print(f"skip {tag}: missing file")
            continue
        ref, _ = _load_rgb_thumbnail(rp, max_side=1920)
        mov, _ = _load_rgb_thumbnail(mp, max_side=1920)
        refw, r = _resize(ref, WORK)
        movw, _ = _resize(mov, WORK)
        pxw = PX / r
        print(f"\n=== {tag}  ({cd8} ↔ {tim3})   {refw.shape[1]}x{refw.shape[0]} @ "
              f"{pxw:.2f} µm/px ===")

        syn = test_synthetic(refw, pxw)
        real = test_real(refw, movw, pxw)
        results[tag] = {"synthetic": syn, "real": real}

        print(f"  {'matcher':<19}{'A: warp n':>10}{'err med':>9}{'gross':>7}"
              f"   |{'B: REAL n':>10}{'resid med':>11}{'max/med':>9}{'gross':>7}{'sec':>6}")
        for t, _ in ARMS:
            s, b = syn.get(t, {}), real.get(t, {})
            if s.get("error") or b.get("error"):
                print(f"  {t:<19}  {s.get('error') or b.get('error')}")
                continue
            sa = (f"{s.get('n',0):>10}{s.get('err_med_um','--'):>9}"
                  f"{s.get('gross_frac','--'):>7}")
            bb = (f"{b.get('n',0):>10}{b.get('resid_med_um','--'):>11}"
                  f"{b.get('max_over_med','--'):>9}{b.get('gross_frac','--'):>7}"
                  f"{b.get('seconds','--'):>6}")
            print(f"  {t:<19}{sa}   |{bb}")
        lm.clear_loftr_caches()

    print("\n" + "=" * 78)
    print("VERDICT — a matcher must survive column B, the real cross-stain pair")
    print("=" * 78)
    usable = {}
    for t, _ in ARMS:
        ns = [results[k]["real"].get(t, {}).get("n", 0) for k in results]
        gs = [results[k]["real"].get(t, {}).get("gross_frac") for k in results]
        gs = [g for g in gs if g is not None]
        usable[t] = {"real_n_per_pair": ns, "min_n": int(min(ns)) if ns else 0,
                     "gross_frac_max": max(gs) if gs else None}
        ok = usable[t]["min_n"] >= 40
        print(f"  {t:<19} real matches {str(ns):<22} "
              f"{'USABLE' if ok else 'FAILS on at least one pair'}"
              + (f", worst gross {100*max(gs):.1f}%" if gs and ok else ""))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"pairs": results, "summary": usable}, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
