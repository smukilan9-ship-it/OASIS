"""
Does PRETRAINED LoFTR register serial histology better than the landmark-fit method,
and is its Fitzpatrick-West error bound calibrated? The experiment that decides whether
we ever need to fine-tune.

WHY THIS IS THE RIGHT TEST. Our previous calibration (validate_fw_anhir_calibration.py)
fit the similarity on one annotator's landmarks (PS) and measured realized error at a
second, independent annotator (JB). LoFTR changes the picture in our favour: it fits the
transform from IMAGE CONTENT, touching no landmarks at all. So BOTH expert annotators are
held out from a LoFTR registration -- the entire expert landmark set is ground truth for
it. That gives three numbers we can put side by side, per pair:

  A. LoFTR realized error     fit M from LoFTR matches; measure ||M.expert_mov - expert_ref||.
                              This is the accuracy of the automatic, landmark-free method.
  B. Landmark realized error  fit M from PS landmarks; measure at JB. The PREVIOUS method.
  C. Inter-observer FLE floor PS vs JB clicking the same points. The best ANY method can do;
                              no registration can beat the noise in its own ground truth.

  If A ~ C  -> LoFTR is as good as the tissue's ground truth allows. Fine-tuning buys nothing.
  If A > B  -> LoFTR is WORSE than the old landmark method. That is the evidence for fine-tuning.
  If A < B  -> LoFTR beats manual landmarks. The headline result.

And the bound question, exactly as before but now on LoFTR's own residuals:
  predicted_p90 (TRE_pred (+) deformation (+) FLE)  must be >= realized_p90. Ratio > 1.15
  means the bound UNDER-states realized error -- anti-conservative, do not ship.

PIXEL SIZE IS LOAD-BEARING AND IS VERIFIED, NOT ASSUMED. LoFTR is not scale-invariant, and
every tolerance in the pipeline lives in microns. Each tissue has a native um/px (ANHIR
Table I); at scale-50pc that doubles. Landmarks are stored at their own scale (lung-lesion
at 50pc, mammary/lung-lobes at 100pc) which need not equal the image scale, so we RESCALE
landmarks to image pixels and ASSERT they land inside the image before trusting anything.
The final micron error is invariant to the working resolution we run LoFTR at -- the harness
downscales big slides for tractable LoFTR compute and the answer must not move; that
invariance is itself a check that the bookkeeping is right.

Run:  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python validation/validate_loftr_anhir.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from oasis.spatial import serial_registration as sr          # noqa: E402
from oasis.spatial import loftr_matcher as lm                # noqa: E402

# native um/px at full resolution (ANHIR Table I); scale-50pc images are 2x this.
NATIVE_PX = {"lung-lesion": 0.174, "lung-lobes": 1.274, "mammary-gland": 2.294}
# the scale (pc) at which each tissue's PS/JB landmark CSVs are stored locally.
LANDMARK_SCALE_PC = {"lung-lesion": 50, "lung-lobes": 100, "mammary-gland": 100}
IMAGE_SCALE_PC = 50
WORK_MAX_DIM = 2000          # cap LoFTR input (long side, post tissue-crop)

IMG_ROOT = "/Volumes/Expansion/registration/anhir_cima"

# (tissue, fixed basename (no ext), moving basename) -- both annotated by PS AND JB.
PAIRS = [
    ("lung-lesion_3", "29-041-Izd2-w35-He-les3", "29-041-Izd2-w35-proSPC-4-les3"),
    ("mammary-gland_1", "s1_37-HE_A4926-4L", "s1_40-PR_A4926-4L"),
    ("mammary-gland_2", "s2_63-HE_A4926-4L", "s2_68-ER-A4962-4L"),
]


def tissue_prefix(tissue):
    for pref in NATIVE_PX:
        if tissue.startswith(pref):
            return pref
    raise KeyError(tissue)


def px_50pc(tissue):
    # NATIVE_PX is full-resolution (100pc) um/px; a 50pc image has half the pixels
    # per axis, so each pixel spans twice the tissue: px(50pc) = native * 2.
    return NATIVE_PX[tissue_prefix(tissue)] * 2.0


def _ann_roots():
    for p in (os.path.join(HERE, "public_landmarks", "annotations"),
              os.path.expanduser("~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations")):
        if os.path.isdir(p):
            yield p


def load_xy(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]
    xi = hdr.index("x") if "x" in hdr else -2
    yi = hdr.index("y") if "y" in hdr else -1
    pts = []
    for r in rows[1:]:
        try:
            pts.append([float(r[xi]), float(r[yi])])
        except (ValueError, IndexError):
            pass
    return np.array(pts, float)


def find_landmarks(tissue, user, base):
    for root in _ann_roots():
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
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_image_pixels(pts, tissue):
    """Landmark coords (stored at LANDMARK_SCALE_PC) -> scale-50pc image pixels."""
    factor = IMAGE_SCALE_PC / LANDMARK_SCALE_PC[tissue_prefix(tissue)]
    return pts * factor


def tissue_bbox(rgb, pad=32):
    """Bounding box of non-white tissue. ANHIR slides carry large white margins that
    waste LoFTR's fixed resolution budget; cropping to tissue lets the SAME pixel cap
    hold far more architecture. Returns (x0, y0) offset and the crop."""
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (g < 220).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.where(mask)
    if len(xs) < 100:
        return (0, 0), rgb
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, rgb.shape[1])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, rgb.shape[0])
    return (x0, y0), rgb[y0:y1, x0:x1]


def prep_pair_for_loftr(rgb_r, rgb_m, px_full):
    """Crop both images to tissue, then downscale BOTH by ONE common factor r so they
    share a working um/px (LoFTR is not scale-invariant -- ref and mov MUST be matched
    in scale). Returns small images plus a closure mapping working coords in each image
    back to full-resolution 50pc pixels: full = offset + work / r."""
    import cv2
    (ox_r, oy_r), crop_r = tissue_bbox(rgb_r)
    (ox_m, oy_m), crop_m = tissue_bbox(rgb_m)
    long_side = max(crop_r.shape[0], crop_r.shape[1], crop_m.shape[0], crop_m.shape[1])
    r = min(1.0, WORK_MAX_DIM / float(long_side))
    def _rs(im):
        return cv2.resize(im, (max(int(im.shape[1] * r), 8), max(int(im.shape[0] * r), 8)),
                          interpolation=cv2.INTER_AREA) if r < 1.0 else im
    small_r, small_m = _rs(crop_r), _rs(crop_m)
    px_work = px_full / r
    back_r = lambda p: np.asarray(p, float) / r + np.array([ox_r, oy_r])
    back_m = lambda p: np.asarray(p, float) / r + np.array([ox_m, oy_m])
    return small_r, small_m, px_work, r, back_r, back_m


def realized_um(M, mov_pts, ref_pts, px):
    return np.linalg.norm(sr._apply_affine(mov_pts, M) - ref_pts, axis=1) * px


def one_pair(tissue, fixed_base, moving_base):
    print(f"\n{'-'*78}\n{tissue}")
    img_r_path, img_m_path = find_image(tissue, fixed_base), find_image(tissue, moving_base)
    if not img_r_path or not img_m_path:
        print(f"  images not on disk yet ({fixed_base} / {moving_base}) -- skipped")
        return None
    # landmarks: PS and JB, on both fixed (ref) and moving images
    lp = {(u, k): find_landmarks(tissue, u, b) for u in ("PS", "JB")
          for k, b in (("ref", fixed_base), ("mov", moving_base))}
    if any(v is None for v in lp.values()):
        print("  missing PS/JB annotations -- skipped")
        return None
    P = {k: load_xy(v) for k, v in lp.items()}
    n = min(len(v) for v in P.values())
    ps_r, ps_m = to_image_pixels(P[("PS", "ref")][:n], tissue), to_image_pixels(P[("PS", "mov")][:n], tissue)
    jb_r, jb_m = to_image_pixels(P[("JB", "ref")][:n], tissue), to_image_pixels(P[("JB", "mov")][:n], tissue)

    px = px_50pc(tissue)
    rgb_r, rgb_m = load_rgb(img_r_path), load_rgb(img_m_path)
    Hr, Wr = rgb_r.shape[:2]
    Hm, Wm = rgb_m.shape[:2]

    # VERIFY pixel-size bookkeeping: ref landmarks must sit in the REF image, mov in the
    # MOV image (the two sections have different pixel dimensions).
    for name, pts, W, H in (("PS_ref", ps_r, Wr, Hr), ("JB_ref", jb_r, Wr, Hr),
                            ("PS_mov", ps_m, Wm, Hm), ("JB_mov", jb_m, Wm, Hm)):
        if not (pts[:, 0].min() >= -3 and pts[:, 1].min() >= -3
                and pts[:, 0].max() <= W + 3 and pts[:, 1].max() <= H + 3):
            print(f"  [SCALE ERROR] {name} extent {pts[:,0].max():.0f}x{pts[:,1].max():.0f} "
                  f"exceeds image {W}x{H} -- scale bookkeeping wrong; refusing to certify")
            return None
    print(f"  ref {Wr}x{Hr}px  mov {Wm}x{Hm}px  px(50pc)={px:.3f} um  n_landmarks={n}  "
          f"[landmarks @ {LANDMARK_SCALE_PC[tissue_prefix(tissue)]}pc, image @ 50pc]")

    # ---- drop discordant landmarks (uses only the two annotations, never a transform) ----
    f_ref = sr.fle_from_repeat(ps_r, jb_r, px)
    f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    if f_ref["fle_um"] is None or f_mov["fle_um"] is None:
        print("  FLE undetermined -- skipped"); return None
    keep = np.array(f_ref["concordant"]) & np.array(f_mov["concordant"])
    ps_r, ps_m, jb_r, jb_m = ps_r[keep], ps_m[keep], jb_r[keep], jb_m[keep]
    dropped = int((~keep).sum())
    f_ref = sr.fle_from_repeat(ps_r, jb_r, px); f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    fle = float(np.sqrt(np.mean([f_ref["fle_um"] ** 2, f_mov["fle_um"] ** 2])))
    print(f"  inter-observer FLE (floor)   {fle:6.2f} um/coord   ({dropped} discordant dropped)")

    # ================= A. LoFTR registration (landmark-free) =================
    # Match at a common working um/px on tissue crops, then map correspondences BACK to
    # full-resolution 50pc pixels. ALL geometry below is in full-res px -- no working-
    # coordinate confusion, and the answer is invariant to WORK_MAX_DIM (a bookkeeping check).
    small_r, small_m, pxw, r, back_r, back_m = prep_pair_for_loftr(rgb_r, rgb_m, px)
    print(f"  LoFTR working {small_r.shape[1]}x{small_r.shape[0]} & {small_m.shape[1]}x{small_m.shape[0]}px  "
          f"px_work={pxw:.2f} um (r={r:.3f})")
    M_lmk0 = sr._fit_similarity_robust(ps_m, ps_r)
    reB0 = realized_um(M_lmk0, jb_m, jb_r, px)          # landmark baseline, always available
    c = lm.loftr_correspondences(small_r, small_m, pixel_size_um=pxw, weights="outdoor")
    if not c["ok"]:
        print(f"  LoFTR: {c['msg']} -- NO MATCHES, cannot register")
        print(f"  B. landmark(PS) realized   p90 {np.percentile(reB0,90):6.2f} um (LoFTR failed here)")
        return dict(tissue=tissue, loftr_ok=False, fle=fle, B_p90=float(np.percentile(reB0, 90)))
    lr, lm_ = back_r(c["ref_points"]), back_m(c["mov_points"])   # -> full-res 50pc px
    print(f"  LoFTR: {c['msg']}")
    M_loftr = sr._fit_similarity_robust(lm_, lr)      # moving -> ref, full-res px

    reA = np.concatenate([realized_um(M_loftr, ps_m, ps_r, px),
                          realized_um(M_loftr, jb_m, jb_r, px)])  # all experts held out

    # ================= B. previous method: fit on PS landmarks (computed above) =======
    reB = reB0                                        # scored at independent JB

    # ================= FW bound + matcher-tail diagnosis =================
    fle_l = lm.loftr_fle(small_r, small_m, c["ref_points"], c["mov_points"], pixel_size_um=pxw)
    fle_loftr = fle_l["fle_um"] if fle_l["fle_um"] else fle
    dec = sr.deformation_from_landmarks(lr, lm_, M_loftr, px, fle_loftr, method="robust")
    deform = max(dec["deformation_p90_um"] or 0.0, dec["deformation_rms_um"] or 0.0)
    # Split tissue deformation from the LoFTR mismatch tail: residual of each match after M.
    resid = np.linalg.norm(sr._apply_affine(lm_, M_loftr) - lr, axis=1) * px
    med = float(np.median(resid)); mad = float(np.median(np.abs(resid - med))) * 1.4826
    inl = resid <= med + 3.0 * max(mad, 1e-6)
    deform_inl = float(np.percentile(resid[inl], 90)) if inl.sum() > 5 else deform
    outlier_frac = float(1.0 - inl.mean())
    tre = sr.transform_prediction_error(lr, fle_loftr * np.sqrt(2.0), np.vstack([ps_r, jb_r]))
    pred = None if tre is None else np.sqrt(tre ** 2 + deform ** 2 + 2.0 * fle ** 2)
    assay = sr.residual_field_assay(lr, lm_, M_loftr, px, fle_loftr)

    # -------- report --------
    print(f"  A. LoFTR realized error    p90 {np.percentile(reA,90):6.2f}  median {np.median(reA):6.2f} um "
          f"(n={len(reA)} expert pts, all held out)")
    print(f"  B. landmark(PS) realized   p90 {np.percentile(reB,90):6.2f}  median {np.median(reB):6.2f} um "
          f"(prev method, scored at JB)")
    print(f"  C. inter-observer floor    ~   {fle*sr._RAYLEIGH_P90:6.2f}  (FLE {fle:.2f} um/coord)")
    print(f"  LoFTR FLE {fle_loftr:5.2f} um   deform(raw p90) {deform:6.2f}   "
          f"deform(inlier p90) {deform_inl:6.2f}   matcher-tail {100*outlier_frac:4.1f}%")
    if pred is not None:
        ratio = float(np.percentile(reA, 90) / np.percentile(pred, 90))
        cov = float(np.mean(reA <= pred))
        print(f"  FW bound: predicted p90 {np.percentile(pred,90):6.2f}  realized p90 {np.percentile(reA,90):6.2f} "
              f"-> ratio {ratio:.2f} ({'UNDER-states' if ratio>1.15 else 'ok'})  coverage {100*cov:.0f}%")
    else:
        ratio, cov = float('nan'), float('nan')
        print("  FW bound: degenerate design, not computed")
    print(f"  residual_field_assay: {assay.get('verdict','?')}  (Moran I "
          f"{assay.get('moran_i','?')}, p {assay.get('p_value','?')})")

    return dict(tissue=tissue, loftr_ok=True, n_loftr=c["n"], fle=fle,
                A_p90=float(np.percentile(reA, 90)), A_med=float(np.median(reA)),
                B_p90=float(np.percentile(reB, 90)), B_med=float(np.median(reB)),
                fle_loftr=fle_loftr, deform=deform, ratio=ratio, cov=cov,
                assay=assay.get("verdict", "?"))


def main():
    print("=" * 78)
    print("PRETRAINED LoFTR vs landmark registration on ANHIR, with FW-bound calibration")
    print("=" * 78)
    if not os.path.isdir(IMG_ROOT):
        print(f"image root {IMG_ROOT} not found -- attach drive / run downloader first")
        return 2
    res = [r for r in (one_pair(*p) for p in PAIRS) if r]
    ok = [r for r in res if r.get("loftr_ok")]
    if not ok:
        print("\nno pairs with LoFTR matches -- cannot conclude")
        return 1
    print("\n" + "=" * 78)
    A = np.array([r["A_p90"] for r in ok]); B = np.array([r["B_p90"] for r in ok])
    ratios = np.array([r["ratio"] for r in ok])
    print(f"pairs registered by LoFTR: {len(ok)}/{len(PAIRS)}")
    print(f"  LoFTR   realized p90 (um):  {', '.join(f'{x:.2f}' for x in A)}")
    print(f"  landmark realized p90 (um): {', '.join(f'{x:.2f}' for x in B)}")
    better = int((A <= B + 0.5).sum())
    print(f"\n[{'PASS' if better==len(ok) else 'PARTIAL' if better else 'FAIL'}] "
          f"LoFTR matches-or-beats landmark registration on {better}/{len(ok)} pairs")
    safe = np.all(ratios[~np.isnan(ratios)] <= 1.15)
    print(f"[{'PASS' if safe else 'FAIL'}] FW bound does not under-state realized error "
          f"(all ratios <= 1.15)")
    print("\nINTERPRETATION:")
    print("  A<=B and A near the FLE floor  -> pretrained LoFTR suffices; DO NOT fine-tune.")
    print("  A>B                            -> fine-tuning is justified (this is the evidence).")
    print("=" * 78)
    return 0 if (better == len(ok) and safe) else 1


if __name__ == "__main__":
    sys.exit(main())
