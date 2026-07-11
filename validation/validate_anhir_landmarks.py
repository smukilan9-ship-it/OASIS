"""
validate_anhir_landmarks.py — run the landmark-driven registration CERTIFICATION
(`serial_registration.landmark_register_and_verify`) against REAL expert landmarks
from the public ANHIR / CIMA histology-landmarks dataset.

WHY (and how this relates to §18-19 + §19.7):
  §19.7 chose HyReCo as the public method-validation anchor but its landmark CSVs are
  distributed ONLY inside 233 GB+ login-gated IEEE-DataPort archives (verified Jun 2026),
  so the "real-CSV run" could not be completed. The ANHIR/CIMA landmark repo
  (github.com/Borda/dataset-histology-landmarks, the official ANHIR-challenge landmark
  ground truth, CC-BY 4.0, NO login) ships the SAME class of data: expert anatomical
  landmarks on real consecutive multi-stain histology serial sections, ~80 points/image,
  with a SECOND annotator on some pairs. Our certification operates on landmark
  COORDINATES alone, so this validates the METHOD exactly as a HyReCo run would — on real
  expert ground truth — independent of any TIM-3 biology.

WHAT IT TESTS (honestly):
  These are CONSECUTIVE sections (different physical planes) and ANHIR is by construction a
  NON-RIGID registration benchmark. Our certification fits a SIMILARITY only (no warp,
  to keep the cross-K distance metric valid). So the truthful expectation is:
    • pairs whose deformation is well-approximated by a similarity  -> CERTIFIED / LOCALLY
    • pairs needing real non-rigid warp                            -> DEFORMED
    • pairs with incoherent / too-few correspondence               -> NOT_CERTIFIABLE
  No thresholds are tuned. We report whatever comes out.

VALIDATION GRADE:
  • single-annotator pairs  -> leave-one-out (fit-unbiased, NOT annotator-independent).
  • two-annotator pairs     -> fit on annotator PS, score held-out on annotator JB's
                               independent corresponding landmarks => ANHIR-grade,
                               annotator-INDEPENDENT TRE (the strongest test we can run).

Pixel sizes (ANHIR Table I native µm/px ÷ stored landmark scale):
  lung-lesion  0.174 µm/px @100pc  -> 0.348 µm/px @ scale-50pc
  lung-lobes   1.274 µm/px @100pc  -> 1.274 µm/px @ scale-100pc
  mammary      2.294 µm/px @100pc  -> 2.294 µm/px @ scale-100pc
"""
import os, sys, csv, glob, itertools, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from oasis.spatial.serial_registration import landmark_register_and_verify  # noqa

LM = os.path.join(HERE, "public_landmarks", "annotations")

# native µm/px (ANHIR Table I) and the scale the landmark CSVs are stored at
TISSUE_PX = {  # tissue-prefix -> (native_um_per_px, landmark_scale_fraction)
    "lung-lesion":   (0.174, 0.50),
    "lung-lobes":    (1.274, 1.00),
    "mammary-gland": (2.294, 1.00),
}
# image_wh (width, height) in landmark-scale (50pc) px, for coverage / LOCALLY_CERTIFIED
# ROI. From ANHIR/CIMA 100pc sizes (H,W) halved; les3 measured from the 5pc images.
IMAGE_WH = {
    "lung-lesion_1": (8972, 7394),   # 100pc (14789,17944)=(H,W) -> 50pc
    "lung-lesion_2": (13911, 9470),  # 100pc (18940,27823)=(H,W) -> 50pc
    "lung-lesion_3": (8920, 6610),   # 5pc 892x661 -> 50pc
}


def px_for(tissue_dir):
    for pref, (native, scale) in TISSUE_PX.items():
        if tissue_dir.startswith(pref):
            return native / scale
    raise KeyError(tissue_dir)


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


def stain_of(fname):
    m = re.search(r"-(CD31|Cc10|He|HE|Ki67|ki67|proSPC|Pro-SPC|cd31|cc10|"
                  r"ER|PR|CNEU)[-_.]", fname)
    if m:
        return m.group(1)
    return os.path.splitext(fname)[0]


def run():
    results = []
    tissues = sorted(d for d in os.listdir(LM) if os.path.isdir(os.path.join(LM, d)))
    for tissue in tissues:
        try:
            px = px_for(tissue)
        except KeyError:
            continue
        ps_dirs = glob.glob(os.path.join(LM, tissue, "user-PS_scale-*"))
        if not ps_dirs:
            continue
        ps_dir = ps_dirs[0]
        csvs = sorted(glob.glob(os.path.join(ps_dir, "*.csv")))
        # de-duplicate stains (mammary has duplicate HE captures) — keep first per stain
        by_stain = {}
        for c in csvs:
            s = stain_of(os.path.basename(c))
            by_stain.setdefault(s, c)
        stains = sorted(by_stain)
        wh = IMAGE_WH.get(tissue)
        for a, b in itertools.combinations(stains, 2):
            ref = load_xy(by_stain[a]); mov = load_xy(by_stain[b])
            n = min(len(ref), len(mov))
            if n < 6:
                continue
            res = landmark_register_and_verify(ref[:n], mov[:n], px, image_wh=wh)
            results.append(dict(tissue=tissue, fixed=a, moving=b, n=n,
                                px=round(px, 4), grade="LOO", **slim(res)))
    # cross-annotator (annotator-INDEPENDENT) for lung-lesion_3 He<->proSPC
    results += cross_annotator_lung3()
    return results


def cross_annotator_lung3():
    out = []
    t = "lung-lesion_3"; px = px_for(t); wh = IMAGE_WH[t]
    ps = os.path.join(LM, t, "user-PS_scale-50pc")
    jb = os.path.join(LM, t, "user-JB_scale-50pc")
    pairs = [("He", "proSPC")]
    for a, b in pairs:
        pa = glob.glob(os.path.join(ps, f"*{a}*.csv"))[0]
        pb = glob.glob(os.path.join(ps, f"*{b}*.csv"))[0]
        ja = glob.glob(os.path.join(jb, f"*{a}*.csv"))[0]
        jbb = glob.glob(os.path.join(jb, f"*{b}*.csv"))[0]
        ref, mov = load_xy(pa), load_xy(pb)
        vref, vmov = load_xy(ja), load_xy(jbb)
        n = min(len(ref), len(mov)); vn = min(len(vref), len(vmov))
        res = landmark_register_and_verify(
            ref[:n], mov[:n], px,
            val_ref_pts=vref[:vn], val_mov_pts=vmov[:vn], image_wh=wh)
        out.append(dict(tissue=t, fixed=a, moving=b, n=n, px=round(px, 4),
                        grade=f"CROSS-ANNOTATOR (fit PS n={n}, val JB n={vn})",
                        **slim(res)))
    return out


def slim(res):
    return {k: res[k] for k in ("verdict", "tre_median_um", "tre_p90_um", "tre_max_um",
                                "fit_residual_um", "est_scale", "coverage_frac",
                                "n_good", "validation", "reason")}


def main():
    res = run()
    print("=" * 100)
    print("REAL EXPERT-LANDMARK CERTIFICATION — ANHIR/CIMA histology-landmarks dataset")
    print("=" * 100)
    cur = None
    for r in res:
        if r["tissue"] != cur:
            cur = r["tissue"]; print(f"\n### {cur}  (px={r['px']} µm)")
        print(f"  {r['fixed']:>7} <- {r['moving']:<8} n={r['n']:>3} [{r['grade']}]")
        print(f"      VERDICT {r['verdict']:<17} TREmed={r['tre_median_um']} "
              f"p90={r['tre_p90_um']} max={r['tre_max_um']} µm  "
              f"fit-resid={r['fit_residual_um']} µm  scale={r['est_scale']} "
              f"cover={r['coverage_frac']}")
    # summary
    from collections import Counter
    c = Counter(r["verdict"] for r in res)
    print("\n" + "=" * 100)
    print("SUMMARY verdict counts:", dict(c), f"  ({len(res)} pairs)")
    json.dump(res, open(os.path.join(HERE, "anhir_certification_results.json"), "w"),
              indent=2)
    print("written -> validation/anhir_certification_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
