"""
Is the Fitzpatrick-West error bound CALIBRATED? The only experiment that can answer it.

THE QUESTION. `landmark_register_and_verify(..., fle_um=...)` predicts the error a cell
experiences: sqrt(TRE_pred^2 + deformation^2). Every test so far was one I designed, on
synthetic deformation, on a single pair. A predictive bound is worthless unless the errors
it predicts are the errors that actually occur. If it says 4 um and the truth is 8 um, it
is an over-confident safety check and must not ship -- worse than the flawed gate it
replaces.

THE DATA. ANHIR/CIMA is the only source here with TWO INDEPENDENT ANNOTATORS (PS and JB)
marking the same anatomical landmarks on the same images. That gives, at once:

  FLE       measured, not assumed: PS and JB clicking the same point differ by the sum of
            two independent draws of the localisation error, so Var(difference) = 2*FLE^2.
            This is the inter-observer FLE -- the one that bounds what a READER could
            reproduce, and the number to quote in a paper (the ANHIR convention).

  TRUTH     an annotator-INDEPENDENT held-out error. Fit the similarity on PS's landmarks
            only; measure the realized error at JB's landmarks. JB never touched the fit.

THE TEST. Predicted error at a JB landmark must account for JB's own clicking noise, since
that is what the measurement carries:

    predicted_i = sqrt( TRE_pred(p_i)^2  +  deformation^2  +  2*FLE^2 )
                          estimation          model            JB's own FLE, both images

    realized_i  = || M_PS . JB_mov_i  -  JB_ref_i || * px

Calibrated means realized_p90 ~ predicted_p90, and per-point coverage ~ 90%. Ratio > 1 means
the bound UNDER-states error (anti-conservative -- do not ship). Ratio < 1 means it
over-states (safe, but the gate is needlessly strict).

The legacy leave-one-out gate is run alongside, on the same PS landmarks, for contrast.

Run:  .venv/bin/python validation/validate_fw_anhir_calibration.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import serial_registration as sr  # noqa: E402

# native um/px (ANHIR Table I) and the scale the landmark CSVs are stored at
TISSUE_PX = {"lung-lesion": (0.174, 0.50), "lung-lobes": (1.274, 1.00),
             "mammary-gland": (2.294, 1.00)}
IMAGE_WH = {"lung-lesion_3": (8920, 6610), "mammary-gland_1": (10000, 8000),
            "mammary-gland_2": (10000, 8000)}
# (tissue, fixed csv basename, moving csv basename) -- both annotated by PS and JB
PAIRS = [
    ("lung-lesion_3", "29-041-Izd2-w35-He-les3.csv", "29-041-Izd2-w35-proSPC-4-les3.csv"),
    ("mammary-gland_1", "s1_37-HE_A4926-4L.csv", "s1_40-PR_A4926-4L.csv"),
    ("mammary-gland_2", "s2_63-HE_A4926-4L.csv", "s2_68-ER-A4962-4L.csv"),
]


def _roots():
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


def find(tissue, user, base):
    for root in _roots():
        hits = glob.glob(os.path.join(root, tissue, f"user-{user}_scale-*", base))
        if hits:
            return hits[0]
    return None


def px_for(tissue):
    for pref, (native, scale) in TISSUE_PX.items():
        if tissue.startswith(pref):
            return native / scale
    raise KeyError(tissue)


def one_pair(tissue, fixed, moving):
    paths = {(u, k): find(tissue, u, b) for u in ("PS", "JB")
             for k, b in (("ref", fixed), ("mov", moving))}
    if any(v is None for v in paths.values()):
        print(f"{tissue}: missing annotations, skipped")
        return None
    P = {k: load_xy(v) for k, v in paths.items()}
    n = min(len(v) for v in P.values())
    ps_r, ps_m = P[("PS", "ref")][:n], P[("PS", "mov")][:n]
    jb_r, jb_m = P[("JB", "ref")][:n], P[("JB", "mov")][:n]
    px, wh = px_for(tissue), IMAGE_WH.get(tissue)

    # ── FLE: inter-observer, pooled over both images of the pair ──────────────
    # DISCORDANT landmarks (the two experts marked different structures) are dropped. That
    # decision uses ONLY the two annotations, never the transform, so it cannot bias the
    # certification it is about to test. A row where two experts disagree by 150 um is not
    # a landmark, and no error model should be asked to cover it.
    f_ref = sr.fle_from_repeat(ps_r, jb_r, px)
    f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    if f_ref["fle_um"] is None or f_mov["fle_um"] is None:
        print(f"{tissue}: FLE undetermined, skipped")
        return None
    keep = np.array(f_ref["concordant"]) & np.array(f_mov["concordant"])
    dropped = int((~keep).sum())
    ps_r, ps_m, jb_r, jb_m = ps_r[keep], ps_m[keep], jb_r[keep], jb_m[keep]
    n = int(keep.sum())
    f_ref = sr.fle_from_repeat(ps_r, jb_r, px)
    f_mov = sr.fle_from_repeat(ps_m, jb_m, px)
    fle = float(np.sqrt(np.mean([f_ref["fle_um"] ** 2, f_mov["fle_um"] ** 2])))

    # ── Fit on PS only. JB is never seen by the fit. ──────────────────────────
    M = sr._fit_similarity_robust(ps_m, ps_r)
    dec = sr.deformation_from_landmarks(ps_r, ps_m, M, px, fle, method="robust")
    # The gate's quantity: the p90 of the deformation FIELD, read off the empirical
    # residual distribution, not an RMS. (The RMS summary under-states this by ~1.6x --
    # a smooth field's magnitudes are not Rayleigh distributed.)
    deform = max(dec["deformation_p90_um"] or 0.0, dec["deformation_rms_um"] or 0.0)
    deform_ub = max(dec["deformation_p90_ub_um"] or 0.0, dec["deformation_rms_ub_um"] or 0.0)

    # ── Predicted error at each JB landmark, incl. JB's own clicking noise ────
    tre = sr.transform_prediction_error(ps_r, fle * np.sqrt(2.0), jb_r)
    if tre is None:
        print(f"{tissue}: degenerate design, skipped")
        return None
    ann = 2.0 * fle ** 2                     # JB clicks BOTH images
    pred = np.sqrt(tre ** 2 + deform ** 2 + ann)
    pred_ub = np.sqrt(tre ** 2 + deform_ub ** 2 + ann)

    # ── Realized, annotator-independent error ────────────────────────────────
    real = np.linalg.norm(sr._apply_affine(jb_m, M) - jb_r, axis=1) * px

    cov = float(np.mean(real <= pred_ub))
    ratio = float(np.percentile(real, 90) / np.percentile(pred, 90))

    legacy = sr.landmark_register_and_verify(ps_r, ps_m, px, image_wh=wh)
    fw = sr.landmark_register_and_verify(ps_r, ps_m, px, image_wh=wh, fle_um=fle)

    print(f"\n{tissue}  (n={n} concordant, {dropped} discordant dropped, {px:.3f} um/px)")
    print(f"  inter-observer FLE (PS vs JB)   {fle:8.2f} um/coord   "
          f"[ref {f_ref['fle_um']}, mov {f_mov['fle_um']}]")
    print(f"  deformation from PS residuals   {deform:8.2f} um  (95% UB {deform_ub:.2f})")
    print(f"  predicted p90 (point est)       {np.percentile(pred, 90):8.2f} um")
    print(f"  realized  p90 (JB, held out)    {np.percentile(real, 90):8.2f} um")
    print(f"  realized  median                {np.median(real):8.2f} um   "
          f"(predicted {np.median(pred):.2f})")
    print(f"  calibration ratio realized/predicted (p90)   {ratio:5.2f}   "
          f"{'UNDER-states error' if ratio > 1.15 else 'ok' if ratio > 0.7 else 'over-states'}")
    print(f"  per-point coverage by the 95% bound          {100*cov:4.0f}%  (target ~90-95%)")
    print(f"  verdicts:  legacy={legacy['verdict']:<18} FW={fw['verdict']}")
    return dict(tissue=tissue, fle=fle, ratio=ratio, cov=cov,
                real_p90=float(np.percentile(real, 90)),
                pred_p90=float(np.percentile(pred, 90)))


def main():
    print("=" * 78)
    print("Fitzpatrick-West bound: calibration against a SECOND, independent annotator")
    print("=" * 78)
    res = [r for r in (one_pair(*p) for p in PAIRS) if r]
    if not res:
        print("\nno usable pairs found")
        return 1
    print("\n" + "=" * 78)
    ratios = np.array([r["ratio"] for r in res])
    covs = np.array([r["cov"] for r in res])
    print(f"pairs: {len(res)}   calibration ratio  min {ratios.min():.2f}  "
          f"median {np.median(ratios):.2f}  max {ratios.max():.2f}")
    print(f"                    95%-bound coverage  min {100*covs.min():.0f}%  "
          f"median {100*np.median(covs):.0f}%")
    safe = bool((ratios <= 1.15).all())
    covered = bool((covs >= 0.85).all())
    print(f"\n[{'PASS' if safe else 'FAIL'}] the bound does not UNDER-state realized error "
          f"on any pair (ratio <= 1.15)")
    print(f"[{'PASS' if covered else 'FAIL'}] the 95% bound covers >=85% of held-out "
          f"landmark errors on every pair")
    if not safe:
        print("\n  => ANTI-CONSERVATIVE. Do not wire this gate into the app.")
    print("=" * 78)
    return 0 if (safe and covered) else 1


if __name__ == "__main__":
    sys.exit(main())
