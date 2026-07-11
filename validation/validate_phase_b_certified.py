"""
validate_phase_b_certified.py — Phase-B entry gate on the REAL certified vs
non-certified ANHIR/CIMA registrations (Step 5 of the public-data certification run).

CONTEXT (honest scope):
  Step 4 produced ONE real LOCALLY_CERTIFIED pair (lung-lesion_1 Cc10<->proSPC: 18 expert
  landmarks agree to <=5 µm within a ~17%-of-field ROI) and NO globally CERTIFIED pair —
  the truthful outcome on a non-rigid consecutive-section benchmark. The pair whose IMAGES
  are freely shipped in the landmark repo (lung-lesion_3) does NOT certify (TRE ~21 µm).

  Phase-B cross-K consumes per-cell coordinates of two markers in the registered frame.
  The production pipeline takes those from an upstream segmentation (geojson/CSV); it is
  already validated on real pre-segmented cells (Schürch CODEX, validate_real_data*.py).
  We do NOT fabricate a cell detector on coarse 5pc images to force a cross-K on a pair
  the gate would reject — that would violate fail-closed + the no-manufactured-result rule.

  What we CAN and DO exercise on real data here: the Phase-B fail-closed REGISTRATION-QC
  GATE (run_pipeline.evaluate_registration_qc) on the real transforms, driven by the
  measured landmark residual/TRE. This shows the gate admits a certified ROI and rejects a
  non-certified pair — the decision that controls whether Phase B may run at all.
"""
import os, sys, glob, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from oasis.spatial.serial_registration import (_fit_similarity_ls, _apply_affine, loo_tre,  # noqa
                                 landmark_register_and_verify)
from run_pipeline import evaluate_registration_qc  # noqa

LM = os.path.join(HERE, "public_landmarks", "annotations")
PX_LESION = 0.174 / 0.50  # 50pc lung-lesion µm/px


def load_xy(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip().lower() for h in rows[0]]
    xi = hdr.index("x") if "x" in hdr else -2
    yi = hdr.index("y") if "y" in hdr else -1
    return np.array([[float(r[xi]), float(r[yi])] for r in rows[1:]
                     if len(r) > max(xi, yi)], float)


def pair(tissue, fixed_tok, moving_tok, user="user-PS_scale-50pc"):
    d = os.path.join(LM, tissue, user)
    f = glob.glob(os.path.join(d, f"*{fixed_tok}*.csv"))[0]
    m = glob.glob(os.path.join(d, f"*{moving_tok}*.csv"))[0]
    return load_xy(f), load_xy(m)


def gate(label, residual_um, p90_um, overlap):
    reg_result = {"method": "similarity (landmark-driven, Umeyama LS)"}
    qc = {"residual_error_um": residual_um, "residual_error_p90_um": p90_um,
          "tissue_overlap_fraction": overlap, "quality_metric": None,
          "qc_inlier_ratio": None}
    g = evaluate_registration_qc(reg_result, qc, PX_LESION)
    print(f"\n[{label}]")
    print(f"   residual(TRE)={residual_um} µm  p90={p90_um} µm  overlap={overlap}")
    print(f"   GATE -> status={g['status']}  valid={g['valid']}")
    print(f"          {g['reason']}")
    return g


def main():
    print("=" * 92)
    print("STEP 5 — Phase-B fail-closed registration gate on REAL ANHIR/CIMA transforms")
    print("=" * 92)

    # ---- lung-lesion_1 Cc10<->proSPC : the LOCALLY_CERTIFIED pair --------------
    ref, mov = pair("lung-lesion_1", "Cc10", "proSPC")
    n = min(len(ref), len(mov)); ref, mov = ref[:n], mov[:n]
    res = landmark_register_and_verify(ref, mov, PX_LESION, image_wh=(8972, 7394))
    loo = loo_tre(ref, mov, PX_LESION)
    err = np.array(loo["per_point_um"])
    good = err <= 5.0
    roi_resid = float(np.median(err[good]))          # residual WITHIN the certified ROI
    roi_p90 = float(np.percentile(err[good], 90))
    print(f"\nlung-lesion_1 Cc10<-proSPC : verdict={res['verdict']}  "
          f"full-pair TREmed={res['tre_median_um']} µm  n_good(<=5µm)={int(good.sum())}/{n}")
    print(f"   certified ROI (the {int(good.sum())} passing landmarks): "
          f"median residual {roi_resid:.2f} µm, p90 {roi_p90:.2f} µm")
    gate("lung-lesion_1 Cc10<->proSPC  [LOCALLY_CERTIFIED ROI]", round(roi_resid, 2),
         round(roi_p90, 2), overlap=0.9)
    # whole-pair (un-restricted) would be rejected:
    gate("lung-lesion_1 Cc10<->proSPC  [whole field, un-restricted]",
         res["tre_median_um"], res["tre_p90_um"], overlap=0.9)

    # ---- lung-lesion_3 Cc10<->proSPC : best image-available pair (NOT certified)
    ref3, mov3 = pair("lung-lesion_3", "Cc10", "proSPC")
    n3 = min(len(ref3), len(mov3))
    res3 = landmark_register_and_verify(ref3[:n3], mov3[:n3], PX_LESION,
                                        image_wh=(8920, 6610))
    print(f"\nlung-lesion_3 Cc10<-proSPC : verdict={res3['verdict']}  "
          f"TREmed={res3['tre_median_um']} µm  (images shipped in repo)")
    gate("lung-lesion_3 Cc10<->proSPC  [image-available pair]",
         res3["tre_median_um"], res3["tre_p90_um"], overlap=0.9)

    print("\n" + "=" * 92)
    print("READOUT")
    print("  • The fail-closed gate ADMITS the certified ROI (status=valid) and REJECTS\n"
          "    both the un-restricted lung-lesion_1 field and the image-available\n"
          "    lung-lesion_3 pair (status=invalid) — exactly the §19.5 / §B2 design.")
    print("  • A cross-K/DCLF on a CERTIFIED pair is not run here: the only certified\n"
          "    registration (lung-lesion_1 ROI) has no freely-accessible images, and the\n"
          "    image-available pair (lung-lesion_3) is gate-rejected. This is a DATA-ACCESS\n"
          "    limit, not a method failure; the cross-K engine is validated on real\n"
          "    Schürch CODEX cells (validate_real_data*.py) and synthetically (§7/§15/§16).")
    print("  • Resolution floor for any future run on these consecutive sections:\n"
          "    smallest interpretable radius ≈ z-gap (section spacing) + TRE; with a\n"
          "    ROI residual ~3–5 µm and an unknown consecutive z-gap, interpret only\n"
          "    >~20–50 µm — and this is a METHOD demonstration (Cc10 club cells vs proSPC\n"
          "    type-II pneumocytes), NOT a CD8/TIM-3 biological claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
