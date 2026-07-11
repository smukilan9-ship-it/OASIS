"""
phase_a_finalize.py — LANDMARK-DRIVEN Gate-A certification (4-state).

Landmarks DEFINE each pair's registration (least-squares similarity —
distance-preserving, no non-rigid warp). Accuracy is measured on HELD-OUT points
(independent second-annotator set if supplied, else leave-one-out — fit-unbiased but
NOT annotator-independent). Verdict (a failed pair is reported, never forced):

  CERTIFIED         n≥6, held-out TRE median ≤5 µm, fit-residual ≤5 µm → eligible for Phase B
  LOCALLY_CERTIFIED only a spatial ROI passes → analyse that ROI only
  DEFORMED          confident correspondences but a similarity can't fit within tolerance
  NOT_CERTIFIABLE   too few unambiguous correspondences to measure accuracy
                    (NOT positive evidence the sections are unrelated)

Inputs:  phase_a_qc/registration.json (pixel sizes + scale bars),
         phase_a_qc/landmarks.json (pasted from landmark_tool.html),
         optional phase_a_qc/landmarks_val.json (independent second-annotator set).
Outputs: gate_a_table.md, gate_a_final.json, certified_transforms.json (Phase B input).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial.serial_registration import landmark_register_and_verify  # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_a_qc")
IMAGE_WH = (1920, 1440)
MIN_N, TARGET_N, LOO_MAX, FIT_MAX, DEFORMED_LOO, SCALE_TOL = 6, 12, 5.0, 5.0, 15.0, 0.03


def _load(path):
    if not os.path.exists(path):
        return {}
    t = open(path).read().strip()
    if t.startswith("LANDMARKS"):
        t = t[len("LANDMARKS"):].strip()
    return json.loads(t) if t else {}


def main():
    lm_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "landmarks.json")
    reg = json.load(open(os.path.join(OUT, "registration.json")))
    lm = _load(lm_path)
    val = _load(os.path.join(OUT, "landmarks_val.json"))   # optional 2nd annotator

    rows, certified = [], {}
    for sid, r in reg.items():
        px = r["pixel_size_um"]
        row = {"sample_id": sid, "pixel_size_um": px}
        pts = lm.get(sid, {}).get("points", [])
        if not pts:
            row.update(n=0, verdict="NOT_CERTIFIABLE", reason="no landmarks supplied")
            rows.append(row); continue
        p = np.array(pts, float)
        vp = np.array(val[sid]["points"], float) if sid in val and val[sid].get("points") else None
        res = landmark_register_and_verify(
            p[:, 0:2], p[:, 2:4], px,
            val_ref_pts=(vp[:, 0:2] if vp is not None else None),
            val_mov_pts=(vp[:, 2:4] if vp is not None else None),
            image_wh=IMAGE_WH, min_n=MIN_N, target_n=TARGET_N,
            loo_max_um=LOO_MAX, fit_max_um=FIT_MAX, deformed_loo_um=DEFORMED_LOO)
        row.update(n=res["n"], est_scale=res["est_scale"], coverage_frac=res["coverage_frac"],
                   fit_residual_um=res["fit_residual_um"], tre_median_um=res["tre_median_um"],
                   tre_p90_um=res["tre_p90_um"], tre_max_um=res["tre_max_um"],
                   validation=res["validation"], verdict=res["verdict"], reason=res["reason"])
        if r.get("ref_bar_px") and r.get("mov_bar_px") and res["est_scale"]:
            row["scale_xcheck_ok"] = abs(res["est_scale"] - r["ref_bar_px"] / r["mov_bar_px"]) <= SCALE_TOL
        else:
            row["scale_xcheck_ok"] = None
        if res["verdict"] in ("CERTIFIED", "LOCALLY_CERTIFIED"):
            certified[sid] = {"pixel_size_um": px, "matrix": res["matrix"],
                              "tre_um": res["tre_median_um"], "verdict": res["verdict"],
                              "roi_polygon": res["roi_polygon"], "coverage_frac": res["coverage_frac"],
                              "ref_path": r["ref_path"], "mov_path": r["mov_path"]}
        rows.append(row)

    hdr = ["pair", "px_um", "n", "cover%", "fit_um", "TRE_med", "TRE_p90", "TRE_max",
           "scale", "xchk", "verdict"]
    def fmt(x):
        cov = f"{x['coverage_frac']*100:.0f}" if x.get("coverage_frac") else "-"
        return [x["sample_id"], x["pixel_size_um"], x.get("n"), cov, x.get("fit_residual_um"),
                x.get("tre_median_um"), x.get("tre_p90_um"), x.get("tre_max_um"),
                x.get("est_scale"), x.get("scale_xcheck_ok"), x["verdict"]]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    md += ["| " + " | ".join(str(v) for v in fmt(x)) + " |" for x in rows]
    table = "\n".join(md)
    print(table)
    for x in rows:
        print(f"  {x['sample_id']} [{x['verdict']}] {x.get('validation','')}: {x['reason']}")

    open(os.path.join(OUT, "gate_a_table.md"), "w").write(table + "\n")
    json.dump(rows, open(os.path.join(OUT, "gate_a_final.json"), "w"), indent=2)
    json.dump(certified, open(os.path.join(OUT, "certified_transforms.json"), "w"), indent=2)
    cert = list(certified)
    print(f"\nCERTIFIED / LOCALLY_CERTIFIED (eligible for Phase B): {cert if cert else 'none'}")
    print("Wrote gate_a_table.md, gate_a_final.json, certified_transforms.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
