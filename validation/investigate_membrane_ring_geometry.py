"""
investigate_membrane_ring_geometry.py — does ring PLACEMENT limit faint-membranous TIM-3?

THE QUESTION. The hardened membrane path segments NUCLEI (InstanSeg) and measures DAB in a
Voronoi-clipped expanded ring. On faint tissue (92290_TIM3_IM) it drops to F1 0.30. Would a
different segmenter that gives the TRUE cell/cytoplasm boundary (e.g. Cellpose-cyto) recover
it — i.e. is the failure a ring-PLACEMENT problem, or a pixel-level stain-CONTRAST floor?

THE TEST (label-free, no new segmenter needed). Cellpose changes WHERE the ring sits. So we
sweep the ring geometry ourselves — expansion 1→6 µm — and ask whether the SEPARABILITY of
the positive-cell population responds. If separability is flat across the whole geometry
sweep on every image, then no ring placement (Cellpose included) unlocks the signal, and the
failure is a stain-contrast floor a segmenter cannot fix. If separability jumps at some
geometry, ring placement matters and Cellpose-cyto is worth a real trial.

Metric: Ashman's D of per-cell ring DAB OD (cytoplasm_dab_mean) across cells — how bimodal
(positive vs negative) the population is. Plus the production positive-rate (membrane_pos_frac
≥ frac_min) and the DAB>H dominance fraction as context.

Data: TIM3_CRC_ICM labeling set (_seg_in images + _seg_out InstanSeg detections). 92290_IM
is the known faint failure; 92658_IM / 92625_CT / 9212046_CT are adequately-stained controls.
Usage:  python validation/investigate_membrane_ring_geometry.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from validation.datasets.resolve import dataset_dir
from oasis.quant.cell_expansion import measure_cytoplasm_dab
from oasis.quant.nuclear_classify import gmm_1d_two

PIX_THR, FRAC_MIN, PX = 0.30, 0.14, 0.5          # production membrane cutoffs
EXPANSIONS = [1.0, 2.0, 3.0, 4.0, 6.0]
IMAGES = [("92290_TIM3_IM", "FAINT (known F1 0.30)"),
          ("92658_TIM3_IM", "control"), ("92625_TIM3_CT", "control"),
          ("9212046_TIM3_CT", "control")]


def main():
    L = os.path.join(str(dataset_dir("tim3_crc_icm")), "inputs", "labeling")
    print(f"Membrane ring-geometry investigation — {L}")
    print(f"{'image':18s} {'exp':>4} {'ncell':>6} {'pos_rate':>9} {'dabDom%':>8} "
          f"{'ringDAB_med':>11} {'sep(Ashman D)':>13}")
    for stem, tag in IMAGES:
        img = os.path.join(L, "_seg_in", stem + ".jpg")
        geo = os.path.join(L, "_seg_out", stem + "_detections.geojson")
        if not (os.path.exists(img) and os.path.exists(geo)):
            print(f"{stem}: missing image/geojson"); continue
        print(f"--- {stem}  [{tag}] ---")
        seps = []
        for e in EXPANSIONS:
            res = measure_cytoplasm_dab(img, geo, PX, expansion_um=e,
                                        membrane_pix_thr=PIX_THR, keep_ring_values=True,
                                        estimate_stains=True, dab_dominance_gate=True)
            posf, ringmean, dom_num, dom_den = [], [], 0, 0
            for r in res:
                if not r:
                    continue
                if r.get("membrane_pos_frac") is not None:
                    posf.append(r["membrane_pos_frac"])
                if r.get("cytoplasm_dab_mean") is not None:
                    ringmean.append(r["cytoplasm_dab_mean"])
                rv, rh = r.get("ring_values"), r.get("ring_h_values")
                if rv and rh:
                    rv, rh = np.asarray(rv, float), np.asarray(rh, float)
                    n = min(len(rv), len(rh))
                    dom_num += int((rv[:n] > rh[:n]).sum()); dom_den += n
            posf, ringmean = np.array(posf), np.array(ringmean)
            pos_rate = float((posf >= FRAC_MIN).mean()) if posf.size else 0.0
            dom = 100.0 * dom_num / dom_den if dom_den else 0.0
            sep = gmm_1d_two(ringmean)["ashman_d"] if ringmean.size >= 10 else 0.0
            seps.append(sep)
            print(f"{'':18s} {e:>4.0f} {len(posf):>6} {pos_rate:>9.3f} {dom:>8.1f} "
                  f"{(np.median(ringmean) if ringmean.size else 0):>11.3f} {sep:>13.2f}")
        if seps:
            print(f"{'':18s}   separability range across geometry: "
                  f"{min(seps):.2f}–{max(seps):.2f}  (Δ={max(seps)-min(seps):.2f})")
    print("\nVERDICT: if Δ separability ≈ 0 across the geometry sweep, ring PLACEMENT is not "
          "the bottleneck → a different segmenter (Cellpose) will not recover the faint case.")


if __name__ == "__main__":
    main()
