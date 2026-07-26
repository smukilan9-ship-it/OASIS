"""
validate_nuclear_classifier.py — benchmark nuclear-DAB positivity methods against
DeepLIIF's IF-derived per-cell ground truth (Ki67, a nuclear marker).

WHAT THIS ISOLATES. Segmentation (InstanSeg) is held fixed — we read the already-
generated GeoJSON detections and their QuPath "DAB: Mean" — and only the CLASSIFICATION
step varies. Each detected cell is matched one-to-one to a GT cell (red=positive,
blue=negative in the DeepLIIF SegMask), then scored under every method on the SAME cells.
So the F1/AUC/κ differences are purely classification quality, not detection recall.

TWO CHANNELS, per the "benchmark both, F1 decides" decision:
  • dabmean  — QuPath's fixed-vector "DAB: Mean" (the current channel)
  • macenko  — per-image Macenko stain vectors, parity-selected vs QuPath's DAB
               (cell_expansion.measure_cytoplasm_dab's nucleus_dab_mean)

FOUR methods:
  • fixed@T          — DAB:Mean > T (the legacy default)
  • otsu_dabmean     — per-image Otsu on DAB:Mean (the current 'adaptive')
  • gmm_dabmean      — per-image 2-component GMM valley + abstain, on DAB:Mean
  • gmm_macenko      — per-image 2-component GMM valley + abstain, on Macenko nuclear OD

Data resolves via the validation framework (validation_data_dir), NOT ~/Desktop.

Usage:  python validation/validate_nuclear_classifier.py [n_images] [--fixed 0.2] [--ashman 2.0]
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.datasets.resolve import dataset_dir
from oasis.quant.nuclear_classify import classify_nuclear, otsu_threshold, gmm_1d_two


# ── geometry / matching ──────────────────────────────────────────────────────
def _feat_centroid(feat):
    g = feat.get("geometry", {})
    if g.get("type") != "Polygon" or not g.get("coordinates"):
        return None
    ring = np.asarray(g["coordinates"][0], float)
    return float(ring[:, 0].mean()), float(ring[:, 1].mean())


def _dabmean(feat):
    m = (feat.get("properties", {}).get("measurements", {}) or {})
    v = m.get("DAB: Mean", m.get("Nucleus: DAB OD mean"))
    return float(v) if isinstance(v, (int, float)) else np.nan


def _greedy_match(gts, pred_cen, tol=15.0):
    """One-to-one GT↔pred by ascending centroid distance. Returns {gt_idx: pred_idx}."""
    pairs = []
    for gi, g in enumerate(gts):
        gx, gy = g["xy"]
        for pj in range(len(pred_cen)):
            d2 = (pred_cen[pj][0] - gx) ** 2 + (pred_cen[pj][1] - gy) ** 2
            if d2 <= tol * tol:
                pairs.append((d2, gi, pj))
    pairs.sort()
    gu, pu, match = set(), set(), {}
    for _, gi, pj in pairs:
        if gi in gu or pj in pu:
            continue
        gu.add(gi); pu.add(pj); match[gi] = pj
    return match


# ── metrics ──────────────────────────────────────────────────────────────────
def _auc(scores, labels):
    """Rank-based ROC-AUC (Mann-Whitney), no sklearn."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _prfk(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    # Cohen's kappa
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0.0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                accuracy=round(acc, 3), kappa=round(kappa, 3),
                tp=tp, fp=fp, fn=fn, tn=tn)


def _confusion(pred_pos, gt_pos):
    tp = int(np.sum(pred_pos & gt_pos)); fp = int(np.sum(pred_pos & ~gt_pos))
    fn = int(np.sum(~pred_pos & gt_pos)); tn = int(np.sum(~pred_pos & ~gt_pos))
    return tp, fp, fn, tn


# ── per-image cell gathering ─────────────────────────────────────────────────
def _macenko_nuclear_od(image_path, geojson_path):
    """Per-feature parity-selected nuclear DAB OD, aligned to the GeoJSON features."""
    from oasis.quant.cell_expansion import measure_cytoplasm_dab
    res = measure_cytoplasm_dab(image_path, geojson_path, pixel_size_um=0.5,
                                estimate_stains=True, dab_dominance_gate=False)
    return [(r.get("nucleus_dab_mean") if r else None) for r in res]


def gather(root, cond, gt_all, n, want_macenko=True):
    out_dir = os.path.join(root, cond, "_pipeline_out")
    img_dir = os.path.join(root, cond, "changed_inputs")
    per_image = []
    eligible = sorted(s for s in gt_all if glob.glob(os.path.join(out_dir, f"{s}*.geojson")))
    # Even stride across the alphabetically-sorted stems so the sample SPANS tissue
    # types (Bladder/Lung/…) instead of taking only the first block.
    if n < len(eligible):
        idx = np.linspace(0, len(eligible) - 1, n).round().astype(int)
        stems = [eligible[i] for i in sorted(set(idx))]
    else:
        stems = eligible
    for k, stem in enumerate(stems):
        geo = glob.glob(os.path.join(out_dir, f"{stem}*.geojson"))[0]
        feats = json.load(open(geo)).get("features", [])
        cen = [_feat_centroid(f) for f in feats]
        dab = np.array([_dabmean(f) for f in feats], float)
        mac = np.full(len(feats), np.nan)
        if want_macenko:
            img = os.path.join(img_dir, stem + ".png")
            if os.path.exists(img):
                try:
                    vals = _macenko_nuclear_od(img, geo)
                    mac = np.array([v if isinstance(v, (int, float)) else np.nan
                                    for v in vals], float)
                except Exception as e:
                    print(f"    macenko failed on {stem}: {e}")
        keep = [i for i, c in enumerate(cen) if c is not None]
        cen = [cen[i] for i in keep]; dab = dab[keep]; mac = mac[keep]
        match = _greedy_match(gt_all[stem], cen)
        gi = np.array(sorted(match.keys()), int)
        if gi.size == 0:
            continue
        pj = np.array([match[g] for g in gi], int)
        per_image.append(dict(
            stem=stem,
            all_dab=dab, all_mac=mac,                       # per-image full distributions
            m_dab=dab[pj], m_mac=mac[pj],                   # matched-cell values
            m_gt=np.array([gt_all[stem][g]["label"] for g in gi], int)))
        print(f"  [{k+1}/{len(stems)}] {stem}: {len(feats)} cells, {gi.size} matched to GT")
    return per_image


# ── scoring per method ───────────────────────────────────────────────────────
def score_all(per_image, fixed_thr, ashman_min):
    methods = {}

    def _accumulate(name, per_img_call):
        TP = FP = FN = TN = 0
        n_abst = 0; abst_tp = abst_fp = abst_fn = abst_tn = 0
        for im in per_image:
            gt_pos = im["m_gt"] == 1
            pred_pos, abstained = per_img_call(im)
            if pred_pos is None:                            # image-level abstain
                continue
            tp, fp, fn, tn = _confusion(pred_pos, gt_pos)
            if abstained:
                n_abst += 1; abst_tp += tp; abst_fp += fp; abst_fn += fn; abst_tn += tn
            else:
                TP += tp; FP += fp; FN += fn; TN += tn
        r = _prfk(TP, FP, FN, TN)
        r["n_abstained_images"] = n_abst
        if n_abst:
            r["abstained_set"] = _prfk(abst_tp, abst_fp, abst_fn, abst_tn)
        methods[name] = r

    # fixed @ T on DAB:Mean
    _accumulate("fixed@%.2f_dabmean" % fixed_thr,
                lambda im: (im["m_dab"] > fixed_thr, False))

    # Otsu on DAB:Mean (current 'adaptive'), threshold from ALL cells
    def _otsu_call(im):
        vals = im["all_dab"][np.isfinite(im["all_dab"])]
        if vals.size < 20:
            return (im["m_dab"] > fixed_thr, False)
        thr = otsu_threshold(vals)
        return (im["m_dab"] > thr, False)
    _accumulate("otsu_dabmean", _otsu_call)

    # GMM valley + abstain, on each channel; threshold from ALL cells
    def _gmm_call(channel_all, channel_matched):
        def _f(im):
            vals = im[channel_all][np.isfinite(im[channel_all])]
            dec = classify_nuclear(vals, fixed_threshold=fixed_thr,
                                   ashman_min=ashman_min, allow_fixed_fallback=False)
            if dec["abstain"] or dec["threshold"] is None:
                # score the abstained image under the fixed fallback, flagged
                return (im[channel_matched] > fixed_thr, True)
            return (im[channel_matched] > dec["threshold"], False)
        return _f
    _accumulate("gmm_dabmean", _gmm_call("all_dab", "m_dab"))
    _accumulate("gmm_macenko", _gmm_call("all_mac", "m_mac"))

    # channel AUC (threshold-free separability), pooled over matched cells
    all_gt = np.concatenate([im["m_gt"] for im in per_image])
    auc_dab = _auc(np.concatenate([im["m_dab"] for im in per_image]), all_gt)
    auc_mac = _auc(np.concatenate([im["m_mac"] for im in per_image]), all_gt)
    return methods, dict(auc_dabmean=round(auc_dab, 3), auc_macenko=round(auc_mac, 3),
                         n_images=len(per_image),
                         n_matched_cells=int(all_gt.size),
                         gt_positive_frac=round(float((all_gt == 1).mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=40)
    ap.add_argument("--fixed", type=float, default=0.20)
    ap.add_argument("--ashman", type=float, default=2.0)
    ap.add_argument("--cond", default="raw_instanseg")
    ap.add_argument("--no-macenko", action="store_true")
    a = ap.parse_args()

    ddir = dataset_dir("deepliif")
    root = os.path.join(str(ddir), "_generated_outputs", "pipeline_validation")
    gt_path = os.path.join(root, "ground_truth.json")
    if not os.path.exists(gt_path):
        print(f"ERROR: ground truth not found at {gt_path}"); sys.exit(1)
    gt_all = json.load(open(gt_path))
    print(f"Nuclear classifier benchmark — DeepLIIF Ki67, cond={a.cond}, "
          f"fixed={a.fixed}, ashman_min={a.ashman}")
    print(f"  data: {root}\n  GT images: {len(gt_all)}\n")

    per_image = gather(root, a.cond, gt_all, a.n, want_macenko=not a.no_macenko)
    if not per_image:
        print("No images with matched GT — nothing to score."); sys.exit(1)
    methods, meta = score_all(per_image, a.fixed, a.ashman)

    print(f"\n{'='*78}\nRESULTS  ({meta['n_images']} images, {meta['n_matched_cells']} "
          f"matched cells, GT positive frac {meta['gt_positive_frac']})")
    print(f"  channel AUC (separability):  DAB:Mean {meta['auc_dabmean']}   "
          f"Macenko {meta['auc_macenko']}")
    print(f"{'-'*78}")
    print(f"  {'method':22s} {'F1':>6} {'prec':>6} {'rec':>6} {'acc':>6} {'kappa':>6}  abstain")
    for name, r in methods.items():
        ab = f"{r['n_abstained_images']} imgs" if r.get("n_abstained_images") else "-"
        print(f"  {name:22s} {r['f1']:>6} {r['precision']:>6} {r['recall']:>6} "
              f"{r['accuracy']:>6} {r['kappa']:>6}  {ab}")
        if r.get("abstained_set"):
            asr = r["abstained_set"]
            print(f"    └ abstained-set F1 {asr['f1']} (proof the gate withholds the bad ones)")
    print(f"{'='*78}")

    out = os.path.join(root, "nuclear_classifier_benchmark.json")
    json.dump(dict(meta=meta, methods=methods,
                   params=dict(fixed=a.fixed, ashman_min=a.ashman, cond=a.cond)),
              open(out, "w"), indent=2)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
