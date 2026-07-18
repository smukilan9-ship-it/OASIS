"""
valis_bench/common.py — shared, dependency-light core for the VALIS-vs-OASIS
serial-section registration benchmark on ANHIR/CIMA landmarks.

WHY THIS FILE IS SEPARATE AND TINY (numpy + PIL + stdlib only):
Both sides of the benchmark import THIS SAME module so the landmark loading, the
pair enumeration, and above all the rTRE metric are byte-for-byte identical —
`run_ours.py` (main .venv, torch/kornia/SimpleITK) and `run_valis.py` (isolated
~/valis_runtime venv). If the scorer differed between the two, the comparison
would be meaningless. Keep this importable in BOTH envs: no torch, no cv2, no valis.

ANTI-CIRCULARITY CONTRACT (enforced by construction, not by comment):
  1. Expert landmarks are NEVER passed to any registration or optimisation. They
     are loaded here and used ONLY by `rtre()` for scoring. Registration sees images.
  2. Both methods are scored by the SAME `rtre()` on the SAME held-out landmarks.
  3. `initial_rtre` (identity transform) is always reported, so neither method is
     credited for pre-alignment already present in the data.
  4. rTRE is RELATIVE (normalised by the fixed-image diagonal), so it is invariant
     to the working scale (5pc / 10pc / …) — the ANHIR MMrTRE convention.
"""
import os
import re
import csv
import glob
import itertools
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # CIMA jpgs are small; disable the decompression-bomb guard


# ──────────────────────────────────────────────────────────────────────────────
def _scale_pc(path_component: str) -> float:
    """Parse 'scale-5pc' / 'user-PS_scale-50pc' → 5.0 / 50.0."""
    m = re.search(r"scale-(\d+(?:\.\d+)?)pc", path_component)
    if not m:
        raise ValueError(f"no scale-Npc token in {path_component!r}")
    return float(m.group(1))


def load_landmarks(csv_path: str, to_scale_factor: float) -> np.ndarray:
    """Load an ANHIR landmark CSV ( ,X,Y ) and rescale into the working-image space.

    to_scale_factor multiplies the stored coordinates (which live in the annotation
    scale, e.g. 50pc) into the image scale actually on disk (e.g. 5pc ⇒ factor 0.1).
    """
    pts = []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r, None)  # header ' ,X,Y'
        for row in r:
            if len(row) >= 3 and row[1] != "" and row[2] != "":
                pts.append((float(row[1]), float(row[2])))
    return np.asarray(pts, float) * float(to_scale_factor)


def load_image(path: str) -> np.ndarray:
    """RGB uint8 array."""
    return np.asarray(Image.open(path).convert("RGB"))


def image_diag(wh) -> float:
    w, h = wh
    return float(np.hypot(w, h))


def apply_affine_2x3(pts: np.ndarray, M) -> np.ndarray:
    """Apply a 2x3 or 3x3 affine (moving→fixed) to Nx2 points."""
    M = np.asarray(M, float)
    A = M[:2, :2]
    t = M[:2, 2]
    return pts @ A.T + t


# ── real ANHIR pixel sizes (µm/px at FULL resolution, from the challenge tissue table) ──
ANHIR_PX_UM_FULL = {
    "lung-lesion": 0.174, "lung-lobes": 1.274, "mammary-gland": 2.294,
    "mice-kidney": 0.227, "COAD": 0.468, "gastric": 0.2528, "breast": 0.2528,
    "kidney": 0.2528,
}


def px_um_for(set_name: str, img_scale_pc: float):
    """Real µm/px for a set at a given scale (e.g. lung-lesion @25pc → 0.174/0.25 = 0.696).
    Returns None for an unknown tissue prefix (caller falls back to a nominal)."""
    prefix = set_name.rsplit("_", 1)[0]
    base = ANHIR_PX_UM_FULL.get(prefix)
    if base is None:
        return None
    return base / (float(img_scale_pc) / 100.0)


def _predict_local_affine(query_pts, lm_ref, lm_mov, k=6, max_near_px=np.inf):
    """Ground-truth displacement field from sparse expert landmarks: for each query ref
    point, fit a LOCAL affine to its k nearest landmark correspondences and predict where it
    maps in the moving image. Local (not global) so it tracks real serial-section deformation
    — the point is to isolate LoFTR error from genuine tissue warp. Returns predicted moving
    points (NaN where the nearest landmark is farther than max_near_px)."""
    query_pts = np.asarray(query_pts, float)
    lm_ref = np.asarray(lm_ref, float); lm_mov = np.asarray(lm_mov, float)
    out = np.full_like(query_pts, np.nan)
    kk = min(k, len(lm_ref))
    if kk < 3:
        return out
    for i, q in enumerate(query_pts):
        d = np.linalg.norm(lm_ref - q, axis=1)
        idx = np.argsort(d)[:kk]
        if d[idx[0]] > max_near_px:
            continue
        src = np.hstack([lm_ref[idx], np.ones((kk, 1))])   # kk x 3
        A, *_ = np.linalg.lstsq(src, lm_mov[idx], rcond=None)  # 3 x 2
        out[i] = np.array([q[0], q[1], 1.0]) @ A
    return out


def correspondence_quality(loftr_ref, loftr_mov, lm_ref, lm_mov, pixel_size_um,
                           tol_um=10.0, max_near_px=None):
    """DIRECT validation of LoFTR correspondences against expert landmarks (non-circular).

    For each LoFTR match (r→m), predict the ground-truth moving location for r from the LOCAL
    expert-landmark affine, then measure ‖m − predicted‖. This is LoFTR's own error at that
    point, with real tissue deformation already removed by the local GT. The landmarks are the
    independent truth (never used to make the matches).

    Returns n_eval, median/p90 error (µm), inlier_rate (err ≤ tol_um), and the match count.
    """
    loftr_ref = np.asarray(loftr_ref, float); loftr_mov = np.asarray(loftr_mov, float)
    if len(loftr_ref) == 0 or len(lm_ref) < 3:
        return {"n_matches": int(len(loftr_ref)), "n_eval": 0, "median_um": None,
                "p90_um": None, "inlier_rate": None, "tol_um": tol_um}
    if max_near_px is None:
        # default catch radius: ~1.5× median nearest-neighbour landmark spacing
        nn = []
        for i in range(len(lm_ref)):
            dd = np.linalg.norm(lm_ref - lm_ref[i], axis=1); dd[i] = np.inf
            nn.append(dd.min())
        max_near_px = 1.5 * float(np.median(nn))
    pred = _predict_local_affine(loftr_ref, lm_ref, lm_mov, max_near_px=max_near_px)
    ok = ~np.isnan(pred[:, 0])
    if ok.sum() == 0:
        return {"n_matches": int(len(loftr_ref)), "n_eval": 0, "median_um": None,
                "p90_um": None, "inlier_rate": None, "tol_um": tol_um,
                "max_near_px": float(max_near_px)}
    err_um = np.linalg.norm(loftr_mov[ok] - pred[ok], axis=1) * float(pixel_size_um)
    return {"n_matches": int(len(loftr_ref)), "n_eval": int(ok.sum()),
            "median_um": float(np.median(err_um)), "p90_um": float(np.percentile(err_um, 90)),
            "mean_um": float(np.mean(err_um)),
            "inlier_rate": float(np.mean(err_um <= tol_um)), "tol_um": tol_um,
            "max_near_px": float(max_near_px)}


def rtre(warped_mov_lm: np.ndarray, fixed_lm: np.ndarray, fixed_wh) -> dict:
    """Relative target registration error (ANHIR convention).

    warped_mov_lm : moving landmarks after the method's transform, in FIXED-image px.
    fixed_lm      : the corresponding fixed-image landmarks, in FIXED-image px.
    Returns median / mean / max / p90 rTRE (fraction of the fixed-image diagonal).
    """
    warped_mov_lm = np.asarray(warped_mov_lm, float)
    fixed_lm = np.asarray(fixed_lm, float)
    n = min(len(warped_mov_lm), len(fixed_lm))
    d = np.linalg.norm(warped_mov_lm[:n] - fixed_lm[:n], axis=1)
    diag = image_diag(fixed_wh)
    r = d / diag
    return {"median": float(np.median(r)), "mean": float(np.mean(r)),
            "p90": float(np.percentile(r, 90)), "max": float(np.max(r)),
            "n": int(n), "diag_px": diag,
            "median_px": float(np.median(d)), "max_px": float(np.max(d))}


def initial_rtre(mov_lm, fixed_lm, fixed_wh) -> dict:
    """rTRE with NO registration (identity) — the honest baseline."""
    return rtre(mov_lm, fixed_lm, fixed_wh)


def _emit_pairs(tset, img_pc, lm_pc, stem_to_img, lm, annotator):
    """Form every usable directed pair from a set's matched images+landmarks."""
    out = []
    usable = [s for s in lm if s in stem_to_img]
    for a, b in itertools.combinations(sorted(usable), 2):
        if len(lm[a]) != len(lm[b]) or len(lm[a]) < 4:
            continue  # non-corresponding or too few points
        for fixed, moving in ((a, b), (b, a)):
            out.append({
                "set": tset, "annotator": annotator,
                "img_scale_pc": img_pc, "lm_scale_pc": lm_pc,
                "fixed_stain": fixed, "moving_stain": moving,
                "fixed_img": stem_to_img[fixed], "moving_img": stem_to_img[moving],
                "fixed_lm": lm[fixed].tolist(), "moving_lm": lm[moving].tolist(),
                "pair_id": f"{tset}|{annotator}|{moving}->{fixed}",
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
def enumerate_pairs(dataset_root: str, annotations_root: str = None, target_scale_pc=None):
    """Yield every usable directed registration pair (images + matched landmarks).

    Layout-agnostic, handles BOTH:
      • real ANHIR grand-challenge: dataset/<set>/scale-Npc/{stem.jpg, stem.csv}
        (landmarks co-located with images, same scale — factor 1.0);
      • split CIMA export: dataset/<set>/scale-Npc/*.jpg  +
        annotations/<set>/user-XX_scale-Mpc/*.csv (landmarks rescaled M→N).

    target_scale_pc : prefer that scale dir (e.g. 25) when a set has several; else the first.
    A pair is usable iff both stains share a landmark CSV with the SAME point count AND both
    have an image. Directed both ways so each method is tested symmetrically.
    """
    pairs = []
    if not os.path.isdir(dataset_root):
        return pairs
    sets = sorted(d for d in os.listdir(dataset_root)
                  if os.path.isdir(os.path.join(dataset_root, d)))
    for tset in sets:
        scale_dirs = sorted(glob.glob(os.path.join(dataset_root, tset, "scale-*pc")))
        if not scale_dirs:
            continue
        # choose the requested scale if present, else the first available
        img_dir = scale_dirs[0]
        if target_scale_pc is not None:
            match = [d for d in scale_dirs if _scale_pc(os.path.basename(d)) == float(target_scale_pc)]
            if match:
                img_dir = match[0]
        img_pc = _scale_pc(os.path.basename(img_dir))
        stem_to_img = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(img_dir, "*"))
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))}

        # (1) real ANHIR: CSVs co-located with the images
        colocated = {os.path.splitext(os.path.basename(c))[0]: c
                     for c in glob.glob(os.path.join(img_dir, "*.csv"))}
        if colocated:
            lm = {s: load_landmarks(colocated[s], 1.0) for s in colocated if s in stem_to_img}
            pairs += _emit_pairs(tset, img_pc, img_pc, stem_to_img, lm, "colocated")
            continue

        # (2) split CIMA: separate annotations root, possibly multiple annotators/scales
        if annotations_root and os.path.isdir(os.path.join(annotations_root, tset)):
            for ann_dir in sorted(glob.glob(os.path.join(annotations_root, tset, "*scale-*pc"))):
                lm_pc = _scale_pc(os.path.basename(ann_dir))
                factor = img_pc / lm_pc
                csvs = {os.path.splitext(os.path.basename(c))[0]: c
                        for c in glob.glob(os.path.join(ann_dir, "*.csv"))}
                lm = {s: load_landmarks(csvs[s], factor) for s in csvs if s in stem_to_img}
                pairs += _emit_pairs(tset, img_pc, lm_pc, stem_to_img, lm,
                                     os.path.basename(ann_dir))
    return pairs


def enumerate_pairs_anhir(csv_path: str, root: str, status_filter=None):
    """Enumerate ANHIR pairs from the official dataset_medium.csv (the canonical spec).

    Columns: Source image, Source landmarks, Target image, Target landmarks, status,
    Image diagonal [pixels]. ANHIR convention: register SOURCE→TARGET, warp source
    landmarks into target space, normalise by the TARGET image diagonal.

    Keeps only pairs where all four files exist on disk (post-challenge, most landmarks
    are released; evaluation-set landmarks may or may not be present). status_filter e.g.
    'training' restricts to the always-public training landmarks.
    """
    pairs = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            status = (row.get("status") or "").strip()
            if status_filter and status != status_filter:
                continue
            si = row["Source image"].strip(); sl = row["Source landmarks"].strip()
            ti = row["Target image"].strip(); tl = row["Target landmarks"].strip()
            paths = {k: os.path.join(root, v) for k, v in
                     (("si", si), ("sl", sl), ("ti", ti), ("tl", tl))}
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            try:
                mov_lm = load_landmarks(paths["sl"], 1.0)
                fix_lm = load_landmarks(paths["tl"], 1.0)
            except Exception:
                continue
            if len(mov_lm) != len(fix_lm) or len(mov_lm) < 4:
                continue
            tset = si.split("/")[0]                       # e.g. COAD_01
            m = re.search(r"scale-(\d+)pc", si)
            img_pc = float(m.group(1)) if m else 25.0
            diag = None
            try:
                diag = float(row.get("Image diagonal [pixels]") or 0) or None
            except Exception:
                pass
            pairs.append({
                "set": tset, "annotator": "anhir", "status": status,
                "img_scale_pc": img_pc, "lm_scale_pc": img_pc,
                "moving_stain": os.path.splitext(os.path.basename(si))[0],
                "fixed_stain": os.path.splitext(os.path.basename(ti))[0],
                "moving_img": paths["si"], "fixed_img": paths["ti"],
                "moving_lm": mov_lm.tolist(), "fixed_lm": fix_lm.tolist(),
                "diag_px_official": diag,
                "pair_id": f"{tset}|{status}|{os.path.splitext(os.path.basename(si))[0]}->"
                           f"{os.path.splitext(os.path.basename(ti))[0]}",
            })
    return pairs


# default roots
CIMA_ROOT = os.path.expanduser("~/oasis_validation_datasets/CIMA_ANHIR/inputs")
ANHIR_ROOT = os.path.expanduser("~/oasis_validation_datasets/ANHIR_medium/images")
ANHIR_CSV = os.path.expanduser("~/oasis_validation_datasets/ANHIR_medium/images/dataset_medium.csv")
DATASET_ROOT = os.path.join(CIMA_ROOT, "dataset")
ANN_ROOT = os.path.join(CIMA_ROOT, "annotations")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def tissue_of(set_name: str) -> str:
    """'lung-lesion_2' → 'lung-lesion'; 'COAD_01' → 'COAD'."""
    return re.sub(r"_\d+$", "", set_name)


def stratified_pairs(pairs, per_tissue=7, seed=0):
    """Representative subset: up to `per_tissue` pairs per tissue type, spread evenly across
    the sorted pair list so a mix of sets and easy/hard (similar-stain vs H&E↔IHC) is kept.
    Deterministic."""
    from collections import defaultdict
    byt = defaultdict(list)
    for p in pairs:
        byt[tissue_of(p["set"])].append(p)
    out = []
    for t, ps in sorted(byt.items()):
        ps = sorted(ps, key=lambda x: x["pair_id"])
        if len(ps) <= per_tissue:
            out += ps
        else:
            idx = np.linspace(0, len(ps) - 1, per_tissue).round().astype(int)
            out += [ps[i] for i in sorted(set(idx.tolist()))]
    return out


def get_pairs(status_filter=None):
    """Auto-select the dataset: full ANHIR (dataset_medium.csv) if present, else CIMA.
    All runners call this so pointing at the real ANHIR data needs no code change."""
    csv_env = os.environ.get("ANHIR_CSV", ANHIR_CSV)
    root_env = os.environ.get("ANHIR_ROOT", ANHIR_ROOT)
    sf = os.environ.get("ANHIR_STATUS", status_filter) or status_filter
    if os.path.exists(csv_env) and os.path.isdir(root_env):
        return enumerate_pairs_anhir(csv_env, root_env, status_filter=sf)
    return enumerate_pairs(DATASET_ROOT, ANN_ROOT)


if __name__ == "__main__":
    ps = get_pairs()
    print(f"usable pairs: {len(ps)}")
    from collections import Counter
    byset = Counter(p["set"] for p in ps)
    for s, n in sorted(byset.items()):
        print(f"  {s}: {n} pairs")
    if ps:
        p = ps[0]
        print("example:", p["pair_id"], "N_lm=", len(p["fixed_lm"]),
              "img_pc=", p["img_scale_pc"], "status=", p.get("status"))
