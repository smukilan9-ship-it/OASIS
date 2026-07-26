"""
classifier.py — a positivity classifier fitted to ONE cohort's own labelled cells.

Why per-cohort and not a shipped model: segmentation generalises because a nucleus looks
like a nucleus, but positivity does not, because positivity is a property of the assay
rather than of the morphology. DAB is not quantitative and cutoffs do not transfer across
antibody or scanner (ihc.md § 3.3). The best published universal IHC model reaches κ 0.578
on unseen stains; per-cohort fitting on this project's own data reaches held-out AUC 0.90.
So the model is fitted where the variation actually lives — inside one cohort.

What this adds over the fixed cutoff is not "it is trained". It is features a single global
optical density cannot express: local background correction (a cell judged against its
neighbours, which survives a staining gradient across a section), the DAB-over-haematoxylin
gate, and for membranous markers the *contiguity* of the stained ring rather than merely
how much of it is stained.

Three design decisions worth stating plainly.

**Leave-one-IMAGE-out, never leave-one-cell-out.** Cells within a slide share a staining
run, illumination, section thickness and operator. Holding out one cell scores a model
whose own slide is still in the training set, which flatters it. The cost is measurable
here: pooled leave-one-image-out gives F1 0.83 while the faint slide alone gives 0.30. The
per-fold spread is therefore reported alongside the mean, because an average of 0.83 over
folds of {0.91, 0.88, 0.30} is a different object from a tight 0.83.

**Logistic regression, in numpy.** Roughly a dozen engineered features and a few hundred
labelled cells. It will not overfit at that n, its coefficients are readable — so the tab
can say "this is mostly ring contiguity and local contrast" to a pathologist who has to
approve it — and it yields calibrated probabilities, which the abstain band needs. A forest
would add a dependency to a bundle already near its size guard, overfit at this n, and buy
nothing interpretable.

**Two gates, because a trained rule fails open.** On faint tissue a fixed cutoff is *safer,
not more accurate*: it under-calls. An adaptive or trained rule finds the best split of
noise and manufactures positives — measured, not asserted: the retired GMM called 28.8 % of
one TIM-3 slide positive against 0.1 % at the fixed cut. So cells near the decision boundary
abstain, and an image whose features fall outside the training range is refused outright and
handed back to the fixed cutoff.
"""
import hashlib
import json
import time

import numpy as np

FEATURE_SET_VERSION = 2

# Feature contracts. Order is part of the contract — persisted models store the names and
# refuse to score a cell vector built from a different list.
FEATURES_NUCLEAR = (
    "dab_mean",             # the signal itself
    "dab_p90",              # brightest part of the nucleus; survives partial staining
    "dab_minus_local_bg",   # vs the median of nearby cells — cancels staining gradients
    "dab_over_h",           # DAB minus haematoxylin; counterstain cross-talk guard
    "hema_mean",            # catches over-stained nuclei masquerading as DAB-positive
    "area_px",              # debris and merged nuclei sit at the extremes
)

FEATURES_MEMBRANE = (
    "ring_mean",
    "ring_p90",             # the brightest arc, undiluted by the empty part of the ring
    "ring_minus_local_bg",
    "ring_minus_nucleus",   # a real membrane is brighter than the cell's own nucleus
    "membrane_pos_frac",    # completeness: how much of the ring is stained
    "membrane_connectivity",  # contiguity: how much of it is stained *continuously*
    "membrane_arc_count",   # one or two arcs is a membrane; many is speckle
    "dab_over_h",
    "area_px",
)


def feature_names(kind):
    if kind == "membrane":
        return list(FEATURES_MEMBRANE)
    if kind == "nuclear":
        return list(FEATURES_NUCLEAR)
    raise ValueError(f"unknown classifier kind {kind!r} (use 'nuclear' or 'membrane')")


# ── Feature extraction ────────────────────────────────────────────────────────

def local_background(centroids, values, k=12):
    """Median value of each cell's k nearest neighbours, itself excluded.

    This is what lets a classifier survive a staining gradient across one section: a cell
    is judged against its own neighbourhood rather than against a single number chosen for
    the whole slide. A global cutoff cannot express this at all.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.zeros(0)
    if n <= 2:
        return np.full(n, float(np.nanmedian(values)) if n else 0.0)
    from scipy.spatial import cKDTree
    kk = int(min(k + 1, n))                      # +1: the query point matches itself
    _, idx = cKDTree(centroids).query(centroids, k=kk)
    idx = np.atleast_2d(idx)
    out = np.empty(n)
    for i in range(n):
        nb = idx[i][idx[i] != i][: kk - 1]
        out[i] = np.nanmedian(values[nb]) if len(nb) else values[i]
    return out


def _get(cell, *names, default=np.nan):
    for nm in names:
        v = cell.get(nm)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    return default


def extract_features(cells, kind):
    """Feature matrix for one image's cells.

    `cells` is a list of per-cell dicts as produced by the measurement path — nuclear runs
    supply `dab_mean`/`hema_mean`/`area_px`/`centroid`, membrane runs additionally supply
    the ring measurements from `cell_expansion`. Missing values become NaN here and are
    imputed at fit/score time against the training medians, so one absent measurement
    degrades a cell rather than dropping it.
    """
    names = feature_names(kind)
    n = len(cells)
    if n == 0:
        return np.zeros((0, len(names))), names

    centroids = np.array([c.get("centroid") or (0.0, 0.0) for c in cells], dtype=np.float64)
    dab = np.array([_get(c, "dab_mean", "DAB: Mean") for c in cells])
    hem = np.array([_get(c, "hema_mean", "Hematoxylin: Mean") for c in cells])
    ring = np.array([_get(c, "cytoplasm_dab_mean", "ring_mean") for c in cells])

    cols = {
        "dab_mean": dab,
        "dab_p90": np.array([_get(c, "dab_p90", "DAB: Max") for c in cells]),
        "dab_over_h": dab - hem,
        "hema_mean": hem,
        "area_px": np.array([_get(c, "area_px", "Nucleus: Area") for c in cells]),
        "ring_mean": ring,
        "ring_p90": np.array([_get(c, "cytoplasm_dab_p90") for c in cells]),
        "ring_minus_nucleus": ring - dab,
        "membrane_pos_frac": np.array([_get(c, "membrane_pos_frac") for c in cells]),
        "membrane_connectivity": np.array([_get(c, "membrane_connectivity") for c in cells]),
        "membrane_arc_count": np.array([_get(c, "membrane_arc_count") for c in cells]),
    }
    # Local background is computed on whichever channel the marker is actually read in.
    base = ring if kind == "membrane" else dab
    finite = np.where(np.isfinite(base), base, np.nanmedian(base[np.isfinite(base)])
                      if np.isfinite(base).any() else 0.0)
    cols["dab_minus_local_bg"] = finite - local_background(centroids, finite)
    cols["ring_minus_local_bg"] = cols["dab_minus_local_bg"]

    X = np.column_stack([cols[nm] for nm in names])
    return X, names


# ── Logistic regression (numpy) ───────────────────────────────────────────────

def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic(X, y, l2=1.0, max_iter=100, tol=1e-8):
    """Ridge-penalised logistic regression by IRLS. Returns (weights, intercept).

    The L2 term is not optional decoration: with a feature like `membrane_connectivity`
    that can separate the classes almost perfectly on a small labelled set, unpenalised
    IRLS diverges as the weights run to infinity. The penalty keeps the fit finite and the
    probabilities calibrated enough to abstain on.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n)])
    w = np.zeros(d + 1)
    # Do not penalise the intercept — it only sets the base rate, and shrinking it would
    # bias a rare-positive marker toward calling nothing.
    pen = np.eye(d + 1) * float(l2)
    pen[-1, -1] = 0.0
    for _ in range(int(max_iter)):
        p = _sigmoid(Xb @ w)
        W = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xb.T @ (p - y) + pen @ w
        H = (Xb.T * W) @ Xb + pen
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w_new = w - step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w[:-1], float(w[-1])


# ── Metrics ───────────────────────────────────────────────────────────────────

def roc_auc(scores, labels):
    """AUC by rank, ties averaged. None when one class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within tie groups, else identical scores bias the statistic.
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[np.isin(s, su[i])] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def prf1(pred, y):
    pred = np.asarray(pred, dtype=int)
    y = np.asarray(y, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ── The model ─────────────────────────────────────────────────────────────────

class CellClassifier:
    """A fitted per-cohort classifier, with everything needed to reproduce its verdict."""

    def __init__(self, kind, names, weights, intercept, mean, scale, medians,
                 metrics=None, meta=None, train_ranges=None):
        self.kind = kind
        self.names = list(names)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.medians = np.asarray(medians, dtype=np.float64)
        self.metrics = metrics or {}
        self.meta = meta or {}
        self.train_ranges = train_ranges or {}

    def _prep(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        if X.shape[1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} features, got {X.shape[1]}")
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.medians, np.where(bad)[1])
        return (X - self.mean) / self.scale

    def predict_proba(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return _sigmoid(self._prep(X) @ self.weights + self.intercept)

    def predict(self, X, abstain_band=0.0):
        """Labels, and an abstain mask when `abstain_band` > 0.

        Returns (labels, abstained). A cell inside the band around 0.5 is not forced to a
        call — it is counted and reported, because a confident wrong call on an ambiguous
        cell is worse than an admitted one.
        """
        p = self.predict_proba(X)
        labels = (p > 0.5).astype(int)
        lo, hi = 0.5 - abstain_band / 2.0, 0.5 + abstain_band / 2.0
        return labels, (p >= lo) & (p <= hi) if abstain_band > 0 else np.zeros(len(p), bool)

    def applicable(self, X, tol=0.25):
        """Is this image close enough to the training data to be scored at all?

        The gate that matters. A model fitted on well-stained slides will render a
        confident verdict on a faint one unless something stops it. Compares this image's
        median feature values against the training range, widened by `tol` of that range;
        an image outside it is refused and handed back to the fixed cutoff.
        """
        if len(X) == 0:
            return False, "no cells to score"
        X = np.array(X, dtype=np.float64, copy=True)
        offenders = []
        for j, nm in enumerate(self.names):
            rng = self.train_ranges.get(nm)
            col = X[:, j]
            col = col[np.isfinite(col)]
            if not rng or col.size == 0:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            pad = max((hi - lo) * float(tol), 1e-9)
            med = float(np.median(col))
            if med < lo - pad or med > hi + pad:
                offenders.append(f"{nm}={med:.4g} outside [{lo:.4g}, {hi:.4g}]")
        if offenders:
            return False, "; ".join(offenders[:3])
        return True, "within training range"

    # ── persistence ──
    def to_dict(self):
        return {"schema": "oasis.cell_classifier/1", "kind": self.kind,
                "feature_set_version": FEATURE_SET_VERSION, "features": self.names,
                "weights": self.weights.tolist(), "intercept": self.intercept,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "medians": self.medians.tolist(), "train_ranges": self.train_ranges,
                "metrics": self.metrics, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        if d.get("feature_set_version") != FEATURE_SET_VERSION:
            raise ValueError(
                f"classifier was fitted against feature set v{d.get('feature_set_version')}, "
                f"this build measures v{FEATURE_SET_VERSION} — refit it")
        return cls(d["kind"], d["features"], d["weights"], d["intercept"], d["mean"],
                   d["scale"], d["medians"], d.get("metrics"), d.get("meta"),
                   d.get("train_ranges"))

    def fingerprint(self):
        """Stable hash of the decision function — the provenance a result cites."""
        payload = json.dumps({k: self.to_dict()[k] for k in
                              ("kind", "features", "weights", "intercept", "mean", "scale")},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def coefficient_report(self):
        """Which features the model actually leans on, largest standardised weight first.

        Features are standardised, so the magnitudes are directly comparable. This is what
        makes the model answerable to a pathologist rather than merely accurate.
        """
        order = np.argsort(-np.abs(self.weights))
        return [{"feature": self.names[i], "weight": round(float(self.weights[i]), 4)}
                for i in order]


def _standardise(X):
    """Column means/scales and medians, computed on training data only."""
    med = np.array([np.nanmedian(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
                    for c in X.T])
    Xf = np.where(np.isfinite(X), X, med)
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0)
    scale[scale < 1e-9] = 1.0          # a constant feature contributes nothing, not inf
    return mean, scale, med, Xf


def fit(X, y, kind, names, l2=1.0, meta=None):
    """Fit on everything given. Use `leave_one_image_out` for the honest estimate."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    mean, scale, med, Xf = _standardise(X)
    w, b = fit_logistic((Xf - mean) / scale, y, l2=l2)
    ranges = {nm: [float(np.nanmin(Xf[:, j])), float(np.nanmax(Xf[:, j]))]
              for j, nm in enumerate(names)}
    return CellClassifier(kind, names, w, b, mean, scale, med,
                          meta={**(meta or {}), "n_cells": int(len(y)),
                                "n_positive": int(y.sum()),
                                "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                            time.gmtime())},
                          train_ranges=ranges)


def leave_one_image_out(X, y, image_ids, kind, names, l2=1.0):
    """Hold out a whole IMAGE at a time — the only honest estimate for this problem.

    Cells inside one slide are not independent, so a cell-wise fold scores a model whose
    own slide is still in training. Returns the pooled held-out metrics **and** the
    per-fold breakdown, because the spread is the finding: a mean of 0.83 over folds of
    {0.91, 0.88, 0.30} says something a single number hides.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    image_ids = np.asarray(image_ids)
    uniq = list(dict.fromkeys(image_ids.tolist()))
    if len(uniq) < 3:
        return {"ok": False, "reason": f"leave-one-image-out needs >= 3 images, got {len(uniq)}"}

    held = np.full(len(y), np.nan)
    folds = []
    for img in uniq:
        te = image_ids == img
        tr = ~te
        y_tr = y[tr]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            folds.append({"image": str(img), "n": int(te.sum()), "skipped": "one class in training"})
            continue
        mean, scale, med, Xf = _standardise(X[tr])
        w, b = fit_logistic((Xf - mean) / scale, y_tr, l2=l2)
        Xte = np.where(np.isfinite(X[te]), X[te], med)
        p = _sigmoid(((Xte - mean) / scale) @ w + b)
        held[te] = p
        f = prf1((p > 0.5).astype(int), y[te])
        folds.append({"image": str(img), "n": int(te.sum()),
                      "n_positive": int(y[te].sum()),
                      "auc": (round(a, 4) if (a := roc_auc(p, y[te])) is not None else None),
                      **f})

    scored = np.isfinite(held)
    if not scored.any():
        return {"ok": False, "reason": "every fold was skipped"}
    pooled = prf1((held[scored] > 0.5).astype(int), y[scored])
    auc = roc_auc(held[scored], y[scored])
    f1s = [f["f1"] for f in folds if "f1" in f]
    return {"ok": True, "n_images": len(uniq), "n_cells": int(scored.sum()),
            "pooled": {**pooled, "auc": round(auc, 4) if auc is not None else None},
            "fold_f1_min": round(min(f1s), 4) if f1s else None,
            "fold_f1_max": round(max(f1s), 4) if f1s else None,
            "fold_f1_mean": round(float(np.mean(f1s)), 4) if f1s else None,
            "fold_f1_std": round(float(np.std(f1s)), 4) if f1s else None,
            "folds": folds}
