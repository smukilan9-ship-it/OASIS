"""
Tier 1 — per-cohort cell classifier (quant.classifier). Synthetic cells; no images.

The load-bearing test here is `test_leave_one_cell_out_is_optimistic...`: it demonstrates
on constructed data why the honesty metric had to move from cells to images.
"""
import numpy as np
import pytest

from oasis.quant import classifier as C


def _cohort(n_images=6, n_per=200, pos_rate=0.2, image_shift=0.0, seed=0, sep=0.30):
    """A cohort where positives separate on feature 0, plus a per-image staining offset."""
    rng = np.random.default_rng(seed)
    X, y, ids = [], [], []
    for img in range(n_images):
        shift = rng.normal(0.0, image_shift)
        yy = (rng.random(n_per) < pos_rate).astype(int)
        x0 = np.where(yy == 1, rng.normal(0.05 + sep, 0.08, n_per),
                      rng.normal(0.05, 0.04, n_per)) + shift
        rest = [rng.normal(0, 1, n_per) for _ in range(len(C.FEATURES_NUCLEAR) - 1)]
        X.append(np.column_stack([x0] + rest))
        y.append(yy)
        ids += [f"img{img}"] * n_per
    return np.vstack(X), np.concatenate(y), ids


# ── the reason leave-one-image-out replaced leave-one-cell-out ────────────────

def _loo_cell_f1(X, y, names, folds=60, seed=0):
    """Leave-one-cell-out, subsampled — the metric calibration.py used."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(folds, len(y)), replace=False)
    preds, truth = [], []
    for i in idx:
        tr = np.ones(len(y), bool); tr[i] = False
        mean, scale, med, Xf = C._standardise(X[tr])
        w, b = C.fit_logistic((Xf - mean) / scale, y[tr])
        xi = np.where(np.isfinite(X[i]), X[i], med)
        p = C._sigmoid(((xi - mean) / scale) @ w + b)
        preds.append(int(p > 0.5)); truth.append(int(y[i]))
    return C.prf1(preds, truth)["f1"]


def test_leave_one_cell_out_is_optimistic_when_images_differ():
    """The finding that motivated the change.

    With a large per-image staining offset, holding out one CELL leaves its own slide in
    the training set, so the model has already seen that slide's offset. Holding out the
    whole IMAGE does not. The cell-wise number is therefore the more flattering one, and
    it is flattering about exactly the thing that decides whether a cohort-wide rule holds.
    """
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(image_shift=0.25, sep=0.18, seed=3)

    f1_cell = _loo_cell_f1(X, y, names)
    f1_image = C.leave_one_image_out(X, y, ids, "nuclear", names)["pooled"]["f1"]

    assert f1_cell > f1_image, (
        f"expected leave-one-cell-out ({f1_cell}) to flatter the model relative to "
        f"leave-one-image-out ({f1_image})")


def test_leave_one_image_out_is_not_more_optimistic_than_in_sample():
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(image_shift=0.15, seed=1)
    model = C.fit(X, y, "nuclear", names)
    in_sample = C.prf1(model.predict(X)[0], y)["f1"]
    held = C.leave_one_image_out(X, y, ids, "nuclear", names)["pooled"]["f1"]
    assert held <= in_sample + 1e-9


def test_per_fold_spread_is_reported():
    """An average of 0.83 over folds of {0.91, 0.88, 0.30} is a different object from a
    tight 0.83. The spread has to survive into the report."""
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(seed=2)
    r = C.leave_one_image_out(X, y, ids, "nuclear", names)
    for key in ("fold_f1_min", "fold_f1_max", "fold_f1_mean", "fold_f1_std"):
        assert r[key] is not None
    assert len(r["folds"]) == r["n_images"]
    assert r["fold_f1_min"] <= r["fold_f1_mean"] <= r["fold_f1_max"]


def test_one_bad_slide_shows_up_as_spread_not_just_a_lower_mean():
    """The faint-slide case: one image far from the rest must be visible in the minimum
    fold, not averaged away."""
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(n_images=5, seed=4)
    ids = list(ids)
    bad = np.array(ids) == "img4"
    X[bad, 0] = np.random.default_rng(0).normal(0.05, 0.04, int(bad.sum()))  # signal erased
    r = C.leave_one_image_out(X, y, ids, "nuclear", names)
    assert r["fold_f1_min"] < 0.5
    assert r["fold_f1_max"] > 0.8
    assert r["fold_f1_std"] > 0.15


def test_loio_refuses_fewer_than_three_images():
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(n_images=2, seed=5)
    r = C.leave_one_image_out(X, y, ids, "nuclear", names)
    assert r["ok"] is False and "3 images" in r["reason"]


def test_a_fold_with_one_class_in_training_is_skipped_not_silently_scored():
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(n_images=3, n_per=40, seed=6)
    y[:] = 0
    y[np.array(ids) == "img0"] = 1          # all positives live in one image
    r = C.leave_one_image_out(X, y, ids, "nuclear", names)
    skipped = [f for f in r.get("folds", []) if "skipped" in f]
    assert r["ok"] is False or skipped


# ── the applicability gate ────────────────────────────────────────────────────

def test_faint_image_is_refused_rather_than_confidently_scored():
    """A model fitted on well-stained slides must not render a verdict on a faint one.
    This is the gate that keeps a trained rule from failing open the way the retired GMM
    did (28.8 % positive against 0.1 % at the fixed cut)."""
    names = C.feature_names("nuclear")
    X, y, ids = _cohort(seed=7)
    model = C.fit(X, y, "nuclear", names)
    ok, _ = model.applicable(X[:100])
    assert ok is True
    faint = X[:100].copy(); faint[:, 0] -= 1.0
    ok, reason = model.applicable(faint)
    assert ok is False and "dab_mean" in reason


def test_applicable_reports_no_cells():
    names = C.feature_names("nuclear")
    X, y, _ = _cohort(n_images=3, seed=8)
    model = C.fit(X, y, "nuclear", names)
    ok, reason = model.applicable(np.zeros((0, len(names))))
    assert ok is False and "no cells" in reason


def test_abstain_band_flags_boundary_cells_without_forcing_a_call():
    names = C.feature_names("nuclear")
    X, y, _ = _cohort(seed=9)
    model = C.fit(X, y, "nuclear", names)
    _, none_abstained = model.predict(X, abstain_band=0.0)
    _, some_abstained = model.predict(X, abstain_band=0.4)
    assert none_abstained.sum() == 0
    assert some_abstained.sum() > 0
    p = model.predict_proba(X)
    assert np.all(np.abs(p[some_abstained] - 0.5) <= 0.2 + 1e-9)


# ── numerics and plumbing ─────────────────────────────────────────────────────

def test_ridge_keeps_a_perfectly_separable_fit_finite():
    """Complete separation sends unpenalised logistic weights to infinity. Membrane
    connectivity can separate cleanly on a small labelled set, so this is a real case."""
    X = np.column_stack([np.r_[np.zeros(50), np.ones(50)], np.random.default_rng(0).normal(0, 1, 100)])
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    w, b = C.fit_logistic(X, y, l2=1.0)
    assert np.all(np.isfinite(w)) and np.isfinite(b)
    assert np.abs(w).max() < 1e3


def test_local_background_cancels_a_staining_gradient():
    """The feature a global cutoff cannot express: identical cells on opposite sides of a
    gradient must score the same once judged against their neighbours."""
    xs = np.linspace(0, 1000, 400)
    centroids = np.column_stack([xs, np.zeros_like(xs)])
    gradient = 0.05 + xs / 1000.0 * 0.20          # background rises across the section
    vals = gradient.copy()
    vals[50] += 0.25                               # one positive cell in the dim region
    vals[350] += 0.25                              # one in the bright region
    corrected = vals - C.local_background(centroids, vals)
    assert corrected[50] == pytest.approx(corrected[350], abs=0.02)
    assert corrected[50] > 0.2
    # untouched neighbours sit near zero despite the gradient spanning 0.05 -> 0.25
    assert abs(corrected[200]) < 0.02


def test_roc_auc_matches_known_values():
    assert C.roc_auc([0, 1, 2, 3], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert C.roc_auc([3, 2, 1, 0], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert C.roc_auc([1, 1, 1, 1], [0, 0, 1, 1]) == pytest.approx(0.5)
    assert C.roc_auc([1, 2], [1, 1]) is None


def test_persistence_round_trip_preserves_the_decision_function():
    names = C.feature_names("membrane")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, len(names)))
    y = (X[:, 0] + rng.normal(0, 0.4, 300) > 0).astype(int)
    m = C.fit(X, y, "membrane", names)
    m2 = C.CellClassifier.from_dict(m.to_dict())
    assert m2.fingerprint() == m.fingerprint()
    assert np.allclose(m2.predict_proba(X), m.predict_proba(X))


def test_a_model_from_a_different_feature_set_is_refused():
    """Silently scoring cells with a stale feature contract would produce plausible,
    meaningless numbers."""
    names = C.feature_names("nuclear")
    X, y, _ = _cohort(n_images=3, seed=10)
    d = C.fit(X, y, "nuclear", names).to_dict()
    d["feature_set_version"] = C.FEATURE_SET_VERSION - 1
    with pytest.raises(ValueError, match="feature set"):
        C.CellClassifier.from_dict(d)


def test_scoring_the_wrong_number_of_features_raises():
    names = C.feature_names("nuclear")
    X, y, _ = _cohort(n_images=3, seed=11)
    m = C.fit(X, y, "nuclear", names)
    with pytest.raises(ValueError, match="expected"):
        m.predict_proba(np.zeros((5, len(names) + 1)))


def test_missing_measurements_degrade_a_cell_rather_than_dropping_it():
    names = C.feature_names("nuclear")
    X, y, _ = _cohort(n_images=3, seed=12)
    m = C.fit(X, y, "nuclear", names)
    Xn = X[:10].copy()
    Xn[0, 2] = np.nan
    p = m.predict_proba(Xn)
    assert len(p) == 10 and np.all(np.isfinite(p))


def test_extract_features_produces_the_declared_contract():
    cells = [{"centroid": (i * 10.0, 0.0), "dab_mean": 0.1 + 0.01 * i, "hema_mean": 0.3,
              "dab_p90": 0.2, "area_px": 120.0,
              "cytoplasm_dab_mean": 0.15, "cytoplasm_dab_p90": 0.3,
              "membrane_pos_frac": 0.2, "membrane_connectivity": 0.4,
              "membrane_arc_count": 1} for i in range(30)]
    for kind in ("nuclear", "membrane"):
        X, names = C.extract_features(cells, kind)
        assert names == C.feature_names(kind)
        assert X.shape == (30, len(names))
        assert np.isfinite(X).all()


def test_extract_features_on_no_cells():
    X, names = C.extract_features([], "nuclear")
    assert X.shape == (0, len(names))


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown classifier kind"):
        C.feature_names("cytoplasmic-ish")
