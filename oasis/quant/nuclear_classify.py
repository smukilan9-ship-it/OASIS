"""
nuclear_classify.py — adaptive, model-based nuclear-DAB positivity.

SINGLE SOURCE OF TRUTH for nuclear-marker positivity, called by BOTH the pipeline
post-step (`_apply_nuclear_reclassification`) and the validation harness
(`validation/validate_nuclear_classifier.py`) so the benchmark measures the exact
code that runs in production.

Adaptive is a NUCLEAR-only tool. A membranous stain has no bimodal nuclear-DAB signal,
so this module is never used for membranous markers (those keep the cytoplasmic-ring
completeness path in cell_expansion.py).

The method, in one paragraph. Fit a two-component Gaussian mixture to THIS image's
per-cell DAB signal (negative mode + positive mode). If the mixture is genuinely
bimodal — 2 components beat 1 by BIC AND the modes are separated (Ashman's D ≥ cut) —
the threshold is the posterior-0.5 crossover between the two modes (a principled valley,
not Otsu's between-class-variance heuristic, which still fires on a single mode). If the
distribution is unimodal or the modes overlap, the marker cannot be adaptively thresholded
on this image: we DO NOT guess. We either fall back to a trusted fixed cutoff (flagged
low-quality) or ABSTAIN and let the caller decide. That abstention is the whole point —
it is what a faint, uncallable slide should trigger instead of a fabricated split.

No sklearn dependency: the 1-D two-component EM is ~40 lines of numpy.
"""
import numpy as np


# ── Otsu (with an outlier cap so one bright artifact can't degrade the cut) ──────
def otsu_threshold(values, nbins: int = 256) -> float:
    """Between-class-variance threshold on a 1-D array. The histogram range is capped
    at the 99.5th percentile (not the raw max) so a single very bright cell cannot
    compress all real signal into the low bins."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(x[0]) if x.size else 0.0
    lo = float(np.min(x))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        hi = float(np.max(x))
        if hi <= lo:
            return lo
    hist, edges = np.histogram(np.clip(x, lo, hi), bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    tot = hist.sum()
    if tot <= 0:
        return float(np.median(x))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w_b = np.cumsum(hist)
    w_f = tot - w_b
    sum_tot = np.cumsum(hist * centers)
    grand = sum_tot[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_b = sum_tot / w_b
        m_f = (grand - sum_tot) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
    between[~np.isfinite(between)] = -1.0
    k = int(np.argmax(between))
    return float(centers[k])


# ── Two-component 1-D Gaussian mixture, EM ──────────────────────────────────────
def _gauss(x, mu, var):
    var = max(float(var), 1e-9)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def gmm_1d_two(values, n_iter: int = 200, tol: float = 1e-7) -> dict:
    """Fit a two-component 1-D GMM by EM (deterministic median-split init).

    Returns a dict:
      means (2,), vars (2,), weights (2,)   — sorted so means[0] < means[1]
      threshold                             — posterior-0.5 crossover in (mu0, mu1)
      ashman_d                              — √2·|μ0−μ1| / √(σ0²+σ1²), bimodality separation
      bic1, bic2                            — BIC of the 1- and 2-component fits (lower = better)
      loglik2                               — 2-component log-likelihood
      ok                                    — fit produced a usable in-between threshold
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    out = {"means": None, "vars": None, "weights": None, "threshold": None,
           "ashman_d": 0.0, "bic1": np.inf, "bic2": np.inf, "loglik2": -np.inf,
           "ok": False, "n": int(n)}
    if n < 10:
        return out

    # 1-component reference (for BIC model selection)
    mu_all = float(x.mean())
    var_all = float(x.var()) + 1e-9
    ll1 = float(np.sum(np.log(np.clip(_gauss(x, mu_all, var_all), 1e-300, None))))
    out["bic1"] = -2.0 * ll1 + 2.0 * np.log(n)          # k = 2 params (mu, var)

    # 2-component EM, median-split init
    med = float(np.median(x))
    lo, hi = x[x <= med], x[x > med]
    if lo.size < 2 or hi.size < 2:
        return out
    mu = np.array([lo.mean(), hi.mean()], float)
    var = np.array([lo.var() + 1e-6, hi.var() + 1e-6], float)
    w = np.array([lo.size, hi.size], float) / n
    ll_prev = -np.inf
    for _ in range(n_iter):
        r0 = w[0] * _gauss(x, mu[0], var[0])
        r1 = w[1] * _gauss(x, mu[1], var[1])
        denom = r0 + r1
        denom[denom <= 0] = 1e-300
        g1 = r1 / denom
        g0 = 1.0 - g1
        Nk = np.array([g0.sum(), g1.sum()])
        Nk[Nk <= 0] = 1e-9
        w = Nk / n
        mu = np.array([(g0 * x).sum(), (g1 * x).sum()]) / Nk
        var = np.array([(g0 * (x - mu[0]) ** 2).sum(),
                        (g1 * (x - mu[1]) ** 2).sum()]) / Nk
        var = np.maximum(var, 1e-6)                     # variance floor
        ll = float(np.sum(np.log(np.clip(denom, 1e-300, None))))
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll

    order = np.argsort(mu)
    mu, var, w = mu[order], var[order], w[order]
    out["means"], out["vars"], out["weights"] = mu, var, w
    out["loglik2"] = ll_prev
    out["bic2"] = -2.0 * ll_prev + 5.0 * np.log(n)      # k = 5 params (2μ, 2σ², 1 weight)
    out["ashman_d"] = float(np.sqrt(2.0) * abs(mu[1] - mu[0]) / np.sqrt(var[0] + var[1]))

    # posterior-0.5 crossover between the two means (quadratic in x)
    thr = _crossover(mu, var, w)
    if thr is not None and mu[0] < thr < mu[1]:
        out["threshold"] = float(thr)
        out["ok"] = True
    else:
        out["threshold"] = float(0.5 * (mu[0] + mu[1]))  # geometric midpoint fallback
        out["ok"] = out["means"] is not None
    return out


def _crossover(mu, var, w):
    """x where w0·N(mu0,var0) = w1·N(mu1,var1); the root that lies between mu0 and mu1."""
    v0, v1 = float(var[0]), float(var[1])
    a = 0.5 * (1.0 / v0 - 1.0 / v1)
    b = (mu[1] / v1 - mu[0] / v0)
    c = (0.5 * mu[0] ** 2 / v0 - 0.5 * mu[1] ** 2 / v1
         + np.log((w[0] / np.sqrt(v0)) / (w[1] / np.sqrt(v1))))
    if abs(a) < 1e-12:                      # equal variances → linear
        if abs(b) < 1e-12:
            return None
        return -c / b
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (1.0, -1.0)]
    inside = [r for r in roots if mu[0] < r < mu[1]]
    return inside[0] if inside else None


# ── The public decision ─────────────────────────────────────────────────────────
def classify_nuclear(values, *, fixed_threshold: float = 0.2, min_cells: int = 20,
                     ashman_min: float = 2.0, bic_margin: float = 0.0,
                     allow_fixed_fallback: bool = True) -> dict:
    """Decide a per-image nuclear-DAB threshold and the per-cell positive calls.

    `values`         per-cell scalar DAB signal (QuPath DAB:Mean, or Macenko nuclear DAB OD).
    `fixed_threshold`the trusted fixed cutoff used as the fallback / fixed method.
    `ashman_min`     minimum bimodality separation to trust an adaptive cut (operating point,
                     tuned in A4).
    `allow_fixed_fallback`  when the image is not adaptively thresholdable, use `fixed_threshold`
                     (flagged low-quality) instead of abstaining. When False → abstain.

    Returns:
      threshold     float | None (None ⇒ abstain)
      method        'gmm' | 'otsu' | 'fixed' | 'abstain'
      labels        bool ndarray (len == finite values) | None
      separability  Ashman's D
      quality       'ok' | 'low'
      abstain       bool
      reason        str
      n             number of finite values used
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    xf = x[finite]
    base = {"threshold": None, "method": "abstain", "labels": None,
            "separability": 0.0, "quality": "low", "abstain": True,
            "reason": "", "n": int(xf.size)}

    if xf.size < min_cells:
        # Too few cells to fit anything → not adaptively callable: flag (abstain) and,
        # if allowed, still emit provisional fixed calls so the caller has numbers to show.
        if allow_fixed_fallback:
            base.update(threshold=float(fixed_threshold), method="fixed", quality="low",
                        abstain=True, reason=f"<{min_cells} cells; provisional fixed cutoff",
                        labels=(x > fixed_threshold))
        else:
            base["reason"] = f"<{min_cells} cells; cannot fit an adaptive threshold"
        return base

    g = gmm_1d_two(xf)
    sep = g["ashman_d"]
    bimodal = (g["ok"] and g["bic2"] + bic_margin < g["bic1"] and sep >= ashman_min)

    if bimodal:
        thr = g["threshold"]
        return {"threshold": float(thr), "method": "gmm", "labels": (x > thr),
                "separability": float(sep), "quality": "ok", "abstain": False,
                "reason": (f"bimodal (Ashman D={sep:.2f}, "
                           f"ΔBIC={g['bic1'] - g['bic2']:.0f}); GMM crossover"),
                "n": int(xf.size)}

    # Not adaptively thresholdable: unimodal or poorly separated → flag for the user.
    why = (f"unimodal / poorly separated (Ashman D={sep:.2f}, "
           f"ΔBIC={g['bic1'] - g['bic2']:.0f} favors 1 component)")
    if allow_fixed_fallback:
        # Flag as abstain (so the UI asks whether to proceed) but still provide
        # provisional fixed-cutoff calls so a "proceed" needs no re-run.
        base.update(threshold=float(fixed_threshold), method="fixed", quality="low",
                    abstain=True, separability=float(sep),
                    reason=f"{why}; provisional fixed cutoff — flagged low quality",
                    labels=(x > fixed_threshold))
    else:
        base.update(separability=float(sep),
                    reason=f"{why}; abstained (no trustworthy call on this image)")
    return base
