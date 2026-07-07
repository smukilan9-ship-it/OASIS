#!/usr/bin/env Rscript
# spatstat_crossval.R — reference computation for cross-validating our Python
# intensity-reweighted inhomogeneous cross-K against spatstat.
#
# Reads a case directory written by validate_spatstat_crossval.py and writes back:
#   r_lambdaA.csv / r_lambdaB.csv  — spatstat density.ppp LOO intensity at A / B
#                                    points (Stage A: intensity surface).
#   r_K_<corr>.csv                 — Kcross.inhom(r) using the PASSED Python lambda
#                                    (Stage B: estimator), for corr in
#                                    none / border / translate / isotropic.
#
# Everything is in PIXELS (Python sets pixel_size=1) so there is no unit mismatch.
# All inputs (points, window, r-grid, bandwidth, lambda) are supplied by Python so
# both tools operate on byte-identical inputs.

suppressMessages({
  ok <- require(spatstat.geom) & require(spatstat.explore)
  if (!ok) ok <- require(spatstat)   # meta-package fallback
})
if (!exists("Kcross.inhom")) {
  cat("ERROR: spatstat not available\n"); quit(status = 3)
}

args <- commandArgs(trailingOnly = TRUE)
dir  <- args[1]
rd   <- function(f) read.csv(file.path(dir, f))

pts    <- rd("points.csv")              # columns: type (A/B), x, y
win    <- rd("window.csv")              # polygon vertices x,y (anticlockwise)
params <- rd("params.csv")              # h (bandwidth px), area (px^2)
rgrid  <- rd("rgrid.csv")$r             # radii (px)
lamA   <- rd("lambdaA.csv")$lambda      # Python LOO intensity at A points
lamB   <- rd("lambdaB.csv")$lambda      # Python LOO intensity at B points
h      <- params$h[1]
area_py<- params$area[1]

# ── Observation window: build from the supplied polygon (anticlockwise) ──
poly <- list(x = win$x, y = win$y)
W <- tryCatch(owin(poly = poly), error = function(e) {
  # fall back to bounding box if the polygon is degenerate for owin
  owin(c(min(win$x), max(win$x)), c(min(win$y), max(win$y)))
})
cat(sprintf("  R owin area = %.4f   (python area = %.4f, ratio %.6f)\n",
            area(W), area_py, area(W) / area_py))

A <- pts[pts$type == "A", ]
B <- pts[pts$type == "B", ]
Xa <- ppp(A$x, A$y, window = W, checkdup = FALSE)
Xb <- ppp(B$x, B$y, window = W, checkdup = FALSE)

# Combined marked pattern for Kcross.inhom
allx <- c(A$x, B$x); ally <- c(A$y, B$y)
marks_ <- factor(c(rep("A", nrow(A)), rep("B", nrow(B))), levels = c("A", "B"))
X <- ppp(allx, ally, window = W, marks = marks_, checkdup = FALSE)

# ── Stage A: spatstat LOO Gaussian intensity (edge=FALSE to match our bare sum) ──
sa <- density.ppp(Xa, sigma = h, kernel = "gaussian",
                  leaveoneout = TRUE, edge = FALSE, at = "points")
sb <- density.ppp(Xb, sigma = h, kernel = "gaussian",
                  leaveoneout = TRUE, edge = FALSE, at = "points")
write.csv(data.frame(lambda = as.numeric(sa)),
          file.path(dir, "r_lambdaA.csv"), row.names = FALSE)
write.csv(data.frame(lambda = as.numeric(sb)),
          file.path(dir, "r_lambdaB.csv"), row.names = FALSE)

# ── Stage B: Kcross.inhom using the PASSED Python lambda, identical r-grid ──
# Bare estimator (1/|W|) Σ 1(d<=r)/(λI λJ); correction varied. spatstat names the
# uncorrected column "un" (not "none"), so we extract the single estimate column
# robustly (everything that is not r/theo) rather than guessing its name.
for (corr in c("none", "border", "translate", "isotropic")) {
  k <- tryCatch(
    Kcross.inhom(X, "A", "B", lambdaI = lamA, lambdaJ = lamB,
                 r = rgrid, correction = corr),
    error = function(e) { cat(sprintf("  Kcross.inhom %s failed: %s\n",
                                      corr, conditionMessage(e))); NULL })
  if (!is.null(k)) {
    estcols <- setdiff(colnames(k), c("r", "theo"))
    est <- as.numeric(k[[estcols[length(estcols)]]])
    write.csv(data.frame(r = k$r, K = est),
              file.path(dir, paste0("r_K_", corr, ".csv")), row.names = FALSE)
  }
}
cat("  R done.\n")
