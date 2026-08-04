# Run KAMP (Wrobel & Song, arXiv:2412.08498) on CODEX CRC, using THEIR source unmodified.
#
# KAMP's bivariate null is the label-permutation ("proportional intensity") null: the two
# cell types are treated as random subsets of the POOLED cell population, so the reference
# pattern is every cell in the sample, not just the two types of interest. That is why the
# ppp built here carries all cells with a third "background" mark.
#
# Two arms:
#   real      — the actual cell-type labels
#   permuted  — labels shuffled within the spot. This is KAMP's own null, so it is a check
#               that the closed-form moments are right, NOT a test of the proportional-
#               intensity assumption (that assumption is not testable from the data alone).

suppressMessages({
  library(spatstat.geom); library(spatstat.explore)
  library(dplyr); library(tibble); library(purrr)
})

args   <- commandArgs(trailingOnly = TRUE)
CSVIN  <- args[1]
OUTCSV <- args[2]
CHUNK  <- if (length(args) >= 3) as.integer(args[3]) else NA
NCHUNK <- if (length(args) >= 4) as.integer(args[4]) else NA
NSPOT  <- if (length(args) >= 5) as.integer(args[5]) else NA

# KAMP's own source, unmodified and NOT vendored here — it is the authors' code under their
# terms, and the point of this script is to run theirs rather than a copy that can drift.
# Clone https://github.com/JuliaWrobel/KAMP and point KAMP_SOURCE at its source/ directory.
# This used to be an absolute path into the working directory of whoever last ran it, which
# meant the script was broken for everybody else and said so only as a file-not-found.
KAMP_SRC <- Sys.getenv("KAMP_SOURCE", unset = "")
if (!nzchar(KAMP_SRC)) {
  stop("set KAMP_SOURCE to KAMP's source/ directory (git clone JuliaWrobel/KAMP)")
}
kamp_file <- file.path(KAMP_SRC, "get_permutation_distribution.R")
if (!file.exists(kamp_file)) {
  stop(sprintf("no get_permutation_distribution.R under KAMP_SOURCE=%s", KAMP_SRC))
}
source(kamp_file)

PX        <- 0.3775
RADII_UM  <- c(10, 20, 50)
RADII_PX  <- RADII_UM / PX
MIN_CELLS <- 40
PAIRS <- list(c("CD8+ T cells", "CD4+ T cells CD45RO+"),
              c("CD8+ T cells", "tumor cells"),
              c("CD8+ T cells", "CD68+CD163+ macrophages"),
              c("tumor cells", "vasculature"),
              c("stroma", "smooth muscle"))

cat("reading CODEX table...\n"); flush.console()
d <- read.csv(CSVIN, stringsAsFactors = FALSE)[, c("spots", "X.X", "Y.Y", "ClusterName")]
names(d) <- c("spot", "x", "y", "type")
spots <- sort(unique(d$spot))
if (!is.na(NSPOT)) spots <- spots[seq_len(min(NSPOT, length(spots)))]
if (!is.na(CHUNK) && !is.na(NCHUNK))
  spots <- spots[(seq_along(spots) - 1) %% NCHUNK == CHUNK]
cat(length(spots), "spots\n"); flush.console()

res <- list(); k <- 0
t0 <- Sys.time()
for (si in seq_along(spots)) {
  s  <- spots[si]
  sub <- d[d$spot == s, ]
  W  <- owin(range(sub$x), range(sub$y))

  for (arm in c("real", "permuted")) {
    set.seed(3000000 + si)
    lab <- if (arm == "permuted") sample(sub$type) else sub$type

    for (pi in seq_along(PAIRS)) {
      ta <- PAIRS[[pi]][1]; tb <- PAIRS[[pi]][2]
      na <- sum(lab == ta); nb <- sum(lab == tb)
      if (na < MIN_CELLS || nb < MIN_CELLS) next

      marks <- ifelse(lab == ta, "immune1", ifelse(lab == tb, "immune2", "background"))
      P <- ppp(sub$x, sub$y, window = W, marks = factor(marks))

      out <- tryCatch(
        map_dfr(RADII_PX, get_permutation_distribution, ppp_obj = P, bivariate = TRUE),
        error = function(e) NULL)
      if (is.null(out)) next

      k <- k + 1
      res[[k]] <- out %>%
        mutate(spot = s, arm = arm, a = ta, b = tb,
               n_a = na, n_b = nb, n_total = nrow(sub),
               r_um = r * PX)
    }
  }
  if (si %% 5 == 0) {
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    cat(sprintf("  %d/%d spots  %.1f min  eta %.1f min\n", si, length(spots), el,
                el / si * (length(spots) - si))); flush.console()
  }
}

out <- bind_rows(res)
write.csv(out, OUTCSV, row.names = FALSE)
cat("\nwrote", OUTCSV, "-", nrow(out), "rows\n")

cat("\nKAMP one-sided p (clustering), size at alpha=0.05\n\n")
cat(sprintf("  %-10s %7s %6s %7s %10s\n", "arm", "r (um)", "n", "size", "median p"))
for (a in c("permuted", "real")) {
  for (ru in RADII_UM) {
    x <- out[out$arm == a & abs(out$r_um - ru) < 1e-6 & is.finite(out$pvalue), ]
    if (!nrow(x)) next
    cat(sprintf("  %-10s %7.0f %6d %7.3f %10.3f\n", a, ru, nrow(x),
                mean(x$pvalue <= 0.05), median(x$pvalue)))
  }
}
