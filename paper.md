---
title: 'OASIS: certified spatial association analysis for serial-section single-plex IHC'
tags:
  - Python
  - digital pathology
  - immunohistochemistry
  - image registration
  - spatial statistics
authors:
  - name: Mukilan   # TODO: full name as it should appear in the citation
    orcid: 0000-0000-0000-0000   # TODO: required by JOSS — register at orcid.org
    affiliation: 1
affiliations:
  - name: TODO — institution, country
    index: 1
date: 25 July 2026
bibliography: paper.bib
---

<!--
DRAFTING NOTE — delete before submission.

This is a STRUCTURAL skeleton, not a finished paper. It has the sections JOSS
requires (notably a clearly labelled "Statement of need") and marks what only the
author can supply.

Numbers are deliberately NOT baked in here. Every quantitative claim should be
copied from `research/ihc.md` / the `validation_reports/` bundles at the time of writing,
so the paper cites figures that were re-measured against the submitted commit
rather than remembered. Placeholders read [FIG: ...].

JOSS papers are short — roughly 250-1000 words. Resist expanding this into a
methods paper; depth belongs in the repository documentation, which reviewers
also read.
-->

# Summary

Immunohistochemistry (IHC) stained on **serial sections** cannot establish that two
markers occur in the same cell: adjacent sections are different tissue planes, so
a cell in one is not the same cell in the other. Studies nonetheless need to ask
whether two cell populations are spatially *associated* — whether, for example,
CD8+ T cells sit closer to TIM-3+ cells than chance would predict.

OASIS is a desktop and command-line tool that answers that question and refuses to
answer questions it cannot support. It segments nuclei, quantifies DAB
positivity, registers a pair of serial sections into a shared coordinate frame,
and measures cross-type spatial association using Ripley's K and the
pair-correlation function against Monte Carlo null models. Crucially, it treats
the registration itself as something to be *certified*: when alignment cannot be
shown accurate at cell scale, the spatial statistics are marked invalid rather
than reported with a caveat.

# Statement of need

<!-- JOSS requires this heading verbatim. It must state the problem, the target
     audience, and the relation to other work. -->

Existing digital-pathology tools solve adjacent problems. QuPath provides
segmentation and quantification but no certified cross-section spatial
statistics. Registration packages align images but report their own internal
error, which is not the same as accuracy at the scale of a cell. Spatial-statistics
libraries assume a point pattern that is already correct, and so inherit any
registration error silently.

The gap OASIS fills is the join between them: a distance-preserving registration
whose accuracy is *certified against independent landmarks*, feeding a
population-level spatial statistic that is only reported when that certification
holds. [TODO: name the specific comparator tools and cite them.]

The intended users are researchers analysing archival single-plex IHC, where
multiplexed imaging is unavailable and serial sections are the only option.

A second need is epistemic. Single-cell co-expression is not recoverable from
serial sections, but tooling that reports a per-cell "double-positive" count
invites exactly that misreading. OASIS reports a population statistic and states
in its output what the statistic does and does not license. [TODO: one sentence on
the fail-closed gates — registration certification, bandwidth verdict, dense-tissue
null selection — citing `research/ihc.md`.]

# Implementation

OASIS runs the InstanSeg `brightfield_nuclei` model [@goldsborough2024instanseg]
in-process via TorchScript, so no external application is required. Stain
separation uses H-DAB colour deconvolution; positivity is thresholded on DAB
optical density, with an optional membrane-completeness classifier for membranous
markers where a ring-mean dilutes thin stained arcs.

Registration proceeds through a cascade and is certified by held-out target
registration error. Spatial association is measured with the cross-type Ripley's
K and g(r) against a tissue-mask-bounded Monte Carlo null, with a global
DCLF envelope test. Three null models are available; the choice is reported.
[TODO: cite Ripley, Diggle, and the DCLF test properly.]

# Validation

Every scientific claim in OASIS is backed by a registered validation that can be
run from the desktop UI or the command line, producing a report bundle with
metrics, software and dataset provenance, and the git SHA:

```bash
python -m validation.run --list
python -m validation.run all --tier instant
```

[TODO: summarise the headline validations in one short table — segmentation
against IF-derived ground truth, membrane classification against expert labels,
registration certification against public landmark datasets. Take the numbers from
the current `validation_reports/` bundles, not from memory.]

Alternatives that were evaluated and rejected are documented rather than
discarded; `research/valis.md` records why a registration backend that outperformed the
chosen one on a public benchmark was nonetheless removed, because it could not be
certified honestly within this architecture.

# Acknowledgements

[TODO: funding, data providers, and any restricted datasets used under agreement.]

# References
