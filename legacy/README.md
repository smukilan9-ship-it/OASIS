# legacy/

Quarantined material that is no longer part of the shipping pipeline but is kept for
reference. Nothing here is imported by `oasis/`, the entrypoints, or the validation suite.

## Contents
- `research/next_session.md`, `paper_skeleton.md` — session-scratch notes superseded by `research/ihc.md`.
- `nuclear_adaptive/` — the per-image adaptive nuclear threshold (GMM valley + Ashman's D
  abstain gate) and its harnesses. Retired in favour of one fixed cutoff per stain across
  the cohort; the evidence is `research/ihc.md` § 11 and `threshold_audit_ll477_results.json` here.
  Short version: positives are rare (1–6 %), so the DAB distribution is unimodal with a
  tail and a two-mode statistic has no second mode to find. At the operating point the
  pipeline actually shipped (1.25) the cut fired at roughly half the trusted value. The
  scripts still run from this directory; they are not imported by `oasis/` or the
  validation suite.

## Feature-level legacy (not files — removed during the Spatial/Quant UI rebuild)
These are methods/flows still wired into the live UI at restructure time; they are removed
as the rebuilt tabs replace them, not moved here:
- `send_chat` (dead AI-chat method in `oasis/webui/api.py`; no UI reference).
- Pre-LoFTR landmark flows: `propose_landmarks`, `guide_landmark_candidates`,
  `suggest_moving_landmark` — superseded by the LoFTR-in-ROI certification path.

See `research/ihc.md` for the current architecture.
