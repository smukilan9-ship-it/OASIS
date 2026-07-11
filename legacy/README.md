# legacy/

Quarantined material that is no longer part of the shipping pipeline but is kept for
reference. Nothing here is imported by `oasis/`, the entrypoints, or the validation suite.

## Contents
- `next_session.md`, `paper_skeleton.md` — session-scratch notes superseded by `ihc.md`.

## Feature-level legacy (not files — removed during the Spatial/Quant UI rebuild)
These are methods/flows still wired into the live UI at restructure time; they are removed
as the rebuilt tabs replace them, not moved here:
- `send_chat` (dead AI-chat method in `oasis/webui/api.py`; no UI reference).
- Pre-LoFTR landmark flows: `propose_landmarks`, `guide_landmark_candidates`,
  `suggest_moving_landmark` — superseded by the LoFTR-in-ROI certification path.

See `ihc.md` for the current architecture.
