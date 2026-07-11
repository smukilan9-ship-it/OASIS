# Spatial UI UX Implementation

Observed first in fullscreen on the current Spatial Association screen:

- Left rail navigation, then a long single-page Spatial workflow.
- Step 2 is Landmark Certification with image loading, colour toggle, expert import, fullscreen landmarking, ROI drawing, auto-propose, certify, undo, and clear controls.
- Step 3 is the 75 um bandwidth validity pre-flight.
- Analysis parameters are visible as ordinary fields/toggles with short helper text.
- Results are rendered after analysis, but parameters and scientific interpretation are scattered between helper text, logs, output JSON, and result cards.

Implemented in the biologist-facing workflow:

- A compact "What this result means" drawer now accompanies spatial result cards and distinguishes population-level association from same-cell co-expression.
- "Parameters used" disclosures now follow quantification outputs, certification, bandwidth pre-flight pairs, spatial overlays, density maps, association plots, and complete spatial result cards.
- Parameter labels and disclosure values have hover definitions covering their exact scientific role and caveats.
- Guided landmarking now gives a short fixed-to-moving instruction in the fullscreen tool without adding a tutorial panel.
- Certification results show an analysis-window chip for full field, Certification ROI, automatic local ROI, or ROI intersected with landmark support.
- The 75 um pre-flight now displays "ok", "caution", "dense tissue", or "unknown: <exact reason>" with cell count, scale, bandwidth, and reason in the tooltip and parameter record.
- Dense morphology-conditioned null selection is shown as a non-error purple chip; unavailable primary-null states are visibly fail-closed.
- Keep the current sparse, calm visual style. Do not add a heavy wizard, large explanatory cards, or always-open scientific text; biologists should see the decision first and inspect details when needed.
