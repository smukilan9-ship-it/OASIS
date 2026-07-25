# VALIS certification: why the structural self-cert is unsafe, and what to do

**Date:** 2026-07-19. **Context:** wiring VALIS as a 3rd registration engine. Validation on
real ANHIR held-out expert landmarks exposed a certification-honesty failure; this documents
the research and the recommended resolution.

## The failure
On a fine-resolution similar-stain pair (ANHIR lung-lesion_1, Cc10→CD31, 0.348 µm/px),
`auto_certify_regions(engine="valis")` marked **every** region certified (structural cell-error
0.5–14 µm), but the **held-out expert landmarks inside those regions were 17–36 µm off**. The
structural cert was optimistic by 2–3×. Certifying that would set the spatial-analysis
registration-radius floor too low and manufacture associations — a fail-closed violation.

## Root causes (measured)
1. **A single global rigid can't hit cell-scale on deforming serial sections.** VALIS's own
   clean rigid transform (via the verified `warp_xy_from_to`) is **24.65 µm** median held-out
   TRE here. LoFTR reaches 3.66 µm on this tissue *only* because it fits a **separate local
   rigid per ROI**; VALIS emits one global rigid for the whole slide.
2. **Structural residual ≠ anatomical correspondence.** `cv2.phaseCorrelate` on the near-
   identical hematoxylin texture of two serial sections locks onto local texture and "explains
   away" real anatomical displacement — the residual *shrinks toward 0 as resolution increases*
   (21 µm → 0.3 µm over max_side 1200→3400) while the true TRE stays 24.65 µm.
3. **The anatomical truth is not in the images.** VALIS's *non-rigid* warp (its best possible
   alignment) is still **16.8 µm** off the expert landmarks on this pair. The correspondence the
   experts encode is physically absent from the pixels for hard/cross-modal pairs.

## Fixes tried, and why they are insufficient
- **Resolution floor (`px_floor=2 µm`) + lumen-TRE veto** in `_structural_certify`: makes the
  cert *resolution-stable and fail-closed on gross errors* (region-max 18–37 µm → NOT_CERTIFIED
  for the whole field — correct). **Kept** (it is a real improvement and a valid gross-error QC),
  but it does **not** make the cert trustworthy at cell scale — local regions still over-passed.
- **Rigid-vs-non-rigid gap** as a texture-independent cert signal: `corr(gap, true_TRE)=0.61`,
  and points where rigid≈non-rigid (gap ≤ 3 µm) are **still 24 µm** off. Rejected.
- **Feed VALIS matches to the validated FW gate** (like LoFTR): VALIS's matches are sparse (177
  whole-slide here) and live in a **mixed multi-detector internal frame** (features at 512 px
  *and* 542 px, combined) → not cleanly mappable to original per-ROI coords. Not viable as built.

## Conclusion
Cell-scale certification requires measuring agreement with **independent, anatomically-faithful
correspondences**. LoFTR supplies these for **similar stains** and its Fitzpatrick–West gate is
validated — which is exactly OASIS's real use case (CD8 vs TIM-3, both brown DAB). For
**cross-modal / hard** pairs no automatic method supplies them (texture ≠ anatomy), so only
**expert landmarks** can certify. VALIS's structural residual can only **fail closed on gross
misregistration**; it must not be presented as a cell-scale certificate.

## Recommendation (implemented separately)
Position VALIS as a **cross-modal REGISTRATION engine**, not a self-certifying gate:
- VALIS produces an accurate global transform where `register_similarity` fails cross-modal.
- Certification of analysis regions stays with the **validated gates**: LoFTR-in-ROI (similar
  stains) or **manual landmarks** (the transform gives an excellent starting alignment, so a few
  landmarks confirm it quickly — VALIS *accelerates* the validated manual path for cross-modal).
- VALIS's structural residual is surfaced as a **fail-closed QC** ("gross-error check", not a
  cell-scale certificate), never as the basis for marking a region analysis-certified.

Evidence scripts (scratchpad, not committed): `verify_per_region.py`, `diag_struct_res.py`,
`research_rigid_vs_nonrigid.py`, `research_valis_corr.py`.
