/* Guided / auto-proposed landmarks — RETIRED from the UI 2026-07-28. Kept for the record.
 *
 * WHY THIS IS GONE. The manual landmark path exists for exactly one reason: OASIS does not
 * guess the correspondence. An auto-placed match the operator never confirmed is an
 * unverified correspondence, and in featureless tissue an NCC/LoFTR snap lands arbitrarily,
 * corrupting the transform and every landmark suggested after it. That is written into the
 * click handler this file was extracted from, and it is the whole scientific claim of the
 * manual path (research/ihc.md 3.5).
 *
 * "Auto-propose landmarks" and "Guide next" contradicted that claim from inside it: they
 * proposed correspondences automatically and then asked the operator to bless them. Once
 * LoFTR-in-ROI became the primary automatic engine, they were also redundant — a second,
 * weaker automatic path living inside the manual one. So the split is now clean:
 *
 *     automatic  -> LoFTR in a region        (default)
 *     manual     -> the operator marks both points, always, with no suggestions
 *
 * The backend methods these called (API.propose_landmarks, API.suggest_moving_landmark,
 * API.guide_landmark_candidates) still exist and are still tested; nothing in the shipped UI
 * reaches them. Restore this file if the manual path is ever given a suggestion mode again —
 * but read the paragraph above first.
 *
 * Note: I had restored the accept/reject buttons for these earlier the same day, because a
 * wiring audit reported "function with no control". That was the wrong repair — the audit was
 * right that something was inconsistent, and the correct fix was to remove the functions, not
 * to add the buttons.
 */

async function spatialCertPropose() {
  const w = spatialCertWork;
  if (!w.pair || !w.ref) { showToast('Load images for landmarking first'); return; }
  const px = spatialCertPixelSize();
  if (!(px > 0)) { showToast('Set a valid pixel size first'); return; }
  if ((w.refPoints.length || w.pending) &&
      !confirm('Replace the current landmarks with auto-proposed ones?')) return;
  if (w.roiMode) { showToast('Finish the certification ROI first (Finish ROI)'); return; }
  const roi = (w.roiPoints||[]).length >= 3 ? w.roiPoints : null;
  const btn = document.getElementById('spatial-cert-propose');
  btn.disabled = true; btn.textContent = 'Proposing…';
  const r = await window.pywebview.api.propose_landmarks(w.pair.path_a, w.pair.path_b, px, 8, roi);
  btn.disabled = false; btn.textContent = 'Auto-propose landmarks';
  if (r.status !== 'ok') { showToast(r.error || 'Could not propose landmarks — place them manually'); return; }
  spatialCertWork.refPoints = r.ref_points || [];
  spatialCertWork.movPoints = r.mov_points || [];
  spatialCertWork.movRoi = r.mov_roi_polygon || spatialCertWork.movRoi;
  spatialCertWork.pending = null;
  spatialCertWork.guideCandidate = null;
  spatialCertWork.guideCandidates = [];
  spatialCertForgetCurrentCertification();
  spatialCertRender();
  const cov = (typeof r.coverage_frac === 'number') ? (r.coverage_frac*100).toFixed(0)+'%' : '—';
  const res = (typeof r.fit_residual_um === 'number') ? r.fit_residual_um.toFixed(1)+' µm' : '—';
  const modeTxt = r.mode === 'roi'
    ? `<b style="color:#b45309">ROI mode</b> — consistent only within a ${cov} region; the certification will likely be LOCALLY_CERTIFIED to that ROI.`
    : `<b style="color:#166534">Field-wide</b> — ${cov} coverage.`;
  document.getElementById('spatial-cert-result').innerHTML =
    `<span style="color:#0e7490">Proposed ${r.n} correspondence${r.n===1?'':'s'} (lumens ref/mov ${r.n_lumen_ref}/${r.n_lumen_mov}). ${modeTxt} Median self-residual ${res}. <b>Verify each pair against the images and delete any mismatches before certifying.</b></span>`;
  spatialCertGuideContinue('auto-proposed landmarks');
}

function spatialCertCandidateKey(c) {
  if (!c) return '';
  const r = c.ref_point || [];
  const m = c.mov_point || [];
  return [r[0], r[1], m[0], m[1]].map(v => Math.round(Number(v) || 0)).join(':');
}

function spatialCertCandidateAt(refPt) {
  const w = spatialCertWork;
  const cands = w.guideCandidates || [];
  if (!cands.length || !w.ref) return null;
  const img = document.getElementById('spatial-cert-ref');
  const iw = img.naturalWidth || w.ref.width;
  const u = w.ref.width / iw;
  const sc = (lmPanes.ref && lmPanes.ref.scale) || 1;
  const hitRadius = Math.max(16 * u / sc, 20);
  let best = null, bestD = Infinity;
  cands.forEach(c => {
    const p = c.ref_point;
    if (!p) return;
    const d = Math.hypot(refPt[0] - p[0], refPt[1] - p[1]);
    if (d < bestD) { bestD = d; best = c; }
  });
  return bestD <= hitRadius ? best : null;
}

async function spatialCertGuideRequestNext(userRequested) {
  const w = spatialCertWork;
  if (!w.pair || !w.ref) { if (userRequested) showToast('Load images for landmarking first'); return false; }
  if (w.roiMode) { if (userRequested) showToast('Finish the certification ROI first'); return false; }
  if (w.pending) { if (userRequested) showToast('Finish the current manual pair first'); return false; }
  if ((w.guideCandidates || []).length || w.guideCandidate) {
    if (userRequested) showToast('Pick one of the shown candidate spots, or reject the highlighted one');
    return true;
  }
  const px = spatialCertPixelSize();
  if (!(px > 0)) { if (userRequested) showToast('Set a valid pixel size first'); return false; }
  w.guiding = true;
  spatialCertRender();
  document.getElementById('spatial-cert-result').innerHTML =
    `<span style="color:#64748b">Searching for the next landmark that improves certification coverage…</span>`;
  const roi = (w.roiPoints||[]).length >= 3 ? w.roiPoints : null;
  let r = null;
  try {
    r = await window.pywebview.api.guide_landmark_candidates(
      w.pair.path_a, w.pair.path_b, px, w.refPoints, w.movPoints, roi, 24,
      [w.ref.width, w.ref.height]);
  } catch (err) {
    r = { status:'error', error:String(err) };
  }
  w.guiding = false;
  if (!r || r.status !== 'ok') {
    w.guideExhausted = true;
    if (userRequested) showToast((r && r.error) || 'No further guided landmarks found');
    document.getElementById('spatial-cert-result').innerHTML =
      `<span style="color:#b45309">No further guided landmark candidates were found. OASIS will now rely on the current accepted landmarks for the final certification verdict.</span>`;
    spatialCertRender();
    return false;
  }
  if (r.certification_gates) spatialCertGates = { ...spatialCertGates, ...r.certification_gates };
  const rejected = new Set(w.guideRejected || []);
  const candidates = (r.candidates || []).filter(c => !rejected.has(spatialCertCandidateKey(c)));
  if (!candidates.length) {
    w.guideExhausted = true;
    document.getElementById('spatial-cert-result').innerHTML =
      `<span style="color:#b45309">All guided candidates have been accepted or rejected. OASIS will now rely on the current accepted landmarks for the final certification verdict.</span>`;
    spatialCertRender();
    return false;
  }
  w.guideCandidates = candidates;
  w.guideCandidate = candidates[0];
  w.guideExhausted = false;
  const certMsg = r.certification_ready
    ? ' Every shown C# spot is predicted to reach a field-wide <b>CERTIFIED</b> verdict if accepted; C1 is the best-ranked option.'
    : ' No single spot certifies the whole field yet — these are ranked by how far each one moves the global fit toward CERTIFIED.';
  document.getElementById('spatial-cert-result').innerHTML =
    `<span style="color:#0e7490">Showing ${candidates.length} candidate spots.${certMsg} Click the best C# anatomical correspondence in the reference pane; the matching C# is shown on moving tissue.</span>`;
  spatialCertRender();
  return true;
}

async function spatialCertAcceptCandidate(c) {
  const w = spatialCertWork;
  if (!c) return;
  w.refPoints.push(c.ref_point);
  w.movPoints.push(c.mov_point);
  w.guideCandidate = null;
  w.guideCandidates = [];
  spatialCertForgetCurrentCertification();
  spatialCertRender();
  await spatialCertGuideContinue('accepted guided landmark');
}

async function spatialCertAcceptGuide() {
  await spatialCertAcceptCandidate(spatialCertWork.guideCandidate);
}

async function spatialCertRejectGuide() {
  const w = spatialCertWork;
  if (!w.guideCandidate) return;
  w.guideRejected = w.guideRejected || [];
  w.guideRejected.push(spatialCertCandidateKey(w.guideCandidate));
  w.guideCandidates = (w.guideCandidates || []).filter(c =>
    spatialCertCandidateKey(c) !== spatialCertCandidateKey(w.guideCandidate));
  w.guideCandidate = w.guideCandidates[0] || null;
  spatialCertRender();
  // Rejecting a suggestion hands control back to the operator. OASIS does not
  // immediately push another one — place the pair by hand, or press "Guide next".
  if (!w.guideCandidate) {
    document.getElementById('spatial-cert-result').innerHTML =
      `<span style="color:#64748b">Suggestion rejected. Click the fixed section, then the matching structure on the moving section to place the pair yourself — or press <b>Guide next</b> for another suggestion.</span>`;
    await spatialCertLiveCertify('manual landmarks');
  }
}

async function spatialCertGuideContinue(source) {
  const w = spatialCertWork;
  if (!w.pair || w.roiMode || (w.guideCandidates || []).length || w.guideCandidate || w.pending) return;
  // A field-wide CERTIFIED verdict is the only outcome that ends the guided search
  // early. LOCALLY_CERTIFIED is strictly a fallback: it is accepted only in the final
  // attempt below, once no further landmark can be proposed and a global
  // certification has therefore been shown to be unreachable.
  if (w.refPoints.length >= spatialCertGates.min_n) {
    const cert = await spatialCertTryAutoCertify(source, false, { provisional: true });
    if (cert && cert.status === 'CERTIFIED') return;
  }
  const hasNext = await spatialCertGuideRequestNext(false);
  if (!hasNext && w.refPoints.length >= spatialCertGates.min_n) {
    await spatialCertTryAutoCertify('final guided attempt', true, { provisional: false });
  }
}
