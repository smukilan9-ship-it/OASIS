/* Calibrate-tab fitting flow — RETIRED 2026-07-27, with the tab itself.

These are the functions that owned the Calibrate tab's own cohort form and its cutoff fit:
adding slides, segmenting them, fitting membrane completeness cutoffs, and saving the
result. All of it is superseded by the Classifier tab (clsAddImage / clsSegment / clsFit /
clsSave), which does the same job for the same compartment and reports a leave-one-IMAGE-out
estimate instead of leave-one-cell-out.

NOT retired, and still live in index.html: the labelling canvas itself (calibBuild,
calibShow, calibSetView, calibZoom, calibSet, calibUpdateCounts and friends). The Classifier
tab labels cells on that shared canvas -- it is the good part of the retired tab.

The backend endpoints these called (calibration_fit, calibration_fit_multi, save_calibration)
are unchanged; saved calibrations on disk are still read by the membranous path in Quant.
*/


function calibDefaultPx(){ return parseFloat(document.getElementById('calib-px').value) || 0.5; }


async function calibAddImage() {
  const f = await window.pywebview.api.pick_file();
  if (!f) return;
  if (calib.images.some(im => im.path === f)) { showToast('That image is already added'); return; }
  calib.images.push({ path:f, px:calibDefaultPx(), data:null, state:[] });
  calibRenderImageList();
}


function calibRemoveImage(i){ calib.images.splice(i,1); calibRenderImageList(); }


function calibRenderImageList() {
  const box = document.getElementById('calib-image-list');
  if (!calib.images.length) {
    box.innerHTML = '<div style="font-size:12px;color:var(--text3)">No images yet — add at least one representative stained slide.</div>';
    return;
  }
  box.innerHTML = calib.images.map((im,i)=>`
    <div style="display:flex;gap:8px;align-items:center">
      <span style="flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(im.path)}">${esc(im.path.split('/').pop())}${im.data?' ✓':''}</span>
      <label style="font-size:11px;color:var(--text3)">µm/px</label>
      <input type="number" value="${im.px}" step="0.01" min="0.01" style="width:78px"
             onchange="calib.images[${i}].px=parseFloat(this.value)||calib.images[${i}].px">
      <button class="btn sm" onclick="calibRemoveImage(${i})">Remove</button>
    </div>`).join('');
}


async function calibSegment() {
  const marker = document.getElementById('calib-marker').value.trim();
  if (!marker) { showToast('Enter the marker name'); return; }
  if (!calib.images.length) { showToast('Add at least one stained image'); return; }
  calib.marker = marker;
  const btn = document.getElementById('calib-seg-btn');
  btn.disabled = true;
  for (let i=0;i<calib.images.length;i++) {
    const im = calib.images[i];
    if (im.data) continue;                                  // already segmented
    btn.textContent = `Segmenting ${i+1}/${calib.images.length}… (InstanSeg, ~1 min each)`;
    const r = await window.pywebview.api.calibration_prepare(im.path, im.px);
    if (!r || !r.ok) {
      btn.disabled=false; btn.textContent='Segment & label →';
      showToast(`Segmentation failed on ${im.path.split('/').pop()}: ${(r&&r.msg)||''}`); return;
    }
    im.data = r; im.state = new Array(r.cells.length).fill(0);
  }
  btn.disabled=false; btn.textContent='Segment & label →';
  calib.cur = 0;
  calibShow('label');            // make the stage visible FIRST so it has a size
  requestAnimationFrame(() => calibBuild());
}


async function calibFit() {
  const items = []; let tp=0, tn=0;
  calib.images.forEach(im => {
    if (!im.data) return;
    const pos=[], neg=[];
    im.state.forEach((s,i)=>{ if(s===1)pos.push(i); else if(s===2)neg.push(i); });
    tp+=pos.length; tn+=neg.length;
    items.push({ image_path:im.path, geojson_path:im.data.geojson,
                 pixel_size:im.px, pos_idx:pos, neg_idx:neg });
  });
  if (tp<8 || tn<8) { showToast(`Need ~8+ of each pooled across images (have ${tp} pos / ${tn} neg)`); return; }
  showToast('Fitting cutoffs…');
  const r = await window.pywebview.api.calibration_fit_multi(items);
  if (!r || !r.ok) { showToast('Fit failed: ' + (r&&r.msg||'')); return; }
  calib.fit = r;
  const held  = r.loo_auc != null;                          // held-out numbers available?
  const auc   = held ? r.loo_auc : r.auc;
  const f1    = held ? r.loo_f1  : r.f1;
  document.getElementById('calib-metrics').innerHTML = `
    <div class="metric cyan"><div class="metric-label">${held?'Held-out separability (leave-one-cell-out AUC)':'Separability (AUC · in-sample)'}</div>
      <div class="metric-value cyan">${auc.toFixed(2)}</div>
      <div class="metric-sub">${held?'LOO ':''}F1 ${f1.toFixed(2)} · ${r.n_pos}+/${r.n_neg}− · ${r.n_images} image${r.n_images===1?'':'s'}</div></div>
    <div class="metric green"><div class="metric-label">Fitted cutoffs</div>
      <div class="metric-value green" style="font-size:20px">${r.membrane_pix_thr.toFixed(3)} / ${r.membrane_frac_min.toFixed(2)}</div>
      <div class="metric-sub">pixel thr / min ring fraction${held?` · in-sample AUC ${r.auc.toFixed(2)}`:''}</div></div>`;
  document.getElementById('calib-perimage').innerHTML = (r.per_image||[])
    .map(p=>`${esc(p.image)}: <b style="color:#ff5a5a">${p.n_pos}+</b> / <b style="color:#4aa3ff">${p.n_neg}−</b>`).join(' &nbsp;·&nbsp; ');
  const q = held ? 'held-out ' : '';
  document.getElementById('calib-verdict').innerHTML = r.callable
    ? `<div class="info-box" style="border-color:#2e7d32">✓ This marker is callable (${q}AUC ${auc.toFixed(2)}). Save the cutoffs and Quant will use them for <b>${esc(calib.marker)}</b> in membrane mode.</div>`
    : `<div class="info-box amber">⚠ Low ${q}separability (AUC ${auc.toFixed(2)}) — positives and negatives don't separate on unseen cells. The staining may be too faint to call, or the labels need review. Saving is allowed but treat results as unverified.</div>`;
  calibShow('results');
}


async function calibSave() {
  if (!calib.fit) return;
  const name = prompt('Name this calibration (so you can switch between them in Quant):',
                      `${calib.marker} — ${new Date().toLocaleDateString()}`);
  if (!name) return;
  const btn=document.getElementById('calib-save-btn'); btn.disabled=true;
  const r = await window.pywebview.api.save_calibration(name, calib.marker, calib.fit);
  btn.disabled=false;
  if (r && r.ok) { showToast(`Saved "${name}" ✓ — pick it under Cutoff profile in Quant`); calibShow('config'); }
  else showToast('Save failed');
}