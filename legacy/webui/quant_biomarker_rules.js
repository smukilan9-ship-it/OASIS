/* Mixed-stain batch rules — REMOVED from the Quant tab 2026-07-27.

WHY IT WAS REMOVED
The card let a batch define per-marker DAB cutoffs matched by a token in the filename
("CD8" -> 0.20, "TIM-3" -> 0.10). It was removed for two reasons, not for simplicity:

  1. It set a SECOND cutoff that silently disagreed with the slider on the same screen.
     An image matching a token ignored the visible cohort value, so what the UI showed
     and what the run used could differ with nothing marking it.
  2. The post-segmentation review already covers the real need. Every image is shown with
     its own DAB histogram and can take its own cutoff there, chosen against the actual
     distribution rather than guessed from a filename before any data exists.

The backend key it wrote (`stain_thresholds`) still exists and is still honoured by
run_pipeline, so a config file can set it for headless runs.
*/

let quantBiomarkers = [];   // [{name, threshold}]

function renderQuantBiomarkers() {
  const el = document.getElementById('q-biomarkers-list');
  if (quantBiomarkers.length === 0) {
    el.innerHTML = '<div style="color:var(--text3);font-size:11px;padding:2px 0">No biomarkers defined — all images use the global DAB threshold (nuclear)</div>';
  } else {
    el.innerHTML = quantBiomarkers.map((b,i) => `
      <div class="override-item">
        <span class="override-name">${esc(b.name)}</span>
        <span class="badge amber">${b.threshold} OD</span>
        <button class="btn sm" onclick="removeQuantBiomarker(${i})">Remove</button>
      </div>
    `).join('');
  }
}

function addQuantBiomarker() {
  const name = document.getElementById('q-bm-name').value.trim();
  const thr  = parseFloat(document.getElementById('q-bm-thr').value);
  if (!name || isNaN(thr)) { showToast('Enter a biomarker name and DAB threshold'); return; }
  quantBiomarkers.push({ name, threshold: thr });
  document.getElementById('q-bm-name').value = '';
  document.getElementById('q-bm-thr').value  = '';
}

function removeQuantBiomarker(i) {
  quantBiomarkers.splice(i, 1);
}

function addQuantBiomarkerPreset(which) {
  if (which === 'CD8')  quantBiomarkers.push({ name: 'CD8',   threshold: 0.20 });
  else                  quantBiomarkers.push({ name: 'TIM-3', threshold: 0.10 });
}

let quantMode = 'single';
