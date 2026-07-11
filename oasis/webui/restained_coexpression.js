/* Isolated UI for same-section restained co-expression. */
(function () {
  const navButton = document.createElement('button');
  navButton.className = 'nav-btn';
  navButton.id = 'nav-restained';
  navButton.onclick = () => showPage('restained');
  navButton.innerHTML = `
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="8" cy="8" r="3"/><circle cx="16" cy="16" r="3"/><path d="M10.5 10.5l3 3M16 5v4M14 7h4M5 16h4M7 14v4"/>
    </svg>
    Restained`;
  // Place Restained directly below Spatial (a top-group tab, not pinned to the
  // bottom). Fall back to before Settings if Spatial isn't present.
  const spatialButton = document.getElementById('nav-spatial');
  const settingsButton = document.getElementById('nav-settings');
  if (spatialButton) spatialButton.insertAdjacentElement('afterend', navButton);
  else settingsButton.parentNode.insertBefore(navButton, settingsButton);

  const page = document.createElement('div');
  page.className = 'page';
  page.id = 'page-restained';
  page.innerHTML = `
    <div class="page-header">
      <h1>Restained co-expression</h1>
      <span class="sub">same physical section · one segmentation · two AEC markers</span>
      <div class="header-right">
        <span id="restained-status" style="display:none;align-items:center;gap:6px;font-size:11px;color:var(--text3)"><span class="sdot b"></span><span>Running</span></span>
        <button class="btn danger" id="restained-stop-btn" style="display:none" onclick="stopPipeline()">Stop</button>
        <button class="btn" id="restained-new-btn" style="display:none" onclick="showView('restained','config')">← New analysis</button>
      </div>
    </div>

    <div class="view active" id="restained-config">
      <div class="scroll">
        <div class="info-box amber" style="margin-bottom:14px">
          This workflow is only for <b>already-corresponding images from the same physical section</b>. It does not register images. All three images must have identical dimensions; otherwise it fails closed. Hematoxylin nuclei are segmented once and the same cell polygons are measured on both AEC restains.
        </div>
        <div class="card">
          <div class="card-title">Analysis mode</div>
          <div class="seg-tabs">
            <button class="seg-tab active" id="r-mode-single" onclick="restainedSetMode('single')">Single section</button>
            <button class="seg-tab" id="r-mode-batch" onclick="restainedSetMode('batch')">Batch folder</button>
          </div>
          <div id="r-single" style="margin-top:12px">
            ${fileField('r-h', 'Hematoxylin reference', false)}
            ${fileField('r-a', 'Marker A AEC image', false)}
            ${fileField('r-b', 'Marker B AEC image', false)}
            ${fileField('r-mask', 'Expert nuclear mask (optional)', false)}
          </div>
          <div id="r-batch" style="display:none;margin-top:12px">
            ${folderField('r-folder', 'Folder containing all three image types')}
            <div class="row3">
              <div class="field"><label>Hematoxylin suffix</label><input id="r-h-token" value="_Hematoxylin"></div>
              <div class="field"><label>Marker A suffix</label><input id="r-a-token" value="_CD8"></div>
              <div class="field"><label>Marker B suffix</label><input id="r-b-token" value="_FoxP3"></div>
            </div>
            ${folderField('r-mask-folder', 'Expert-mask folder (optional)')}
            <button class="btn sm" onclick="restainedPreview()">Preview complete bundles</button>
            <div id="r-preview" class="muted" style="font-size:11px;margin-top:8px"></div>
          </div>
          <div style="margin-top:12px">${folderField('r-output', 'Output folder')}</div>
        </div>

        <div class="card">
          <div class="card-title">Segmentation reference</div>
          <div style="display:flex;align-items:center;justify-content:space-between;gap:16px">
            <div><b style="font-size:12px">Faint-nucleus preprocessing</b><div class="muted" style="font-size:11px;margin-top:3px">Fixed H/AEC colour deconvolution + 1st–99th percentile H-OD stretch. Applied only in this tab.</div></div>
            <div class="toggle on" id="r-preprocess" onclick="toggleEl('r-preprocess')"></div>
          </div>
          <div class="divider"></div>
          <div class="field"><label>Pixel size (µm/px)</label><input type="number" id="r-pixel-size" value="0.5" step="0.01" min="0.01"></div>
          <div class="info-box" style="margin-top:8px">The current brightfield-nuclei InstanSeg engine is reused unchanged. Preprocessing creates an auditable intermediate PNG; the source image is never overwritten.</div>
        </div>

        <div class="card">
          <div class="card-title">AEC marker classification</div>
          <div class="info-box warn" style="margin-bottom:12px"><b>AEC thresholds are required.</b> These are not DAB thresholds and no old pipeline cutoff is inserted. Until validated against manual marker labels, co-expression classes are exploratory even when nucleus segmentation has expert-mask validation.</div>
          <div class="row2">
            <div>
              <div class="field"><label>Marker A label</label><input id="r-label-a" value="CD8"></div>
              <div class="field"><label>Marker A AEC OD threshold</label><input type="number" id="r-threshold-a" placeholder="required" min="0" step="0.01"></div>
              <div class="field"><label>Marker A compartment</label><select id="r-compartment-a"><option value="ring" selected>Membrane ring</option><option value="nucleus">Nucleus</option></select></div>
            </div>
            <div>
              <div class="field"><label>Marker B label</label><input id="r-label-b" value="FOXP3"></div>
              <div class="field"><label>Marker B AEC OD threshold</label><input type="number" id="r-threshold-b" placeholder="required" min="0" step="0.01"></div>
              <div class="field"><label>Marker B compartment</label><select id="r-compartment-b"><option value="nucleus" selected>Nucleus</option><option value="ring">Membrane ring</option></select></div>
            </div>
          </div>
          <div class="field"><label>Membrane-ring expansion (µm)</label><input type="number" id="r-expansion" value="2.0" min="0.5" step="0.5"></div>
        </div>
        <div class="card">
          <div class="card-title">Correspondence certification (required)</div>
          <div class="info-box warn" style="margin-bottom:12px"><b>Equal image dimensions do NOT verify shared cell coordinates.</b> A grossly non-corresponding tile with matching dimensions can produce a false double-positive signal (ihc.md §21.6, tile Case2_S3_1_1). Co-expression statistics are <b>fail-closed</b> until you confirm the three captures are the same physical section in shared coordinates. Verify visually (overlay/landmarks) before certifying. A hematoxylin cross-correlation diagnostic is reported per tile to inform this — it is advisory, not an automatic pass.</div>
          <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer">
            <input type="checkbox" id="r-correspondence-certified">
            I have verified these images share cell coordinates (same physical section) and certify correspondence.
          </label>
        </div>
        <div class="action-row">
          <button class="btn primary" onclick="restainedRun()">Run restained co-expression</button>
        </div>
      </div>
    </div>

    <div class="view" id="restained-running" style="flex-direction:column;min-height:0">
      <div class="progress-area">
        <div class="progress-label"><span class="plabel" id="restained-progress-label">Starting…</span><span class="ppct" id="restained-pct">0%</span></div>
        <div class="progress-track"><div class="progress-fill" id="restained-progress-fill"></div></div>
      </div>
      <div class="terminal-wrap">
        <div class="terminal-header"><div class="terminal-dots"><span></span><span></span><span></span></div><div class="terminal-title">ACTIVITY LOG</div><button class="btn sm" style="margin-left:auto" onclick="copyLog('restained-log')">Copy</button></div>
        <div class="terminal-body" id="restained-log"></div>
      </div>
    </div>

    <div class="view" id="restained-results">
      <div class="scroll">
        <div id="restained-summary"></div>
        <div id="restained-result-cards"></div>
        <div class="action-row" id="restained-actions"></div>
      </div>
    </div>`;
  document.querySelector('.pages').appendChild(page);

  let mode = 'single';

  function fileField(id, label) {
    return `<div class="field"><label>${label}</label><div class="input-with-btn"><input id="${id}" readonly placeholder="select a file…"><button class="btn sm" onclick="restainedPickFile('${id}')">Browse</button></div></div>`;
  }
  function folderField(id, label) {
    return `<div class="field"><label>${label}</label><div class="input-with-btn"><input id="${id}" placeholder="select a folder…"><button class="btn sm" onclick="restainedPickFolder('${id}')">Browse</button></div></div>`;
  }

  window.restainedSetMode = function (next) {
    mode = next;
    document.getElementById('r-mode-single').classList.toggle('active', next === 'single');
    document.getElementById('r-mode-batch').classList.toggle('active', next === 'batch');
    document.getElementById('r-single').style.display = next === 'single' ? '' : 'none';
    document.getElementById('r-batch').style.display = next === 'batch' ? '' : 'none';
  };

  window.restainedPickFile = async function (id) {
    const path = await window.pywebview.api.pick_file();
    if (path) document.getElementById(id).value = path;
  };
  window.restainedPickFolder = async function (id) {
    const path = await window.pywebview.api.pick_folder();
    if (path) document.getElementById(id).value = path;
  };

  window.restainedPreview = async function () {
    const result = await window.pywebview.api.preview_restained_bundles(
      value('r-folder'), value('r-h-token'), value('r-a-token'), value('r-b-token'), value('r-mask-folder') || null);
    const box = document.getElementById('r-preview');
    if (!result.ok) { box.textContent = 'Preview failed: ' + result.error; return; }
    box.innerHTML = `<b>${result.bundles.length} complete bundle(s)</b> · ${result.incomplete.length} incomplete` +
      (result.bundles.length ? '<br>' + result.bundles.slice(0, 8).map(x => esc(x.sample_id)).join(', ') : '');
  };

  function value(id) { return document.getElementById(id).value.trim(); }
  function numberValue(id) {
    const raw = value(id);
    return raw === '' ? null : Number(raw);
  }

  window.restainedRun = async function () {
    const thresholdA = numberValue('r-threshold-a');
    const thresholdB = numberValue('r-threshold-b');
    if (thresholdA === null || thresholdB === null || thresholdA < 0 || thresholdB < 0) {
      showToast('Enter valid AEC OD thresholds for both markers'); return;
    }
    const correspondenceCertified = !!(document.getElementById('r-correspondence-certified') || {}).checked;
    if (!correspondenceCertified) {
      showToast('Certify correspondence first — equal dimensions do not verify shared coordinates (§21.6)');
      return;
    }
    const config = {
      mode, output_dir: value('r-output'), pixel_size_um: numberValue('r-pixel-size'),
      preprocess_hematoxylin: isOn('r-preprocess'),
      label_a: value('r-label-a') || 'CD8', label_b: value('r-label-b') || 'FOXP3',
      threshold_a: thresholdA, threshold_b: thresholdB,
      compartment_a: value('r-compartment-a'), compartment_b: value('r-compartment-b'),
      cell_expansion_um: numberValue('r-expansion') || 2.0,
      correspondence_certified: correspondenceCertified,
    };
    if (!config.output_dir || !config.pixel_size_um || config.pixel_size_um <= 0) {
      showToast('Choose an output folder and valid pixel size'); return;
    }
    if (mode === 'single') {
      Object.assign(config, {hematoxylin_image: value('r-h'), marker_a_image: value('r-a'), marker_b_image: value('r-b'), reference_mask: value('r-mask') || null});
      if (!config.hematoxylin_image || !config.marker_a_image || !config.marker_b_image) {
        showToast('Select hematoxylin and both AEC marker images'); return;
      }
    } else {
      Object.assign(config, {input_folder: value('r-folder'), hematoxylin_token: value('r-h-token'), marker_a_token: value('r-a-token'), marker_b_token: value('r-b-token'), reference_mask_folder: value('r-mask-folder') || null});
      if (!config.input_folder) { showToast('Choose the batch image folder'); return; }
    }
    activeMode = 'restained';
    showView('restained', 'running');
    document.getElementById('restained-log').innerHTML = '';
    setProgress('restained', 0, 'Starting same-section workflow…');
    appendLog('restained-log', 'Registration disabled by design · exact dimension check required', 'info');
    appendLog('restained-log', `AEC thresholds: ${config.label_a} ${thresholdA.toFixed(3)} · ${config.label_b} ${thresholdB.toFixed(3)} OD`, 'warn');
    const accepted = await window.pywebview.api.run_restained_coexpression(config);
    if (!accepted || !accepted.ok) appendLog('restained-log', 'Could not start workflow', 'error');
  };

  function fmt(value, digits = 3) { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'; }

  function showResults(data) {
    const cohort = data.cohort_coexpression || {};
    document.getElementById('restained-summary').innerHTML = `
      <div class="info-box amber" style="margin-bottom:12px"><b>Interpretation boundary:</b> segmentation can be validated against an expert nuclear mask. Marker positivity and co-expression remain exploratory until the entered AEC thresholds are independently validated.</div>
      <div class="metrics-row c4" style="margin-bottom:14px">
        <div class="metric"><div class="metric-label">Cells</div><div class="metric-value">${(cohort.total_cells || 0).toLocaleString()}</div><div class="metric-sub">${data.n_scored ?? data.n_samples ?? 0} scored${data.n_blocked_uncertified ? ` · ${data.n_blocked_uncertified} blocked` : ''}</div></div>
        <div class="metric"><div class="metric-label">Double positive</div><div class="metric-value">${fmt(cohort.double_positive_pct,1)}%</div><div class="metric-sub">${cohort.double_positive || 0} cells</div></div>
        <div class="metric"><div class="metric-label">Enrichment</div><div class="metric-value">${fmt(cohort.double_positive_enrichment,2)}×</div><div class="metric-sub">vs independence</div></div>
        <div class="metric"><div class="metric-label">Fisher p</div><div class="metric-value">${fmt(cohort.fisher_p_value,4)}</div><div class="metric-sub">cohort table</div></div>
      </div>`;
    document.getElementById('restained-result-cards').innerHTML = (data.results || []).map(result => {
      const corr = result.correspondence || {};
      if (corr.certified === false) {
        const d = corr.diagnostic || {};
        const mc = (typeof d.min_corr === 'number') ? d.min_corr.toFixed(3) : '—';
        return `<div class="card" style="border-color:#dc2626">
          <div class="card-title">${esc(result.sample_id)} <span class="badge red">BLOCKED — correspondence not certified</span></div>
          <div class="info-box" style="border-color:#dc2626;color:#991b1b;background:#fef2f2;margin-top:8px">
            No co-expression statistics computed (fail-closed). ${esc(corr.reason || '')}
            <div class="muted" style="margin-top:6px">Hematoxylin cross-correlation diagnostic (advisory): min NCC = ${mc}.</div>
          </div>
        </div>`;
      }
      const c = result.coexpression || {}, v = result.segmentation?.ground_truth_validation;
      const a = result.markers?.a || {}, b = result.markers?.b || {};
      const validation = v ? `<div class="metrics-row c4" style="margin-top:10px">
        <div class="metric"><div class="metric-label">Segmentation F1</div><div class="metric-value">${fmt(v.f1)}</div></div>
        <div class="metric"><div class="metric-label">Precision</div><div class="metric-value">${fmt(v.precision)}</div></div>
        <div class="metric"><div class="metric-label">Recall</div><div class="metric-value">${fmt(v.recall)}</div></div>
        <div class="metric"><div class="metric-label">Pixel Dice</div><div class="metric-value">${fmt(v.pixel_dice)}</div></div>
      </div>` : '<div class="info-box" style="margin-top:10px">No expert nuclear mask supplied for this section.</div>';
      const overlay = result.artifacts?.overlay;
      return `<div class="card">
        <div class="card-title">${esc(result.sample_id)} <span class="badge gray">same section · no registration</span></div>
        <div class="metrics-row c4">
          <div class="metric"><div class="metric-label">${esc(a.label)} only</div><div class="metric-value">${c.marker_a_only || 0}</div></div>
          <div class="metric"><div class="metric-label">${esc(b.label)} only</div><div class="metric-value">${c.marker_b_only || 0}</div></div>
          <div class="metric"><div class="metric-label">Double +</div><div class="metric-value">${c.double_positive || 0}</div></div>
          <div class="metric"><div class="metric-label">BH q</div><div class="metric-value">${fmt(c.fisher_q_value_bh,4)}</div></div>
        </div>
        ${validation}
        <div class="muted" style="font-size:11px;margin:10px 0 4px">Overlay: gray neither · vivid red ${esc(a.label)} only · vivid blue ${esc(b.label)} only · magenta double positive</div>
        ${overlay ? `<img src="file://${esc(overlay)}?t=${Date.now()}" style="width:100%;border:1px solid var(--border);border-radius:var(--r-sm)">` : ''}
        <div class="info-box" style="margin-top:10px">Segmentation: ${esc(result.segmentation?.preprocessing?.method || '—')} · ${esc(a.label)} ${fmt(a.threshold_od)} OD (${esc(a.compartment)}) · ${esc(b.label)} ${fmt(b.threshold_od)} OD (${esc(b.compartment)})</div>
      </div>`;
    }).join('');
    document.getElementById('restained-actions').innerHTML = `
      <button class="btn primary" onclick="window.pywebview.api.open_folder('${esc(data.output_dir)}')">Open output folder</button>
      <button class="btn" onclick="showView('restained','config')">New analysis</button>`;
    showView('restained', 'results');
  }

  const previousHandler = window.onPipelineEvent;
  window.onPipelineEvent = function (event) {
    const type = event.type, data = event.data || {};
    if (type === 'restained_log') {
      appendLog('restained-log', data.msg, data.level || 'normal'); return;
    }
    if (type === 'restained_progress') {
      setProgress('restained', data.pct || 0, data.msg || 'Running…');
      if (data.msg) appendLog('restained-log', data.msg, 'info');
      return;
    }
    if (type === 'restained_complete') {
      setProgress('restained', 100, 'Complete');
      appendLog('restained-log', 'Restained co-expression complete ✓', 'ok');
      showResults(data); return;
    }
    if (type === 'restained_failed') {
      appendLog('restained-log', data.msg || 'Workflow failed', 'error');
      if (data.stderr) appendLog('restained-log', data.stderr, 'error');
      showToast('Restained co-expression failed'); return;
    }
    previousHandler(event);
  };

  window.addEventListener('pywebviewready', async () => {
    const home = await window.pywebview.api.get_home();
    if (!value('r-output')) document.getElementById('r-output').value = home + '/Desktop/ihc_restained_results';
  });
})();
