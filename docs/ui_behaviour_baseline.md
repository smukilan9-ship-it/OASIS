# UI behaviour baseline — captured at tag `pre-ui-rebuild`

Generated from `oasis/webui/index.html` before the page-by-page rebuild.
Every control below exists TODAY. During the rebuild each row must end as one of:
**kept** (same behaviour), **changed** (deliberately, note why), or **dropped**
(note why). A control that simply disappears is a regression, not a simplification.

A blank handler means the control is decorative or driven by JS elsewhere.

## page-quant

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| Stop | `quant-stop-btn` | `stopPipeline()` | |
| ← New analysis | `quant-new-btn` | `showView('quant','config')` | |
| 1 Inputs | `` | `quantWizardGoTo(1)` | |
| 2 Calibration | `` | `quantWizardGoTo(2)` | |
| 3 Classification | `` | `quantWizardGoTo(3)` | |
| 4 Review & run | `` | `quantWizardGoTo(4)` | |
| Single image | `q-mode-single-btn` | `quantSetMode('single')` | |
| Batch (folder of images) | `q-mode-batch-btn` | `quantSetMode('batch')` | |
| Browse | `` | `quantPickSingleImage()` | |
| Browse | `` | `pickInto('q-input', onQuantFolderPicked)` | |
| Browse | `` | `pickInto('q-output', null)` | |
| Select scale image… | `` | `quantPickScaleImage()` | |
| Preview matched pairs | `` | `previewQuantScaleMatches()` | |
| Add | `` | `addQuantOverride()` | |
| + CD8 (0.20) | `` | `addQuantBiomarkerPreset('CD8')` | |
| + TIM-3 (0.10) | `` | `addQuantBiomarkerPreset('TIM-3')` | |
| Add | `` | `addQuantBiomarker()` | |
| ← Back | `quant-wizard-back` | `quantWizardGoTo(quantWizardStep - 1)` | |
| Next → | `quant-wizard-next` | `quantWizardGoTo(quantWizardStep + 1)` | |
| Run Analysis | `quant-run-btn` | `runQuantAnalysis()` | |
| Show activity log | `` | `toggleTechnicalLog('quant-tech-log',this)` | |
| Copy | `` | `copyLog('quant-log')` | |
| Clear overrides | `` | `qrResetOverrides()` | |
| Continue — generate results | `` | `qrApply()` | |
| Export CSV | `` | `exportQuantCSV()` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input | text | `q-single-image` | `` | |
| input | text | `q-input` | `` | |
| input | text | `q-output` | `` | |
| input | radio | `q-px` | `onQuantPxChange()` | |
| input | number | `q-px-x` | `quantPxTouched()` | |
| input | radio | `q-px` | `onQuantPxChange()` | |
| input | radio | `q-px` | `onQuantPxChange()` | |
| input | radio | `q-px` | `onQuantPxChange()` | |
| input | number | `q-px-scalematch-default` | `quantPxTouched()` | |
| select |  | `q-ov-file` | `` | |
| input | number | `q-ov-px` | `` | |
| select |  | `q-stain` | `quantStainChanged()` | |
| input |  | `q-stain-other` | `` | |
| input | range | `q-thresh` | `document.getElementById('q-thresh-displa` | |
| select |  | `q-calib-profile` | `` | |
| select |  | `q-classifier` | `quantClassifierChanged()` | |
| input | text | `q-bm-name` | `` | |
| input | number | `q-bm-thr` | `` | |
| input | range | `qr-cohort` | `qrCohortChanged(this.value/100)` | |

## page-spatial

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| Stop | `spatial-stop-btn` | `stopPipeline()` | |
| ← New analysis | `spatial-new-btn` | `showView('spatial','config')` | |
| 1 Inputs | `` | `spatialWizardGoTo(1)` | |
| 2 Certify registration | `` | `spatialWizardGoTo(2)` | |
| 3 Settings | `` | `spatialWizardGoTo(3)` | |
| 4 Validation check & run | `` | `spatialWizardGoTo(4)` | |
| Single Pair | `spatial-mode-single-btn` | `spatialAssocSetMode('single')` | |
| Batch | `spatial-mode-batch-btn` | `spatialAssocSetMode('batch')` | |
| Select scale image… | `spatial-session-scale-btn` | `spatialAssocPickSessionScale()` | |
| Browse | `` | `spatialAssocPickImage('a')` | |
| Select scale image… | `` | `spatialAssocPickOverrideScale('a')` | |
| × Reset to default | `` | `spatialAssocClearOverride('a')` | |
| Browse | `` | `spatialAssocPickImage('b')` | |
| Select scale image… | `` | `spatialAssocPickOverrideScale('b')` | |
| × Reset to default | `` | `spatialAssocClearOverride('b')` | |
| Two Folders | `spatial-fmode-two-btn` | `spatialAssocSetFolderMode('two_folder')` | |
| Single Folder (auto-detect) | `spatial-fmode-single-btn` | `spatialAssocSetFolderMode('single_folder')` | |
| Browse | `` | `pickInto('spatial-folder-a', null)` | |
| Browse | `` | `pickInto('spatial-folder-b', null)` | |
| Browse | `` | `pickInto('spatial-folder-single', null)` | |
| Preview Matches | `spatial-preview-btn` | `spatialAssocPreview()` | |
| Load images for landmarking | `spatial-cert-load` | `spatialCertLoad()` | |
| Show colour | `spatial-cert-colour` | `spatialCertToggleColour()` | |
| LoFTR · automatic (default) | `cert-mode-loftr-tab` | `spatialCertSetMode('loftr')` | |
| Manual landmarks | `cert-mode-landmark-tab` | `spatialCertSetMode('landmark')` | |
| ⛶ Fullscreen | `spatial-cert-fs-btn` | `spatialCertToggleFullscreen()` | |
| Reset zoom | `` | `spatialCertZoomFit()` | |
| Draw certification ROI | `spatial-cert-roi-draw` | `spatialCertRoiToggle()` | |
| Finish ROI | `` | `spatialCertRoiFinish()` | |
| Clear ROI | `` | `spatialCertRoiClear()` | |
| ✦ Auto-certify | `spatial-loftr-auto` | `loftrAutoFind()` | |
| ✎ Polygon | `spatial-loftr-draw` | `loftrRoiToggle()` | |
| ✐ Freehand | `spatial-loftr-freehand` | `loftrFreehandToggle()` | |
| Finish region | `` | `loftrRoiFinish()` | |
| Clear all | `` | `loftrRoiClearAll()` | |
| Certify drawn regions | `spatial-loftr-certify` | `loftrRoiCertifyAll()` | |
| Undo | `` | `spatialCertUndo()` | |
| Clear | `` | `spatialCertClear()` | |
| Fit transform & certify | `` | `spatialCertRun()` | |
| Browse | `` | `pickInto('spatial-output', null)` | |
| Validate 75 µm bandwidth | `spatial-bw-btn` | `spatialBandwidthCheck()` | |
| ← Back | `spatial-wizard-back` | `spatialWizardGoTo(spatialWizardStep - 1)` | |
| Next → | `spatial-wizard-next` | `spatialWizardGoTo(spatialWizardStep + 1)` | |
| Run Certified Spatial Analysis | `spatial-run-btn` | `spatialAssocRun()` | |
| Show activity log | `` | `toggleTechnicalLog('spatial-tech-log',this)` | |
| Copy | `` | `copyLog('spatial-log')` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input | radio | `spatial-session-px` | `spatialAssocSetSessionPxMode('manual')` | |
| input | radio | `spatial-session-px` | `spatialAssocSetSessionPxMode('scale')` | |
| input | number | `spatial-session-px-val` | `spatialCertInvalidate()` | |
| input | text | `spatial-label-a` | `` | |
| input | text | `spatial-image-a` | `` | |
| input | radio | `spatial-ov-mode-a` | `spatialAssocSetOverrideMode('a','manual'` | |
| input | radio | `spatial-ov-mode-a` | `spatialAssocSetOverrideMode('a','scale')` | |
| input | number | `spatial-ov-px-a` | `spatialOvManualChanged('a')` | |
| input | text | `spatial-label-b` | `` | |
| input | text | `spatial-image-b` | `` | |
| input | radio | `spatial-ov-mode-b` | `spatialAssocSetOverrideMode('b','manual'` | |
| input | radio | `spatial-ov-mode-b` | `spatialAssocSetOverrideMode('b','scale')` | |
| input | number | `spatial-ov-px-b` | `spatialOvManualChanged('b')` | |
| input | text | `spatial-folder-a` | `` | |
| input | text | `spatial-folder-b` | `` | |
| input | text | `spatial-folder-single` | `` | |
| select |  | `spatial-cert-pair` | `spatialCertSelectPair()` | |
| input | number | `spatial-loftr-regionum` | `` | |
| input | number | `spatial-loftr-tol` | `` | |
| input | checkbox | `spatial-loftr-showcorr` | `spatialCertRender()` | |
| input | number | `spatial-max-dist` | `` | |
| input | number | `spatial-thresh-a` | `` | |
| input | number | `spatial-thresh-b` | `` | |
| select |  | `spatial-calib-a` | `` | |
| select |  | `spatial-calib-b` | `` | |
| input | text | `spatial-output` | `` | |

## page-settings

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| Save settings | `` | `saveSetup()` | |
| Browse | `` | `pickInto('s-model', null)` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input | text | `s-microscope` | `` | |
| input | text | `s-camera` | `` | |
| input | text | `s-scanner` | `` | |
| input | number | `s-px-x` | `` | |
| input | number | `s-px-y` | `` | |
| select |  | `s-objective` | `` | |
| input | text | `s-model` | `` | |
| select |  | `s-device` | `` | |
| input | number | `s-threads` | `` | |

## page-classifier

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| + Add slide | `` | `clsAddImage()` | |
| Segment & label → | `cls-seg-btn` | `clsSegment()` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input |  | `cls-marker` | `` | |
| select |  | `cls-kind` | `` | |
| input | number | `cls-px` | `` | |

## page-calibrate

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| + Add image | `` | `calibAddImage()` | |
| Segment & label → | `calib-seg-btn` | `calibSegment()` | |
| + | `` | `calibZoom(1.25)` | |
| − | `` | `calibZoom(0.8)` | |
| View: Normalized | `calib-viewbtn` | `calibCycleView()` | |
| Fit → | `` | `calib.mode === 'classifier' ? clsFit() : calib` | |
| Cancel | `` | `calibBack()` | |
| Save for this marker | `calib-save-btn` | `calibSave()` | |
| ← Re-label | `` | `calibRelabel()` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input |  | `calib-marker` | `` | |
| input | number | `calib-px` | `` | |

## page-validation

### Buttons

| label | id | onclick | outcome |
|---|---|---|---|
| ↻ Refresh | `` | `loadValidations()` | |
| Change root | `` | `changeValidationDataDir()` | |
| Reset | `` | `lightboxFit()` | |
| ✕ Close | `` | `closeLightbox()` | |
| Download | `` | `showDatasetHelp('${d.name}')` | |
| Run anyway | `` | `runValidation('${v.id}', true)` | |
| Run validation | `` | `runValidation('${v.id}')` | |
| Log | `` | `toggleVLog('${v.id}')` | |
| Open report | `` | `openValidationReport('${v.id}')` | |
| Remove | `` | `calibRemoveImage(${i})` | |
| Remove | `` | `clsRemoveImage(${i})` | |
| Save classifier | `` | `clsSave()` | |
| ◀ | `` | `calibGoImage(${calib.cur-1})` | |
| ▶ | `` | `calibGoImage(${calib.cur+1})` | |
| Manual | `` | `quantApplyScaleManual(${i})` | |
| Use global | `` | `quantUseGlobalForScale(${i})` | |
| Remove | `` | `removeQuantOverride('${esc(f)}')` | |
| Remove | `` | `removeQuantBiomarker(${i})` | |
| use cohort | `` | `qrClearOverride(${i})` | |
| Open Dashboard | `` | `window.pywebview.api.open_file('${esc(r.dashbo` | |
| Open Excel | `` | `window.pywebview.api.open_file('${esc(r.excel_` | |
| View Overlays | `` | `window.pywebview.api.open_folder('${esc(r.over` | |
| New Analysis | `` | `showView('quant','config')` | |
| Remove | `` | `loftrRoiRemove(${i})` | |
| Open Output Folder | `` | `window.pywebview.api.open_folder('${esc(outDir` | |
| New Analysis | `` | `showView('spatial','config')` | |

### Inputs

| element | type | id/name | handler | outcome |
|---|---|---|---|---|
| input | number | `` | `calib.images[${i}].px=parseFloat(this.va` | |
| input | number | `` | `cls.images[${i}].px=parseFloat(this.valu` | |
| input | number | `` | `` | |
| input | number | `` | `qrOverride(${i}, this.value)` | |

