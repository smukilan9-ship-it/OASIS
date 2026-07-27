# Wiring audit — every API method vs every UI call site

Captured at tag `pre-ui-rebuild`, before the page-by-page rebuild.
Method list: `^    def <name>(self` in `oasis/webui/api.py` (+ `restained_api.attach_restained_api`).
Call sites: `pywebview.api.<name>` in `oasis/webui/index.html` and `restained_coexpression.js`.

**56 public API methods · 41 unique UI call sites · 0 UI calls with no backend · 15 backend
methods the UI never calls _through_ `pywebview.api`.**

That last count is an upper bound, not a verdict: the API also pushes results to the UI as
events (`self._emit`), so a method can be fully wired and still never appear in a call-site
grep. `get_review_data` is exactly that case and is listed under A, not B.

Zero missing backends is the good news: nothing in the interface calls something that
does not exist. The 15 below are the real finding, and they are *not* all dead code —
three distinct things are mixed together, and they need opposite treatment.

## A. Live, called from outside the UI — keep, do not touch

| method | called by |
|---|---|
| `set_window` | `app.py`, `serve.py` (host wiring) |
| `get_home` | `oasis/webui/restained_coexpression.js` |
| `certify_expert_landmarks` | validation harnesses; the UI import button was deliberately removed (see the note at `index.html` ~4850) |
| `get_review_data` | `api.py:677` — pushed to the UI as a `review` event after segmentation, not fetched by the UI. A `pywebview.api.*` grep cannot see server-push endpoints; this one is fully wired and drives the threshold-review screen. |

## B. Features that exist in the backend but were never given a control — WIRE THESE

These are the ones worth acting on. Each is finished, tested backend work that a user
cannot reach.

| method | what it does | rebuild action |
|---|---|---|
| `preview_threshold` | what a candidate cutoff would call positive, writing nothing | **Quant.** The backend for a live cutoff preview, already written and never surfaced. The review screen currently re-renders from cached per-cell OD values instead. |
| `delete_classifier` | removes a saved classifier | **Classifier.** Tab can save but not delete. |
| `delete_calibration` | removes a saved calibration | Folds into Classifier when Calibrate retires. |
| `auto_certify_regions` | tile the tissue and certify each region automatically | **Spatial.** Confirm whether `certify_spatial_auto` superseded it; if so it belongs in D, not here. |
| `get_spatial_association_results` | reload results incl. null stats + QC overlay paths | **Spatial.** Reopening a finished run currently has no path. |

## C. Superseded by the Classifier tab — retire with Calibrate

| method | note |
|---|---|
| `calibration_fit` | fits membrane cutoffs from hand labels; `classifier.py` covers `kind="membrane"` with a stricter leave-one-image-out estimate |

## D. Small helpers and stale state — decide during the rebuild

| method | note |
|---|---|
| `get_standard_pixel_size` | one-line dict lookup; belongs in the single pixel-size resolver |
| `is_first_run` | no first-run flow exists any more |
| `get_experiments` / `save_experiments` | an experiments store nothing reads or writes |
| `suggest_moving_landmark` | `serial_registration.suggest_moving_landmark` is exercised by tests; the API wrapper is not called |

## Rule for the rebuild

Nothing in this table gets deleted silently. Each row ends as **wired**, **retired to
`legacy/` with a reason**, or **kept with a note saying who calls it**. An endpoint that
simply disappears is a lost feature, not a simplification.
