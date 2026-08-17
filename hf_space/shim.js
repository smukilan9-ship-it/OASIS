/*
 * shim.js — the pywebview bridge, in a browser.
 *
 * index.html talks to the backend as `window.pywebview.api.<method>(...)` and waits for a
 * `pywebviewready` event before it starts. Neither exists in a browser, so this script is
 * injected into <head> by hf_space/app.py and supplies both over HTTP. It is the same idea as
 * the shim in serve.py, with one addition that matters on a Space.
 *
 * BROWSE. The UI's Browse buttons call `pick_folder()` / `pick_file()`, which open a native
 * dialog on the desktop and return a path the user then sees in the text box. On a Space
 * there is no native dialog and, more to the point, no access to the visitor's disk — in
 * serve.py those calls return null and Browse quietly does nothing. Here they are
 * intercepted before the fetch: they open the browser's own file picker, upload what was
 * chosen, and return the path it landed on server-side. The UI puts that in its text box
 * exactly as it would a native dialog's answer, so Browse works and index.html is untouched.
 */
(function () {
  'use strict';

  // ── which session this page belongs to ─────────────────────────────────────
  //
  // NOT A COOKIE. On huggingface.co the Space is shown inside an iframe served from
  // <space>.hf.space, so a cookie set by this app is a THIRD-PARTY cookie — which Safari
  // blocks outright and other browsers are phasing out. The server would then mint a fresh
  // session for every single request: the upload lands in one, the run happens in a second,
  // and the event poll listens on a third, so the page waits forever on a job that already
  // finished. That is exactly what happened on the first deploy, and it only reproduces in
  // the iframe — testing the .hf.space URL directly makes the cookie first-party and hides it.
  //
  // So the id is generated here and sent explicitly on every request. A header is subject to
  // no storage policy at all.
  var SESSION = (function () {
    var KEY = 'oasis_sid';
    var id = null;
    try { id = window.sessionStorage.getItem(KEY); } catch (e) {}
    if (!id) {
      id = (window.crypto && window.crypto.randomUUID)
        ? window.crypto.randomUUID().replace(/-/g, '').slice(0, 16)
        : String(Date.now()) + Math.random().toString(36).slice(2, 10);
      // sessionStorage is partitioned rather than blocked in an iframe, so this normally
      // survives a reload. When even that is denied the id stays in memory for the life of
      // the page, which still keeps one page's requests consistent with each other.
      try { window.sessionStorage.setItem(KEY, id); } catch (e) {}
    }
    return id;
  })();

  // ── the API bridge ─────────────────────────────────────────────────────────
  function call(name, args) {
    return fetch('/api/' + name, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-OASIS-Session': SESSION },
      body: JSON.stringify(args)
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (res && res.__error) { throw new Error(res.__error); }
      return res ? res.__result : undefined;
    });
  }

  var LOCAL = {
    pick_folder: function () { return pickAndUpload(true); },
    pick_file: function () { return pickAndUpload(false); }
  };

  window.pywebview = window.pywebview || {};
  window.pywebview.api = new Proxy({}, {
    get: function (_t, name) {
      if (Object.prototype.hasOwnProperty.call(LOCAL, name)) return LOCAL[name];
      return function () {
        return call(name, Array.prototype.slice.call(arguments));
      };
    }
  });

  // ── a run that produced nothing must say so ────────────────────────────────
  //
  // When every image fails to segment, the worker still exits 0 ("No results to
  // summarize"), so api.py takes its review branch and pushes a `review` event carrying an
  // empty image list. The review screen then paints its header and cohort slider with no
  // cards under them — a blank page where the histogram should be. It is indistinguishable
  // from a rendering bug, and it is what a visitor sees the moment they run out of ZeroGPU
  // quota, which on a public Space is often.
  //
  // Measured on this Space: segmentation raised "You have exceeded your ZeroGPU runs
  // limit", the log said so, and the screen still showed an empty review.
  //
  // The event is rewritten into the failure it actually was. The real fix belongs in
  // api.py — it should not report a run with no measured images as a review — but this is
  // the Space's own bridge and nothing in the app changes.
  function guardEmptyReview(handler) {
    return function (event) {
      if (event && event.type === 'review') {
        var imgs = (event.data && event.data.images) || [];
        if (!imgs.length) {
          return handler({ type: 'done', data: { ok: false, msg:
            'No image could be measured, so there is nothing to review. The activity log '
            + 'above says why — on this Space it is usually the ZeroGPU daily limit.' } });
        }
      }
      return handler(event);
    };
  }

  // ── the push channel ───────────────────────────────────────────────────────
  // The backend updates the UI by evaluating JS in the window (onPipelineEvent /
  // onValidationEvent / the restained events). Server-side those strings go into a
  // per-session buffer; this poll drains it and evaluates them in the page, which
  // reproduces the desktop push semantics without changing a line of index.html.
  var seen = 0;
  function poll() {
    fetch('/__events?since=' + seen, { headers: { 'X-OASIS-Session': SESSION } })
      .then(function (r) { return r.json(); }).then(function (d) {
      seen = d.n;
      (d.events || []).forEach(function (js) {
        try { (0, eval)(js); } catch (e) { console.error('event bridge', e); }
      });
      setTimeout(poll, 10);
    }).catch(function () { setTimeout(poll, 1000); });
  }
  poll();

  document.addEventListener('DOMContentLoaded', function () {
    // Wrap after the page has defined its handler, not before.
    if (typeof window.onPipelineEvent === 'function') {
      window.onPipelineEvent = guardEmptyReview(window.onPipelineEvent);
    }
    try { window.dispatchEvent(new Event('pywebviewready')); } catch (e) {}
  });

  // ── Browse -> upload ───────────────────────────────────────────────────────
  function pickAndUpload(isFolder) {
    return new Promise(function (resolve) {
      var input = document.createElement('input');
      input.type = 'file';
      if (isFolder) { input.webkitdirectory = true; input.multiple = true; }
      input.style.display = 'none';
      document.body.appendChild(input);

      var settled = false;
      function done(value) {
        if (settled) return;
        settled = true;
        if (input.parentNode) input.parentNode.removeChild(input);
        resolve(value);
      }

      input.addEventListener('cancel', function () { done(null); });
      input.addEventListener('change', function () {
        var files = Array.prototype.slice.call(input.files || []);
        if (!files.length) { done(null); return; }
        upload(files, isFolder).then(done).catch(function (e) {
          hideOverlay();
          done(null);
          setTimeout(function () { alert('Upload failed: ' + e.message); }, 0);
        });
      });
      input.click();
    });
  }

  function upload(files, isFolder) {
    var name = isFolder
      ? ((files[0].webkitRelativePath || 'upload').split('/')[0] || 'upload')
      : files[0].name;

    var form = new FormData();
    form.append('kind', isFolder ? 'folder' : 'file');
    form.append('name', name);
    var bytes = 0;
    files.forEach(function (f) {
      bytes += f.size;
      // The third argument carries the path INSIDE the chosen folder, so a nested
      // layout survives the round trip. The server treats it as untrusted.
      form.append('files', f, f.webkitRelativePath || f.name);
    });

    showOverlay(files.length, bytes);
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/__upload');
      xhr.setRequestHeader('X-OASIS-Session', SESSION);
      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) setOverlayProgress(e.loaded / e.total);
      };
      xhr.onload = function () {
        hideOverlay();
        var res;
        try { res = JSON.parse(xhr.responseText || '{}'); }
        catch (e) { reject(new Error('bad response from server')); return; }
        if (res.__error) { reject(new Error(res.__error)); return; }
        resolve(res.path);
      };
      xhr.onerror = function () { hideOverlay(); reject(new Error('network error')); };
      xhr.send(form);
    });
  }

  // ── upload progress ────────────────────────────────────────────────────────
  // Uploading a folder of slides takes long enough that a silent Browse button reads as
  // broken. Built here rather than in index.html so the desktop app carries none of it.
  var overlay = null, bar = null, label = null;

  function showOverlay(count, bytes) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.setAttribute('role', 'status');
      overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;display:flex;' +
        'align-items:center;justify-content:center;background:rgba(15,18,22,.45);' +
        'backdrop-filter:blur(2px);font:13px/1.5 system-ui,-apple-system,sans-serif';
      var card = document.createElement('div');
      card.style.cssText = 'min-width:290px;padding:20px 22px;border-radius:10px;' +
        'background:var(--bg1,#fff);color:var(--text1,#12161c);' +
        'box-shadow:0 12px 40px rgba(0,0,0,.28)';
      label = document.createElement('div');
      label.style.cssText = 'margin-bottom:12px;font-weight:500';
      var track = document.createElement('div');
      track.style.cssText = 'height:5px;border-radius:3px;background:var(--bg3,#e6e9ee);' +
        'overflow:hidden';
      bar = document.createElement('div');
      bar.style.cssText = 'height:100%;width:0%;border-radius:3px;' +
        'background:var(--accent,#2f6fed);transition:width .15s ease';
      track.appendChild(bar);
      card.appendChild(label);
      card.appendChild(track);
      overlay.appendChild(card);
      document.body.appendChild(overlay);
    }
    label.textContent = 'Uploading ' + count + (count === 1 ? ' file' : ' files') +
      ' (' + humanBytes(bytes) + ')…';
    bar.style.width = '0%';
    overlay.style.display = 'flex';
  }

  function setOverlayProgress(fraction) {
    if (bar) bar.style.width = Math.round(fraction * 100) + '%';
  }

  function hideOverlay() {
    if (overlay) overlay.style.display = 'none';
  }

  function humanBytes(n) {
    var units = ['B', 'KB', 'MB', 'GB'], i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(1)) + ' ' + units[i];
  }
})();
