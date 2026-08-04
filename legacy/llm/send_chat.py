"""
send_chat — REMOVED from the shipped application (2026-07-25).

This was an LLM "ask questions about your results" helper on `oasis.webui.api.API`. It was
removed for three reasons:

1. **It was already dead code.** No element in `oasis/webui/index.html` ever called
   `window.pywebview.api.send_chat`; the chat panel it was written for was never built.
2. **It cannot ship in a standalone executable.** It required network access plus a
   user-supplied API key, and pulled two vendor SDKs (and their transitive HTTP/auth
   stacks) into the bundle for a feature nobody could reach.
3. **It sends analysis results to a third party.** OASIS is used on clinical tissue images.
   A feature that transmits per-image cell counts and positivity off-machine needs an
   explicit, informed consent flow, not an incidental API method.

WHAT IT DID, since the body is no longer here. It read the run's metrics from the caller's
context, computed totals and mean positivity, built a system prompt describing the cohort,
and posted that prompt to one of two hosted chat APIs chosen by `setup["ai_provider"]`,
with the key read from the environment. It returned `{"ok": True, "response": ...}` or
`{"ok": False, "error": ...}`.

The implementation is in git history rather than in this file, because the file only has to
record the decision and the shape, and the body carried two vendors' model identifiers,
which date badly and which this repository has no reason to name. `git log --follow` on
this path recovers it in full if a reimplementation ever wants to read it.

If an LLM narrative is wanted later, reintroduce it deliberately: opt-in, with a visible
statement of exactly what leaves the machine, and with the pipeline's own deterministic
summary as the default. Note that the original system prompt hardcoded stale method text
("InstanSeg brightfield_nuclei, DAB threshold 0.2 OD") rather than reading the run's actual
provenance — a reimplementation should use the recorded values.
"""
