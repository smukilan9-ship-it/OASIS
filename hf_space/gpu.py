"""
gpu.py — the ZeroGPU seam.

ZeroGPU does not give the Space a GPU. It gives a GPU to a `@spaces.GPU`-decorated
function, for the duration of that call, in a process the scheduler owns. Three
consequences drive everything in this file:

  1. ARGUMENTS AND RETURN VALUES ARE PICKLED across a process boundary. A decorated
     function may not take a torch module, a file handle, or a callback. That rules out
     wrapping `segment.segment_labels(rgb, model, ...)` — it takes the loaded model — and
     rules out passing the `progress=` callbacks the pipeline uses for per-tile logging.

  2. EVERY CALL COSTS AN ACQUISITION (queue + fork + transfer). Wrapping the hot inner
     function is therefore the wrong instinct: `segment._forward` runs once per 512px tile,
     so a 4 Mpx image would pay ~16 acquisitions to do a few seconds of work. Wrap the
     coarsest call that is still GPU-bound.

  3. THE WRAPPED CALLS NEST. `certify_local_roi` calls `correspondences`, which calls
     `loftr_correspondences`, and `certify_local_roi` also calls `loftr_fle`, which calls
     `loftr_correspondences` again. Decorating all of them naively would have the inner
     ones try to acquire a second GPU from inside the first one's process. `_reentrant`
     below solves that: the flag is set inside the scheduler's process, so any nested
     wrapped call made from there sees it and runs the undecorated function directly, on
     the GPU that has already been acquired.

WHAT IS WRAPPED, and why each is the right altitude:

    segment.segment_image            one image — the unit run_pipeline.py:256 asks for
    sparse_matcher.correspondences   the dispatcher every certification path goes through
    loftr_matcher.loftr_correspondences
    loftr_matcher.certify_local_roi  outermost for an ROI certification, which loops inside
    loftr_matcher.loftr_fle          re-runs the whole matcher n_trials times

Patching is done by rebinding the module attribute. Every caller looks the name up at call
time — `run_pipeline.run_native_segmentation` does `from oasis.quant import segment as sg`
inside the function body, `sparse_matcher.correspondences` imports `loftr_correspondences`
inside its body — so the patch takes effect without any file in `oasis/` changing.

OUTSIDE ZEROGPU this module is inert. `spaces` is documented as effect-free off-platform,
and if it is not installed at all the patching is skipped entirely, so `python hf_space/app.py`
on a laptop runs exactly the code the desktop app runs.
"""
import functools
import os
import sys

# Set inside the scheduler's process by the decorated wrapper — see point 3 above.
_reentrant = False

_PATCHED = []


def on_zerogpu():
    """True when this really is a ZeroGPU Space, not a laptop with `spaces` installed."""
    return bool(os.environ.get("SPACES_ZERO_GPU"))


# ── whose quota the run is charged to ────────────────────────────────────────
#
# ZeroGPU rations GPU time per VISITOR, and identifies them by an `X-IP-Token` header the
# Hub attaches to each request. `spaces` reads it, at the moment a decorated function is
# called, from `gradio.context.LocalContext.request` — a ContextVar. With no request there
# it warns "Falling back to IP-based quotas".
#
# That fallback is what this app would get by default, and it is much worse than it sounds.
# OASIS runs every job on a detached thread (api.py starts one and returns immediately, so
# the UI can stream progress), and a ContextVar set while handling the request is not
# visible from a thread created inside it. So every segmentation would be charged
# anonymously — not to the visitor who asked for it, but to a single pool shared by
# everyone, which a couple of runs a day would exhaust for the whole Space.
#
# The fix is to carry the request across the thread boundary by hand: remember it while
# handling the API call, and re-install it on the job thread before any GPU work starts.
#
# PRECISION, STATED HONESTLY. What is remembered is the most recent API request, not the
# one belonging to a specific job — the job thread is created by api.py, two frames removed
# from the request, and there is nowhere to thread the value through without editing it.
# Since jobs are serialised (inproc.JOB_LOCK) and a run always starts from the API call
# immediately preceding it, the window in which two visitors could be confused is the few
# milliseconds between their two clicks. The cost of losing that race is that one visitor's
# GPU seconds are billed to the other; it cannot produce a wrong result, and when there is
# no request at all the behaviour is simply ZeroGPU's own IP-based fallback.
_request = None


_tokened = 0


def tokened_requests():
    """How many requests since boot carried a ZeroGPU quota token.

    Zero means every run is being charged to the anonymous pool, whatever the visitor is
    signed in as — which is the difference between a Space that works and one that is rate
    limited after a couple of minutes a day.
    """
    return _tokened


def remember_request(request):
    """Record the request being handled, for the job thread to re-install."""
    global _request, _tokened
    if request is not None:
        _request = request
        try:
            if "x-ip-token" in request.headers:
                _tokened += 1
        except Exception:
            pass


def install_request():
    """Re-install the remembered request on this thread. Safe to call anywhere."""
    if _request is None:
        return False
    try:
        from gradio.context import LocalContext
        LocalContext.request.set(_request)
        return True
    except Exception:
        return False


def as_gradio_request(raw):
    """Wrap a Starlette/FastAPI request as the `gr.Request` `spaces` expects.

    `spaces.zero.client._get_headers` wants an object with a `.headers` that also carries a
    `__dict__`, which a bare dict does not have — so this hands back a real `gr.Request`
    rather than a lookalike.
    """
    try:
        import gradio as gr
        return gr.Request(request=raw)
    except Exception:
        return None


def _spaces():
    try:
        import spaces
        return spaces
    except Exception:
        return None


def _is_out_of_quota(exc):
    """True for the ZeroGPU refusals that mean "not now", as opposed to a real failure.

    `spaces` raises these as `gradio.Error` with a title like "ZeroGPU quota exceeded";
    matching on the message keeps this working if the class changes, and keeps a genuine
    segmentation bug from being quietly retried on the CPU and reported as a success.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in (
        "exceeded your zerogpu",       # daily allowance spent
        "zerogpu quota",
        "gpu task aborted",
        "queue timeout",               # the cluster is full right now
        "no gpu is currently available",
        "pending credits",
    ))


def _run_on_cpu(fn, args, kwargs):
    """Run `fn` here, on the CPU, with every route to CUDA closed.

    THE CUDA CALLS MUST NOT HAPPEN. This runs in the main process, and ZeroGPU forks its
    GPU workers from here — a CUDA context initialised in this process is inherited by
    every future worker and permanently breaks them (see the note above `install`). So the
    device is forced two ways, because the callees choose it two different ways:

      segment_image  takes an explicit `device=` (run_pipeline resolves it to "cuda" via
                     device.resolve_device, which sees ZeroGPU's emulated availability)
      the matchers   ask `torch.cuda.is_available()` themselves; loftr_matcher memoizes the
                     answer in a module global, so that has to be reset, not just re-asked

    Patching `torch.cuda.is_available` is process-wide, which is only safe because
    inproc.JOB_LOCK admits one job at a time — no GPU call can be in flight beside this.
    """
    import inspect
    import torch

    kwargs = dict(kwargs)
    try:
        if "device" in inspect.signature(fn).parameters:
            kwargs["device"] = "cpu"
    except (TypeError, ValueError):
        pass

    from oasis.spatial import loftr_matcher, sparse_matcher

    real_is_available = torch.cuda.is_available
    saved_device = getattr(loftr_matcher, "_DEVICE", None)
    torch.cuda.is_available = lambda: False
    loftr_matcher._DEVICE = torch.device("cpu")
    loftr_matcher._MATCHER.clear()
    loftr_matcher.clear_loftr_caches()
    sparse_matcher.clear_sparse_caches()
    try:
        return fn(*args, **kwargs)
    finally:
        torch.cuda.is_available = real_is_available
        # Back to "undecided" rather than to the stale value, so the next GPU-backed call
        # resolves the device again instead of staying pinned to the CPU for the session.
        loftr_matcher._DEVICE = None
        loftr_matcher._MATCHER.clear()
        loftr_matcher.clear_loftr_caches()
        sparse_matcher.clear_sparse_caches()
        del saved_device


def _wrap(fn, duration):
    """`fn` decorated with @spaces.GPU, made nest-safe and callback-safe.

    `duration` is a callable taking the same arguments as `fn` (the dynamic-duration form
    ZeroGPU supports), so a small crop asks for a small slice and keeps its place in the
    queue, while a big one asks for what it needs.
    """
    spaces = _spaces()
    if spaces is None:
        return fn

    @spaces.GPU(duration=duration)
    def _on_gpu(*args, **kwargs):
        global _reentrant
        _reentrant = True
        try:
            return fn(*args, **kwargs)
        finally:
            _reentrant = False

    @functools.wraps(fn)
    def _entry(*args, **kwargs):
        if _reentrant:
            return fn(*args, **kwargs)      # already holding a GPU; don't acquire another
        # A callback cannot be pickled across the process boundary. `progress` is
        # per-tile log decoration only — the pipeline's own per-image progress lines are
        # emitted by run_pipeline, outside this call — so dropping it costs log detail
        # and nothing else. Anything else unpicklable (an `rng`, say) is a signal we do
        # not understand the call, so it runs here rather than being silently altered.
        kwargs.pop("progress", None)
        if any(callable(v) for v in kwargs.values()):
            return fn(*args, **kwargs)
        try:
            return _on_gpu(*args, **kwargs)
        except Exception as exc:
            if not _is_out_of_quota(exc):
                raise
            # A visitor out of GPU seconds should get a slow answer, not no answer. The
            # ZeroGPU allowance is a couple of minutes a day, which one real slide can
            # spend, and before this the run simply died — the operator saw an empty
            # review screen and no reason for it. Printed, not swallowed: this goes to
            # the run's activity log, so the wall-clock has a stated cause and nobody
            # concludes the tool is just slow.
            print(f"  ZeroGPU unavailable ({exc}). Falling back to the CPU for this step "
                  f"— the result is the same, it just takes longer.")
            return _run_on_cpu(fn, args, kwargs)

    return _entry


# ── duration estimates ───────────────────────────────────────────────────────
#
# Too short and the scheduler kills the call mid-image; too long and the Space queues
# worse and burns more of the visitor's daily quota (2 min unauthenticated, 5 min for a
# free account). Both are estimated from the only thing known cheaply up front — how much
# image there is.

MIN_DURATION = 60
MAX_DURATION = 240


def _clamp(seconds):
    return int(max(MIN_DURATION, min(MAX_DURATION, seconds)))


def _segment_duration(image_path, model_dir=None, pixel_size_um=None, *a, **k):
    """~20 s per megapixel, which is many times slower than the measured rate on this
    hardware — the estimate is deliberately generous because overrunning kills the run."""
    try:
        from oasis.quant.segment import image_dimensions
        w, h = image_dimensions(image_path)
        return _clamp(30 + (w * h / 1e6) * 20)
    except Exception:
        return MAX_DURATION


def _pair_duration(ref_rgb=None, mov_rgb=None, *a, **k):
    try:
        px = max(getattr(ref_rgb, "size", 0), getattr(mov_rgb, "size", 0)) / 3.0
        return _clamp(45 + (px / 1e6) * 20)
    except Exception:
        return MAX_DURATION


def _cert_duration(ref_rgb=None, mov_rgb=None, *a, **k):
    """Certification loops the matcher inside one call, so it gets the ceiling."""
    return MAX_DURATION


# DO NOT PRELOAD THE MODEL ONTO CUDA AT STARTUP — measured, 2026-08-04, on the live Space.
#
# The ZeroGPU docs say models should be placed on `cuda` at module level, because outside a
# `@spaces.GPU` function torch runs in a CUDA emulation mode that makes the placement legal
# and cheap. That advice holds for the `diffusers`/`transformers` idiom it is written for
# (`pipe.to('cuda')`). It does NOT hold for `torch.jit.load(..., map_location="cuda")`,
# which is how the InstanSeg TorchScript bundle is loaded: that call goes past the emulation
# to the real CUDA runtime, which is not there yet in the main process.
#
# It does not merely fail. It fails and leaves CUDA PARTIALLY INITIALISED, and ZeroGPU forks
# its GPU workers from this process — so every worker for the life of the Space inherits the
# poisoned state. The first deploy showed exactly that:
#
#     [hf_space] InstanSeg preload skipped (AcceleratorError: CUDA error:
#                no CUDA-capable device is detected)          <- startup, looked harmless
#     ...
#     spaces/zero/torch/patching.py:417 in init
#       torch.Tensor([0]).cuda()
#     RuntimeError: No CUDA GPUs are available                <- every run, forever
#     Segmentation FAILED: Error: 'RuntimeError'
#
# The startup line is caught and printed as a skipped optimisation, which is what makes this
# nasty: the damage is silent and total, and it is done by the code that was trying to help.
# So the weights are loaded lazily INSIDE the decorated call, where a real GPU exists.
# `segment.load_model` caches on (path, device), so it happens once per worker.


def install():
    """Patch the GPU entry points. Returns the list of what was patched."""
    if _spaces() is None:
        print("[hf_space] `spaces` not installed — running without the ZeroGPU seam",
              file=sys.stderr)
        return []

    from oasis.quant import segment
    from oasis.spatial import loftr_matcher, sparse_matcher

    targets = [
        (segment, "segment_image", _segment_duration),
        (sparse_matcher, "correspondences", _pair_duration),
        (loftr_matcher, "loftr_correspondences", _pair_duration),
        (loftr_matcher, "certify_local_roi", _cert_duration),
        (loftr_matcher, "loftr_fle", _cert_duration),
    ]
    for module, name, duration in targets:
        original = getattr(module, name)
        setattr(module, name, _wrap(original, duration))
        _PATCHED.append(f"{module.__name__}.{name}")

    # Nothing touches CUDA here. See the block above `install`'s targets for why any CUDA
    # call in this process permanently breaks every forked GPU worker.
    return list(_PATCHED)
