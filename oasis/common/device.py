"""
device.py — choosing the torch device once, for the whole app.

WHY THIS EXISTS. The shipped default was the literal string "mps", written when the only
machine OASIS ran on was an Apple Silicon Mac. `_torch_device` then resolved an unavailable
request straight to "cpu". On an NVIDIA box — a lab workstation, or the cloud GPU rented to
run the registration validations — that is the worst possible outcome:

    device: "mps"  ->  MPS unavailable  ->  "cpu"          InstanSeg on the CPU
    LoFTR                                                  CUDA (it detects its own)

so half the pipeline uses the accelerator and the other half silently does not. Nothing is
wrong in the output; it just takes hours longer than the hardware allows, and the run looks
like it worked. That is a bad failure because it is invisible: you pay for a GPU and get a
CPU segmentation.

THE POLICY, and the reasoning behind each line:

  "cpu"                     honoured absolutely. It is the escape hatch — the validation
                            corpus was scored on CPU and `validate_native_segmenter_
                            robustness.py` compares CPU against the accelerator, so
                            "cpu" must mean cpu even on a machine with a GPU sitting idle.

  "auto" / None / ""        cuda, else mps, else cpu. The new default.

  "cuda" / "cuda:1" / "mps" a *preference*, not a demand. Honoured when that backend is
                            present; otherwise falls through the auto chain rather than
                            dropping to cpu. The field means "use the accelerator"; when
                            the named one is absent, the next-best accelerator is a far
                            better reading of that intent than no accelerator at all.

The preference-not-demand rule is what repairs installs already in the wild: a settings file
written on a Mac carries `device: mps`, and copying that config to a CUDA workstation now
uses the CUDA card instead of the CPU.

Every caller resolves through here so there is one policy and one place to test it. The
resolved string is recorded in the summary provenance (`segmenter_device`), so a finished
run always says which device actually ran it.

LoFTR deliberately does NOT use this. `loftr_matcher._device()` takes CUDA when present but
requires LOFTR_GPU=1 for MPS, because MPS measured NO speedup for LoFTR — its attention ops
fall back to the CPU anyway. That is a measured, matcher-specific policy, not a second
opinion about the same question.
"""


def _availability():
    """(cuda_available, mps_available). False/False when torch is missing or broken."""
    try:
        import torch
    except Exception:
        return False, False
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    try:
        # A torch built without the MPS backend has no torch.backends.mps at all.
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps = False
    return cuda, mps


def resolve_device(name=None):
    """The torch device string to actually use, given what the config asked for.

    Returns one of "cpu", "mps", "cuda", "cuda:N". Never raises: an unusable request
    degrades to something that works rather than failing a run at its last step.
    """
    want = str(name or "auto").strip().lower()
    if want == "cpu":
        return "cpu"                       # explicit opt-out, honoured on any hardware

    cuda, mps = _availability()
    if want.startswith("cuda") and cuda:
        return want                        # keeps the device index, e.g. "cuda:1"
    if want == "mps" and mps:
        return "mps"
    # "auto", or a preference this machine cannot satisfy: take the best available.
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def device_label(dev):
    """"NVIDIA GPU (CUDA)" for "cuda:1" — the wording the Settings dropdown already uses.

    The raw torch string still goes into the summary provenance (`segmenter_device`); this
    is for the two places a person reads it.
    """
    d = str(dev or "").strip().lower()
    if d.startswith("cuda"):
        n = d.split(":", 1)[1] if ":" in d else None
        return f"NVIDIA GPU (CUDA{', card ' + n if n else ''})"
    if d == "mps":
        return "Apple Silicon GPU (MPS)"
    if d == "cpu":
        return "CPU"
    return d or "unknown"


def describe_device(name=None):
    """What was asked for, what was chosen, and why they differ.

    Shown in the run banner and under the Settings dropdown, so an operator sees the
    substitution happen instead of inferring it from the wall-clock time.
    """
    want = str(name or "auto").strip().lower()
    got = resolve_device(name)
    if want in ("auto", ""):
        return f"{device_label(got)}, auto-detected"
    if got == want:
        return device_label(got)
    return (f"{device_label(got)} — {device_label(want)} was requested but is not "
            f"available on this machine")
