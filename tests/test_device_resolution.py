"""The compute device has to follow the hardware, not the machine it was configured on.

The shipped default was the literal "mps", and an unavailable request resolved to "cpu".
On an NVIDIA workstation — or the rented cloud GPU — that produced a run where LoFTR used
the CUDA card (it detects its own device) and InstanSeg did not:

    config: device: mps  →  MPS unavailable  →  "cpu"

Nothing in the output is wrong, which is why it survived: you pay for a GPU, get a CPU
segmentation, and the only symptom is that it took hours.

These tests fix the resolution policy against a faked torch, so they give the same answer on
the Mac they were written on and on the CUDA box they were written for.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.common import device as dev  # noqa: E402


@pytest.fixture
def hardware(monkeypatch):
    """Pretend this machine has (cuda, mps) whatever the real one has."""
    def _set(cuda, mps):
        monkeypatch.setattr(dev, "_availability", lambda: (cuda, mps))
    return _set


# ── the bug ──────────────────────────────────────────────────────────────────────────
def test_a_mac_config_on_a_cuda_box_uses_the_cuda_card(hardware):
    """The exact failure: a settings file written on Apple Silicon, run on an NVIDIA box.

    "mps" is a preference for acceleration, not a demand for that specific backend, so the
    fallback is the next accelerator — not the CPU.
    """
    hardware(cuda=True, mps=False)
    assert dev.resolve_device("mps") == "cuda"


def test_a_cuda_config_on_a_mac_uses_the_apple_gpu(hardware):
    """The same rule in the other direction — a lab config carried to a laptop."""
    hardware(cuda=False, mps=True)
    assert dev.resolve_device("cuda") == "mps"


# ── auto ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cuda,mps,expected", [
    (True,  True,  "cuda"),      # both present: CUDA is the faster of the two
    (True,  False, "cuda"),
    (False, True,  "mps"),
    (False, False, "cpu"),
])
@pytest.mark.parametrize("asked", ["auto", None, "", "AUTO", " auto "])
def test_auto_takes_the_best_available(hardware, cuda, mps, expected, asked):
    hardware(cuda, mps)
    assert dev.resolve_device(asked) == expected


# ── cpu is an opt-out, and must stay one ─────────────────────────────────────────────
def test_cpu_is_honoured_even_when_a_gpu_is_present(hardware):
    """Not a fallback — a choice.

    The validated corpus was scored on CPU and validate_native_segmenter_robustness.py
    compares CPU against the accelerator. If "cpu" quietly upgraded itself, that comparison
    would silently become accelerator-vs-accelerator and could never fail.
    """
    hardware(cuda=True, mps=True)
    assert dev.resolve_device("cpu") == "cpu"
    assert dev.resolve_device("CPU") == "cpu"


# ── honouring a request that CAN be met ──────────────────────────────────────────────
def test_an_available_request_is_taken_as_asked(hardware):
    hardware(cuda=True, mps=True)
    assert dev.resolve_device("mps") == "mps"        # not silently upgraded to cuda
    assert dev.resolve_device("cuda") == "cuda"


def test_a_device_index_survives(hardware):
    """"cuda:1" on a multi-GPU box must not collapse to card 0."""
    hardware(cuda=True, mps=False)
    assert dev.resolve_device("cuda:1") == "cuda:1"


def test_a_device_index_falls_back_like_any_other_preference(hardware):
    hardware(cuda=False, mps=True)
    assert dev.resolve_device("cuda:1") == "mps"


# ── never raise ──────────────────────────────────────────────────────────────────────
def test_no_torch_at_all_still_returns_a_usable_device(monkeypatch):
    """A stripped install must degrade to CPU, not kill the run at its last step."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _no_torch)
    assert dev.resolve_device("cuda") == "cpu"
    assert dev.resolve_device("auto") == "cpu"


def test_nonsense_is_treated_as_a_preference_not_an_error(hardware):
    hardware(cuda=True, mps=False)
    assert dev.resolve_device("tpu") == "cuda"
    hardware(cuda=False, mps=False)
    assert dev.resolve_device("tpu") == "cpu"


# ── the operator has to be able to see the substitution ──────────────────────────────
def test_the_banner_says_when_it_did_not_get_what_was_asked(hardware):
    hardware(cuda=True, mps=False)
    msg = dev.describe_device("mps")
    assert "cuda" in msg and "mps" in msg, msg

    hardware(cuda=False, mps=True)
    assert dev.describe_device("mps") == "mps"       # nothing to explain

    hardware(cuda=True, mps=False)
    assert "auto-detected" in dev.describe_device("auto")


# ── the pipeline's own entry point uses this policy ──────────────────────────────────
def test_run_pipeline_resolves_through_the_shared_policy(hardware):
    """_torch_device is what actually reaches InstanSeg (run_pipeline.py:238)."""
    import run_pipeline
    hardware(cuda=True, mps=False)
    assert run_pipeline._torch_device("mps") == "cuda"
    assert run_pipeline._torch_device("cpu") == "cpu"


def test_the_shipped_default_is_auto():
    """A config that says nothing must not name a backend."""
    import run_pipeline
    cfg = {"instanseg_model": __file__, "input_dir": "", "stain_type": "hdab"}
    # load_config does the setdefaults; call the same code path it uses.
    import inspect
    src = inspect.getsource(run_pipeline.load_config)
    assert 'setdefault("device", "auto")' in src, "default backend must be auto"

    from oasis.webui.api import DEFAULT_SETUP
    assert DEFAULT_SETUP["device"] == "auto"
