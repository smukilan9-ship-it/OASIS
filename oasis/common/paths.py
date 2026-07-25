"""
paths.py — locating resources that ship with OASIS, from source and when frozen.

WHY THIS EXISTS. The InstanSeg weights used to be read from wherever QuPath had
downloaded them (`~/QuPath/v0.7/instanseg/downloaded/...`). That was the last path in the
app that assumed QuPath was installed at all — after the native-segmenter cutover the
default configuration pointed into a directory that, on a machine that never installed
QuPath, does not exist. A shipped bundle cannot depend on it.

The weights are now vendored at `models/brightfield_nuclei-0.1.1` (Apache-2.0, see
`models/NOTICE.md`) and resolved through here, which is the one place that knows the
difference between a source checkout and a PyInstaller bundle:

    from source :  <repo root>/models/brightfield_nuclei-0.1.1
    frozen      :  <sys._MEIPASS>/models/brightfield_nuclei-0.1.1

`default_model_dir()` still falls back to the old QuPath location if the bundled copy is
somehow absent, so an existing working install keeps working rather than breaking on
upgrade. A user-set `instanseg_model` in the config always wins over both — this only
supplies the default.
"""
import os
import sys

MODEL_NAME = "brightfield_nuclei-0.1.1"

# Where QuPath's InstanSeg extension puts its downloads. Kept only as a fallback for
# installs that predate the vendored model.
LEGACY_QUPATH_MODEL = "~/QuPath/v0.7/instanseg/downloaded/" + MODEL_NAME


def resource_dir():
    """Root for read-only files that ship with OASIS (the repo root, or the bundle)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    # oasis/common/paths.py -> oasis/common -> oasis -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bundled_model_dir():
    """Path to the vendored InstanSeg model, or None when it is not present."""
    path = os.path.join(resource_dir(), "models", MODEL_NAME)
    # rdf.yaml is what the segmenter reads for `scale`; a directory without it is not a
    # usable model, so treat that as absent rather than returning a path that fails later.
    return path if os.path.exists(os.path.join(path, "rdf.yaml")) else None


def default_model_dir():
    """Default `instanseg_model`: the vendored copy, else the legacy QuPath download."""
    return bundled_model_dir() or os.path.expanduser(LEGACY_QUPATH_MODEL)
