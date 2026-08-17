"""
hf_space — running OASIS as a Hugging Face Space on ZeroGPU.

Everything the Space needs and the desktop app does not lives in here. Nothing under
`oasis/`, `run_pipeline.py`, `app.py` or `serve.py` is modified by any of it; the Space is
assembled at startup by rebinding two things in the already-imported app (see `app.py`).

WHY THIS PACKAGE IS NOT CALLED `spaces`. Hugging Face's own package — the one that provides
the `@spaces.GPU` decorator this whole deployment depends on — is imported as `spaces`. A
top-level `spaces/` directory in the repo shadows it: Python 3 treats any directory on
sys.path as an implicit namespace package, and the Space puts the repo root on sys.path, so
`import spaces` would resolve here and `spaces.GPU` would not exist. The failure is quiet in
the worst way — the seam falls back to running without a GPU, which is exactly the outcome
this package exists to prevent. Hence `hf_space`.
"""
