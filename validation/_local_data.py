"""Where a validation script's inputs live, without naming one machine's disk.

These scripts read cohorts that are not, and mostly cannot be, in the repository: some are
unpublished lab data, others are public sets under terms that forbid redistribution. So each
one needs a path from outside.

They used to carry that path as a constant pointing into a Desktop folder. Two costs, and
the second is the one that matters for a public repository:

  * the script was unrunnable for anyone else, failing with a missing-file error that reads
    like a broken dependency rather than "set this to your copy"; and
  * the constants named unpublished images. A filename is not nothing — case identifiers
    travel in them — and every reader of the repository could see them.

Ask for the path by name instead, and say what to set when it is missing.
"""
import os
import sys


def local_dir(env_var, what, default=None):
    """A directory supplied by the caller's environment.

    `what` is shown when the variable is unset, and should say what the directory holds
    rather than restate the variable's name.
    """
    raw = os.environ.get(env_var) or default
    if not raw:
        sys.exit(f"error: set {env_var} to {what}. It is not in this repository.")
    path = os.path.expanduser(raw)
    if not os.path.isdir(path):
        sys.exit(f"error: {env_var}={raw} is not a directory. Expected {what}.")
    return path


def local_file(env_var, what, default=None):
    """A single file supplied by the caller's environment."""
    raw = os.environ.get(env_var) or default
    if not raw:
        sys.exit(f"error: set {env_var} to {what}. It is not in this repository.")
    path = os.path.expanduser(raw)
    if not os.path.isfile(path):
        sys.exit(f"error: {env_var}={raw} is not a file. Expected {what}.")
    return path


def default_dir(env_var, fallback):
    """A directory that has a sensible default and need not exist yet (outputs, caches)."""
    return os.path.expanduser(os.environ.get(env_var) or fallback)
