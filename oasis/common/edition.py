"""
edition.py — which surfaces of OASIS this build exposes.

There are two audiences with different needs, served by one codebase.

**v1 (default)** is what a pathology lab installs to analyse slides. It exposes Quant,
Spatial, Calibrate and Settings. Nothing else is hidden for marketing reasons — the
excluded tabs are genuinely research instruments:

  • Validation runs the registered validation suite against reference datasets that are
    not shipped (and in some cases cannot be redistributed). On a normal install every
    validation would skip for want of data, so the tab is an empty promise.
  • Restained drives the same-section restaining workflow, which is an experimental
    protocol rather than a supported analysis.

**Research** additionally exposes both. Enable with the environment variable
`OASIS_RESEARCH=1`, or `edition: research` in the user's setup.yaml.

Defaulting to v1 rather than research is deliberate: a tab that appears functional and
then cannot do anything is worse than a tab that is absent, and the research surfaces are
the ones most likely to produce numbers a user cannot interpret unaided.
"""
import os

_TRUE = {"1", "true", "yes", "on", "research"}


def is_research(setup: dict | None = None) -> bool:
    """True when the research surfaces (Validation, Restained) should be exposed.

    The environment variable wins over the stored setting, so a researcher can enable it
    for a single launch without editing configuration:

        OASIS_RESEARCH=1 python app.py
    """
    env = os.environ.get("OASIS_RESEARCH")
    if env is not None:
        return env.strip().lower() in _TRUE
    if setup:
        return str(setup.get("edition", "")).strip().lower() in _TRUE
    return False


def edition_name(setup: dict | None = None) -> str:
    return "research" if is_research(setup) else "v1"
