"""The Windows GUI backend, tested the way a user gets it: downloaded.

A user's v0.1.3 download died at startup with

    RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize from
    D:\\Downloads\\OASIS-windows-x64 (1)\\OASIS\\_internal\\pythonnet\\runtime\\Python.Runtime.dll

on the exact line the release check runs green. The published zip was intact — the assembly
is in it at its full size — so the difference was not the bundle but where it had been. A
browser marks what it downloads, Explorer copies that mark onto everything it extracts, and
the .NET Framework will not load a marked assembly. CI has never once run a bundle that was
downloaded, so nothing it checked could see this.

This puts the mark on, confirms the backend then fails the way the user's did, and confirms
`prepare_gui()` gets it loading again. It needs a real Windows and a real .NET, so it skips
everywhere else — but the platform it needs is the platform that was broken.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="the Windows GUI backend needs Windows and .NET")

MOTW = "[ZoneTransfer]\r\nZoneId=3\r\n"       # 3 = downloaded from the internet

# Each case runs in its own process: hosting the CLR is a one-way, once-per-process act, so
# "does it load" cannot be asked twice in the same interpreter.
PLAIN = "import clr; print('CLR OK')"
REPAIRED = ("from oasis.common.winstart import prepare_gui; prepare_gui();"
            " import clr; print('CLR OK')")

# The config the build writes beside OASIS.exe. Tested here against the interpreter running
# the suite, because that is this process's host executable and the setting is read per
# process from <host>.config.
HOST_CONFIG = Path(f"{sys.executable}.config")


def _load_clr(code):
    """Returns (ok, output). Never raises — a failure to load is the thing under test."""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[1]), timeout=180)
    return proc.returncode == 0 and "CLR OK" in proc.stdout, (proc.stdout + proc.stderr)


@pytest.fixture
def marked_assemblies():
    """Mark the .NET assemblies as downloaded, the way Explorer does. Undone afterwards."""
    from oasis.common import winstart
    files = winstart.assembly_files()
    if not files:
        pytest.skip("pythonnet is not installed in this environment")
    written = []
    for path in files:
        try:
            with open(f"{path}:{winstart._MOTW_STREAM}", "w", encoding="utf-8") as fh:
                fh.write(MOTW)
        except OSError:
            continue        # not NTFS, or not writable: nothing to test here
        written.append(path)
    if not written:
        pytest.skip("could not write alternate data streams (not an NTFS volume?)")
    try:
        yield written
    finally:
        for path in written:
            try:
                os.remove(f"{path}:{winstart._MOTW_STREAM}")
            except OSError:
                pass


def test_the_backend_loads_before_anything_is_marked():
    """The premise. If .NET cannot host the assembly at all, the rest proves nothing."""
    ok, output = _load_clr(PLAIN)
    if not ok:
        pytest.skip(f"no working .NET on this runner, so the repair cannot be tested:\n{output}")


def test_the_mark_of_the_web_is_what_broke_the_users_download(marked_assemblies):
    ok, output = _load_clr(PLAIN)
    if ok:
        pytest.skip("this Windows loads marked assemblies, so it cannot reproduce the report")
    assert "Python.Runtime" in output, (
        "the marked assembly failed to load, but not in the way the user saw:\n" + output)


def test_prepare_gui_makes_a_downloaded_copy_start(marked_assemblies):
    """The fix, measured against the failure directly above."""
    ok, output = _load_clr(REPAIRED)
    assert ok, "prepare_gui() did not get the GUI backend loading again:\n" + output


def test_the_shipped_host_config_covers_an_install_we_cannot_write_to(marked_assemblies):
    """The fallback, for an install directory whose files we have no permission to change.

    Kept separate from the repair above so a regression here names the fallback instead of
    being hidden by the unblock that would otherwise have run first. An earlier version of
    this fallback passed the same XML to pythonnet via PYTHONNET_NETFX_CONFIG_FILE and did
    not work: loadFromRemoteSources is read per process from the host executable's config,
    not from the AppDomain clr_loader builds.
    """
    from oasis.common import winstart
    if HOST_CONFIG.exists():
        pytest.skip(f"{HOST_CONFIG} already exists; refusing to overwrite it")
    HOST_CONFIG.write_text(winstart.HOST_CONFIG_XML, encoding="utf-8")
    try:
        ok, output = _load_clr(PLAIN)
    finally:
        HOST_CONFIG.unlink()
    assert ok, ("the shipped OASIS.exe.config would not let .NET load a marked assembly:\n"
                + output)


def test_marked_files_reports_what_is_still_blocked(marked_assemblies):
    """The startup report has to be able to say "still blocked", or it misdiagnoses."""
    from oasis.common import winstart
    assert winstart.marked_files(), "marked files went unnoticed"
    winstart.unblock_assemblies()
    assert not winstart.marked_files(), "unblock_assemblies() left files marked"
