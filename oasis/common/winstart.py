"""
winstart.py — the repairs Windows needs before the app can open its window.

WHY THIS EXISTS. On Windows, pywebview draws the interface with WebView2, which it reaches
through pythonnet, which asks the .NET Framework to load one managed assembly:
`pythonnet/runtime/Python.Runtime.dll`. If that load fails, the app dies at startup with

    RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize from
    ...\\_internal\\pythonnet\\runtime\\Python.Runtime.dll

which names the file but not the reason, because clr_loader's shim reports failure as a
null pointer and throws the .NET exception away.

THE REASON, in the case that reached a user: the Mark of the Web. A file downloaded with a
browser carries an alternate data stream called Zone.Identifier recording that it came from
the internet, and Windows Explorer copies that mark onto every file it extracts from the
zip. The .NET Framework refuses to load an assembly carrying it unless the application
opts in, so `Python.Runtime.dll` fails to load and the window never opens. The same bundle
run from the build directory works, which is exactly why every release check was green: CI
builds the bundle and runs it in place, so nothing it produces has ever been downloaded.

Two independent repairs, because they fail in different circumstances:

  1. Take the mark off the assemblies. Fixes the cause, but needs write access to the
     install directory — not a given if the app was put somewhere privileged.
  2. Tell .NET it may load a marked assembly anyway, via `loadFromRemoteSources` in a
     config file the runtime is pointed at. Needs no write access to the install directory,
     only to the temp directory.

Both are no-ops off Windows, and neither loads the CLR, so calling them costs nothing when
nothing is wrong.
"""
import importlib.util
import os
import sys
from pathlib import Path

# The mark lives in an NTFS alternate data stream, addressed as "<file>:<stream>".
_MOTW_STREAM = "Zone.Identifier"

# Where the .NET pieces sit, relative to the bundle root or to site-packages.
_ASSEMBLY_DIRS = (("pythonnet", "runtime"), ("clr_loader", "ffi", "dlls"))

# `loadFromRemoteSources` is the documented opt-in for loading an assembly that .NET judges
# to have come from somewhere other than this machine, which is what a file carrying the
# Mark of the Web is. It is a <runtime> setting, so it is read once per process from the
# host executable's own config file — OASIS.exe.config, beside OASIS.exe — and NOT from the
# config of the AppDomain clr_loader creates. Handing it to pythonnet through
# PYTHONNET_NETFX_CONFIG_FILE looks like it should work and measurably does not; see
# tests/test_windows_gui_backend.py, which failed on precisely that.
#
# The spec writes this text to OASIS.exe.config at build time. That covers the case
# unblock_assemblies() cannot: an install directory the user has no permission to write to,
# where the mark can never be removed.
HOST_CONFIG_XML = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <runtime>
    <loadFromRemoteSources enabled="true"/>
  </runtime>
</configuration>
"""


def _roots():
    """Directories that may contain the .NET assemblies, frozen or from source."""
    found = []
    # Frozen: PyInstaller puts data beside the executable, under _internal on Windows.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        found.append(Path(meipass))
    # From source: site-packages. find_spec locates the package WITHOUT importing it, so
    # this cannot drag pythonnet in as a side effect of asking where it is.
    for pkg in ("pythonnet", "clr_loader"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        if locations:
            found.append(Path(locations[0]).parent)
    # dict.fromkeys: preserve order, drop the duplicate when both packages share a prefix.
    return list(dict.fromkeys(found))


def assembly_files():
    """Every file whose Mark of the Web could stop the GUI backend from loading."""
    files = []
    for root in _roots():
        for parts in _ASSEMBLY_DIRS:
            directory = root.joinpath(*parts)
            if directory.is_dir():
                files.extend(p for p in directory.rglob("*") if p.is_file())
    return files


def marked_files():
    """Which of those files currently carry the mark. Empty off Windows."""
    if sys.platform != "win32":
        return []
    return [p for p in assembly_files() if Path(f"{p}:{_MOTW_STREAM}").exists()]


def unblock_assemblies():
    """Remove the Mark of the Web from the bundled .NET assemblies.

    Returns the files it cleared. Never raises: an install directory we cannot write to is
    a reason to fall back to the config file below, not to refuse to start.
    """
    if sys.platform != "win32":
        return []
    cleared = []
    for path in assembly_files():
        try:
            os.remove(f"{path}:{_MOTW_STREAM}")
        except OSError:
            continue        # no mark to remove, or no permission to remove it
        cleared.append(path)
    return cleared


def prepare_gui():
    """Everything Windows needs doing before the GUI backend is imported.

    Call before `webview.start()` or `webview.guilib.initialize()`; both import clr, and
    the repair has to be in place before the .NET runtime is asked for the assembly.
    """
    if sys.platform != "win32":
        return {}
    return {"unblocked": unblock_assemblies()}


def _dotnet_release():
    """The installed .NET Framework 4.x release number, or None."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full")
        with key:
            return winreg.QueryValueEx(key, "Release")[0]
    except OSError:
        return None


def gui_failure_report():
    """What to tell someone whose window did not open. Reads state; changes nothing.

    A frozen windowed app has no console, but PyInstaller shows whatever reached stderr in
    the dialog it puts up when startup fails — the same dialog that showed the traceback
    naming Python.Runtime.dll and nothing else.
    """
    if sys.platform != "win32":
        return ""
    lines = ["", "OASIS could not start its interface.", ""]

    files = assembly_files()
    runtime = [p for p in files if p.name == "Python.Runtime.dll"]
    if not runtime:
        lines += ["The .NET support files are missing from this copy of OASIS.",
                  "Download it again and extract the whole folder, not just OASIS.exe.", ""]
    else:
        dll = runtime[0]
        lines.append(f"Found {dll} ({dll.stat().st_size:,} bytes).")

    still_marked = marked_files()
    if still_marked:
        lines += [
            "",
            f"{len(still_marked)} of its files are still blocked by Windows because they",
            "came from a downloaded zip, and OASIS could not unblock them here.",
            "",
            "To fix it: delete this folder, right-click the downloaded zip, tick Unblock",
            "at the bottom of the General tab, apply, and extract it again.",
        ]

    release = _dotnet_release()
    if release is None:
        lines += ["", "No .NET Framework 4 was found. Install .NET Framework 4.8 from",
                  "Microsoft and try again."]
    elif release < 461808:      # 461808 = 4.7.2, the first release that can load netstandard2.0
        lines += ["", f".NET Framework 4 release {release} is installed; OASIS needs 4.7.2",
                  "or newer. Install .NET Framework 4.8 from Microsoft and try again."]

    if not still_marked and release is not None and release >= 461808:
        # Neither known cause fits, so name the remaining prerequisite rather than leaving
        # the traceback below as the only thing on screen.
        lines += ["", "OASIS draws its interface with the Microsoft Edge WebView2 Runtime,",
                  "which is part of current versions of Windows. If this machine does not",
                  "have it, install it from Microsoft and try again."]

    lines.append("")
    return "\n".join(lines)
