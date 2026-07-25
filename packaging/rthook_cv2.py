"""PyInstaller runtime hook: make opencv's loader work inside a macOS .app bundle.

Runs before any application code, so the flag is set before anything imports cv2.

THE FAILURE. Inside OASIS.app, `import cv2` dies with:

    ImportError: ERROR: recursion is detected during loading of "cv2" binary extensions.

WHY. cv2/__init__.py is a loader, not the real module. It pops itself from sys.modules and
calls importlib.import_module("cv2") again, expecting that second import to resolve to the
native extension (cv2.abi3.so) rather than back to the package. It arranges that by putting
the directory holding the .so onto sys.path — but at index **1**, not 0:

    sys.path.insert(1 if not applySysPathWorkaround else 0, p)

In a normal install index 1 is early enough, because sys.path[0] is the script directory.
In a PyInstaller .app, sys.path[0] is Contents/Frameworks, which *contains the cv2 package
directory itself* — and a package always wins over a same-named extension module. So the
second import finds the package again, re-enters the loader, and trips its own recursion
guard. The guard is the symptom; the path ordering is the cause.

opencv anticipated this and left an escape hatch immediately above that line: if
`sys.OpenCV_REPLACE_SYS_PATH_0` is set, it inserts at 0 instead of 1, which puts the .so
ahead of the package and lets the second import resolve correctly. It normally auto-detects
the situation by comparing sys.path[0] against the package's parent, but that check fails
here because the .app splits the package across Contents/Frameworks and Contents/Resources,
so realpath() and sys.path[0] disagree.

Setting the flag ourselves is opencv's own supported mechanism — not a patch of its files.
This is why the vendored config.py / config-3.py must be left ALONE: config-3.py is what
supplies the extension directory in the first place.
"""
import sys

sys.OpenCV_REPLACE_SYS_PATH_0 = True
