# PyInstaller spec for the OASIS desktop app. Cross-platform.
#
# Build with:  ./packaging/build.sh          (macOS/Linux, from the repo root)
# or directly: pyinstaller --noconfirm packaging/OASIS.spec     (any platform)
#
# Output differs by platform and that is deliberate:
#   macOS   -> dist/OASIS.app     (application bundle)
#   Windows -> dist/OASIS/OASIS.exe
#   Linux   -> dist/OASIS/OASIS
#
# Verified on macOS arm64, Python 3.14.0, PyInstaller 6.21.0, torch 2.13.0. MPS works
# inside the frozen binary, so the shipped `device: mps` default survives freezing. On
# Windows and Linux torch has no MPS and `_torch_device` falls back to CPU on its own.
#
# THREE THINGS THAT BREAK A NAIVE BUILD, and why each line below exists:
#
#  1. kornia raises at import:
#        OSError: Can't get source for <function sampson_epipolar_distance>.
#        TorchScript requires source access in order to carry out compilation.
#     PyInstaller compiles modules to .pyc and drops the .py source, but kornia has
#     module-level @torch.jit.script decorators and TorchScript recompiles from source
#     text at import. collect_data_files(..., include_py_files=True) puts the sources back.
#
#  2. torch is imported lazily, inside functions (oasis/quant/segment.py), so PyInstaller's
#     static analysis never sees it. Without the explicit collect_submodules('torch') the
#     build SUCCEEDS and produces a 38 MB bundle with no torch in it — it then fails on a
#     user's machine, not on the build machine. Same for the other lazily-imported readers.
#
#  3. Data files are not code. index.html, its sibling .js, and the InstanSeg weights are
#     read at runtime by path; nothing imports them, so they must be listed explicitly.
#     They are resolved at runtime through oasis/common/paths.py:resource_dir(), which
#     returns sys._MEIPASS when frozen.
#
#  4. cv2 cannot import inside a .app without the runtime hook below — see
#     packaging/rthook_cv2.py for the mechanism. This one appears only in the .app; a
#     plain --onedir build imports cv2 fine, so it cannot be caught before BUNDLE.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), os.pardir))

datas = [
    # UI: index.html loads restained_coexpression.js with a relative <script src>, so the
    # two must stay siblings inside the bundle.
    (os.path.join(ROOT, "oasis", "webui", "index.html"), "oasis/webui"),
    (os.path.join(ROOT, "oasis", "webui", "restained_coexpression.js"), "oasis/webui"),
    # The "Need help" card loads these by relative path (help/<name>.png). They fail the
    # way missing images always do — a broken frame, no error — so a bundle that omits
    # them looks built and works everywhere except the one screen a new user opens first.
    (os.path.join(ROOT, "oasis", "webui", "help"), "oasis/webui/help"),
    # InstanSeg weights (Apache-2.0, see models/NOTICE.md). Vendoring these is what makes
    # the bundle self-contained: no external segmenter install.
    (os.path.join(ROOT, "models", "brightfield_nuclei-0.1.1"),
     "models/brightfield_nuclei-0.1.1"),
    (os.path.join(ROOT, "config.example.yaml"), "."),
]

# See note 1.
datas += collect_data_files("kornia", include_py_files=True)

# See note 2. openslide/tifffile/PIL/cv2 are all imported inside functions on the image
# reading path, and SimpleITK on the registration path.
hiddenimports = collect_submodules("torch") + [
    "PIL", "PIL.Image", "cv2", "SimpleITK", "openslide", "tifffile",
    "scipy.spatial", "scipy.ndimage", "shapely", "yaml",
]

# The GUI backend for THIS platform. pywebview picks one at runtime in guilib.initialize();
# naming it here means a bundle that cannot open a window fails the build rather than the
# user's first double-click. Only the host's own backend is added, so the freeze does not
# chase imports that cannot exist on it.
if sys.platform == "darwin":
    hiddenimports += ["webview.platforms.cocoa"]
elif sys.platform == "win32":
    hiddenimports += ["webview.platforms.winforms", "clr"]
else:
    hiddenimports += ["webview.platforms.gtk", "gi"]
    # gi loads GTK and WebKit2 through gobject-introspection typelibs, which are data, not
    # imports, so nothing pulls them in on their own.
    datas += collect_data_files("gi", include_py_files=True)

excludes = [
    # pywebview uses the native backend on every platform we ship: WebKit via pyobjc on
    # macOS, WebView2 via pythonnet on Windows, GTK/WebKit2 on Linux. Qt is never used and
    # PySide6 alone is ~1.1 GB, so it must never be pulled in.
    "PySide6", "PyQt5", "PyQt6",
    "tensorboard", "tests",
    "matplotlib.tests", "numpy.tests", "scipy.tests",
]

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[os.path.join(os.path.dirname(os.path.abspath(SPEC)), "rthook_cv2.py")],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OASIS",
    debug=False,
    strip=False,
    upx=False,          # UPX corrupts signed macOS dylibs; never enable it here.
    console=False,      # windowed app
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="OASIS",
)

if sys.platform == "win32":
    # OASIS.exe.config, beside OASIS.exe — NOT under _internal, which is where anything
    # listed in `datas` would land, and where the .NET Framework would never look for it.
    # Hence writing it here, after COLLECT has built the directory, rather than declaring
    # it as data.
    #
    # It grants loadFromRemoteSources, which is what lets .NET load an assembly carrying
    # the Mark of the Web. A user's v0.1.3 download died at startup without it: Explorer
    # marks every file it extracts from a downloaded zip, and .NET then refuses to load
    # pythonnet's Python.Runtime.dll, so WebView2 never starts and no window opens.
    #
    # The app also takes the mark off those files itself at startup, which is the repair
    # that runs in practice. This covers the case that cannot: an install directory the
    # user has no permission to write to, where the mark can never be removed.
    sys.path.insert(0, ROOT)
    from oasis.common.winstart import HOST_CONFIG_XML

    with open(os.path.join(coll.name, "OASIS.exe.config"), "w", encoding="utf-8") as fh:
        fh.write(HOST_CONFIG_XML)

# Only macOS wraps the collected tree in an application bundle. On Windows and Linux the
# COLLECT directory *is* the deliverable (OASIS/OASIS.exe, OASIS/OASIS), and asking for a
# BUNDLE there produces nothing useful.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OASIS.app",
        icon=None,
        bundle_identifier="io.github.smukilan9.oasis",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            # The app opens user-selected slide images; no other privileged access.
            "NSPrincipalClass": "NSApplication",
        },
    )
