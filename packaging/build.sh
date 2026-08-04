#!/usr/bin/env bash
# Build the OASIS desktop bundle. Run from anywhere; paths are resolved from this script.
#
#   ./packaging/build.sh
#
# Produces dist/OASIS.app (~875 MB unpacked). The bundle is a GitHub Release asset — it is
# gitignored and must never be committed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: no interpreter at $PY (set PY=/path/to/python to override)" >&2
  exit 1
fi

echo "==> interpreter: $("$PY" -V) at $PY"

if ! "$PY" -c "import PyInstaller" 2>/dev/null; then
  echo "==> installing pyinstaller"
  "$PY" -m pip install pyinstaller
fi

# The vendored weights are what make the bundle self-contained. Building without them
# yields an app that starts and then fails on the first image, so fail here instead.
if [ ! -f "$ROOT/models/brightfield_nuclei-0.1.1/rdf.yaml" ]; then
  echo "error: models/brightfield_nuclei-0.1.1 is missing or incomplete." >&2
  echo "       The InstanSeg weights are vendored in the repo; see models/NOTICE.md." >&2
  exit 1
fi

echo "==> cleaning previous build"
rm -rf "$ROOT/build" "$ROOT/dist"

echo "==> freezing"
"$PY" -m PyInstaller --noconfirm --clean packaging/OASIS.spec

APP="$ROOT/dist/OASIS.app"
[ -d "$APP" ] || { echo "error: $APP was not produced" >&2; exit 1; }

# Smoke-test the frozen bundle without opening a window. app.py dispatches --oasis-worker
# before importing pywebview, so this exercises the real frozen import path (torch, kornia,
# the vendored model) and would catch a missing hidden import that the build itself does not.
echo "==> smoke test (frozen, headless)"
BIN="$APP/Contents/MacOS/OASIS"

# Run the REAL pipeline, and let it fail the build. What stood here was
#     "$BIN" --oasis-worker oasis.common.paths 2>/dev/null || true
# which discarded the output and swallowed the exit code, so it could not fail; the only
# thing that could was the file-existence check below, and "bundle contents OK" then read
# like the app had been exercised. packaging/smoke_test.py — which segments real tissue and
# asserts a plausible cell count — existed the whole time and was wired only into the
# release workflow, so a locally built bundle never ran it.
#
# That gap is not theoretical: deleting a public function from pixel_size_util left every
# spatial and quant run dying on "import failed", and a bundle built from that source would
# have passed this step. The failure is exactly the kind the docstring in smoke_test.py
# describes — invisible from source, green build, broken app.
SMOKE_WD="$(mktemp -d)"
trap 'rm -rf "$SMOKE_WD"' EXIT
"$PY" "$ROOT/packaging/smoke_test.py" prepare --dir "$SMOKE_WD"
"$BIN" --oasis-worker run_pipeline --config "$SMOKE_WD/config.yaml"
"$PY" "$ROOT/packaging/smoke_test.py" check --dir "$SMOKE_WD"
# The UI too. The worker check above shares almost no code with startup, which is how a
# bundle whose window could never open shipped as v0.1.0. No window is opened here.
"$BIN" --oasis-check-ui
"$PY" - "$APP" <<'PY'
import os, sys
app = sys.argv[1]
need = [
    "Contents/MacOS/OASIS",
    "Contents/Frameworks/models/brightfield_nuclei-0.1.1/instanseg.pt",
    "Contents/Frameworks/models/brightfield_nuclei-0.1.1/rdf.yaml",
    "Contents/Frameworks/oasis/webui/index.html",
    "Contents/Frameworks/oasis/webui/restained_coexpression.js",
]
missing = [p for p in need if not os.path.exists(os.path.join(app, p))]
# PyInstaller has moved data between Contents/Frameworks and Contents/Resources across
# versions, so accept either location before declaring anything missing.
missing = [p for p in missing
           if not os.path.exists(os.path.join(app, p.replace("Frameworks", "Resources")))]
if missing:
    print("MISSING FROM BUNDLE:"); [print("  ", m) for m in missing]; sys.exit(1)
print("bundle contents OK")
PY

echo
echo "==> built: $APP"
du -sh "$APP"
echo
echo "Next: sign and notarize before distributing, or macOS Gatekeeper will refuse to open it"
echo "on any machine but this one:"
echo "  codesign --deep --force --options runtime --timestamp \\"
echo "    --sign \"Developer ID Application: <NAME> (<TEAMID>)\" \"$APP\""
echo "  ditto -c -k --keepParent \"$APP\" dist/OASIS.zip"
echo "  xcrun notarytool submit dist/OASIS.zip --apple-id <EMAIL> --team-id <TEAMID> --wait"
echo "  xcrun stapler staple \"$APP\""
