#!/usr/bin/env bash
#
# deploy.sh — assemble and upload the OASIS Space.
#
# The Space repo is NOT a mirror of this one. Hugging Face reads its configuration from the
# YAML front matter of README.md at the repo root, and installs from requirements.txt at the
# repo root — and both of those names are already taken here by the project's own README and
# the desktop app's dependency pins. Pushing this repo as-is would either overwrite the
# project README with a Space card or leave the Space with no configuration at all, and would
# hand the container a requirements.txt containing pywebview.
#
# So the Space is assembled: the files it needs are copied into a staging directory, with
# hf_space/README.md and hf_space/requirements.txt promoted to the root names the platform
# looks for. Nothing in this repository is modified.
#
# Usage:  hf_space/deploy.sh <owner>/<space-name> [--dry-run]
#
# Uploads with the `hf` CLI, which uses the token from `hf auth login` — no git remote and no
# git credentials involved. Create the Space first as a Gradio SDK Space on ZeroGPU hardware:
# ZeroGPU exists only on that SDK, and on any other the pipeline runs on CPU while looking
# perfectly healthy.
#
#     hf repos create <owner>/<name> --type space --sdk gradio --flavor zero-a10g
#     hf buckets create <owner>/<name>-data --region us
#     hf spaces volumes set <owner>/<name> --volume hf://buckets/<owner>/<name>-data:/data

set -euo pipefail

TARGET="${1:-}"
DRY_RUN="${2:-}"

if [[ -z "$TARGET" ]]; then
    echo "usage: hf_space/deploy.sh <owner>/<space-name> [--dry-run]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${OASIS_SPACE_STAGING:-${TMPDIR:-/tmp}/oasis-space-${TARGET//\//-}}"

echo "repo    : $REPO_ROOT"
echo "space   : https://huggingface.co/spaces/${TARGET}"
echo "staging : $STAGING"
echo

# What the Space actually needs to run. Deliberately a list rather than "everything except
# what's ignored": the repo carries validation corpora, reports and build artefacts that
# have no business in a public container, and an exclude-list would ship each new one by
# default. models/ is 16 MB and is required — the segmenter cannot start without it.
PATHS=(
    "oasis"
    "run_pipeline.py"
    "models"
    "hf_space"
    "LICENSE"
)

rm -rf "$STAGING"
mkdir -p "$STAGING"

for path in "${PATHS[@]}"; do
    if [[ ! -e "$REPO_ROOT/$path" ]]; then
        echo "missing: $path" >&2
        exit 1
    fi
    # -L so a symlinked model directory is copied as real files.
    cp -RL "$REPO_ROOT/$path" "$STAGING/$path"
done

# Caches and OS cruft are not deployment artefacts; .DS_Store in particular would be served
# at the Space's root by the static-asset routes if it reached oasis/webui/.
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGING" \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true
rm -rf "$STAGING/hf_space/.local-data"

# The two promotions the platform requires. See the header for why they cannot simply live
# at the root of this repository.
cp "$REPO_ROOT/hf_space/README.md"        "$STAGING/README.md"
cp "$REPO_ROOT/hf_space/requirements.txt" "$STAGING/requirements.txt"

echo "staged:"
(cd "$STAGING" && du -sh "${PATHS[@]}" README.md requirements.txt 2>/dev/null || true)
echo "total: $(du -sh "$STAGING" | cut -f1)"
echo

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "--dry-run: nothing uploaded. Inspect $STAGING"
    exit 0
fi

# --delete '*' so a file removed from this checkout is removed from the Space too; without
# it the Space accumulates orphans from earlier deploys that still import and still run.
hf upload "$TARGET" "$STAGING" . \
    --type space \
    --delete "*" \
    --commit-message "Deploy OASIS $(git -C "$REPO_ROOT" rev-parse --short HEAD)"

echo
echo "Deployed. Watch the build:  hf spaces logs $TARGET --build --follow"
echo "Then confirm it is really on the GPU — a run's *_summary.json must say"
echo "segmenter_device: cuda, and /__health reports the patched entry points."
