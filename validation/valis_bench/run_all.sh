#!/bin/bash
# Sequential ANHIR benchmark chain (RAM-safe on 16GB — never runs two heavy jobs at once).
# Order: ours (stratified) -> valis (stratified) -> compare (REPORT) -> correspondence (full 222).
# The comparison REPORT lands first (~3h); the full LoFTR-correspondence answer after (~5h).
cd "/Users/mukilan/PycharmProjects/ihc-original copy" || exit 1
export PYTHONUNBUFFERED=1
L=validation/valis_bench
STRAT=7

echo "=== [1/4] OURS (stratified $STRAT/tissue) $(date) ==="
ANHIR_STRATIFY=$STRAT .venv/bin/python -m validation.valis_bench.run_ours > "$L/_ours.log" 2>&1
echo "ours done ($(date))"

echo "=== [2/4] VALIS (stratified $STRAT/tissue) $(date) ==="
ANHIR_STRATIFY=$STRAT DYLD_LIBRARY_PATH=/opt/homebrew/lib \
  ~/valis_runtime/venv/bin/python -m validation.valis_bench.run_valis > "$L/_valis.log" 2>&1
echo "valis done ($(date))"

echo "=== [3/4] COMPARE $(date) ==="
.venv/bin/python -m validation.valis_bench.compare > "$L/_compare.log" 2>&1
echo "compare done ($(date))"

echo "=== [4/4] CORRESPONDENCE (full 222) $(date) ==="
.venv/bin/python -m validation.valis_bench.run_correspondence > "$L/_corr.log" 2>&1
echo "correspondence done ($(date))"

echo "=== CHAIN DONE $(date) ==="
