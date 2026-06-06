"""
file_matcher.py
Pairs IHC image files across stains by filename similarity.

Two modes:
  two_folder    — CD8 folder + TIM3 folder; files matched by normalized name
  single_folder — One folder containing both stains; auto-grouped by stain token

Stain token detection handles all common patterns:
  "ll4_liver_cd8_10x.svs"   ← token surrounded by separators
  "Sample001_TIM3.svs"       ← token at end
  "CD8-slide-003.svs"        ← token at start
  "slide003.svs"             ← no token found → unmatched
"""

import os
import re
from pathlib import Path

# Recognized stain identifiers sorted longest-first to prevent partial matches
# (e.g. "tim-3" must be checked before "tim")
DEFAULT_STAIN_TOKENS = sorted([
    "foxp3", "pdl1", "pd-l1", "pd_l1", "cd163", "cd68",
    "panck", "tim-3", "tim3",
    "hdab", "cd8", "cd4", "cd3", "ck",
    "ki67", "ki-67",
    "dab", "he", "h&e",
], key=len, reverse=True)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".svs", ".ndpi", ".png", ".jpg", ".jpeg"}


# ──────────────────────────────────────────────────────────────────────────────
# Core normalizer
# ──────────────────────────────────────────────────────────────────────────────

def normalize_name(stem: str, stain_tokens=DEFAULT_STAIN_TOKENS):
    """
    Strip a stain token from a filename stem.

    Returns:
        (normalized_stem, found_token)
        normalized_stem: lowercase, separators collapsed to "_", stain removed
        found_token:     the stain string that was matched, or None

    Examples:
        "ll4_liver_cd8_10x"  → ("ll4_liver_10x", "cd8")
        "Sample001_TIM3"     → ("sample001", "tim3")
        "CD8-Slide-003"      → ("slide_003", "cd8")
        "slide003"           → ("slide003", None)
    """
    s = stem.lower()
    found = None

    for token in stain_tokens:
        # Match token that is NOT surrounded by other alphanumeric characters
        # Handles: _cd8_, -cd8-, _cd8$, ^cd8_
        pattern = r'(?<![a-z0-9])' + re.escape(token) + r'(?![a-z0-9])'
        if re.search(pattern, s):
            found = token
            s = re.sub(pattern, '', s)
            break  # stop at first match — one stain per filename

    # Collapse leftover separators and trim
    s = re.sub(r'[\-_\s\.]+', '_', s).strip('_')
    return s, found


def get_image_files(folder: str) -> list:
    """Return sorted list of image filenames in folder (non-recursive)."""
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        return []
    return sorted([
        f for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Two-folder matching
# ──────────────────────────────────────────────────────────────────────────────

def match_two_folders(
    folder_a: str,
    folder_b: str,
    stain_tokens: list = DEFAULT_STAIN_TOKENS,
) -> dict:
    """
    Match images from two separate stain folders by normalized filename.

    Both of these will produce the same normalized key "ll4_liver_10x":
        folder_a/ll4_liver_cd8_10x.svs
        folder_b/ll4_liver_tim3_10x.svs

    Returns:
        {
          "pairs": [
            {
              "sample_id":  str,     normalized key
              "path_a":     str,     full path in folder_a
              "path_b":     str,     full path in folder_b
              "stain_a":    str,     detected stain token (uppercase) or "MARKER_A"
              "stain_b":    str,
              "filename_a": str,
              "filename_b": str,
            }, ...
          ],
          "unmatched_a": [filenames without a partner in B],
          "unmatched_b": [filenames without a partner in A],
          "mode": "two_folder",
        }
    """
    files_a = get_image_files(folder_a)
    files_b = get_image_files(folder_b)

    # Build norm → entry maps
    norm_a = {}
    for f in files_a:
        norm, stain = normalize_name(Path(f).stem, stain_tokens)
        # If two files in the same folder normalize to the same key, last one wins
        norm_a[norm] = {
            "file":  f,
            "stain": (stain or "marker_a").upper(),
            "path":  os.path.join(folder_a, f),
        }

    norm_b = {}
    for f in files_b:
        norm, stain = normalize_name(Path(f).stem, stain_tokens)
        norm_b[norm] = {
            "file":  f,
            "stain": (stain or "marker_b").upper(),
            "path":  os.path.join(folder_b, f),
        }

    pairs, matched_b = [], set()
    for norm, ea in norm_a.items():
        if norm in norm_b:
            eb = norm_b[norm]
            pairs.append({
                "sample_id":  norm,
                "path_a":     ea["path"],
                "path_b":     eb["path"],
                "stain_a":    ea["stain"],
                "stain_b":    eb["stain"],
                "filename_a": ea["file"],
                "filename_b": eb["file"],
            })
            matched_b.add(norm)

    unmatched_a = [v["file"] for k, v in norm_a.items() if k not in norm_b]
    unmatched_b = [v["file"] for k, v in norm_b.items() if k not in matched_b]

    print(f"  Two-folder match: {len(pairs)} pairs, "
          f"{len(unmatched_a)} unmatched in A, {len(unmatched_b)} unmatched in B")

    return {
        "pairs":       pairs,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "mode":        "two_folder",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-folder auto-detection
# ──────────────────────────────────────────────────────────────────────────────

def match_single_folder(
    folder: str,
    stain_tokens: list = DEFAULT_STAIN_TOKENS,
) -> dict:
    """
    Auto-detect stain pairs within a single folder.

    Groups files by normalized name; any group with ≥ 2 recognized stain tokens
    becomes a pair (or N-tuple for future multi-marker support).

    Files with no detectable stain token → unmatched.

    Returns same structure as match_two_folders, plus:
        "detected_stains": sorted list of all stain tokens found in the folder
        "groups":          full grouping dict for inspection / N-marker use
    """
    files = get_image_files(folder)

    # norm_name → {stain_token: full_path}
    groups: dict = {}
    no_stain: list = []

    for f in files:
        norm, stain = normalize_name(Path(f).stem, stain_tokens)
        if stain is None:
            no_stain.append(f)
            continue
        groups.setdefault(norm, {})[stain] = os.path.join(folder, f)

    pairs, unmatched, all_stains = [], [], set()
    for norm, stain_map in groups.items():
        all_stains.update(stain_map.keys())
        stain_list = sorted(stain_map.keys())  # deterministic order

        if len(stain_list) >= 2:
            pairs.append({
                "sample_id":  norm,
                "path_a":     stain_map[stain_list[0]],
                "path_b":     stain_map[stain_list[1]],
                "stain_a":    stain_list[0].upper(),
                "stain_b":    stain_list[1].upper(),
                "filename_a": Path(stain_map[stain_list[0]]).name,
                "filename_b": Path(stain_map[stain_list[1]]).name,
                # Full stain map kept for future N-marker support
                "all_stains": {k: Path(v).name for k, v in stain_map.items()},
            })
        else:
            # Only one stain found for this sample — can't pair
            unmatched.extend(Path(v).name for v in stain_map.values())

    # Files with no stain token at all are also unmatched
    unmatched.extend(no_stain)

    print(f"  Single-folder match: {len(pairs)} pairs, "
          f"{len(unmatched)} unmatched, stains found: {sorted(all_stains)}")

    return {
        "pairs":            pairs,
        "unmatched_a":      unmatched,
        "unmatched_b":      [],
        "detected_stains":  sorted(all_stains),
        "groups":           {k: {s: Path(v).name for s, v in sm.items()}
                             for k, sm in groups.items()},
        "mode":             "single_folder",
    }
