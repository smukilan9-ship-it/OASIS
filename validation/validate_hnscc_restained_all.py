"""All-image external validation for HNSCC-mIF-mIHC-comparison v2.

This harness is intentionally isolated from the production pipelines.  It runs
the Restained workflow without altering its algorithms or thresholds, validates
the unchanged InstanSeg detections against the released expert nuclear masks,
and evaluates AEC marker localization against the paired mIF channels.

The release does not contain expert CD8/FoxP3-positive cell labels.  Therefore
two distinct marker results are reported and never conflated:

1. Pixel-level mIF/AEC concordance, reproducing the type of analysis described
   in the dataset paper (mIF Otsu foreground versus AEC background).
2. Conditional cell-classification agreement against a clearly labelled,
   non-expert mIF intensity proxy.  This is a sensitivity analysis, not ground
   truth marker validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.restained.restained_coexpression import (  # noqa: E402
    _bh_adjust,
    _geometry_compartments,
    _load_geojson,
    _segment_reference,
    preprocess_hematoxylin,
    run_bundle,
    stain_channels,
    summarize_coexpression,
    validate_segmentation_reference,
)
from validation.validate_segmentation import match_centroids  # noqa: E402


PIXEL_SIZE_UM = 0.5
VALIDATION_TOLERANCE_UM = 5.0
CELL_EXPANSION_UM = 2.0
CD8_THRESHOLD_OD = 0.19
FOXP3_THRESHOLD_OD = 0.47


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(_json_safe(value), handle, indent=2)


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _describe(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": None, "q1": None, "q3": None,
                "minimum": None, "maximum": None}
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {"n": int(len(array)), "median": float(median), "q1": float(q1),
            "q3": float(q3), "minimum": float(array.min()),
            "maximum": float(array.max())}


def discover_dataset(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    folders = {name: root / name for name in ("mIF_Data", "mIHC_Data", "Segmentation")}
    missing_folders = [str(path) for path in folders.values() if not path.is_dir()]
    if missing_folders:
        raise FileNotFoundError("Missing dataset folder(s): " + ", ".join(missing_folders))

    sample_ids = sorted(path.stem for path in folders["Segmentation"].glob("*.png"))
    rows = []
    for sample_id in sample_ids:
        row = {
            "sample_id": sample_id,
            "case": sample_id.split("_")[0],
            "reference_mask": str(folders["Segmentation"] / f"{sample_id}.png"),
            "hematoxylin": str(folders["mIHC_Data"] / f"{sample_id}_Hematoxylin.png"),
            "mihc_cd8": str(folders["mIHC_Data"] / f"{sample_id}_CD8.png"),
            "mihc_foxp3": str(folders["mIHC_Data"] / f"{sample_id}_FoxP3.png"),
            "mif_dapi": str(folders["mIF_Data"] / f"{sample_id}_DAPI.png"),
            "mif_cd8": str(folders["mIF_Data"] / f"{sample_id}_CD8.png"),
            "mif_foxp3": str(folders["mIF_Data"] / f"{sample_id}_FoxP3.png"),
        }
        row["missing"] = [key for key, path in row.items()
                          if key not in {"sample_id", "case", "missing"}
                          and not Path(path).is_file()]
        row["complete_coexpression"] = not row["missing"]
        rows.append(row)

    png_counts = {name: len(list(path.glob("*.png"))) for name, path in folders.items()}
    missing_files = [{"sample_id": row["sample_id"], "missing": row["missing"]}
                     for row in rows if row["missing"]]
    audit = {
        "dataset_root": str(root),
        "png_counts": png_counts,
        "png_total": int(sum(png_counts.values())),
        "reference_mask_tiles": len(rows),
        "complete_cd8_foxp3_bundles": sum(row["complete_coexpression"] for row in rows),
        "incomplete_bundles": missing_files,
        "tiles_per_case": dict(sorted(Counter(row["case"] for row in rows).items())),
    }
    return rows, audit


def _mask_audit(mask_path):
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    blue = (rgb[..., 2] > 127) & (rgb[..., 0] < 127) & (rgb[..., 1] < 127)
    green = (rgb[..., 1] > 127) & (rgb[..., 0] < 127) & (rgb[..., 2] < 127)
    red = (rgb[..., 0] > 127) & (rgb[..., 1] < 127) & (rgb[..., 2] < 127)
    _, n_blue = ndimage.label(blue, structure=np.ones((3, 3), dtype=np.uint8))
    return {"blue_instances": int(n_blue), "blue_pixels": int(blue.sum()),
            "green_pixels": int(green.sum()), "red_pixels": int(red.sum())}


def _load_gt_labels(mask_path):
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    blue = (rgb[..., 2] > 127) & (rgb[..., 0] < 127) & (rgb[..., 1] < 127)
    labels, n_labels = ndimage.label(blue, structure=np.ones((3, 3), dtype=np.uint8))
    indices = np.arange(1, n_labels + 1)
    centroids_yx = np.asarray(ndimage.center_of_mass(blue, labels, indices), dtype=float)
    gt_xy = centroids_yx[:, ::-1]
    distance, nearest_indices = ndimage.distance_transform_edt(
        labels == 0, return_indices=True)
    nearest_labels = labels[tuple(nearest_indices)]
    expansion_px = CELL_EXPANSION_UM / PIXEL_SIZE_UM
    ring_labels = np.where((labels == 0) & (distance <= expansion_px),
                           nearest_labels, 0).astype(np.int32)
    return labels.astype(np.int32), ring_labels, gt_xy


def _mif_gray(path):
    return np.asarray(Image.open(path).convert("RGB")).max(axis=2).astype(np.uint8)


def _high_range_threshold(image):
    """DeepLIIF's documented automatic marker threshold, used only as proxy."""
    nonzero = image[image != 0]
    if not nonzero.size:
        return 0
    low, high = np.percentile(nonzero, [0.1, 99.9])
    low, high = round(float(low)), round(float(high))
    return round((high - low) * 0.9) + low


def _label_max(image, labels, n_labels):
    return np.asarray(ndimage.maximum(
        image, labels=labels, index=np.arange(1, n_labels + 1)), dtype=float)


def _binary_counts(reference, predicted):
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(reference & predicted))
    fp = int(np.sum(~reference & predicted))
    fn = int(np.sum(reference & ~predicted))
    tn = int(np.sum(~reference & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1}


def _sum_binary_counts(rows, prefix):
    counts = {key: sum(int(row[f"{prefix}_{key}"]) for row in rows)
              for key in ("tp", "fp", "fn", "tn")}
    reconstructed = ([True] * counts["tp"] + [False] * counts["fp"]
                     + [True] * counts["fn"] + [False] * counts["tn"])
    predictions = ([True] * counts["tp"] + [True] * counts["fp"]
                   + [False] * counts["fn"] + [False] * counts["tn"])
    return _binary_counts(reconstructed, predictions)


def _auc(reference, values):
    reference = np.asarray(reference, dtype=bool).ravel()
    values = np.asarray(values, dtype=float).ravel()
    n_positive = int(reference.sum())
    n_negative = int(len(reference) - n_positive)
    if not n_positive or not n_negative:
        return None
    ranks = stats.rankdata(values, method="average")
    return float((ranks[reference].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def pixel_concordance(mif_path, mihc_path):
    mif = _mif_gray(mif_path)
    otsu_threshold, _ = cv2.threshold(
        mif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = mif > otsu_threshold
    _, aec = stain_channels(np.asarray(Image.open(mihc_path).convert("RGB")))
    if not foreground.any() or foreground.all():
        raise ValueError(f"Degenerate mIF Otsu foreground: {mif_path}")
    foreground_mean = float(aec[foreground].mean())
    background_mean = float(aec[~foreground].mean())
    return {
        "mif_otsu_threshold": float(otsu_threshold),
        "mif_foreground_fraction": float(foreground.mean()),
        "aec_foreground_mean": foreground_mean,
        "aec_background_mean": background_mean,
        "aec_foreground_minus_background": foreground_mean - background_mean,
        "aec_pixel_auc": _auc(foreground, aec),
    }


def dapi_hematoxylin_structural_correlation(dapi_path, hematoxylin_path):
    """Zero-lag cross-modal structure diagnostic; not a registration metric.

    DAPI and deconvolved hematoxylin both emphasize nuclei.  Gaussian
    high-pass filtering suppresses illumination/tissue background.  A low or
    negative correlation warns that the expert DAPI-derived mask and the mIHC
    hematoxylin image may not be valid pixel-coordinate counterparts.  No
    cutoff is used to exclude tiles or to alter the primary results.
    """
    dapi = _mif_gray(dapi_path).astype(np.float32)
    hematoxylin, _ = stain_channels(
        np.asarray(Image.open(hematoxylin_path).convert("RGB")))

    def high_pass(image):
        narrow = cv2.GaussianBlur(image, (0, 0), 1)
        broad = cv2.GaussianBlur(image, (0, 0), 8)
        return narrow - broad

    dapi_hp = high_pass(dapi).ravel()
    hematoxylin_hp = high_pass(hematoxylin.astype(np.float32)).ravel()
    if not dapi_hp.std() or not hematoxylin_hp.std():
        return None
    return float(np.corrcoef(dapi_hp, hematoxylin_hp)[0, 1])


def _read_predicted_cells(cells_csv):
    rows = []
    with open(cells_csv) as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "x": float(row["x_px"]), "y": float(row["y_px"]),
                "cd8_positive": row["CD8_positive"].lower() == "true",
                "foxp3_positive": row["FOXP3_positive"].lower() == "true",
            })
    xy = np.asarray([[row["x"], row["y"]] for row in rows], dtype=float).reshape(-1, 2)
    return rows, xy


def proxy_classification_metrics(dataset_row, result):
    labels, ring_labels, gt_xy = _load_gt_labels(dataset_row["reference_mask"])
    n_gt = len(gt_xy)
    mif_cd8 = _mif_gray(dataset_row["mif_cd8"])
    mif_foxp3 = _mif_gray(dataset_row["mif_foxp3"])
    threshold_cd8 = _high_range_threshold(mif_cd8)
    threshold_foxp3 = _high_range_threshold(mif_foxp3)
    reference_cd8 = _label_max(mif_cd8, ring_labels, n_gt) > threshold_cd8
    reference_foxp3 = _label_max(mif_foxp3, labels, n_gt) > threshold_foxp3

    predicted, pred_xy = _read_predicted_cells(result["artifacts"]["cells_csv"])
    pairs, _, _, _ = match_centroids(
        gt_xy, pred_xy, VALIDATION_TOLERANCE_UM / PIXEL_SIZE_UM)
    gt_indices = np.asarray([pair[0] for pair in pairs], dtype=int)
    pred_indices = np.asarray([pair[1] for pair in pairs], dtype=int)
    predicted_cd8 = np.asarray([predicted[index]["cd8_positive"]
                                for index in pred_indices], dtype=bool)
    predicted_foxp3 = np.asarray([predicted[index]["foxp3_positive"]
                                  for index in pred_indices], dtype=bool)
    cd8 = _binary_counts(reference_cd8[gt_indices], predicted_cd8)
    foxp3 = _binary_counts(reference_foxp3[gt_indices], predicted_foxp3)
    reference_coexpression = summarize_coexpression(
        reference_cd8[gt_indices], reference_foxp3[gt_indices])
    predicted_coexpression = summarize_coexpression(predicted_cd8, predicted_foxp3)
    return {
        "matched_cells": len(pairs),
        "mif_proxy_method": (
            "per-image 90% of non-zero 0.1th-to-99.9th percentile range; "
            "cell maximum; CD8 2-um ring; FOXP3 nucleus; non-expert proxy"),
        "mif_proxy_threshold_cd8": threshold_cd8,
        "mif_proxy_threshold_foxp3": threshold_foxp3,
        "mif_proxy_cd8_positive": int(reference_cd8.sum()),
        "mif_proxy_foxp3_positive": int(reference_foxp3.sum()),
        "cd8": cd8,
        "foxp3": foxp3,
        "reference_proxy_coexpression": reference_coexpression,
        "predicted_coexpression_on_matched": predicted_coexpression,
    }


def _segmentation_config(args):
    return {
        "qupath_binary": str(Path(args.qupath_binary).expanduser()),
        "instanseg_model": str(Path(args.instanseg_model).expanduser()),
        "device": args.device,
        "instanseg_threads": args.threads,
        "timeout_seconds": args.timeout_seconds,
        "pixel_size_um": PIXEL_SIZE_UM,
    }


def run_all(args):
    dataset_rows, audit = discover_dataset(args.dataset_root)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    segmentation_root = output / "segmentation"
    pipeline_root = output / "full_pipeline"
    config = _segmentation_config(args)

    if not Path(config["qupath_binary"]).is_file():
        raise FileNotFoundError(f"QuPath binary not found: {config['qupath_binary']}")
    if not Path(config["instanseg_model"]).exists():
        raise FileNotFoundError(f"InstanSeg model not found: {config['instanseg_model']}")

    mask_audits = [_mask_audit(row["reference_mask"]) for row in dataset_rows]
    audit["reference_mask_red_pixels_total"] = sum(row["red_pixels"] for row in mask_audits)
    audit["reference_mask_blue_instances_total"] = sum(
        row["blue_instances"] for row in mask_audits)
    _write_json(output / "dataset_audit.json", audit)

    tile_rows = []
    results = []
    for index, row in enumerate(dataset_rows, start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(dataset_rows)}] {sample_id}", flush=True)
        segmentation_dir = segmentation_root / sample_id
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        processed = segmentation_dir / f"{sample_id}_H_preprocessed.png"
        detection_geojson = segmentation_dir / f"{sample_id}_H_preprocessed_detections.geojson"
        if not detection_geojson.exists():
            preprocess_hematoxylin(row["hematoxylin"], processed)
            _segment_reference(processed, segmentation_dir, config)
        elif not processed.exists():
            raise RuntimeError(f"Resume detection exists without preprocessed image: {sample_id}")

        geojson = _load_geojson(detection_geojson)
        nuclei, _, _ = _geometry_compartments(
            geojson["features"], PIXEL_SIZE_UM, CELL_EXPANSION_UM)
        segmentation = validate_segmentation_reference(
            row["reference_mask"], nuclei, PIXEL_SIZE_UM,
            VALIDATION_TOLERANCE_UM)

        tile = {
            "sample_id": sample_id,
            "case": row["case"],
            "complete_coexpression": row["complete_coexpression"],
            **{f"seg_{key}": value for key, value in segmentation.items()},
        }
        tile["dapi_hematoxylin_structural_correlation"] = (
            dapi_hematoxylin_structural_correlation(
                row["mif_dapi"], row["hematoxylin"]))

        for marker, mif_key, mihc_key in (
                ("cd8", "mif_cd8", "mihc_cd8"),
                ("foxp3", "mif_foxp3", "mihc_foxp3")):
            if Path(row[mif_key]).exists() and Path(row[mihc_key]).exists():
                concordance = pixel_concordance(row[mif_key], row[mihc_key])
                tile.update({f"{marker}_{key}": value
                             for key, value in concordance.items()})

        if row["complete_coexpression"]:
            bundle = {
                "sample_id": sample_id,
                "hematoxylin": row["hematoxylin"],
                "marker_a": row["mihc_cd8"],
                "marker_b": row["mihc_foxp3"],
                "reference_mask": row["reference_mask"],
            }
            run_config = {
                **config,
                "threshold_a": CD8_THRESHOLD_OD,
                "threshold_b": FOXP3_THRESHOLD_OD,
                "label_a": "CD8",
                "label_b": "FOXP3",
                "compartment_a": "ring",
                "compartment_b": "nucleus",
                "cell_expansion_um": CELL_EXPANSION_UM,
                "preprocess_hematoxylin": True,
                "detections_geojson": str(detection_geojson),
                "validation_tolerance_um": VALIDATION_TOLERANCE_UM,
            }
            result_json = pipeline_root / sample_id / f"{sample_id}_restained_result.json"
            if result_json.exists():
                with open(result_json) as handle:
                    result = json.load(handle)
                result.setdefault("artifacts", {})["result_json"] = str(result_json)
            else:
                result = run_bundle(bundle, run_config, pipeline_root)
            proxy = proxy_classification_metrics(row, result)
            coexpression = result["coexpression"]
            tile.update({
                "cells": coexpression["total_cells"],
                "cd8_positive": coexpression["marker_a_positive"],
                "foxp3_positive": coexpression["marker_b_positive"],
                "double_positive": coexpression["double_positive"],
                "fisher_odds_ratio": coexpression["fisher_odds_ratio"],
                "fisher_p": coexpression["fisher_p_value"],
                "proxy_matched_cells": proxy["matched_cells"],
                "proxy_threshold_cd8": proxy["mif_proxy_threshold_cd8"],
                "proxy_threshold_foxp3": proxy["mif_proxy_threshold_foxp3"],
            })
            for marker in ("cd8", "foxp3"):
                tile.update({f"proxy_{marker}_{key}": value
                             for key, value in proxy[marker].items()})
            result["external_validation"] = proxy
            results.append(result)
        tile_rows.append(tile)
        _write_rows(output / "tile_metrics.partial.csv", tile_rows)

    q_values = _bh_adjust([row["fisher_p"] for row in tile_rows
                           if row["complete_coexpression"]])
    q_index = 0
    for row in tile_rows:
        if row["complete_coexpression"]:
            row["fisher_q_bh"] = q_values[q_index]
            results[q_index]["coexpression"]["fisher_q_value_bh"] = q_values[q_index]
            _write_json(results[q_index]["artifacts"]["result_json"], results[q_index])
            q_index += 1

    segmentation_totals = {
        key: sum(int(row[f"seg_{key}"]) for row in tile_rows)
        for key in ("tp", "fp", "fn", "n_gt", "n_pred")
    }
    seg_precision = segmentation_totals["tp"] / (
        segmentation_totals["tp"] + segmentation_totals["fp"])
    seg_recall = segmentation_totals["tp"] / (
        segmentation_totals["tp"] + segmentation_totals["fn"])
    segmentation_totals.update({
        "precision": seg_precision,
        "recall": seg_recall,
        "f1": 2 * seg_precision * seg_recall / (seg_precision + seg_recall),
        "per_tile_f1": _describe([row["seg_f1"] for row in tile_rows]),
        "per_tile_pixel_dice": _describe([row["seg_pixel_dice"] for row in tile_rows]),
    })

    complete_rows = [row for row in tile_rows if row["complete_coexpression"]]
    proxy_totals = {marker: _sum_binary_counts(complete_rows, f"proxy_{marker}")
                    for marker in ("cd8", "foxp3")}
    pixel_summary = {}
    for marker in ("cd8", "foxp3"):
        available = [row for row in tile_rows if f"{marker}_aec_pixel_auc" in row]
        deltas = [row[f"{marker}_aec_foreground_minus_background"] for row in available]
        pixel_summary[marker] = {
            "n_tiles": len(available),
            "foreground_mean_exceeds_background": int(np.sum(np.asarray(deltas) > 0)),
            "foreground_minus_background": _describe(deltas),
            "pixel_auc": _describe([row[f"{marker}_aec_pixel_auc"] for row in available]),
            "mif_foreground_fraction": _describe(
                [row[f"{marker}_mif_foreground_fraction"] for row in available]),
        }

    structure_values = np.asarray(
        [row["dapi_hematoxylin_structural_correlation"] for row in tile_rows],
        dtype=float)
    segmentation_f1_values = np.asarray(
        [row["seg_f1"] for row in tile_rows], dtype=float)
    structure_f1_spearman = stats.spearmanr(
        structure_values, segmentation_f1_values)
    reference_diagnostic = {
        "method": (
            "zero-lag Pearson correlation of Gaussian-high-pass mIF DAPI and "
            "fixed-deconvolved mIHC hematoxylin; diagnostic only; no exclusion cutoff"),
        "structural_correlation": _describe(structure_values),
        "spearman_with_segmentation_f1": float(structure_f1_spearman.statistic),
        "spearman_p_value": float(structure_f1_spearman.pvalue),
        "lowest_correlation_tiles": [
            {"sample_id": row["sample_id"],
             "correlation": row["dapi_hematoxylin_structural_correlation"],
             "segmentation_f1": row["seg_f1"]}
            for row in sorted(
                tile_rows,
                key=lambda item: item["dapi_hematoxylin_structural_correlation"])[:10]
        ],
    }

    # Rebuild aligned vectors from the four pooled categories. This pooled test
    # is descriptive only because cells/tiles within a patient are not
    # independent experimental replicates.
    double = sum(row["double_positive"] for row in complete_rows)
    cd8_only = sum(row["cd8_positive"] - row["double_positive"] for row in complete_rows)
    fox_only = sum(row["foxp3_positive"] - row["double_positive"] for row in complete_rows)
    neither = sum(row["cells"] - row["cd8_positive"] - row["foxp3_positive"]
                  + row["double_positive"] for row in complete_rows)
    aggregate = summarize_coexpression(
        [True] * double + [True] * cd8_only + [False] * fox_only + [False] * neither,
        [True] * double + [False] * cd8_only + [True] * fox_only + [False] * neither)

    def aggregate_external_coexpression(key):
        totals = np.zeros((2, 2), dtype=int)
        for result in results:
            totals += np.asarray(
                result["external_validation"][key]["contingency_table"], dtype=int)
        a, b = (int(value) for value in totals[0])
        c, d = (int(value) for value in totals[1])
        return summarize_coexpression(
            [True] * a + [True] * b + [False] * c + [False] * d,
            [True] * a + [False] * b + [True] * c + [False] * d)

    proxy_reference_coexpression = aggregate_external_coexpression(
        "reference_proxy_coexpression")
    matched_prediction_coexpression = aggregate_external_coexpression(
        "predicted_coexpression_on_matched")
    significant_tiles = sorted(
        [row for row in complete_rows if row.get("fisher_q_bh", 1) < 0.05],
        key=lambda item: item["fisher_q_bh"])

    per_case = {}
    for case in sorted({row["case"] for row in tile_rows}):
        case_rows = [row for row in complete_rows if row["case"] == case]
        per_case[case] = {
            "n_tiles": len(case_rows),
            "segmentation_f1_median": float(np.median(
                [row["seg_f1"] for row in tile_rows if row["case"] == case])),
            "cd8_proxy_f1": _sum_binary_counts(case_rows, "proxy_cd8") if case_rows else None,
            "foxp3_proxy_f1": _sum_binary_counts(case_rows, "proxy_foxp3") if case_rows else None,
        }

    summary = {
        "status": "complete",
        "method": {
            "section_type": "same physical section, mIF followed by stripped/restained AEC mIHC",
            "registration": "not rerun; public 512x512 patches are author-co-registered",
            "segmentation": "fixed H preprocessing plus unchanged QuPath InstanSeg brightfield_nuclei-0.1.1",
            "segmentation_tolerance_um": VALIDATION_TOLERANCE_UM,
            "pixel_size_um": PIXEL_SIZE_UM,
            "cd8": {"threshold_aec_od": CD8_THRESHOLD_OD, "compartment": "2-um ring"},
            "foxp3": {"threshold_aec_od": FOXP3_THRESHOLD_OD, "compartment": "nucleus"},
            "thresholds_tuned_during_all_image_validation": False,
        },
        "dataset_audit": audit,
        "segmentation_ground_truth": segmentation_totals,
        "pixel_level_mif_aec_concordance": pixel_summary,
        "dapi_hematoxylin_reference_correspondence_diagnostic": reference_diagnostic,
        "conditional_cell_classification_vs_nonexpert_mif_proxy": proxy_totals,
        "coexpression": {
            "n_complete_tiles": len(complete_rows),
            "tiles_bh_q_below_0_05": len(significant_tiles),
            "pooled_descriptive_only": aggregate,
            "mif_proxy_pooled_descriptive_only": proxy_reference_coexpression,
            "aec_prediction_on_mif_matched_cells_descriptive_only": (
                matched_prediction_coexpression),
            "most_significant_tile": ({
                key: significant_tiles[0].get(key) for key in (
                    "sample_id", "case", "cells", "cd8_positive",
                    "foxp3_positive", "double_positive", "fisher_p",
                    "fisher_q_bh", "cd8_aec_pixel_auc",
                    "foxp3_aec_pixel_auc", "seg_f1")
            } if significant_tiles else None),
            "validity_verdict": "NOT_VALID_FOR_BIOLOGICAL_INFERENCE",
            "validity_reason": (
                "Binary marker labels lack expert ground truth and the apparent "
                "signal includes grossly non-corresponding tiles; the most "
                "significant tile has weak mIF/AEC pixel AUCs and cannot support "
                "same-cell biological co-expression."),
        },
        "per_case": per_case,
        "interpretation_boundary": (
            "Expert ground truth applies to nuclear segmentation only. Pixel-level paired mIF "
            "supports technical AEC localization. The mIF cell labels are a reproducible "
            "intensity proxy, not pathologist CD8/FoxP3 positivity ground truth; biological "
            "coexpression results are therefore descriptive and not clinically validated."),
    }
    _write_rows(output / "tile_metrics.csv", tile_rows)
    _write_json(output / "all_image_validation_summary.json", summary)
    _write_json(output / "full_pipeline_results.json", {
        "summary": summary, "results": results})
    partial = output / "tile_metrics.partial.csv"
    if partial.exists():
        partial.unlink()
    print("VALIDATION_RESULT=" + str(output / "all_image_validation_summary.json"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="All-image HNSCC restained co-expression validation")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qupath-binary", required=True)
    parser.add_argument("--instanseg-model", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
