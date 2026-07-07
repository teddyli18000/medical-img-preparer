import glob
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from tqdm import tqdm


IMAGE_SUFFIX = "_image_prep.nii.gz"
LABEL_SUFFIX = "_label_prep.nii.gz"


def analyze_prep_headers(config) -> Dict[str, Any]:
    preprocessed_dir = config.preprocessed_dir()
    _validate_input_dir(config, preprocessed_dir)

    image_files = _list_nii_files(preprocessed_dir, config.FILE_GLOB_IMAGE, config.SORT_FILES)
    label_files = _list_nii_files(preprocessed_dir, config.FILE_GLOB_LABEL, config.SORT_FILES)

    label_map = {_sample_id_from_path(path, LABEL_SUFFIX): path for path in label_files}
    image_ids = [_sample_id_from_path(path, IMAGE_SUFFIX) for path in image_files]
    image_id_set = set(image_ids)
    label_ids = set(label_map.keys())

    missing_label_ids = sorted([sample_id for sample_id in image_ids if sample_id not in label_ids])
    orphan_label_ids = sorted([sample_id for sample_id in label_ids if sample_id not in image_id_set])

    samples: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    spacing_rows: List[List[float]] = []
    shape_rows: List[List[int]] = []
    orientation_counter: Counter = Counter()
    image_dtype_counter: Counter = Counter()
    label_dtype_counter: Counter = Counter()
    geometry_mismatch_counts: Counter = Counter()
    mismatch_sample_ids: List[str] = []

    for image_path in tqdm(
        image_files,
        desc=config.PROGRESS_DESC,
        unit=config.PROGRESS_UNIT,
        dynamic_ncols=True,
    ):
        sample_id = _sample_id_from_path(image_path, IMAGE_SUFFIX)
        label_path = label_map.get(sample_id)
        label_exists = label_path is not None

        sample_record = {
            "sample_id": sample_id,
            "image_path": _path_for_output(image_path, config.SAVE_ABSOLUTE_PATHS),
            "label_path": _path_for_output(label_path, config.SAVE_ABSOLUTE_PATHS) if label_path else None,
            "label_exists": bool(label_exists),
            "spacing_xyz": None,
            "shape_xyz": None,
            "orientation": None,
            "image_dtype": None,
            "label_spacing_xyz": None,
            "label_shape_xyz": None,
            "label_orientation": None,
            "label_dtype": None,
            "geometry_match": None,
            "geometry_mismatch_types": [],
            "read_status": "success",
            "error_message": None,
        }

        try:
            nifti_img = cast(nib.Nifti1Image, nib.load(image_path))

            zooms = nifti_img.header.get_zooms()
            if len(zooms) < config.HEADER_DIMENSIONS:
                raise ValueError(
                    f"Header zoom dimensions < {config.HEADER_DIMENSIONS}: {len(zooms)}"
                )
            spacing_xyz = [float(zooms[i]) for i in range(config.HEADER_DIMENSIONS)]

            shape = nifti_img.shape
            if len(shape) < config.HEADER_DIMENSIONS:
                raise ValueError(
                    f"Image shape dimensions < {config.HEADER_DIMENSIONS}: {len(shape)}"
                )
            shape_xyz = [int(shape[i]) for i in range(config.HEADER_DIMENSIONS)]

            orientation_codes = aff2axcodes(nifti_img.affine)
            orientation = "".join(code if code is not None else "?" for code in orientation_codes[:3])

            image_dtype = _dtype_name(nifti_img.get_data_dtype())

            sample_record["spacing_xyz"] = spacing_xyz
            sample_record["shape_xyz"] = shape_xyz
            sample_record["orientation"] = orientation
            sample_record["image_dtype"] = image_dtype

            spacing_rows.append(spacing_xyz)
            shape_rows.append(shape_xyz)
            orientation_counter[orientation] += 1
            image_dtype_counter[image_dtype] += 1

            if label_exists and label_path:
                label_img = cast(nib.Nifti1Image, nib.load(label_path))

                label_zooms = label_img.header.get_zooms()
                if len(label_zooms) < config.HEADER_DIMENSIONS:
                    raise ValueError(
                        f"Label header zoom dimensions < {config.HEADER_DIMENSIONS}: {len(label_zooms)}"
                    )
                label_spacing = [float(label_zooms[i]) for i in range(config.HEADER_DIMENSIONS)]

                label_shape = label_img.shape
                if len(label_shape) < config.HEADER_DIMENSIONS:
                    raise ValueError(
                        f"Label shape dimensions < {config.HEADER_DIMENSIONS}: {len(label_shape)}"
                    )
                label_shape_xyz = [int(label_shape[i]) for i in range(config.HEADER_DIMENSIONS)]

                label_orientation_codes = aff2axcodes(label_img.affine)
                label_orientation = "".join(
                    code if code is not None else "?" for code in label_orientation_codes[:3]
                )

                label_dtype = _dtype_name(label_img.get_data_dtype())

                sample_record["label_spacing_xyz"] = label_spacing
                sample_record["label_shape_xyz"] = label_shape_xyz
                sample_record["label_orientation"] = label_orientation
                sample_record["label_dtype"] = label_dtype

                label_dtype_counter[label_dtype] += 1

                mismatch_types = _compare_geometry(
                    spacing_xyz,
                    shape_xyz,
                    orientation,
                    nifti_img.affine,
                    label_spacing,
                    label_shape_xyz,
                    label_orientation,
                    label_img.affine,
                    config,
                )
                if mismatch_types:
                    sample_record["geometry_match"] = False
                    sample_record["geometry_mismatch_types"] = mismatch_types
                    mismatch_sample_ids.append(sample_id)
                    for item in mismatch_types:
                        geometry_mismatch_counts[item] += 1
                else:
                    sample_record["geometry_match"] = True
        except Exception as exc:  # noqa: BLE001
            sample_record["read_status"] = "failed"
            sample_record["error_message"] = f"{exc.__class__.__name__}: {exc}"
            failures.append(
                {
                    "sample_id": sample_id,
                    "image_path": sample_record["image_path"],
                    "error_message": sample_record["error_message"],
                }
            )

        samples.append(sample_record)

    spacing_stats = _compute_spacing_stats(spacing_rows, config.AXIS_NAMES)
    shape_stats = _compute_shape_stats(shape_rows, config.AXIS_NAMES)

    summary = {
        "dataset_name": config.dataset_name(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessed_dir": preprocessed_dir,
        "image_file_count": int(len(image_files)),
        "label_file_count": int(len(label_files)),
        "counts_match": bool(len(image_files) == len(label_files)),
        "missing_label_count": int(len(missing_label_ids)),
        "orphan_label_count": int(len(orphan_label_ids)),
        "header_read_success_count": int(len(samples) - len(failures)),
        "header_read_failed_count": int(len(failures)),
    }

    stats = {
        "spacing_xyz": spacing_stats,
        "shape_xyz": shape_stats,
        "orientation_distribution": _sorted_counter(orientation_counter),
        "image_dtype_distribution": _sorted_counter(image_dtype_counter),
        "label_dtype_distribution": _sorted_counter(label_dtype_counter),
    }

    geometry_consistency = {
        "mismatch_sample_count": int(len(mismatch_sample_ids)),
        "mismatch_sample_ids": mismatch_sample_ids,
        "mismatch_counts": _sorted_counter(geometry_mismatch_counts),
    }

    conclusions = _build_conclusions(summary, stats, geometry_consistency, config)

    return {
        "summary": summary,
        "stats": stats,
        "consistency": {
            "missing_label_sample_ids": missing_label_ids,
            "orphan_label_sample_ids": orphan_label_ids,
        },
        "geometry_consistency": geometry_consistency,
        "samples": samples,
        "failures": failures,
        "conclusions": conclusions,
        "implementation_notes": {
            "header_only": True,
            "voxel_data_loaded": False,
            "reader": "nibabel",
            "label_header_compared": True,
        },
    }


def _validate_input_dir(config, preprocessed_dir: str) -> None:
    if not config.FAIL_ON_MISSING_DIR:
        return
    if not os.path.isdir(preprocessed_dir):
        raise FileNotFoundError(f"preprocessed directory does not exist: {preprocessed_dir}")


def _list_nii_files(directory: str, file_glob: str, sort_files: bool) -> List[str]:
    files = glob.glob(os.path.join(directory, file_glob), recursive=True)
    return sorted(files) if sort_files else files


def _sample_id_from_path(path: str, suffix: str) -> str:
    name = os.path.basename(path)
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _path_for_output(path: Optional[str], save_absolute_paths: bool) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if save_absolute_paths else path


def _dtype_name(dtype: Any) -> str:
    try:
        return str(np.dtype(dtype))
    except Exception:  # noqa: BLE001
        return str(dtype)


def _compute_spacing_stats(rows: List[List[float]], axis_names) -> Dict[str, Dict[str, Any]]:
    if not rows:
        return {axis: {"min": None, "max": None, "median": None} for axis in axis_names}

    arr = np.asarray(rows, dtype=np.float64)
    result = {}
    for i, axis in enumerate(axis_names):
        col = arr[:, i]
        result[axis] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
        }
    return result


def _compute_shape_stats(rows: List[List[int]], axis_names) -> Dict[str, Dict[str, Any]]:
    if not rows:
        return {axis: {"min": None, "max": None, "median": None} for axis in axis_names}

    arr = np.asarray(rows, dtype=np.int64)
    result = {}
    for i, axis in enumerate(axis_names):
        col = arr[:, i]
        result[axis] = {
            "min": int(np.min(col)),
            "max": int(np.max(col)),
            "median": float(np.median(col)),
        }
    return result


def _sorted_counter(counter: Counter) -> Dict[str, int]:
    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {str(k): int(v) for k, v in sorted_items}


def _build_conclusions(
    summary: Dict[str, Any],
    stats: Dict[str, Any],
    geometry: Dict[str, Any],
    config,
) -> List[str]:
    conclusions = []

    if summary["counts_match"] and summary["missing_label_count"] == 0 and summary["orphan_label_count"] == 0:
        conclusions.append("Image/Label file counts match with no missing or orphan labels.")
    else:
        conclusions.append("Image/Label count mismatch detected. Fix missing/orphan labels first.")

    if summary["header_read_failed_count"] == 0:
        conclusions.append("All headers are readable; no corrupted files detected.")
    else:
        conclusions.append(
            f"Found {summary['header_read_failed_count']} header read failures; fix data quality issues first."
        )

    mismatch_count = geometry.get("mismatch_sample_count", 0)
    if mismatch_count == 0:
        conclusions.append("Image/Label geometry is consistent across all samples.")
    else:
        conclusions.append(
            f"Found {mismatch_count} samples with image/label geometry mismatch; fix before training."
        )

    orientation_dist = stats.get("orientation_distribution", {})
    if len(orientation_dist) <= 1:
        only_orientation = next(iter(orientation_dist.keys()), "N/A")
        conclusions.append(f"Orientation is consistent (dominant: {only_orientation}).")
    else:
        conclusions.append("Orientation has multiple codes; verify preprocessing output.")

    spacing_stats = stats.get("spacing_xyz", {})
    if _spacing_is_uniform(spacing_stats, config.SPACING_RANGE_TOL):
        conclusions.append("Spacing is highly consistent after preprocessing.")
    else:
        conclusions.append("Spacing shows variability; verify resampling output.")

    if config.EXPECTED_SPACING:
        spacing_ok = _spacing_matches_expected(spacing_stats, config.EXPECTED_SPACING, config.SPACING_RANGE_TOL)
        if spacing_ok:
            conclusions.append("Spacing matches expected target spacing.")
        else:
            conclusions.append("Spacing does not match expected target spacing.")

    if config.EXPECTED_ORIENTATION:
        expected = str(config.EXPECTED_ORIENTATION)
        if list(orientation_dist.keys()) == [expected]:
            conclusions.append("Orientation matches expected target orientation.")
        else:
            conclusions.append("Orientation does not match expected target orientation.")

    return conclusions


def _spacing_is_uniform(spacing_stats: Dict[str, Dict[str, Any]], tol: float) -> bool:
    for axis in ("x", "y", "z"):
        axis_info = spacing_stats.get(axis, {})
        min_v = axis_info.get("min")
        max_v = axis_info.get("max")
        if min_v is None or max_v is None:
            return False
        if abs(max_v - min_v) > tol:
            return False
    return True


def _spacing_matches_expected(
    spacing_stats: Dict[str, Dict[str, Any]], expected: Any, tol: float
) -> bool:
    if expected is None:
        return True
    if not isinstance(expected, (list, tuple)) or len(expected) < 3:
        return False
    for axis, target in zip(("x", "y", "z"), expected):
        axis_info = spacing_stats.get(axis, {})
        median_v = axis_info.get("median")
        if median_v is None:
            return False
        if abs(float(median_v) - float(target)) > tol:
            return False
    return True


def _compare_geometry(
    image_spacing: List[float],
    image_shape: List[int],
    image_orientation: str,
    image_affine: np.ndarray,
    label_spacing: List[float],
    label_shape: List[int],
    label_orientation: str,
    label_affine: np.ndarray,
    config,
) -> List[str]:
    mismatch_types: List[str] = []

    if not _match_float_list(image_spacing, label_spacing, config.SPACING_ATOL):
        mismatch_types.append("spacing")

    if not _match_int_list(image_shape, label_shape):
        mismatch_types.append("shape")

    if image_orientation != label_orientation:
        mismatch_types.append("orientation")

    if not _match_affine(image_affine, label_affine, config.AFFINE_ATOL):
        mismatch_types.append("affine")

    return mismatch_types


def _match_float_list(left: List[float], right: List[float], atol: float) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= atol for a, b in zip(left, right))


def _match_int_list(left: List[int], right: List[int]) -> bool:
    if len(left) != len(right):
        return False
    return all(int(a) == int(b) for a, b in zip(left, right))


def _match_affine(left: np.ndarray, right: np.ndarray, atol: float) -> bool:
    if left.shape != right.shape:
        return False
    return bool(np.allclose(left, right, atol=atol))
