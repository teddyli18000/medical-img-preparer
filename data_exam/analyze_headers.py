import glob
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from tqdm import tqdm


NIFTI_SUFFIXES = (".nii.gz", ".nii")


def analyze_dataset_headers(config) -> Dict[str, Any]:
    images_dir = config.images_dir()
    labels_dir = config.labels_dir()

    _validate_input_dirs(config, images_dir, labels_dir)

    image_files = _list_nii_files(images_dir, config.FILE_GLOB, config.SORT_FILES)
    label_files = _list_nii_files(labels_dir, config.FILE_GLOB, config.SORT_FILES)

    label_map = {_sample_id_from_path(path): path for path in label_files}
    image_ids = [_sample_id_from_path(path) for path in image_files]
    image_id_set = set(image_ids)
    label_ids = set(label_map.keys())

    missing_label_ids = sorted([sample_id for sample_id in image_ids if sample_id not in label_ids])
    orphan_label_ids = sorted([sample_id for sample_id in label_ids if sample_id not in image_id_set])

    samples: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    spacing_rows: List[List[float]] = []
    shape_rows: List[List[int]] = []
    orientation_counter: Counter = Counter()

    for image_path in tqdm(
        image_files,
        desc=config.PROGRESS_DESC,
        unit=config.PROGRESS_UNIT,
        dynamic_ncols=True,
    ):
        sample_id = _sample_id_from_path(image_path)
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
            "read_status": "success",
            "error_message": None,
        }

        try:
            # Header-only metadata access. Do not load full voxel array.
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

            sample_record["spacing_xyz"] = spacing_xyz
            sample_record["shape_xyz"] = shape_xyz
            sample_record["orientation"] = orientation

            spacing_rows.append(spacing_xyz)
            shape_rows.append(shape_xyz)
            orientation_counter[orientation] += 1

        except Exception as exc:  # noqa: BLE001 - intentional resilience for bad files
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
        "raw_data_dir": config.raw_data_dir(),
        "images_dir": images_dir,
        "labels_dir": labels_dir,
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
    }

    conclusions = _build_conclusions(summary, stats)

    return {
        "summary": summary,
        "stats": stats,
        "consistency": {
            "missing_label_sample_ids": missing_label_ids,
            "orphan_label_sample_ids": orphan_label_ids,
        },
        "samples": samples,
        "failures": failures,
        "conclusions": conclusions,
        "implementation_notes": {
            "header_only": True,
            "voxel_data_loaded": False,
            "reader": "nibabel",
        },
    }


def _validate_input_dirs(config, images_dir: str, labels_dir: str) -> None:
    if not config.FAIL_ON_MISSING_DIR:
        return

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"imagesTr directory does not exist: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labelsTr directory does not exist: {labels_dir}")


def _list_nii_files(directory: str, file_glob: str, sort_files: bool) -> List[str]:
    files = glob.glob(os.path.join(directory, file_glob))
    return sorted(files) if sort_files else files


def _sample_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _path_for_output(path: Optional[str], save_absolute_paths: bool) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if save_absolute_paths else path


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


def _build_conclusions(summary: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
    conclusions = []

    if summary["counts_match"] and summary["missing_label_count"] == 0 and summary["orphan_label_count"] == 0:
        conclusions.append("Image/Label 文件数量一致，且未发现缺失或多余标签。")
    else:
        conclusions.append(
            "Image/Label 存在数量或配对不一致，请优先修复 missing_label_sample_ids / orphan_label_sample_ids。"
        )

    if summary["header_read_failed_count"] == 0:
        conclusions.append("所有样本 Header 均可读取，未发现损坏文件。")
    else:
        conclusions.append(
            f"发现 {summary['header_read_failed_count']} 个样本 Header 读取失败，建议先处理数据质量问题。"
        )

    orientation_dist = stats.get("orientation_distribution", {})
    if len(orientation_dist) <= 1:
        only_orientation = next(iter(orientation_dist.keys()), "N/A")
        conclusions.append(f"空间方向基本一致（主方向: {only_orientation}）。")
    else:
        conclusions.append("空间方向存在多种坐标系，建议在训练前做统一方向标准化。")

    spacing_stats = stats.get("spacing_xyz", {})
    shape_stats = stats.get("shape_xyz", {})

    spacing_summary = _axis_range_summary(spacing_stats)
    shape_summary = _axis_range_summary(shape_stats)
    conclusions.append(f"Spacing 轴向范围概览: {spacing_summary}。")
    conclusions.append(f"Shape 轴向范围概览: {shape_summary}。")

    return conclusions


def _axis_range_summary(axis_stats: Dict[str, Dict[str, Any]]) -> str:
    chunks = []
    for axis in ("x", "y", "z"):
        axis_info = axis_stats.get(axis, {})
        min_v = axis_info.get("min")
        max_v = axis_info.get("max")
        if min_v is None or max_v is None:
            chunks.append(f"{axis}: N/A")
        else:
            chunks.append(f"{axis}: {min_v}~{max_v}")
    return ", ".join(chunks)
