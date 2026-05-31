from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}

DTYPE_NAMES = {
    2: "uint8",
    4: "int16",
    8: "int32",
    16: "float32",
    64: "float64",
    256: "int8",
    512: "uint16",
    768: "uint32",
}

IMAGE_SUFFIX = "_image_prep.nii.gz"
LABEL_SUFFIX = "_label_prep.nii.gz"


@dataclass
class NiftiHeader:
    path: str
    endian: str
    shape: Tuple[int, ...]
    shape_xyz: Tuple[int, int, int]
    datatype_code: int
    datatype_name: str
    bitpix: int
    spacing_xyz: Tuple[float, float, float]
    affine: Tuple[Tuple[float, float, float, float], ...]
    orientation: str
    vox_offset: int
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int


@dataclass
class SampleResult:
    sample_id: str
    status: str
    image_path: str
    label_path: str
    image_shape: Optional[Tuple[int, int, int]]
    label_shape: Optional[Tuple[int, int, int]]
    image_spacing: Optional[Tuple[float, float, float]]
    label_spacing: Optional[Tuple[float, float, float]]
    image_orientation: Optional[str]
    label_orientation: Optional[str]
    image_dtype: Optional[str]
    label_dtype: Optional[str]
    image_min: Optional[float]
    image_max: Optional[float]
    image_mean: Optional[float]
    image_finite: Optional[bool]
    label_values: Optional[List[float]]
    label_is_integer: Optional[bool]
    label_nonempty: Optional[bool]
    mask_voxels: Optional[int]
    mask_bbox_xyz: Optional[List[List[int]]]
    mask_body_overlap_ratio: Optional[float]
    z_with_largest_mask: Optional[int]
    overlay_path: Optional[str]
    failures: List[str]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Full acceptance validation for preprocessed MSD Task07 data."
    )
    parser.add_argument(
        "--preprocessed-dir",
        default=str(root / "data" / "preprocessed" / "processed_MSD_Task7"),
        help="Directory containing per-sample preprocessed image/label folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(root / "task7_preprocess_validation" / "output"),
        help="Output directory for JSON/CSV/Markdown/overlay artifacts.",
    )
    parser.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        default=(0.8027, 0.8027, 2.5),
        metavar=("X", "Y", "Z"),
        help="Expected preprocessed spacing in mm.",
    )
    parser.add_argument(
        "--spacing-atol",
        type=float,
        default=5.0e-4,
        help="Absolute tolerance for spacing comparisons.",
    )
    parser.add_argument(
        "--affine-atol",
        type=float,
        default=1.0e-3,
        help="Absolute tolerance for affine matrix comparisons.",
    )
    parser.add_argument(
        "--image-min-atol",
        type=float,
        default=1.0e-5,
        help="Allowed negative tolerance for normalized image minimum.",
    )
    parser.add_argument(
        "--image-max-atol",
        type=float,
        default=1.0e-5,
        help="Allowed positive tolerance for normalized image maximum.",
    )
    parser.add_argument(
        "--expected-label-values",
        nargs="+",
        type=float,
        default=(0.0, 1.0, 2.0),
        help="Allowed label values after preprocessing.",
    )
    parser.add_argument(
        "--body-threshold",
        type=float,
        default=0.0,
        help="Image values above this threshold are treated as non-background.",
    )
    parser.add_argument(
        "--min-mask-body-overlap",
        type=float,
        default=0.95,
        help="Minimum fraction of mask voxels overlapping non-background image.",
    )
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=12,
        help="Number of evenly sampled overlay PNGs to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    pairs, pairing_failures = discover_pairs(preprocessed_dir)
    overlay_ids = choose_overlay_ids([sample_id for sample_id, _, _ in pairs], args.overlay_count)

    samples: List[SampleResult] = []
    for index, (sample_id, image_path, label_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] validating {sample_id}")
        overlay_path = overlays_dir / f"{sample_id}_overlay.png" if sample_id in overlay_ids else None
        samples.append(validate_sample(sample_id, image_path, label_path, args, overlay_path))

    summary = build_summary(preprocessed_dir, args, samples, pairing_failures)
    write_outputs(out_dir, summary, samples)
    print(f"Validation complete: {out_dir}")
    print(f"Overall status: {summary['overall_status']}")


def discover_pairs(root: Path) -> Tuple[List[Tuple[str, Path, Path]], List[str]]:
    if not root.exists():
        raise FileNotFoundError(f"Preprocessed directory does not exist: {root}")

    image_files = sorted(root.glob(f"**/*{IMAGE_SUFFIX}"))
    label_files = sorted(root.glob(f"**/*{LABEL_SUFFIX}"))
    image_map = map_by_sample_id(image_files, IMAGE_SUFFIX)
    label_map = map_by_sample_id(label_files, LABEL_SUFFIX)

    failures: List[str] = []
    for sample_id, paths in sorted(image_map.items()):
        if len(paths) != 1:
            failures.append(f"{sample_id}: expected 1 image file, found {len(paths)}")
    for sample_id, paths in sorted(label_map.items()):
        if len(paths) != 1:
            failures.append(f"{sample_id}: expected 1 label file, found {len(paths)}")

    image_ids = set(image_map)
    label_ids = set(label_map)
    for sample_id in sorted(image_ids - label_ids):
        failures.append(f"{sample_id}: missing label file")
    for sample_id in sorted(label_ids - image_ids):
        failures.append(f"{sample_id}: orphan label file")

    pairs = []
    for sample_id in sorted(image_ids & label_ids):
        if len(image_map[sample_id]) == 1 and len(label_map[sample_id]) == 1:
            pairs.append((sample_id, image_map[sample_id][0], label_map[sample_id][0]))
    return pairs, failures


def map_by_sample_id(paths: Iterable[Path], suffix: str) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for path in paths:
        name = path.name
        if not name.endswith(suffix):
            continue
        sample_id = name[: -len(suffix)]
        result.setdefault(sample_id, []).append(path)
    return result


def choose_overlay_ids(sample_ids: Sequence[str], count: int) -> set[str]:
    if count <= 0 or not sample_ids:
        return set()
    if count >= len(sample_ids):
        return set(sample_ids)
    indices = np.linspace(0, len(sample_ids) - 1, count, dtype=int)
    return {sample_ids[int(i)] for i in indices}


def validate_sample(
    sample_id: str,
    image_path: Path,
    label_path: Path,
    args: argparse.Namespace,
    overlay_path: Optional[Path],
) -> SampleResult:
    failures: List[str] = []
    image_header: Optional[NiftiHeader] = None
    label_header: Optional[NiftiHeader] = None
    image: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None
    image_min = image_max = image_mean = None
    image_finite: Optional[bool] = None
    label_values: Optional[List[float]] = None
    label_is_integer: Optional[bool] = None
    label_nonempty: Optional[bool] = None
    mask_voxels: Optional[int] = None
    mask_bbox_xyz: Optional[List[List[int]]] = None
    mask_body_overlap_ratio: Optional[float] = None
    z_with_largest_mask: Optional[int] = None
    saved_overlay: Optional[str] = None

    try:
        image_header = read_nifti_header(image_path)
        label_header = read_nifti_header(label_path)
        validate_headers(image_header, label_header, args, failures)

        image = read_nifti_array(image_path, image_header)
        label = read_nifti_array(label_path, label_header)
        image = squeeze_to_xyz(image)
        label = squeeze_to_xyz(label)

        if image.shape != label.shape:
            failures.append(f"voxel shape mismatch: image={image.shape}, label={label.shape}")
        else:
            image_finite = bool(np.isfinite(image).all())
            image_min = float(np.nanmin(image))
            image_max = float(np.nanmax(image))
            image_mean = float(np.nanmean(image))
            if not image_finite:
                failures.append("image contains NaN or Inf")
            if image_min < -args.image_min_atol:
                failures.append(f"image min below 0: {image_min}")
            if image_max > 1.0 + args.image_max_atol:
                failures.append(f"image max above 1: {image_max}")

            values = np.unique(label)
            label_values = [float(v) for v in values.tolist()]
            rounded = np.round(values)
            label_is_integer = bool(np.all(np.abs(values - rounded) <= 1.0e-6))
            if not label_is_integer:
                failures.append(f"label contains non-integer values: {label_values[:20]}")

            allowed_values = set(float(v) for v in args.expected_label_values)
            unexpected_values = sorted(set(label_values) - allowed_values)
            if unexpected_values:
                failures.append(f"unexpected label values: {unexpected_values[:20]}")

            mask = label > 0
            mask_voxels = int(mask.sum())
            label_nonempty = mask_voxels > 0
            if not label_nonempty:
                failures.append("label mask is empty")
            else:
                coords = np.argwhere(mask)
                mins = coords.min(axis=0).astype(int).tolist()
                maxs = coords.max(axis=0).astype(int).tolist()
                mask_bbox_xyz = [[mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]]
                z_counts = mask.sum(axis=(0, 1))
                z_with_largest_mask = int(np.argmax(z_counts))
                body = image > args.body_threshold
                mask_body_overlap_ratio = float((mask & body).sum() / mask_voxels)
                if mask_body_overlap_ratio < args.min_mask_body_overlap:
                    failures.append(
                        "low mask/body overlap: "
                        f"{mask_body_overlap_ratio:.6f} < {args.min_mask_body_overlap:.6f}"
                    )
                if overlay_path:
                    save_overlay(image, mask, sample_id, z_with_largest_mask, overlay_path)
                    saved_overlay = str(overlay_path)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{exc.__class__.__name__}: {exc}")

    status = "pass" if not failures else "fail"
    return SampleResult(
        sample_id=sample_id,
        status=status,
        image_path=str(image_path),
        label_path=str(label_path),
        image_shape=image_header.shape_xyz if image_header else None,
        label_shape=label_header.shape_xyz if label_header else None,
        image_spacing=image_header.spacing_xyz if image_header else None,
        label_spacing=label_header.spacing_xyz if label_header else None,
        image_orientation=image_header.orientation if image_header else None,
        label_orientation=label_header.orientation if label_header else None,
        image_dtype=image_header.datatype_name if image_header else None,
        label_dtype=label_header.datatype_name if label_header else None,
        image_min=image_min,
        image_max=image_max,
        image_mean=image_mean,
        image_finite=image_finite,
        label_values=label_values,
        label_is_integer=label_is_integer,
        label_nonempty=label_nonempty,
        mask_voxels=mask_voxels,
        mask_bbox_xyz=mask_bbox_xyz,
        mask_body_overlap_ratio=mask_body_overlap_ratio,
        z_with_largest_mask=z_with_largest_mask,
        overlay_path=saved_overlay,
        failures=failures,
    )


def validate_headers(
    image_header: NiftiHeader,
    label_header: NiftiHeader,
    args: argparse.Namespace,
    failures: List[str],
) -> None:
    if image_header.shape_xyz != label_header.shape_xyz:
        failures.append(
            f"header shape mismatch: image={image_header.shape_xyz}, label={label_header.shape_xyz}"
        )
    if not close_tuple(image_header.spacing_xyz, label_header.spacing_xyz, args.spacing_atol):
        failures.append(
            f"image/label spacing mismatch: image={image_header.spacing_xyz}, "
            f"label={label_header.spacing_xyz}"
        )
    if not close_tuple(image_header.spacing_xyz, tuple(args.target_spacing), args.spacing_atol):
        failures.append(
            f"image spacing does not match target: {image_header.spacing_xyz} vs "
            f"{tuple(args.target_spacing)}"
        )
    if not close_tuple(label_header.spacing_xyz, tuple(args.target_spacing), args.spacing_atol):
        failures.append(
            f"label spacing does not match target: {label_header.spacing_xyz} vs "
            f"{tuple(args.target_spacing)}"
        )
    if image_header.orientation != "RAS":
        failures.append(f"image orientation is not RAS: {image_header.orientation}")
    if label_header.orientation != "RAS":
        failures.append(f"label orientation is not RAS: {label_header.orientation}")
    if not close_matrix(image_header.affine, label_header.affine, args.affine_atol):
        failures.append("image/label affine mismatch")


def read_nifti_header(path: Path) -> NiftiHeader:
    with gzip.open(path, "rb") as handle:
        header = handle.read(348)
    if len(header) != 348:
        raise ValueError(f"NIfTI header too short: {path}")

    endian = "<"
    sizeof_hdr = struct.unpack(endian + "i", header[0:4])[0]
    if sizeof_hdr != 348:
        endian = ">"
        sizeof_hdr = struct.unpack(endian + "i", header[0:4])[0]
    if sizeof_hdr != 348:
        raise ValueError(f"Invalid NIfTI sizeof_hdr={sizeof_hdr}: {path}")

    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dim[0])
    if ndim < 3:
        raise ValueError(f"NIfTI ndim < 3: {ndim} in {path}")
    shape = tuple(int(x) for x in dim[1 : 1 + ndim])
    shape_xyz = tuple(int(x) for x in dim[1:4])
    datatype_code = struct.unpack(endian + "h", header[70:72])[0]
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    if datatype_code not in DTYPES:
        raise ValueError(f"Unsupported datatype code {datatype_code}: {path}")

    pixdim = struct.unpack(endian + "8f", header[76:108])
    spacing_xyz = tuple(float(x) for x in pixdim[1:4])
    vox_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
    scl_slope = float(struct.unpack(endian + "f", header[112:116])[0])
    scl_inter = float(struct.unpack(endian + "f", header[116:120])[0])
    qform_code = struct.unpack(endian + "h", header[252:254])[0]
    sform_code = struct.unpack(endian + "h", header[254:256])[0]
    affine = build_affine(header, endian, pixdim, qform_code, sform_code)
    orientation = affine_to_orientation(affine)

    return NiftiHeader(
        path=str(path),
        endian=endian,
        shape=shape,
        shape_xyz=(shape_xyz[0], shape_xyz[1], shape_xyz[2]),
        datatype_code=datatype_code,
        datatype_name=DTYPE_NAMES[datatype_code],
        bitpix=bitpix,
        spacing_xyz=(spacing_xyz[0], spacing_xyz[1], spacing_xyz[2]),
        affine=affine,
        orientation=orientation,
        vox_offset=vox_offset,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        qform_code=qform_code,
        sform_code=sform_code,
    )


def build_affine(
    header: bytes,
    endian: str,
    pixdim: Tuple[float, ...],
    qform_code: int,
    sform_code: int,
) -> Tuple[Tuple[float, float, float, float], ...]:
    if sform_code > 0:
        srow_x = struct.unpack(endian + "4f", header[280:296])
        srow_y = struct.unpack(endian + "4f", header[296:312])
        srow_z = struct.unpack(endian + "4f", header[312:328])
        return (
            tuple(float(x) for x in srow_x),
            tuple(float(x) for x in srow_y),
            tuple(float(x) for x in srow_z),
            (0.0, 0.0, 0.0, 1.0),
        )
    if qform_code > 0:
        return qform_to_affine(header, endian, pixdim)
    return (
        (float(pixdim[1]), 0.0, 0.0, 0.0),
        (0.0, float(pixdim[2]), 0.0, 0.0),
        (0.0, 0.0, float(pixdim[3]), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def qform_to_affine(
    header: bytes,
    endian: str,
    pixdim: Tuple[float, ...],
) -> Tuple[Tuple[float, float, float, float], ...]:
    b = float(struct.unpack(endian + "f", header[256:260])[0])
    c = float(struct.unpack(endian + "f", header[260:264])[0])
    d = float(struct.unpack(endian + "f", header[264:268])[0])
    x = float(struct.unpack(endian + "f", header[268:272])[0])
    y = float(struct.unpack(endian + "f", header[272:276])[0])
    z = float(struct.unpack(endian + "f", header[276:280])[0])
    a_sq = max(0.0, 1.0 - (b * b + c * c + d * d))
    a = math.sqrt(a_sq)
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    dx, dy, dz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3]) * qfac
    r11 = a * a + b * b - c * c - d * d
    r12 = 2.0 * (b * c - a * d)
    r13 = 2.0 * (b * d + a * c)
    r21 = 2.0 * (b * c + a * d)
    r22 = a * a + c * c - b * b - d * d
    r23 = 2.0 * (c * d - a * b)
    r31 = 2.0 * (b * d - a * c)
    r32 = 2.0 * (c * d + a * b)
    r33 = a * a + d * d - c * c - b * b
    return (
        (r11 * dx, r12 * dy, r13 * dz, x),
        (r21 * dx, r22 * dy, r23 * dz, y),
        (r31 * dx, r32 * dy, r33 * dz, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def affine_to_orientation(affine: Tuple[Tuple[float, float, float, float], ...]) -> str:
    matrix = np.array([row[:3] for row in affine[:3]], dtype=float)
    labels = (("L", "R"), ("P", "A"), ("I", "S"))
    used_world_axes: set[int] = set()
    codes: List[str] = []
    for voxel_axis in range(3):
        column = matrix[:, voxel_axis]
        candidates = sorted(
            ((abs(float(column[i])), i) for i in range(3) if i not in used_world_axes),
            reverse=True,
        )
        if not candidates or candidates[0][0] == 0:
            codes.append("?")
            continue
        world_axis = candidates[0][1]
        used_world_axes.add(world_axis)
        sign = 1 if column[world_axis] >= 0 else 0
        codes.append(labels[world_axis][sign])
    return "".join(codes)


def read_nifti_array(path: Path, header: NiftiHeader) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    dtype = np.dtype(DTYPES[header.datatype_code]).newbyteorder(header.endian)
    count = int(np.prod(header.shape))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=header.vox_offset)
    array = data.reshape(header.shape, order="F")
    slope = header.scl_slope
    inter = header.scl_inter
    if not math.isfinite(slope) or slope == 0.0:
        slope = 1.0
    if not math.isfinite(inter):
        inter = 0.0
    if slope != 1.0 or inter != 0.0:
        array = array.astype(np.float32) * slope + inter
    return np.asarray(array)


def squeeze_to_xyz(array: np.ndarray) -> np.ndarray:
    if array.ndim > 3 and 1 in array.shape[3:]:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array after squeeze, got shape={array.shape}")
    return array


def close_tuple(a: Sequence[float], b: Sequence[float], atol: float) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def close_matrix(
    a: Tuple[Tuple[float, float, float, float], ...],
    b: Tuple[Tuple[float, float, float, float], ...],
    atol: float,
) -> bool:
    return bool(np.allclose(np.array(a), np.array(b), atol=atol, rtol=0.0))


def save_overlay(image: np.ndarray, mask: np.ndarray, sample_id: str, z: int, path: Path) -> None:
    base = normalize_slice(image[:, :, z])
    base = np.rot90(base)
    mask_slice = np.rot90(mask[:, :, z])
    rgb = np.stack([base, base, base], axis=-1)
    rgb[mask_slice, 0] = np.maximum(rgb[mask_slice, 0], 230)
    rgb[mask_slice, 1] = (rgb[mask_slice, 1] * 0.45).astype(np.uint8)
    rgb[mask_slice, 2] = (rgb[mask_slice, 2] * 0.45).astype(np.uint8)
    output = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(output)
    ys, xs = np.where(mask_slice)
    if len(xs) > 0:
        draw.rectangle(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            outline=(0, 255, 0),
            width=2,
        )
    draw.text((8, 8), f"{sample_id} z={z}", fill=(255, 255, 0))
    output.save(path)


def normalize_slice(array: np.ndarray) -> np.ndarray:
    data = array.astype(np.float32)
    lo, hi = np.percentile(data, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def build_summary(
    preprocessed_dir: Path,
    args: argparse.Namespace,
    samples: Sequence[SampleResult],
    pairing_failures: Sequence[str],
) -> Dict[str, object]:
    failed_samples = [sample for sample in samples if sample.status != "pass"]
    all_label_values = sorted(
        {
            value
            for sample in samples
            if sample.label_values is not None
            for value in sample.label_values
        }
    )
    spacing_values = sorted(
        {
            tuple(round(float(v), 6) for v in sample.image_spacing)
            for sample in samples
            if sample.image_spacing is not None
        }
    )
    orientation_counts: Dict[str, int] = {}
    label_dtype_counts: Dict[str, int] = {}
    for sample in samples:
        if sample.image_orientation:
            orientation_counts[sample.image_orientation] = orientation_counts.get(sample.image_orientation, 0) + 1
        if sample.label_dtype:
            label_dtype_counts[sample.label_dtype] = label_dtype_counts.get(sample.label_dtype, 0) + 1

    mask_counts = [sample.mask_voxels for sample in samples if sample.mask_voxels is not None]
    overlap_values = [
        sample.mask_body_overlap_ratio
        for sample in samples
        if sample.mask_body_overlap_ratio is not None
    ]
    image_mins = [sample.image_min for sample in samples if sample.image_min is not None]
    image_maxs = [sample.image_max for sample in samples if sample.image_max is not None]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessed_dir": str(preprocessed_dir),
        "overall_status": "pass" if not pairing_failures and not failed_samples else "fail",
        "sample_count": len(samples),
        "passed_sample_count": len(samples) - len(failed_samples),
        "failed_sample_count": len(failed_samples),
        "pairing_failure_count": len(pairing_failures),
        "pairing_failures": list(pairing_failures),
        "failed_sample_ids": [sample.sample_id for sample in failed_samples],
        "target_spacing": list(args.target_spacing),
        "spacing_atol": args.spacing_atol,
        "spacing_values": [list(item) for item in spacing_values],
        "orientation_counts": orientation_counts,
        "label_dtype_counts": label_dtype_counts,
        "label_values": all_label_values,
        "image_min_global": safe_min(image_mins),
        "image_max_global": safe_max(image_maxs),
        "mask_voxels_min": safe_min(mask_counts),
        "mask_voxels_median": safe_median(mask_counts),
        "mask_voxels_max": safe_max(mask_counts),
        "mask_body_overlap_min": safe_min(overlap_values),
        "mask_body_overlap_median": safe_median(overlap_values),
        "overlay_count": sum(1 for sample in samples if sample.overlay_path),
        "acceptance_checks": {
            "all_pairs_present": len(pairing_failures) == 0,
            "all_samples_passed": len(failed_samples) == 0,
            "target_spacing_reached": all(
                sample.image_spacing is not None
                and close_tuple(sample.image_spacing, tuple(args.target_spacing), args.spacing_atol)
                for sample in samples
            ),
            "all_orientation_ras": orientation_counts == {"RAS": len(samples)},
            "all_masks_nonempty": all(sample.label_nonempty for sample in samples),
            "all_images_normalized": all(
                sample.image_min is not None
                and sample.image_max is not None
                and sample.image_min >= -args.image_min_atol
                and sample.image_max <= 1.0 + args.image_max_atol
                for sample in samples
            ),
        },
    }


def write_outputs(out_dir: Path, summary: Dict[str, object], samples: Sequence[SampleResult]) -> None:
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": [asdict(sample) for sample in samples]}, handle, indent=2)
    write_samples_csv(out_dir / "samples.csv", samples)
    write_markdown(out_dir / "report.md", summary, samples)


def write_samples_csv(path: Path, samples: Sequence[SampleResult]) -> None:
    rows = [asdict(sample) for sample in samples]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary: Dict[str, object], samples: Sequence[SampleResult]) -> None:
    failed_samples = [sample for sample in samples if sample.status != "pass"]
    overlays = [sample for sample in samples if sample.overlay_path]
    lines = [
        "# Task7 Preprocessing Acceptance Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Preprocessed dir: `{summary['preprocessed_dir']}`",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Samples: `{summary['sample_count']}` total, "
        f"`{summary['passed_sample_count']}` passed, `{summary['failed_sample_count']}` failed",
        "",
        "## Acceptance Summary",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    checks = summary["acceptance_checks"]
    assert isinstance(checks, dict)
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Key Evidence",
            "",
            f"- Target spacing: `{summary['target_spacing']}` with atol `{summary['spacing_atol']}`.",
            f"- Observed spacing values: `{summary['spacing_values']}`.",
            f"- Orientation counts: `{summary['orientation_counts']}`.",
            f"- Label dtype counts: `{summary['label_dtype_counts']}`.",
            f"- Label values: `{summary['label_values']}`.",
            f"- Image global min/max: `{summary['image_min_global']}` / `{summary['image_max_global']}`.",
            f"- Mask voxel count min/median/max: `{summary['mask_voxels_min']}` / "
            f"`{summary['mask_voxels_median']}` / `{summary['mask_voxels_max']}`.",
            f"- Mask-body overlap min/median: `{summary['mask_body_overlap_min']}` / "
            f"`{summary['mask_body_overlap_median']}`.",
            "",
            "## Pairing Failures",
            "",
        ]
    )
    pairing_failures = summary["pairing_failures"]
    if pairing_failures:
        lines.extend(f"- {item}" for item in pairing_failures)  # type: ignore[union-attr]
    else:
        lines.append("- None.")

    lines.extend(["", "## Failed Samples", ""])
    if failed_samples:
        for sample in failed_samples[:50]:
            lines.append(f"- `{sample.sample_id}`: {'; '.join(sample.failures)}")
        if len(failed_samples) > 50:
            lines.append(f"- ... {len(failed_samples) - 50} more")
    else:
        lines.append("- None.")

    lines.extend(["", "## Overlay Checks", ""])
    for sample in overlays:
        rel_path = os.path.relpath(sample.overlay_path or "", path.parent).replace("\\", "/")
        lines.append(f"![{sample.sample_id}]({rel_path})")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_min(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def safe_max(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def safe_median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return float(np.median(clean)) if clean else None


if __name__ == "__main__":
    main()
