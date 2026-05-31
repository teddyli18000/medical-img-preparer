import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency
    MATPLOTLIB_AVAILABLE = False


def build_distribution_notes(values: List[float], name: str) -> Dict[str, Any]:
    if not values:
        return {
            "name": name,
            "count": 0,
            "notes": "No data available.",
        }

    arr = np.asarray(values, dtype=np.float64)
    unique_count = int(len(np.unique(arr)))
    mean = float(np.mean(arr))
    std = float(np.std(arr)) if arr.size > 1 else 0.0
    centered = arr - mean
    skewness = float(np.mean(centered**3) / (std**3)) if std > 0 else 0.0
    kurtosis = float(np.mean(centered**4) / (std**4) - 3.0) if std > 0 else 0.0

    notes = []
    if unique_count <= 8:
        notes.append("Discrete values, likely protocol-driven.")
    elif abs(skewness) < 0.3:
        notes.append("Roughly symmetric distribution.")
    elif skewness >= 0.3:
        notes.append("Right-skewed distribution (long tail).")
    else:
        notes.append("Left-skewed distribution (long tail).")

    if abs(kurtosis) > 1.0:
        notes.append("Heavy-tailed tendency detected.")

    return {
        "name": name,
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "unique_count": unique_count,
        "notes": " ".join(notes),
    }


def _save_fig(path: str, fig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight")
    plt.close(fig)


def _apply_axes_style(ax, grid: bool) -> None:
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _hist(ax, values: List[float], bins: int, color: str, title: str, xlabel: str, median: float) -> None:
    ax.hist(values, bins=bins, color=color, alpha=0.85)
    ax.axvline(median, color="#c0392b", linestyle="--", linewidth=1.2, label="median")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)


def _bar(ax, labels: List[str], values: List[int], color: str, title: str, xlabel: str) -> None:
    ax.bar(labels, values, color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)


def _prepare_unique_counts(values: List[float], max_items: int, round_to: int) -> Tuple[List[str], List[int]]:
    rounded = [round(v, round_to) for v in values]
    uniq, counts = np.unique(np.asarray(rounded), return_counts=True)
    pairs = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))
    if len(pairs) > max_items:
        pairs = pairs[:max_items]
    labels = [str(item[0]) for item in pairs]
    values_out = [int(item[1]) for item in pairs]
    return labels, values_out


def generate_figures(report_data: Dict[str, Any], config) -> Dict[str, Any]:
    figures: List[Dict[str, str]] = []
    notes: List[Dict[str, Any]] = []

    if not MATPLOTLIB_AVAILABLE:
        return {
            "figures": figures,
            "notes": notes,
            "warnings": ["matplotlib is not available; figures were skipped."],
        }

    samples = report_data.get("samples", [])
    stats = report_data.get("stats", {})
    geometry = report_data.get("geometry_consistency", {})

    spacing_x = [s["spacing_xyz"][0] for s in samples if s.get("spacing_xyz")]
    spacing_y = [s["spacing_xyz"][1] for s in samples if s.get("spacing_xyz")]
    spacing_z = [s["spacing_xyz"][2] for s in samples if s.get("spacing_xyz")]

    shape_x = [s["shape_xyz"][0] for s in samples if s.get("shape_xyz")]
    shape_y = [s["shape_xyz"][1] for s in samples if s.get("shape_xyz")]
    shape_z = [s["shape_xyz"][2] for s in samples if s.get("shape_xyz")]

    fov_x = [s * sp for s, sp in zip(shape_x, spacing_x)]
    fov_y = [s * sp for s, sp in zip(shape_y, spacing_y)]
    fov_z = [s * sp for s, sp in zip(shape_z, spacing_z)]

    notes.append(build_distribution_notes(spacing_x, "spacing_x"))
    notes.append(build_distribution_notes(spacing_y, "spacing_y"))
    notes.append(build_distribution_notes(spacing_z, "spacing_z"))
    notes.append(build_distribution_notes(shape_z, "shape_z"))
    notes.append(build_distribution_notes(fov_z, "fov_z_mm"))

    figs_dir = config.figures_dir()

    def _add_fig(rel_path: str, title: str, caption: str) -> None:
        figures.append({
            "title": title,
            "path": rel_path.replace("\\", "/"),
            "caption": caption,
        })

    fig_size = config.FIGURE_SIZE
    bins = config.FIGURE_BINS
    color = config.FIGURE_COLOR
    grid = config.FIGURE_GRID

    for axis_name, values, median in (
        ("x", spacing_x, stats.get("spacing_xyz", {}).get("x", {}).get("median")),
        ("y", spacing_y, stats.get("spacing_xyz", {}).get("y", {}).get("median")),
        ("z", spacing_z, stats.get("spacing_xyz", {}).get("z", {}).get("median")),
    ):
        if not values:
            continue
        fig, ax = plt.subplots(figsize=fig_size)
        _hist(ax, values, bins, color, f"Spacing {axis_name.upper()}", "mm", median or np.median(values))
        _apply_axes_style(ax, grid)
        filename = f"spacing_{axis_name}_hist.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            f"Spacing {axis_name.upper()} Histogram",
            "Full-sample spacing distribution.",
        )

    for axis_name, values, median in (
        ("x", shape_x, stats.get("shape_xyz", {}).get("x", {}).get("median")),
        ("y", shape_y, stats.get("shape_xyz", {}).get("y", {}).get("median")),
        ("z", shape_z, stats.get("shape_xyz", {}).get("z", {}).get("median")),
    ):
        if not values:
            continue
        fig, ax = plt.subplots(figsize=fig_size)
        _hist(ax, values, bins, color, f"Shape {axis_name.upper()}", "voxels", median or np.median(values))
        _apply_axes_style(ax, grid)
        filename = f"shape_{axis_name}_hist.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            f"Shape {axis_name.upper()} Histogram",
            "Full-sample shape distribution.",
        )

    if spacing_z:
        labels, counts = _prepare_unique_counts(
            spacing_z,
            max_items=config.FIGURE_Z_UNIQUE_MAX,
            round_to=config.FIGURE_VALUE_ROUND,
        )
        fig, ax = plt.subplots(figsize=fig_size)
        _bar(ax, labels, counts, color, "Spacing Z Unique Values", "spacing_z (mm)")
        _apply_axes_style(ax, grid)
        filename = "spacing_z_unique.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            "Spacing Z Unique Values",
            "Top unique spacing values with counts.",
        )

    if fov_z:
        fig, ax = plt.subplots(figsize=fig_size)
        _hist(ax, fov_z, bins, color, "Physical FOV Z", "mm", np.median(fov_z))
        _apply_axes_style(ax, grid)
        filename = "fov_z_hist.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            "Physical FOV Z Histogram",
            "Physical depth distribution (shape_z * spacing_z).",
        )

    orientation = stats.get("orientation_distribution", {})
    if orientation:
        items = sorted(orientation.items(), key=lambda item: (-item[1], item[0]))
        labels = [item[0] for item in items][: config.FIGURE_MAX_CATEGORIES]
        values = [int(item[1]) for item in items][: config.FIGURE_MAX_CATEGORIES]
        fig, ax = plt.subplots(figsize=fig_size)
        _bar(ax, labels, values, color, "Orientation Distribution", "orientation")
        _apply_axes_style(ax, grid)
        filename = "orientation_bar.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            "Orientation Distribution",
            "Orientation codes across all samples.",
        )

    mismatch_counts = geometry.get("mismatch_counts", {})
    if mismatch_counts:
        labels = list(mismatch_counts.keys())
        values = [int(mismatch_counts[k]) for k in labels]
        fig, ax = plt.subplots(figsize=fig_size)
        _bar(ax, labels, values, color, "Geometry Mismatch Breakdown", "mismatch type")
        _apply_axes_style(ax, grid)
        filename = "geometry_mismatch.png"
        path = os.path.join(figs_dir, filename)
        _save_fig(path, fig)
        _add_fig(
            os.path.relpath(path, config.REPORT_DIR),
            "Geometry Mismatch Breakdown",
            "Mismatch types between image and label headers.",
        )

    return {"figures": figures, "notes": notes, "warnings": []}
