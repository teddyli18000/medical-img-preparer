import json
import os
from typing import Any, Dict, List

import numpy as np


def write_json_report(report_data: Dict[str, Any], output_path: str, config) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = _to_builtin(report_data)
    with open(output_path, "w", encoding=config.JSON_ENCODING) as f:
        json.dump(
            payload,
            f,
            indent=config.JSON_INDENT,
            ensure_ascii=config.JSON_ENSURE_ASCII,
        )


def write_markdown_report(report_data: Dict[str, Any], output_path: str, config) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = report_data["summary"]
    stats = report_data["stats"]
    consistency = report_data["consistency"]
    samples = report_data["samples"]
    failures = report_data["failures"]
    conclusions = report_data["conclusions"]

    lines: List[str] = []
    lines.append(f"# {config.MARKDOWN_TITLE} - {summary['dataset_name']}")
    lines.append("")
    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append(f"- Generated At: {summary['generated_at']}")
    lines.append(f"- Raw Data Dir: {summary['raw_data_dir']}")
    lines.append(f"- Images Dir: {summary['images_dir']}")
    lines.append(f"- Labels Dir: {summary['labels_dir']}")
    lines.append("")

    lines.append("## 2. Count Consistency")
    lines.append("")
    lines.extend(
        _render_table(
            ["Metric", "Value"],
            [
                ["image_file_count", summary["image_file_count"]],
                ["label_file_count", summary["label_file_count"]],
                ["counts_match", summary["counts_match"]],
                ["missing_label_count", summary["missing_label_count"]],
                ["orphan_label_count", summary["orphan_label_count"]],
                ["header_read_success_count", summary["header_read_success_count"]],
                ["header_read_failed_count", summary["header_read_failed_count"]],
            ],
        )
    )
    lines.append("")

    lines.append("## 3. Spacing Statistics (mm)")
    lines.append("")
    lines.extend(_render_axis_stats_table(stats["spacing_xyz"]))
    lines.append("")

    lines.append("## 4. Shape Statistics (voxels)")
    lines.append("")
    lines.extend(_render_axis_stats_table(stats["shape_xyz"]))
    lines.append("")

    lines.append("## 5. Orientation Distribution")
    lines.append("")
    orientation_rows = [[k, v] for k, v in stats["orientation_distribution"].items()]
    lines.extend(_render_table(["Orientation", "Count"], orientation_rows))
    lines.append("")

    lines.append("## 6. Missing / Orphan Labels")
    lines.append("")
    lines.append(f"- missing_label_sample_ids ({len(consistency['missing_label_sample_ids'])}):")
    lines.extend(_render_bullet_values(consistency["missing_label_sample_ids"]))
    lines.append(f"- orphan_label_sample_ids ({len(consistency['orphan_label_sample_ids'])}):")
    lines.extend(_render_bullet_values(consistency["orphan_label_sample_ids"]))
    lines.append("")

    lines.append("## 7. Failed Samples")
    lines.append("")
    failed_rows = [
        [item["sample_id"], item["image_path"], item["error_message"]]
        for item in failures
    ]
    lines.extend(_render_table(["sample_id", "image_path", "error_message"], failed_rows))
    lines.append("")

    lines.append("## 8. Sample-Level Details")
    lines.append("")
    sample_rows = []
    for item in samples:
        sample_rows.append(
            [
                item["sample_id"],
                item["image_path"],
                item["label_path"],
                item["label_exists"],
                _format_xyz(item["spacing_xyz"]),
                _format_xyz(item["shape_xyz"]),
                item["orientation"],
                item["read_status"],
                item["error_message"],
            ]
        )

    lines.extend(
        _render_table(
            [
                "sample_id",
                "image_path",
                "label_path",
                "label_exists",
                "spacing_xyz",
                "shape_xyz",
                "orientation",
                "read_status",
                "error_message",
            ],
            sample_rows,
        )
    )
    lines.append("")

    lines.append("## 9. Conclusions")
    lines.append("")
    lines.extend([f"- {item}" for item in conclusions])
    lines.append("")

    with open(output_path, "w", encoding=config.JSON_ENCODING) as f:
        f.write("\n".join(lines))


def _render_axis_stats_table(axis_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    rows = []
    for axis in ("x", "y", "z"):
        row = axis_stats.get(axis, {})
        rows.append([axis, row.get("min"), row.get("max"), row.get("median")])
    return _render_table(["Axis", "Min", "Max", "Median"], rows)


def _render_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["_No data._"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        escaped = [_escape_md_cell(_stringify_cell(cell)) for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")

    return lines


def _render_bullet_values(values: List[str]) -> List[str]:
    if not values:
        return ["  - (none)"]
    return [f"  - {v}" for v in values]


def _format_xyz(value: Any) -> str:
    if value is None:
        return "N/A"
    return "[" + ", ".join(str(v) for v in value) + "]"


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
