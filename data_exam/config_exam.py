import os
import re
from typing import Tuple


class ConfigExam:
    """Independent configuration for dataset examination (EDA)."""

    # -----------------------------
    # Dataset source configuration
    # -----------------------------
    # Root directory that contains imagesTr/labelsTr.
    RAW_DATA_DIR = r"E:\Python_Projects\Swin-UNETR-for-MSD-task7\data\Task07_Pancreas"
    IMAGES_SUBDIR = "imagesTr"
    LABELS_SUBDIR = "labelsTr"
    FILE_GLOB = "*.nii.gz"
    SORT_FILES = True

    # Optional explicit dataset name used in output file names.
    # If None, it will be inferred from RAW_DATA_DIR folder name.
    DATASET_NAME = None

    # -----------------------------
    # Output configuration
    # -----------------------------
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data_exam_report")
    )
    STATS_FILE_PREFIX = "stats"
    REPORT_FILE_PREFIX = "report"
    JSON_INDENT = 2
    JSON_ENCODING = "utf-8"
    JSON_ENSURE_ASCII = False

    # -----------------------------
    # Analysis behavior
    # -----------------------------
    AXIS_NAMES = ("x", "y", "z")
    HEADER_DIMENSIONS = 3
    PROGRESS_DESC = "Reading NIfTI headers"
    PROGRESS_UNIT = "file"
    FAIL_ON_MISSING_DIR = True
    SAVE_ABSOLUTE_PATHS = True

    # -----------------------------
    # Markdown report rendering
    # -----------------------------
    MARKDOWN_TITLE = "MSD Task07 Dataset EDA Report"

    @classmethod
    def raw_data_dir(cls) -> str:
        return os.path.abspath(cls.RAW_DATA_DIR)

    @classmethod
    def images_dir(cls) -> str:
        return os.path.join(cls.raw_data_dir(), cls.IMAGES_SUBDIR)

    @classmethod
    def labels_dir(cls) -> str:
        return os.path.join(cls.raw_data_dir(), cls.LABELS_SUBDIR)

    @classmethod
    def dataset_name(cls) -> str:
        if cls.DATASET_NAME:
            return _sanitize_name(str(cls.DATASET_NAME))
        return _sanitize_name(os.path.basename(cls.raw_data_dir()))

    @classmethod
    def stats_file_name(cls) -> str:
        return f"{cls.STATS_FILE_PREFIX}_{cls.dataset_name()}.json"

    @classmethod
    def report_file_name(cls) -> str:
        return f"{cls.REPORT_FILE_PREFIX}_{cls.dataset_name()}.md"

    @classmethod
    def stats_output_path(cls) -> str:
        return os.path.join(cls.REPORT_DIR, cls.stats_file_name())

    @classmethod
    def report_output_path(cls) -> str:
        return os.path.join(cls.REPORT_DIR, cls.report_file_name())

    @classmethod
    def output_paths(cls) -> Tuple[str, str]:
        return cls.stats_output_path(), cls.report_output_path()


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "dataset"
