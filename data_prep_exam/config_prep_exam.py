import os
import re
from typing import Tuple


class ConfigPrepExam:
    """Independent configuration for preprocessed dataset examination (EDA)."""

    # -----------------------------
    # Dataset source configuration
    # -----------------------------
    PREPROCESSED_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", "processed_MSD_Task7")
    )
    FILE_GLOB_IMAGE = "**/*_image_prep.nii.gz"
    FILE_GLOB_LABEL = "**/*_label_prep.nii.gz"
    SORT_FILES = True

    # Optional explicit dataset name used in output file names.
    DATASET_NAME = None

    # -----------------------------
    # Output configuration
    # -----------------------------
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data_prep_exam_report")
    )
    STATS_FILE_PREFIX = "stats_prep"
    REPORT_FILE_PREFIX = "report_prep"
    JSON_INDENT = 2
    JSON_ENCODING = "utf-8"
    JSON_ENSURE_ASCII = False

    # -----------------------------
    # Analysis behavior
    # -----------------------------
    AXIS_NAMES = ("x", "y", "z")
    HEADER_DIMENSIONS = 3
    PROGRESS_DESC = "Reading preprocessed NIfTI headers"
    PROGRESS_UNIT = "file"
    FAIL_ON_MISSING_DIR = True
    SAVE_ABSOLUTE_PATHS = True

    # -----------------------------
    # Markdown report rendering
    # -----------------------------
    MARKDOWN_TITLE = "Preprocessed Dataset EDA Report"

    # -----------------------------
    # Visualization output (PNG only for MD)
    # -----------------------------
    FIGURES_SUBDIR = "figures"
    FIGURE_DPI = 140
    FIGURE_BINS = 40
    FIGURE_SIZE = (6.5, 4.0)
    FIGURE_COLOR = "#1f4e79"
    FIGURE_GRID = True
    FIGURE_MAX_CATEGORIES = 20
    FIGURE_MAX_MISMATCH_SAMPLES = 30
    FIGURE_Z_UNIQUE_MAX = 30
    FIGURE_VALUE_ROUND = 4

    # -----------------------------
    # Geometry consistency thresholds
    # -----------------------------
    AFFINE_ATOL = 1.0e-3
    SPACING_ATOL = 1.0e-6
    SHAPE_ATOL = 0
    SPACING_RANGE_TOL = 1.0e-4

    # Optional expectations (set to tuple/str to enforce in conclusions)
    EXPECTED_SPACING = None
    EXPECTED_ORIENTATION = None

    @classmethod
    def preprocessed_dir(cls) -> str:
        return os.path.abspath(cls.PREPROCESSED_DIR)

    @classmethod
    def dataset_name(cls) -> str:
        if cls.DATASET_NAME:
            return _sanitize_name(str(cls.DATASET_NAME))
        return _sanitize_name(os.path.basename(cls.preprocessed_dir()))

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

    @classmethod
    def figures_dir(cls) -> str:
        return os.path.join(cls.REPORT_DIR, cls.FIGURES_SUBDIR, cls.dataset_name())


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "dataset"
