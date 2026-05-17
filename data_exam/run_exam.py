import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from analyze_headers import analyze_dataset_headers
from config_exam import ConfigExam
from report_writer import write_json_report, write_markdown_report


def main() -> None:
    stats_path, report_path = ConfigExam.output_paths()

    print("=" * 72)
    print("Start dataset header EDA")
    print(f"Dataset name: {ConfigExam.dataset_name()}")
    print(f"RAW_DATA_DIR: {ConfigExam.raw_data_dir()}")
    print(f"Output JSON: {stats_path}")
    print(f"Output MD: {report_path}")
    print("=" * 72)

    result = analyze_dataset_headers(ConfigExam)
    write_json_report(result, stats_path, ConfigExam)
    write_markdown_report(result, report_path, ConfigExam)

    summary = result["summary"]
    print("EDA finished.")
    print(
        "Summary: "
        f"images={summary['image_file_count']}, "
        f"labels={summary['label_file_count']}, "
        f"success={summary['header_read_success_count']}, "
        f"failed={summary['header_read_failed_count']}"
    )


if __name__ == "__main__":
    main()
