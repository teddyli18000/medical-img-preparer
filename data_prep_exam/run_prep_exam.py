import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from analyze_prep_headers import analyze_prep_headers
from config_prep_exam import ConfigPrepExam
from plotting import generate_figures
from report_writer import write_json_report, write_markdown_report


def main() -> None:
    stats_path, report_path = ConfigPrepExam.output_paths()

    print("=" * 72)
    print("Start preprocessed dataset header EDA")
    print(f"Dataset name: {ConfigPrepExam.dataset_name()}")
    print(f"PREPROCESSED_DIR: {ConfigPrepExam.preprocessed_dir()}")
    print(f"Output JSON: {stats_path}")
    print(f"Output MD: {report_path}")
    print("=" * 72)

    result = analyze_prep_headers(ConfigPrepExam)
    figure_payload = generate_figures(result, ConfigPrepExam)

    md_payload = dict(result)
    md_payload["figures"] = figure_payload.get("figures", [])
    md_payload["distribution_notes"] = figure_payload.get("notes", [])
    md_payload["figure_warnings"] = figure_payload.get("warnings", [])

    write_json_report(result, stats_path, ConfigPrepExam)
    write_markdown_report(md_payload, report_path, ConfigPrepExam)

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
