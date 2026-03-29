import builtins
import datetime as dt
import glob
import logging
import os
import re
import sys
import traceback
from typing import Optional


class PrintMirrorLogger:
    """Mirror all print output to a structured log file without changing console behavior."""

    def __init__(self, module_name: str, dataset_name: str, project_root: str, logs_folder_name: str = "logs"):
        self.module_name = _sanitize_for_filename(module_name) or "module"
        self.dataset_name = _sanitize_for_filename(dataset_name) or "dataset"
        self.project_root = os.path.abspath(project_root)
        self.logs_dir = os.path.join(self.project_root, logs_folder_name)
        self._file_prefix = f"{self.module_name}_{self.dataset_name}_"

        self.start_time = dt.datetime.now()
        self.end_time: Optional[dt.datetime] = None

        self._counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
        self._original_print = builtins.print
        self._original_stdout = None
        self._original_stderr = None
        self._is_active = False
        self._is_finalized = False
        self._logging_handler = None
        self._attached_loggers = []
        self._attached_logger_ids = set()
        self._original_get_logger = None

        temp_name = f".{self.module_name}_{os.getpid()}_{int(self.start_time.timestamp())}.tmp"
        self._temp_log_path = os.path.join(self.logs_dir, temp_name)

    def __enter__(self):
        os.makedirs(self.logs_dir, exist_ok=True)
        self._cleanup_stale_temp_files()
        self._is_active = True
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _StreamMirror(self, self._original_stdout, default_level="INFO")
        sys.stderr = _StreamMirror(self, self._original_stderr, default_level="WARN")
        self._attach_logging_capture()
        self._safe_write_record("INFO", f"Logging started for module={self.module_name}, dataset={self.dataset_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb)).rstrip("\n")
                self._safe_write_record("ERROR", "Program terminated with an exception.")
                for line in tb.splitlines():
                    self._safe_write_record("ERROR", line)
                status = "ABNORMAL"
            else:
                status = "SUCCESS"

            if isinstance(sys.stdout, _StreamMirror):
                sys.stdout.flush()
            if isinstance(sys.stderr, _StreamMirror):
                sys.stderr.flush()

            self._safe_finalize(status=status)
        finally:
            self._detach_logging_capture()
            # Always restore runtime streams, even if logging itself fails.
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
            if self._original_stderr is not None:
                sys.stderr = self._original_stderr
            builtins.print = self._original_print
            self._is_active = False
        return False

    def _resolve_level(self, message: str, default_level: str) -> str:
        inferred = _infer_level(message)
        if inferred == "INFO" and default_level in {"WARN", "ERROR"}:
            return default_level
        return inferred

    def _safe_write_record(self, level: str, message: str):
        try:
            self._write_record(level, message)
        except Exception:
            # Logging must never break the main pipeline.
            pass

    def _write_record(self, level: str, message: str):
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_message = message.rstrip("\n")
        self._counts[level] += 1

        os.makedirs(self.logs_dir, exist_ok=True)

        with open(self._temp_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level}] {safe_message}\n")

    def _safe_finalize(self, status: str):
        try:
            self._finalize(status=status)
        except Exception:
            # Finalization failures should not mask real pipeline exceptions.
            pass

    def _finalize(self, status: str):
        if self._is_finalized:
            return

        self.end_time = dt.datetime.now()

        body = ""
        if os.path.exists(self._temp_log_path):
            with open(self._temp_log_path, "r", encoding="utf-8") as f:
                body = f.read()

        header_lines = [
            "===== RUN SUMMARY =====",
            f"module_name: {self.module_name}",
            f"dataset_name: {self.dataset_name}",
            f"status: {status}",
            f"start_time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"end_time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"info_count: {self._counts['INFO']}",
            f"warn_count: {self._counts['WARN']}",
            f"error_count: {self._counts['ERROR']}",
            "=======================",
            "",
        ]

        run_token = self.end_time.strftime("%Y%m%d%H%M%S")
        final_name = f"{self.module_name}_{self.dataset_name}_{run_token}.txt"
        final_log_path = os.path.join(self.logs_dir, final_name)

        with open(final_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            f.write(body)

        if os.path.exists(self._temp_log_path):
            os.remove(self._temp_log_path)

        # Keep only the latest completed log for the same module+dataset pair.
        self._cleanup_historical_logs(keep_path=final_log_path)

        self._is_finalized = True

    def _cleanup_stale_temp_files(self):
        pattern = os.path.join(self.logs_dir, f".{self.module_name}_*.tmp")
        for path in glob.glob(pattern):
            if os.path.abspath(path) == os.path.abspath(self._temp_log_path):
                continue
            try:
                os.remove(path)
            except Exception:
                pass

    def _cleanup_historical_logs(self, keep_path: str):
        pattern = os.path.join(self.logs_dir, f"{self._file_prefix}*.txt")
        keep_abs = os.path.abspath(keep_path)
        for path in glob.glob(pattern):
            if os.path.abspath(path) == keep_abs:
                continue
            try:
                os.remove(path)
            except Exception:
                pass

    def _attach_logging_capture(self):
        handler = _PythonLoggingMirrorHandler(self)
        handler.setLevel(logging.NOTSET)
        self._logging_handler = handler

        self._original_get_logger = logging.getLogger

        # Attach to all current logger objects so non-propagating third-party loggers are captured.
        self._attach_handler_to_logger(logging.getLogger())
        for logger_obj in logging.root.manager.loggerDict.values():
            if isinstance(logger_obj, logging.Logger):
                self._attach_handler_to_logger(logger_obj)

        def _capturing_get_logger(name=None):
            logger = self._original_get_logger(name)
            self._attach_handler_to_logger(logger)
            return logger

        logging.getLogger = _capturing_get_logger

    def _attach_handler_to_logger(self, logger: logging.Logger):
        if self._logging_handler is None:
            return
        logger_id = id(logger)
        if logger_id in self._attached_logger_ids:
            return
        logger.addHandler(self._logging_handler)
        self._attached_loggers.append(logger)
        self._attached_logger_ids.add(logger_id)

    def _detach_logging_capture(self):
        if self._original_get_logger is not None:
            logging.getLogger = self._original_get_logger
            self._original_get_logger = None
        if self._logging_handler is None:
            return
        for logger in self._attached_loggers:
            try:
                logger.removeHandler(self._logging_handler)
            except Exception:
                pass
        self._attached_loggers = []
        self._attached_logger_ids = set()
        self._logging_handler = None


def infer_dataset_name(path_or_name: str) -> str:
    if not path_or_name:
        return "unknown_dataset"

    normalized = os.path.normpath(path_or_name)
    base = os.path.basename(normalized)

    # Prefer TaskNN style to map into MSD_taskN format.
    match_task = re.search(r"task\s*0*(\d+)", path_or_name, flags=re.IGNORECASE)
    if match_task:
        return f"MSD_task{int(match_task.group(1))}"

    # Fallback to a clean folder name.
    cleaned = _sanitize_for_filename(base)
    return cleaned or "unknown_dataset"


def _infer_level(message: str) -> str:
    lower = message.lower()

    error_keywords = ["error", "exception", "traceback", "fatal", "fail", "异常", "报错", "失败"]
    warn_keywords = ["warn", "warning", "注意", "警告", "跳过"]

    if any(k in lower for k in error_keywords):
        return "ERROR"
    if any(k in lower for k in warn_keywords):
        return "WARN"
    return "INFO"


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


class _StreamMirror:
    """Tee stream writes to console and logger with line buffering."""

    def __init__(self, logger: PrintMirrorLogger, stream, default_level: str):
        self._logger = logger
        self._stream = stream
        self._default_level = default_level
        self._buffer = ""

    def write(self, data):
        text = str(data)
        self._stream.write(text)
        if not text:
            return 0

        self._buffer += text
        self._drain_buffer()
        return len(text)

    def flush(self):
        self._flush_pending()
        self._stream.flush()

    def _drain_buffer(self):
        while True:
            idx_nl = self._buffer.find("\n")
            idx_cr = self._buffer.find("\r")

            candidates = [i for i in (idx_nl, idx_cr) if i != -1]
            if not candidates:
                break

            idx = min(candidates)
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]

            if line:
                level = self._logger._resolve_level(line, self._default_level)
                self._logger._safe_write_record(level, line)

    def _flush_pending(self):
        pending = self._buffer.strip()
        if pending:
            level = self._logger._resolve_level(pending, self._default_level)
            self._logger._safe_write_record(level, pending)
        self._buffer = ""

    def isatty(self):
        return self._stream.isatty() if hasattr(self._stream, "isatty") else False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, item):
        return getattr(self._stream, item)


class _PythonLoggingMirrorHandler(logging.Handler):
    """Capture standard logging records so third-party library logs are not missed."""

    def __init__(self, logger: PrintMirrorLogger):
        super().__init__()
        self._logger = logger

    def emit(self, record):
        try:
            timestamp = dt.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            rendered = f"{timestamp} {record.levelname} {record.filename}:{record.lineno} - {record.getMessage()}"
            level = self._map_level(record.levelno)
            self._logger._safe_write_record(level, rendered)
        except Exception:
            pass

    @staticmethod
    def _map_level(levelno: int) -> str:
        if levelno >= logging.ERROR:
            return "ERROR"
        if levelno >= logging.WARNING:
            return "WARN"
        return "INFO"


