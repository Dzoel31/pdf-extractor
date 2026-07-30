import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path(os.getenv("PDF_EXTRACTOR_LOG_DIR", "app/logs"))
LOG_LEVEL = os.getenv("PDF_EXTRACTOR_LOG_LEVEL", "INFO").upper()

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|credential|password)(\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(key=)([A-Za-z0-9_\-+/=]{12,})"),
]


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in SECRET_PATTERNS:
            if pattern.groups >= 3:
                message = pattern.sub(r"\1\2[REDACTED]", message)
            else:
                message = pattern.sub(r"\1[REDACTED]", message)
        record.msg = message
        record.args = ()
        return True


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if getattr(root, "_pdf_extractor_configured", False):
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redaction_filter = SecretRedactionFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redaction_filter)

    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redaction_filter)

    root.handlers.clear()
    root.setLevel(LOG_LEVEL)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root._pdf_extractor_configured = True

    for logger_name in ("azure", "httpx", "urllib3", "PIL"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
