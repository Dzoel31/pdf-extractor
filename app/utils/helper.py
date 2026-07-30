import os
import random
import re
from pathlib import Path
import time
from typing import Any, Generator, Optional, Union, List
from urllib.parse import urlparse

import requests
import pandas as pd
import json
import pymupdf
import csv

from app.config import TEMP_DIR, TEMP_DIR_MAP
from app.logging_config import get_logger

logger = get_logger(__name__)


def ensure_temp_dir(dir_name: Union[str, Path, List[str], List[Path]] = TEMP_DIR):
    """Ensure the temporary directory exists."""
    if isinstance(dir_name, list):
        for sub_dir in dir_name:
            Path(sub_dir).mkdir(parents=True, exist_ok=True)
    else:
        Path(dir_name).mkdir(parents=True, exist_ok=True)


def log_process(status: str, message: str) -> dict:
    """
    Logs the process status and message.

    This function creates a dictionary containing the status and message
    of a process, which can be used for logging or displaying information
    in a Streamlit application.

    Args:
        status (str): The status of the process (e.g., "success", "error").
        message (str): A descriptive message about the process.

    Returns:
        dict: A dictionary containing the status and message.
    """
    return {
        "status": status,
        "message": message,
    }


def safe_filename(filename: str, fallback: str = "downloaded_file") -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")
    return name or fallback

def check_json_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Checks if a JSON file exists and has content.

    Args:
        file_path (str | Path): The path to the JSON file.

    Returns:
        bool: True if the file exists and has content, False otherwise.
    """
    file_path = Path(file_path)
    if file_path.exists():
        try:
            json_content = json.loads(file_path.read_text(encoding="utf-8"))
            content_count = len(json_content.get("content", [])) or 0
            expected_count = (
                json_content.get("total_pages")
                or json_content.get("total_segments")
                or 0
            )
            return bool(expected_count) and expected_count == content_count
        except ValueError:
            return False
    return False

def read_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a dataset from a file and returns it as a DataFrame.

    Args:
        file_path (str | Path): The path to the dataset file.

    Returns:
        pd.DataFrame: The dataset as a DataFrame.
    """
    file_path = Path(file_path)
    if file_path.suffix == ".csv":
        # Try to detect delimiter automatically using csv.Sniffer
        with open(file_path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
            if sample:
                try:
                    delimiter = csv.Sniffer().sniff(sample).delimiter
                except Exception:
                    delimiter = ","
            else:
                delimiter = ","
        return pd.read_csv(file_path, sep=delimiter)
    elif file_path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type for dataset.")


class Downloader:
    """
    Initialize the Downloader with a URL, timeout, and stream option.

    Args:
        url (str): The URL to download the file or from the dataset.
        url_column (str): The column name in the dataset containing URLs. Mandatory if `url` is a dataset.
        timeout (int): The timeout for the download request.
        stream (bool): Whether to stream the download.
    """
    def __init__(self, url: str | pd.DataFrame, url_column: Optional[str] = None, url_id: Optional[str] = None, timeout: int = 20, stream: bool = True):
        self.timeout = timeout
        self.stream = stream

        if isinstance(url, str) and os.path.isfile(url):
            if not url_column:
                raise ValueError("The 'url_column' parameter must be provided for dataset URLs.")
            
            if url.endswith(".csv"):
                self.df = pd.read_csv(url)
            elif url.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(url)
            else:
                raise ValueError("Unsupported file type for dataset.")
            
            if url_column not in self.df.columns:
                raise ValueError(f"Column '{url_column}' not found in the dataset.")
            self.urls = self.df[url_column].tolist()
        elif isinstance(url, str):  # If it's a single URL
            self.urls = [url]

        elif isinstance(url, pd.DataFrame):  # If it's a DataFrame
            if url_column not in url.columns:
                raise ValueError(f"Column '{url_column}' not found in the DataFrame.")
            self.urls = url[url_column].tolist()
        else:
            raise ValueError("The 'url' parameter must be a string or a DataFrame.")
        self.url_id = url_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
        )
        self.max_retries = 3

    def download(self) -> Generator[dict[Any, Any], Any, None]:
        """Download files from the specified URL(s), either from a dataset or a single URL."""
        ensure_temp_dir(TEMP_DIR)

        for url in self.urls:
            if not isinstance(url, str) or not url.strip():
                logger.warning("Skipping empty URL from dataset.")
                yield log_process("error", "Empty URL found. Skipping download...")
                continue
            self.url = url
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"}:
                logger.warning("Unsupported URL scheme for %s", parsed_url.geturl())
                yield log_process("error", f"Unsupported URL scheme: {parsed_url.scheme}")
                continue

            self.filepath = Path(parsed_url.path)
            self.filename = safe_filename(self.filepath.name)
            try:
                self.dirpath = TEMP_DIR / next(
                    key
                    for key, value in TEMP_DIR_MAP.items()
                    if self.filepath.suffix.split(".")[-1].lower() in value
                )
            except StopIteration:
                logger.error(f"Unsupported file extension {self.url}: {self.filepath.suffix}")
                yield log_process(
                    "error",
                    f"❌ Unsupported file extension {self.url}: {self.filepath.suffix}. Skipping download...",
                )
                continue

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.session.get(
                        self.url, timeout=self.timeout, stream=self.stream
                    )
                    response.raise_for_status()  # Raise an error for bad responses

                    if self.url_id:
                        self.filename = safe_filename(f"{self.url_id}{self.filepath.suffix}")

                    ensure_temp_dir(self.dirpath)

                    with open(self.dirpath / self.filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    time.sleep(
                        random.uniform(0.5, 1.5)
                    )  # Random sleep to avoid rate limiting
                    logger.info("Downloaded %s successfully.", self.filename)
                    yield log_process("success", f"Downloaded {self.filename}")

                    # Check if the downloaded file is a valid PDF (only for PDF files)
                    if self.filepath.suffix.lower() == ".pdf":
                        if not self._is_pdf_valid(self.dirpath / self.filename):
                            logger.error("Invalid PDF file: %s", self.filename)
                            yield log_process(
                                "error",
                                f"Invalid PDF file: {self.filename}. Trying to download again...",
                            )
                            os.remove(self.dirpath / self.filename)
                            continue  # Retry download
                    break

                except requests.RequestException as e:
                    logger.warning(
                        "Download attempt %s/%s failed for %s: %s",
                        attempt,
                        self.max_retries,
                        self.filename,
                        e,
                    )
                    if attempt == self.max_retries:
                        yield log_process(
                            "error",
                            f"Failed to download {self.filename} after {self.max_retries} attempts.",
                        )
                    time.sleep(random.uniform(0.5, 1.5))

    def _is_pdf_valid(self, file_path: Path) -> bool:
        """Check if the PDF file is valid."""
        try:
            with pymupdf.open(file_path) as doc:
                doc.load_page(0)  # Try to load the first page
                logs = pymupdf.TOOLS.mupdf_warnings()

                if re.search(r"(repairing PDF document|format error)", logs, re.IGNORECASE):
                    return False
            return True
        except Exception as e:
            logger.exception("Error validating PDF %s: %s", file_path, e)
            return False
