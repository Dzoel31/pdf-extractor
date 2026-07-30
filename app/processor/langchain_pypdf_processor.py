import json
import time
import warnings
from pathlib import Path
from typing import Generator

from pypdf import PdfReader

from app.config import OUTPUT_DIR
from app.logging_config import get_logger
from app.utils.helper import check_json_file_exists, ensure_temp_dir, log_process


logger = get_logger(__name__)


def _pypdf_loader_class():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`langchain-community` is being sunset.*",
            category=DeprecationWarning,
        )
        from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader


class LangChainPyPDFProcessor:
    def __init__(self, input_path: str | Path, overwrite: bool = False):
        self.input_path = Path(input_path)
        self.overwrite = overwrite
        self.output_dir = OUTPUT_DIR / "langchain_pypdf"
        ensure_temp_dir(self.output_dir)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

    def process_pdf(self) -> Generator[dict[str, str], None, None]:
        base_name = self.input_path.stem
        output_file = self.output_dir / f"{base_name}.json"

        if not self.overwrite and check_json_file_exists(output_file):
            logger.info("JSON result already exists at %s. Skipping processing.", output_file)
            yield log_process(
                "skip",
                f"JSON result already exists for {base_name}. Skipping processing.",
            )
            return

        try:
            PyPDFLoader = _pypdf_loader_class()
            total_start = time.perf_counter()
            total_pages = self._page_count()
            result_json = {
                "content": [],
                "total_pages": total_pages,
                "total_time": 0.0,
                "method": "LangChain + PyPDF",
            }
            loader = PyPDFLoader(str(self.input_path), mode="page")
            documents = iter(loader.lazy_load())
            idx = 1

            while True:
                page_start = time.perf_counter()
                try:
                    document = next(documents)
                except StopIteration:
                    break

                page_number = int(document.metadata.get("page", idx - 1)) + 1
                content = document.page_content or ""
                duration = round(time.perf_counter() - page_start, 2)

                result_json["content"].append(
                    {
                        "page": page_number,
                        "content": content,
                        "duration": duration,
                        "source": document.metadata.get("source", str(self.input_path)),
                    }
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=2)

                yield log_process(
                    "info",
                    f"Processed page {page_number}/{total_pages} of {base_name} in {duration:.2f} seconds.",
                )
                idx += 1

            result_json["total_time"] = round(time.perf_counter() - total_start, 2)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)

            yield log_process(
                "success",
                f"Processed {base_name} with {total_pages} pages in {result_json['total_time']:.2f} seconds.",
            )

        except Exception as exc:
            logger.exception("Error processing PDF %s with LangChain + PyPDF: %s", self.input_path, exc)
            yield log_process(
                "error",
                f"Error processing PDF {self.input_path}: {exc}",
            )
            raise

    def _page_count(self) -> int:
        reader = PdfReader(self.input_path)
        return len(reader.pages)
