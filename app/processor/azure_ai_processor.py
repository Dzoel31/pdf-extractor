import logging
import os
import re
import time
import json
import gc
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

import fitz  # pymupdf
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from app.utils.helper import (
    log_process,
    check_json_file_exists,
    ensure_temp_dir,
)
from app.config import TEMP_DIR, OUTPUT_DIR
from app.logging_config import get_logger
from app.processor.yolo_processor import YoloProcessor

load_dotenv()

logger = get_logger(__name__)

for name in logging.root.manager.loggerDict:
    if name.startswith("azure"):
        logging.getLogger(name).setLevel(logging.WARNING)


class AzureAIProcessor:
    def __init__(self, endpoint: str, key: str, yolo_model_path="app/yolo/best-1.pt"):
        if not endpoint or not key:
            raise ValueError(
                "Azure Document Intelligence credentials are missing. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY."
            )
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        self.endpoint = endpoint
        self.output_dir = OUTPUT_DIR / "azure_document_intelligence"
        self.temp_pdf_dir = TEMP_DIR / "processed_pdf"
        self.yolo_model = YoloProcessor(yolo_model_path).model

    @staticmethod
    def clean_text(text):
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        return text.strip()

    @staticmethod
    def extract_table_contents(table_dict):
        table = next(iter(table_dict.values()))
        row_count = table.get("rowCount") or table.get("row_count") or 0
        col_count = table.get("columnCount") or table.get("column_count") or 0
        table_contents = [["" for _ in range(col_count)] for _ in range(row_count)]
        for cell in table.get("cells", []):
            row = cell.get("rowIndex", cell.get("row_index"))
            col = cell.get("columnIndex", cell.get("column_index"))
            content = cell.get("content", "")
            if row is None or col is None:
                continue
            if "columnSpan" in cell or "column_span" in cell:
                span = cell.get("columnSpan", cell.get("column_span", 1))
                combined = f"[{content}]"
                for c in range(col, col + span):
                    if row < row_count and c < col_count:
                        table_contents[row][c] = combined
            else:
                if row < row_count and col < col_count:
                    table_contents[row][col] = content
        return table_contents

    @staticmethod
    def download_pdf_if_url(url_pdf, output_folder="temp"):
        if not url_pdf.startswith("http"):
            return url_pdf
        try:
            os.makedirs(output_folder, exist_ok=True)
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url_pdf, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            filename = url_pdf.split("/")[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"[DOWNLOADED] PDF to local directory: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[ERROR] Failed to download PDF: {e}")
            return None

    @staticmethod
    def mask_pdf_with_fitz(local_pdf_path, page_number, bounding_polygons, output_folder="temp", zoom=1.0):
        os.makedirs(output_folder, exist_ok=True)
        if not local_pdf_path:
            logger.error("[ERROR] PDF masking cannot be processed.")
            return None
        filename = f"masked_table_page_{page_number}.pdf"
        output_path = os.path.join(output_folder, filename)
        with fitz.open(local_pdf_path) as original_doc, fitz.open() as new_doc:
            new_doc.insert_pdf(
                original_doc,
                from_page=page_number - 1,
                to_page=page_number - 1,
                links=False,
                widgets=False,
            )
            new_page = new_doc[-1]
            for poly in bounding_polygons:
                if len(poly) < 8:
                    logger.warning("Skipping anomalous polygon coordinates: %s", poly)
                    continue
                x_coords, y_coords = poly[0::2], poly[1::2]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                dpi = 72
                rect = fitz.Rect(x1 * dpi, y1 * dpi, x2 * dpi, y2 * dpi)
                new_page.add_redact_annot(rect, fill=(1, 1, 1))
            new_page.apply_redactions()
            new_doc.save(output_path)
        logger.info("Masked table page saved to %s", output_path)
        return output_path

    @staticmethod
    def page_to_image(pdf_path, page_number=0, dpi=300):
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples), zoom

    def mask_image_with_yolo(self, image, page_number, output_folder="temp"):
        import cv2
        os.makedirs(output_folder, exist_ok=True)
        filename = f"masked_page_yolo{page_number}.pdf"
        output_path = os.path.join(output_folder, filename)
        pil_image = image[0] if isinstance(image, tuple) else image
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results = self.yolo_model.predict(img_cv, verbose=False)
        label_masking = ["Non-Text"]
        masked_cv = img_cv.copy()
        for r in results:
            if getattr(r.boxes, "xyxy", None) is None or len(r.boxes.xyxy) == 0:
                logger.warning("[WARNING] No exclude objects detected")
                continue
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label_id = int(cls)
                    label_name = self.yolo_model.names[label_id]
                    if label_name in label_masking:
                        import cv2
                        cv2.rectangle(masked_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)
            except Exception as e:
                logger.error(f"[ERROR] Error while masking exclude object: {e}")
                continue
        masked_pil = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))
        masked_pil.save(output_path, "PDF", resolution=300.0)
        logger.info("Masked YOLO page saved to %s", output_path)
        return masked_pil, output_path

    def get_doc_intelligent_result(self, file, model="prebuilt-layout"):
        try:
            if isinstance(file, str) and file.startswith("http"):
                request = AnalyzeDocumentRequest(url_source=file)
                poller = self.client.begin_analyze_document(model, request)
            else:
                with open(file, "rb") as f:
                    poller = self.client.begin_analyze_document(
                        model,
                        body=f,
                        content_type="application/pdf",
                    )
        except Exception as e:
            logger.exception("Failed to create Azure poller: %s", e)
            return None
        return poller.result()

    @staticmethod
    def extract_text_from_azure_result(result):
        if hasattr(result, "pages") and result.pages:
            return "\n".join(
                line.content for page in result.pages for line in page.lines
            )
        return str(result)

    def process_page_with_table(self, page_number, detected_tables, local_pdf_path, temp_dir):
        table_content = [
            table["content"] for table in detected_tables if table["page"] == page_number
        ]
        table_polygon = [
            table["boundingPolygon"] for table in detected_tables if table["page"] == page_number
        ]
        page_text = []
        for tbl in table_content:
            table_str = "\n".join(["\t".join(map(str, row)) for row in tbl])
            page_text.append("\n[Teks di dalam tabel:]\n" + table_str)
        mask_table_pdf_path = self.mask_pdf_with_fitz(
            local_pdf_path, page_number, table_polygon, output_folder=temp_dir
        )
        mask_table_to_img = self.page_to_image(mask_table_pdf_path)
        _, output_path_exclude_object = self.mask_image_with_yolo(
            mask_table_to_img, page_number, output_folder=temp_dir
        )
        text_without_table = self.get_doc_intelligent_result(
            str(output_path_exclude_object)
        )
        ocr_text = self.extract_text_from_azure_result(text_without_table)
        page_text.append("\n[Teks di luar tabel:]\n" + self.clean_text(ocr_text))
        return page_text

    def process_page_without_table(self, page, local_pdf_path, temp_dir):
        page_text = []
        mask_table_to_img = self.page_to_image(local_pdf_path, page_number=page.page_number - 1)
        _, output_path_exclude_object = self.mask_image_with_yolo(
            mask_table_to_img, page.page_number, output_folder=temp_dir
        )
        text_without_table = self.get_doc_intelligent_result(
            str(output_path_exclude_object)
        )
        if text_without_table is None:
            return page_text
        for p in text_without_table.pages:
            for line in p.lines:
                page_text.append(line.content)
        return page_text

    @staticmethod
    def compute_avg_confidence(page):
        """ Compute the average confidence score for words in a page.
        Args:
            page (DocumentPage): The page object containing words.
        Returns:
            float: The average confidence score.
        """
        all_confidences = [word.confidence for word in page.words]
        return round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

    @staticmethod
    def _as_dict(value):
        if hasattr(value, "as_dict"):
            return value.as_dict()
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value

    def transcribe_pdf(self, file: str, overwrite=True):
        """ Transcribe a PDF file using Azure Document Intelligence.
        
        Args:
            file (str): Path to the PDF file or URL.
            overwrite (bool, optional): Whether to overwrite existing JSON files. Defaults to True.
        Yields:
            Generator of log messages indicating the progress and status of the transcription.
        """
        file_name = Path(file).stem
        output_folder = self.output_dir
        os.makedirs(output_folder, exist_ok=True)
        output_json_path = Path(output_folder) / f"{file_name}.json"

        if not overwrite and check_json_file_exists(output_json_path):
            yield log_process(
                "skip", f"[SKIP] JSON result already exists for {file_name}, skipping."
            )
            return

        try:
            result_json = {
                "content": [],
                "total_time": 0,
            }
            result = self.get_doc_intelligent_result(file)
            if result is None:
                raise RuntimeError("Azure Document Intelligence did not return a result.")
            page_count = len(result.pages)
            logger.info("Starting Azure extraction for %s with %s pages.", file_name, page_count)
            detected_tables = []
            if getattr(result, "tables", None):
                for i, table in enumerate(result.tables, 1):
                    table_data = self._as_dict(table)
                    table_key = f"table_{i}"
                    bounding_regions = (
                        table_data.get("boundingRegions")
                        or table_data.get("bounding_regions")
                        or [{}]
                    )
                    bounding_region = bounding_regions[0]
                    page_number = (
                        bounding_region.get("pageNumber")
                        or bounding_region.get("page_number")
                    )
                    polygon = bounding_region.get("polygon", [])
                    table_dict = {table_key: table_data}
                    content = self.extract_table_contents(table_dict)
                    detected_tables.append(
                        {
                            "page": page_number,
                            "table_id": table_key,
                            "boundingPolygon": polygon,
                            "content": content,
                        }
                    )
            result_json["total_pages"] = page_count
            pages_with_tables = {table["page"] for table in detected_tables}
            pages_str = ", ".join(str(page) for page in sorted(pages_with_tables))
            logger.info("File %s contains tables on pages: %s", file_name, pages_str or "-")

            local_pdf_path = self.download_pdf_if_url(file) if isinstance(file, str) and file.startswith("http") else file

            total_page_time = 0
            for page in result.pages:
                logger.info("Processing Azure page %s/%s for %s", page.page_number, page_count, file_name)
                page_number = page.page_number
                page_start_time = time.time()
                if page_number in pages_with_tables:
                    logger.info("Page %s contains table(s).", page_number)
                    page_text = self.process_page_with_table(
                        page_number, detected_tables, local_pdf_path, self.temp_pdf_dir
                    )
                else:
                    logger.info("Page %s does not contain tables.", page_number)
                    page_text = self.process_page_without_table(
                        page, local_pdf_path, self.temp_pdf_dir
                    )
                full_text = self.clean_text("\n".join(page_text))
                avg_confidence = self.compute_avg_confidence(page)
                logger.info("Page %s average confidence score: %s", page_number, avg_confidence)

                elapsed_time = round(time.time() - page_start_time, 2)
                total_page_time += elapsed_time
                result_json["content"].append(
                    {
                        "page": page_number,
                        "content": full_text,
                        "confidence": avg_confidence,
                        "duration": elapsed_time,
                    }
                )

                yield log_process(
                    "info",
                    f"Processed page {page_number}/{page_count} of {file_name} in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
                )

                del page_text, full_text, avg_confidence
                gc.collect()

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=2)

            result_json["total_time"] = round(total_page_time, 2)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)

            total_minutes, total_seconds = divmod(result_json["total_time"], 60)
            yield log_process(
                "success",
                f"Total time for all pages: {int(total_minutes)} minutes {total_seconds:.2f} seconds",
            )
            logger.info("Azure extraction results for %s saved to %s", file_name, output_json_path)

        except Exception as e:
            yield log_process(
                "error",
                f"An error occurred during transcription: {str(e)}"
            )
            logger.exception("An error occurred during Azure transcription: %s", e)

if __name__ == "__main__":
    processor = AzureAIProcessor(
        endpoint=os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
        key=os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", ""),
        yolo_model_path="app/yolo/best-1.pt"
    )
    # Example usage
    for log in processor.transcribe_pdf("sample-layout.pdf"):
        print(log)
