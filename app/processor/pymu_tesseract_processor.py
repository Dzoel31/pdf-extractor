import json
import re
import time
import gc
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import pymupdf
import pytesseract

from app.config import OUTPUT_DIR, TEMP_DIR, TEMP_DIR_MAP
from app.processor.yolo_processor import YoloProcessor
from app.utils.helper import (
    ensure_temp_dir,
    check_json_file_exists,
    log_process,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


class PymuTesseractProcessor:
    def __init__(self, input_path: Union[str, Path], overwrite: bool = True):
        self.input_path = Path(input_path)
        self.overwrite = overwrite
        self.output_dir = OUTPUT_DIR / "pymu_tesseract"
        ensure_temp_dir(self.output_dir)

        # Tambahkan atribut penting lain jika diperlukan
        self.temp_dir = TEMP_DIR
        self.temp_dir_map = TEMP_DIR_MAP

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

    @staticmethod
    def mask_and_redact_non_text(image, model, page, zoom):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(img_cv, verbose=False)
        target_labels = list(model.names.values())
        bounding_boxes = {label: [] for label in target_labels}
        label_masking = ["Non-Text"]
        masked_cv = img_cv.copy()
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                label_id = int(cls)
                label_name = model.names[label_id]
                bounding_boxes[label_name].append((x1, y1, x2, y2))
                if label_name in label_masking:
                    cv2.rectangle(masked_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)
                    rect = pymupdf.Rect(x1 // zoom, y1 // zoom, x2 // zoom, y2 // zoom)
                    page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        masked_pil = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))
        return masked_pil, bounding_boxes

    def process_pdf(self):
        base_name = self.input_path.stem
        output_file = self.output_dir / f"{base_name}.json"
        model = YoloProcessor().model

        if not self.overwrite and check_json_file_exists(output_file):
            logger.info("JSON result already exists at %s. Skipping processing.", output_file)
            yield log_process(
                "skip",
                f"JSON result already exists for {base_name}. Skipping processing.",
            )
            return

        try:
            with pymupdf.open(self.input_path) as doc:
                result_json = {"content": [], "total_pages": doc.page_count}
                total_pages = doc.page_count
                total_times = 0.0

                for i, page in enumerate(doc.pages()):
                    start_time = time.time()
                    yield log_process(
                        "info",
                        f"Processing page {i + 1}/{total_pages} of {base_name}",
                    )

                    # Placeholder for actual extraction logic
                    content, confidence = self.extract_pdf_single_page(
                        doc, base_name, model, i
                    )

                    duration = round(time.time() - start_time, 2)
                    total_times += duration

                    result_json["content"].append({
                        "page": i + 1,
                        "content": content,
                        "confidence": confidence,
                        "duration": duration,
                    })

                    with open(output_file, "w+", encoding="utf-8") as f:
                        json.dump(result_json, f, indent=2, ensure_ascii=False)

                result_json["total_time"] = round(total_times, 2)

                with open(output_file, "w+", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)

                yield log_process(
                    "success",
                    f"Processed {total_pages} pages in {base_name}.",
                )

        except Exception as e:
            logger.exception("Error processing PDF %s: %s", self.input_path, e)
            yield log_process(
                "error",
                f"Error processing PDF {self.input_path}: {e}",
            )
            return

    def _extract_text_tesseract(self, image) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            image=image,
            config="--oem 3 --psm 4",
            lang="eng+ind",
            output_type=pytesseract.Output.DICT,
        )
        confidence = [float(data['conf'][i]) for i in range(len(data['text'])) if data['conf'][i] != '-1']
        avg_confidence = round(sum(confidence) / len(confidence), 2) if confidence else 0.0

        text = " ".join([
            data['text'][i]
            for i in range(len(data['text']))
            if data['conf'][i] != '-1' and data['text'][i].strip() != ""
        ])

        return text, avg_confidence

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = "\n".join([line.strip() for line in text.splitlines()])
        text = text.replace("\t", " ")
        return text.strip()

    @staticmethod
    def mask_image_with_yolo(image, model):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(img_cv, verbose=False)

        target_labels = list(model.names.values())
        bounding_boxes = {label: [] for label in target_labels}
        label_masking = ["Non-Text"]
        masked_cv = img_cv.copy()

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                label_id = int(cls)
                label_name = model.names[label_id]
                bounding_boxes[label_name].append((x1, y1, x2, y2))

                if label_name in label_masking:
                    cv2.rectangle(masked_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)

        masked_pil = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))
        return masked_pil, bounding_boxes

    @staticmethod
    def page_to_image(page, dpi=300):
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples), zoom

    def extract_pdf_single_page(self, doc, base_name, model_yolo, page_number):
        page = doc.load_page(page_number)
        img, zoom = self.page_to_image(page)
        mask_image, bounding_boxes = self.mask_and_redact_non_text(img, model_yolo, page, zoom)

        label_count = {}
        text_bounding_box = {}
        for label, bboxes in bounding_boxes.items():
            if bboxes and label == "Text":
                for bbox in bboxes:
                    label_count[label] = label_count.get(label, 0) + 1
                    unique_label = f"{label}{label_count[label]}"
                    text_bounding_box[unique_label] = [bbox]

        table_bounding_box = {}
        tables = page.find_tables(strategy="lines_strict").tables
        if tables:
            for index, x in enumerate(tables, start=1):
                table_key = f"Table{index}" if len(tables) > 1 else "Table"
                if table_key not in table_bounding_box:
                    table_bounding_box[table_key] = {"bounding_box": [], "objects": []}
                table_bounding_box[table_key]["bounding_box"].append((x.bbox[0], x.bbox[1], x.bbox[2], x.bbox[3]))
                table_bounding_box[table_key]["objects"].append(x)

        combined_data = {}
        if table_bounding_box:
            for label, info in table_bounding_box.items():
                if info["bounding_box"]:
                    scaled_bounding_boxes = [
                        (int(x1 * zoom), int(y1 * zoom), int(x2 * zoom), int(y2 * zoom))
                        for x1, y1, x2, y2 in info["bounding_box"]
                    ]
                    combined_data[label] = {
                        "bbox": scaled_bounding_boxes[0],
                        "objects": info["objects"]
                    }
        if text_bounding_box:
            for label, bboxes in text_bounding_box.items():
                if bboxes:
                    combined_data[label] = {
                        "bbox": bboxes[0],
                        "objects": []
                    }

        sorted_combined = dict(
            sorted(combined_data.items(), key=lambda item: item[1]["bbox"][1])
        )

        if sorted_combined:
            combined_content = ""
            confidences = []
            working_image1 = mask_image.copy()
            draw1 = ImageDraw.Draw(working_image1)

            for label, info in sorted_combined.items():
                bbox = info["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                draw1.rectangle([x1, y1, x2, y2], fill="white", outline="red")

            for label, info in sorted_combined.items():
                bbox = info["bbox"]
                if label.startswith("Text"):
                    working_image2 = mask_image.copy()
                    draw2 = ImageDraw.Draw(working_image2)
                    for other_label, other_info in sorted_combined.items():
                        if other_label != label:
                            other_bbox = other_info["bbox"]
                            if isinstance(other_bbox, (list, tuple)) and len(other_bbox) == 4:
                                x1, y1, x2, y2 = map(int, other_bbox)
                                draw2.rectangle([x1, y1, x2, y2], fill="white", outline="red")
                    raw_text, confidence = self._extract_text_tesseract(working_image2)
                    confidences.append(confidence)
                    combined_content += f"\n\n{raw_text}\n\n"
                    del working_image2, draw2
                elif label.startswith("Table"):
                    for bbox, table in zip(info["bbox"], info["objects"]):
                        rows = table.extract()
                        if rows:
                            combined_content += f"\n\n{label}:\n\n"
                            for row in rows:
                                combined_content += f"{row}\n\n"

            raw_text, confidence = self._extract_text_tesseract(working_image1)
            combined_content += f"\n\n{raw_text}\n\n"

            combined_content = self._clean_text(combined_content)
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            del working_image1, draw1, table_bounding_box, text_bounding_box, combined_data
            gc.collect()

            return combined_content, avg_confidence

        else:
            combined_content = ""
            raw_text, confidence = self._extract_text_tesseract(mask_image)
            combined_content += f"\n\n{raw_text}\n\n"
            combined_content = self._clean_text(combined_content)
            return combined_content, confidence

if __name__ == "__main__":
    processor = PymuTesseractProcessor("sample-layout.pdf")
    for log in processor.process_pdf():
        print(log)
