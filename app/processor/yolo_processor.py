from pathlib import Path
from ultralytics import YOLO
from app.config import YOLO_DIR
from pymupdf import Page, Rect, Matrix
from PIL import Image
from typing import Union, List, Tuple, Optional
from functools import lru_cache
from app.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_yolo_model(model_path: str):
    return YOLO(model_path)

class YoloProcessor:
    def __init__(self, model_path: Optional[str | Path] = None):
        if not model_path:
            model_path = self.get_latest_model()
        model_path = Path(model_path).resolve()
        logger.info("Loading YOLO model: %s", model_path.name)
        self.model = _load_yolo_model(str(model_path))

    def get_latest_model(self) -> Path:
        """Get the latest YOLO model file from the YOLO directory."""
        yolo_files = sorted(YOLO_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)

        if not yolo_files:
            raise FileNotFoundError("No YOLO model files found in the directory.")
        return yolo_files[0]
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        conf: float = 0.5,
        verbose: bool = False,
        ):
        """Run YOLO model prediction on the given image.

        Args:
            image (Union[str, Path]): Path to the image file.
            conf (float): Confidence threshold for predictions.
            verbose (bool): If True, print additwith bounding boxes and labels.
        """
        results = self.model.predict(image, conf=conf, verbose=verbose)
        result_dict = {
            "cls": results[0].boxes.cls.cpu().tolist(), # type: ignore
            "box": results[0].boxes.xyxy.cpu().tolist(), # type: ignore
        }
        if verbose:
            logger.info(f"Predictions: {result_dict}")
        return result_dict
    
    @staticmethod
    def normalize_bbox(bbox: List[List[float]], zoom: float) -> List[Rect]:
        """Normalize YOLO bounding box coordinates to PyMuPDF format.
        Args:
            bbox (List[List[float]]): List of bounding boxes in YOLO format [x1, y1, x2, y2].
            zoom (float): Zoom level of the page.
        Returns:
            List[Rect]: List of normalized bounding boxes in PyMuPDF format.
        """
        rects = []
        for box in bbox:
            x1, y1, x2, y2 = box
            rect = Rect(
                x0=x1 // zoom,
                y0=y1 // zoom,
                x1=x2 // zoom,
                y1=y2 // zoom,
            )
            rects.append(rect)
        return rects

    def exclude_object(
        self,
        page: Page,
        class_names: Union[int, str],
        zoom: float = 3.0,
        color: Tuple[int, int, int] = (1,1,1), # White color
    ) -> Page:
        """Exclude objects from the page based on bounding boxes.
        Args:
            page (Page): The PyMuPDF page object.
            class_names (Union[int, str]): Class name or index to exclude.
            bbox (List[Rect]): List of bounding boxes to exclude.
            color (Tuple[int, int, int]): Color to fill the redaction area.
        Returns:
            Page: The modified PyMuPDF page with redacted areas.
        """
        class_mapping = self.model.names if isinstance(class_names, int) else {i: class_names for i in range(len(self.model.names))}

        mat = Matrix(zoom, zoom)
        page_pixmap = page.get_pixmap(matrix=mat) # type: ignore

        page_to_image = Image.frombytes(
            "RGBA" if page_pixmap.alpha else "RGB",
            [page_pixmap.width, page_pixmap.height],
            page_pixmap.samples,
        )

        results = self.model.predict(page_to_image, conf=0.5, verbose=False)

        result_dict = {
            "cls": results[0].boxes.cls.cpu().tolist(),  # type: ignore
            "box": results[0].boxes.xyxy.cpu().tolist(),  # type: ignore
        }

        bbox = []
        for i, cls_value in enumerate(result_dict["cls"]):
            if isinstance(class_names, int) and cls_value == class_names:
                bbox.append(result_dict["box"][i])
            elif isinstance(class_names, str) and class_mapping[cls_value] == class_names:
                bbox.append(result_dict["box"][i])

        bbox = self.normalize_bbox(bbox, zoom)

        for rect in bbox:
            page.add_redact_annot(rect, fill=color)
        page.apply_redactions() # type: ignore
        return page
    
    @property
    def class_names(self) -> dict[int, str]:
        """Return class names (labels) from the loaded YOLO model."""
        return self.model.names
    
    
if __name__ == "__main__":
    from PIL import Image
    import pymupdf

    with pymupdf.open("kap_pjj_penilaian_aset_tak_berwujud_djp_bagi_pegawai_djp.pdf") as doc:
        page = doc[0]
        yolo_processor = YoloProcessor()
        print(yolo_processor.class_names)  # Will print the class names from the YOLO

        # Example usage
        result = yolo_processor.exclude_object(
            page=page,
            class_names=0,
        )

        # Preview the modified page
        result_pixmap = result.get_pixmap()
        img = Image.frombytes("RGB", [result_pixmap.width, result_pixmap.height], result_pixmap.samples)
        img.show()  # Display the image with redacted areas

