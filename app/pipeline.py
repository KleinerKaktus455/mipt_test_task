import base64
import os
import threading
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from russian_docs_ocr.document_processing import Pipeline as RussianDocsPipeline
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "russian-docs-ocr is required. Make sure it is installed via requirements.txt."
    ) from e


_RUSSIAN_DOCS_PIPELINE: RussianDocsPipeline | None = None
_PIPELINE_LOCK = threading.Lock()


def _get_pipeline() -> RussianDocsPipeline:
    global _RUSSIAN_DOCS_PIPELINE
    if _RUSSIAN_DOCS_PIPELINE is None:
        model_format = os.getenv("RUS_DOCS_MODEL_FORMAT", "ONNX")
        device = os.getenv("RUS_DOCS_DEVICE", "cpu")
        _RUSSIAN_DOCS_PIPELINE = RussianDocsPipeline(model_format=model_format, device=device)
    return _RUSSIAN_DOCS_PIPELINE


def _load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def _encode_image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _cleanup_whitespace(s: str | None) -> str | None:
    if not s:
        return None
    out = " ".join(str(s).split()).strip()
    return out or None


def _extract_structured_fields(ocr_dict: Dict[str, Any]) -> Dict[str, Any]:
    last_name = _cleanup_whitespace(ocr_dict.get("Last_name_ru"))
    first_name = _cleanup_whitespace(ocr_dict.get("First_name_ru"))
    middle_name = _cleanup_whitespace(ocr_dict.get("Middle_name_ru"))
    full_name = " ".join(p for p in [last_name, first_name, middle_name] if p) or None

    date_of_birth = _cleanup_whitespace(ocr_dict.get("Birth_date"))
    if date_of_birth:
        date_of_birth = date_of_birth.replace("-", ".").replace("/", ".")

    document_number = _cleanup_whitespace(ocr_dict.get("Licence_number"))

    return {
        "full_name": full_name,
        "date_of_birth": date_of_birth,
        "document_number": document_number,
    }


def _to_bbox(entry: Any) -> Tuple[int, int, int, int] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 4:
        return None

    coords = []
    for value in entry[:4]:
        if isinstance(value, (int, float)):
            coords.append(int(value))
        else:
            return None

    x1, y1, x2, y2 = coords
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return (left, top, right, bottom)


def _build_detections_and_lines(text_fields_meta: Dict[str, Any], ocr_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    detections: List[Dict[str, Any]] = []
    bboxes = text_fields_meta.get("bbox") if isinstance(text_fields_meta, dict) else None

    if isinstance(bboxes, list):
        for entry in bboxes:
            bbox = _to_bbox(entry)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            text = entry[-1] if isinstance(entry[-1], str) else ""
            confidence = 0.0
            if len(entry) > 4 and isinstance(entry[4], (int, float)):
                confidence = float(entry[4])

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "confidence": confidence,
                }
            )

    raw_lines: List[str] = []
    if isinstance(ocr_dict, dict):
        for value in ocr_dict.values():
            if isinstance(value, str):
                clean = _cleanup_whitespace(value)
                if clean:
                    raw_lines.append(clean)

    return detections, raw_lines


def _annotate_image(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        text = det.get("text") or ""
        conf = float(det.get("confidence") or 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if text:
            label = f"{text} ({conf:.2f})"
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - 20)),
                (x1 + min(len(label) * 8, 240), y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return annotated


def process_document_image(image_bytes: bytes) -> Dict[str, Any]:
    original_bgr = _load_image_from_bytes(image_bytes)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    pipeline = _get_pipeline()
    docconf = float(os.getenv("RUS_DOCS_DOCCONF", "0.5"))
    img_size = int(os.getenv("RUS_DOCS_IMG_SIZE", "1500"))

    with _PIPELINE_LOCK:
        results = pipeline(
            img_path=original_rgb,
            ocr=True,
            get_doc_borders=True,
            find_text_fields=True,
            check_quality=False,
            low_quality=True,
            docconf=docconf,
            img_size=img_size,
        )

    ocr_dict = results.ocr or {}
    text_fields_meta = results.text_fields_meta or {}
    structured = _extract_structured_fields(ocr_dict)
    detections, raw_lines = _build_detections_and_lines(text_fields_meta, ocr_dict)

    # RussianDocsOCR works in RGB; convert to BGR for OpenCV drawing/encoding.
    best_rgb = results.img_with_fixed_perspective if results.img_with_fixed_perspective is not None else results.rotated_image
    if best_rgb is None:
        best_rgb = original_rgb
    best_bgr = cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR)

    annotated = _annotate_image(best_bgr, detections)
    annotated_b64 = _encode_image_to_base64(annotated)

    return {
        "angle": 0.0,
        "structured_fields": structured,
        "raw_lines": raw_lines,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "document_type": results.doctype,
        "quality": results.quality,
    }
