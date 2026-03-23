import base64
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

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


def _deskew_image(image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Fine deskew after optional coarse 90° rotation.
    Returns (rotated_bgr, applied_angle, debug_info).
    """
    debug: Dict[str, Any] = {}
    angle, meta = _estimate_best_skew_angle(image_bgr)
    debug.update(meta)

    if angle is None:
        return image_bgr, 0.0, debug

    min_skew_to_rotate = float(os.getenv("DESKEW_MIN_ANGLE", "0.5"))
    max_skew_to_rotate = float(os.getenv("DESKEW_MAX_ANGLE", "60.0"))
    if abs(angle) < min_skew_to_rotate or abs(angle) > max_skew_to_rotate:
        debug["skipped_reason"] = "outside_threshold"
        debug["threshold_min"] = min_skew_to_rotate
        debug["threshold_max"] = max_skew_to_rotate
        return image_bgr, 0.0, debug

    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    debug["applied_angle"] = float(angle)
    return rotated, float(angle), debug


def _normalize_min_area_rect_angle(width: float, height: float, raw_angle: float) -> float | None:
    if width <= 0 or height <= 0:
        return None

    angle = raw_angle + 90.0 if width < height else raw_angle
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return float(angle)


def _build_edge_maps(gray: np.ndarray) -> List[np.ndarray]:
    """Several edge maps — wood grain / glare break a single Canny pass."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    e1 = cv2.Canny(blur, 40, 120)
    e2 = cv2.Canny(blur, 60, 180)
    adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (9, 9), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2,
    )
    e3 = cv2.Canny(adapt, 50, 150)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return [
        cv2.morphologyEx(e1, cv2.MORPH_CLOSE, ker, iterations=2),
        cv2.morphologyEx(e2, cv2.MORPH_CLOSE, ker, iterations=2),
        cv2.morphologyEx(e3, cv2.MORPH_CLOSE, ker, iterations=2),
    ]


def _angle_and_score_from_contour(contour: np.ndarray, image_area: float) -> Tuple[Optional[float], float]:
    """
    Score contour as document-like; no approxPolyDP (rounded card corners break it).
    """
    area = float(cv2.contourArea(contour))
    min_a = float(os.getenv("DESKEW_MIN_CONTOUR_AREA_FRAC", "0.012"))
    max_a = float(os.getenv("DESKEW_MAX_CONTOUR_AREA_FRAC", "0.92"))
    if area < image_area * min_a or area > image_area * max_a:
        return None, 0.0

    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), raw_angle = rect
    if w <= 0 or h <= 0:
        return None, 0.0

    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / short_side
    # At ~45° tilt minAreaRect is often nearly square (diamond in AABB) — do not reject.
    amin = float(os.getenv("DESKEW_MIN_ASPECT", "1.0"))
    amax = float(os.getenv("DESKEW_MAX_ASPECT", "3.8"))
    if aspect < amin or aspect > amax:
        return None, 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    smin = float(os.getenv("DESKEW_MIN_SOLIDITY", "0.45"))
    if solidity < smin:
        return None, 0.0

    angle = _normalize_min_area_rect_angle(w, h, raw_angle)
    if angle is None:
        return None, 0.0

    # Prefer larger, more rectangular blobs.
    score = area * solidity * (1.0 + 0.15 * min(aspect, 2.5))
    return angle, score


def _estimate_angle_from_contours(image_bgr: np.ndarray) -> Tuple[Optional[float], Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    best_angle: Optional[float] = None
    best_score = float("-inf")
    per_map: List[Dict[str, Any]] = []

    for idx, edges in enumerate(_build_edge_maps(gray)):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            per_map.append({"map": idx, "candidates": 0})
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        local_best: Optional[float] = None
        local_score = float("-inf")
        tried = 0
        for contour in contours[:25]:
            ang, sc = _angle_and_score_from_contour(contour, image_area)
            tried += 1
            if ang is None:
                continue
            if sc > local_score:
                local_score = sc
                local_best = ang
            if sc > best_score:
                best_score = sc
                best_angle = ang
        per_map.append(
            {
                "map": idx,
                "contours_checked": tried,
                "best_angle": local_best,
                "best_score": local_score if local_score > float("-inf") else None,
            }
        )

    return best_angle, {
        "contour_maps": per_map,
        "contour_best_angle": best_angle,
        "contour_best_score": float(best_score) if best_score > float("-inf") else None,
    }


def _estimate_angle_hough(edges: np.ndarray) -> Tuple[Optional[float], Dict[str, Any]]:
    h, w = edges.shape[:2]
    min_len = max(30, int(min(h, w) * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, min(h, w) // 25),
        minLineLength=min_len,
        maxLineGap=12,
    )
    if lines is None or len(lines) == 0:
        return None, {"hough_lines": 0}

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ang = ((ang + 90.0) % 180.0) - 90.0
        if abs(ang) <= 89.0:
            angles.append(ang)

    if len(angles) < 5:
        return None, {"hough_lines": len(lines), "usable_angles": len(angles)}

    hist, bin_edges = np.histogram(angles, bins=72, range=(-90.0, 90.0))
    peak = int(np.argmax(hist))
    lo, hi = bin_edges[peak], bin_edges[peak + 1]
    center = float((lo + hi) / 2.0)
    return center, {"hough_lines": len(lines), "hough_peak_angle": center, "hough_peak_count": int(hist[peak])}


def _row_projection_score(gray_rot: np.ndarray) -> float:
    """Higher when horizontal text lines align with image rows (classic deskew cue)."""
    _, bw1 = cv2.threshold(gray_rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw2 = cv2.threshold(gray_rot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    s1 = float(np.var(bw1.sum(axis=1).astype(np.float64)))
    s2 = float(np.var(bw2.sum(axis=1).astype(np.float64)))
    return max(s1, s2)


def _estimate_angle_projection_profile(image_bgr: np.ndarray) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Robust for photos on textured backgrounds: scan rotation on a central crop
    and maximize row-wise projection variance (document text becomes 'horizontal').
    """
    if os.getenv("DESKEW_USE_PROJECTION", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return None, {"projection": "disabled"}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    frac = float(os.getenv("DESKEW_PROJ_CROP_FRAC", "0.78"))
    ch = max(80, int(h * frac))
    cw = max(80, int(w * frac))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    crop = gray[y0 : y0 + ch, x0 : x0 + cw]
    h2, w2 = crop.shape
    center = (w2 // 2, h2 // 2)

    a_lo = float(os.getenv("DESKEW_PROJ_ANGLE_MIN", "-55"))
    a_hi = float(os.getenv("DESKEW_PROJ_ANGLE_MAX", "55"))
    step = float(os.getenv("DESKEW_PROJ_STEP", "0.5"))

    best_sc = float("-inf")
    best_a = 0.0
    for ang in np.arange(a_lo, a_hi + step * 0.5, step, dtype=np.float64):
        matrix = cv2.getRotationMatrix2D(center, float(ang), 1.0)
        rot = cv2.warpAffine(
            crop,
            matrix,
            (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sc = _row_projection_score(rot)
        if sc > best_sc:
            best_sc = sc
            best_a = float(ang)

    # Fine search around the coarse peak
    for ang in np.arange(best_a - 3.0, best_a + 3.05, 0.1, dtype=np.float64):
        matrix = cv2.getRotationMatrix2D(center, float(ang), 1.0)
        rot = cv2.warpAffine(
            crop,
            matrix,
            (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sc = _row_projection_score(rot)
        if sc > best_sc:
            best_sc = sc
            best_a = float(ang)

    if not math.isfinite(best_sc):
        return None, {"projection_angle": None, "projection_score": None}

    return best_a, {
        "projection_angle": best_a,
        "projection_score": best_sc,
        "projection_crop_frac": frac,
    }


def _estimate_best_skew_angle(image_bgr: np.ndarray) -> Tuple[Optional[float], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    c_ang, c_meta = _estimate_angle_from_contours(image_bgr)
    meta.update(c_meta)

    t_ang = _estimate_text_like_angle(image_bgr)
    meta["text_angle"] = t_ang

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = _build_edge_maps(gray)[0]
    h_ang, h_meta = _estimate_angle_hough(edges)
    meta["hough"] = h_meta

    p_ang, p_meta = _estimate_angle_projection_profile(image_bgr)
    meta["projection"] = p_meta

    # Phone photos: contour/Hough often latch onto wood grain; projection on text rows is more reliable.
    primary = os.getenv("DESKEW_PRIMARY", "projection").strip().lower()
    if primary in {"projection", "proj", "profile"} and p_ang is not None:
        meta["picked"] = "projection_primary"
        return p_ang, meta

    tol = float(os.getenv("DESKEW_AGREE_TOL", "15.0"))
    disagree = float(os.getenv("DESKEW_CONTOUR_PROJ_DISAGREE", "18.0"))
    on_disagree = os.getenv("DESKEW_ON_DISAGREE_USE", "projection").strip().lower()

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    # Projection is strong on real phone photos (wood grain breaks pure contour methods).
    if p_ang is not None:
        if c_ang is None:
            meta["picked"] = "projection"
            return p_ang, meta
        if abs(c_ang - p_ang) <= disagree:
            pool = [c_ang, p_ang]
            if t_ang is not None and _close(t_ang, c_ang):
                pool.append(t_ang)
            if h_ang is not None and _close(h_ang, c_ang):
                pool.append(h_ang)
            meta["picked"] = "contour+projection+avg" if len(pool) > 2 else "contour+projection"
            meta["picked_angles"] = pool
            return float(np.mean(pool)), meta
        meta["picked"] = f"disagree_use_{on_disagree}"
        meta["contour_angle"] = c_ang
        meta["projection_angle_chosen"] = p_ang
        if on_disagree in {"projection", "proj"}:
            return p_ang, meta
        return c_ang, meta

    if c_ang is not None:
        pool = [c_ang]
        if t_ang is not None and _close(t_ang, c_ang):
            pool.append(t_ang)
        if h_ang is not None and _close(h_ang, c_ang):
            pool.append(h_ang)
        meta["picked"] = "contour+avg" if len(pool) > 1 else "contour"
        meta["picked_angles"] = pool
        return float(np.mean(pool)), meta

    if h_ang is not None:
        meta["picked"] = "hough"
        return h_ang, meta
    if t_ang is not None:
        meta["picked"] = "text"
        return t_ang, meta

    meta["picked"] = None
    return None, meta


def _estimate_text_like_angle(image_bgr: np.ndarray) -> float | None:
    """
    Fallback skew estimate from text-like foreground points.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(thresh > 0)
    if xs.size < 200:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    (_, _), (w, h), raw_angle = cv2.minAreaRect(points)
    return _normalize_min_area_rect_angle(w, h, raw_angle)


def _coarse_rotate_90(image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Coarse orientation correction in 90-degree steps.
    We pick orientation with the widest text-line projection profile.
    """
    candidates: List[Tuple[np.ndarray, float]] = [
        (image_bgr, 0.0),
        (cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), 90.0),
        (cv2.rotate(image_bgr, cv2.ROTATE_180), 180.0),
        (cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), -90.0),
    ]

    best_img = image_bgr
    best_angle = 0.0
    best_score = float("-inf")

    for candidate_img, candidate_angle in candidates:
        gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        # Text strokes become bright in the binary map.
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_sums = bw.sum(axis=1).astype(np.float32)
        if row_sums.size == 0:
            continue

        # Prefer orientation with stronger row-wise variation (horizontal text lines).
        score = float(np.std(row_sums))
        if score > best_score:
            best_score = score
            best_img = candidate_img
            best_angle = candidate_angle

    return best_img, best_angle


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


def _angle_from_field_bboxes_hull(bboxes: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Fallback: minAreaRect over all box corners. Often ~0 when detector returns axis-aligned
    boxes in image coordinates — use together with row-based method.
    """
    meta: Dict[str, Any] = {"method": "hull_corners"}
    if not isinstance(bboxes, list) or len(bboxes) < 2:
        return None, meta

    corners: List[Tuple[float, float]] = []
    for entry in bboxes:
        bb = _to_bbox(entry)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        corners.extend(
            [
                (float(x1), float(y1)),
                (float(x2), float(y1)),
                (float(x2), float(y2)),
                (float(x1), float(y2)),
            ]
        )

    if len(corners) < 8:
        return None, {**meta, "reason": "few_corners"}

    pts = np.array(corners, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), raw = rect
    ang = _normalize_min_area_rect_angle(rw, rh, raw)
    meta["field_boxes_used"] = len(bboxes)
    meta["fields_corner_count"] = len(corners)
    meta["hull_angle"] = ang
    return ang, meta


def _angle_from_field_rows(bboxes: Any, img_h: int, img_w: int) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Primary for TextFieldsDetector: boxes are axis-aligned in image space, but *rows* of fields
    follow the document baseline — angle of (left→right) center vectors ≈ in-plane skew.
    """
    meta: Dict[str, Any] = {"method": "row_center_vectors"}
    if not isinstance(bboxes, list) or len(bboxes) < 2:
        return None, meta

    centers_heights: List[Tuple[float, float, float]] = []
    for entry in bboxes:
        bb = _to_bbox(entry)
        if bb is None:
            continue
        left, top, right, bottom = bb
        w_box = right - left
        h_box = bottom - top
        if w_box < 8 or h_box < 6:
            continue
        centers_heights.append(((left + right) / 2.0, (top + bottom) / 2.0, float(h_box)))

    if len(centers_heights) < 2:
        return None, {**meta, "reason": "few_boxes", "n": len(centers_heights)}

    heights = [t[2] for t in centers_heights]
    med_h = float(np.median(heights)) if heights else 24.0
    row_thresh = max(
        10.0,
        float(os.getenv("DESKEW_FIELDS_ROW_THRESH_FRAC", "0.38")) * med_h,
        0.018 * float(max(img_h, img_w)),
    )
    meta["row_thresh_px"] = row_thresh

    centers_heights.sort(key=lambda t: t[1])
    rows_pts: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = [(centers_heights[0][0], centers_heights[0][1])]
    for i in range(1, len(centers_heights)):
        _, cy, _ = centers_heights[i]
        _, py, _ = centers_heights[i - 1]
        if abs(cy - py) <= row_thresh:
            cur.append((centers_heights[i][0], centers_heights[i][1]))
        else:
            rows_pts.append(cur)
            cur = [(centers_heights[i][0], centers_heights[i][1])]
    rows_pts.append(cur)

    row_angles: List[float] = []
    for row in rows_pts:
        if len(row) < 2:
            continue
        row.sort(key=lambda p: p[0])
        x0, y0 = row[0]
        x1, y1 = row[-1]
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 10.0:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90.0:
            ang -= 180.0
        elif ang < -90.0:
            ang += 180.0
        row_angles.append(float(ang))

    if not row_angles:
        return None, {**meta, "reason": "no_row_vectors", "rows": len(rows_pts)}

    ang_med = float(np.median(row_angles))
    meta["row_angles_sample"] = row_angles[:12]
    meta["row_angle_median"] = ang_med
    meta["row_groups"] = len(rows_pts)
    return ang_med, meta


def _angle_from_field_bboxes_combined(bboxes: Any, img_h: int, img_w: int) -> Tuple[Optional[float], Dict[str, Any]]:
    row_ang, row_meta = _angle_from_field_rows(bboxes, img_h, img_w)
    hull_ang, hull_meta = _angle_from_field_bboxes_hull(bboxes)
    meta: Dict[str, Any] = {"rows": row_meta, "hull": hull_meta}

    agree = float(os.getenv("DESKEW_FIELDS_ROW_HULL_AGREE", "14.0"))

    if row_ang is not None and hull_ang is not None and abs(row_ang - hull_ang) <= agree:
        out = float(np.mean([row_ang, hull_ang]))
        meta["merged"] = "row+hull_mean"
        meta["final_angle"] = out
        return out, meta
    if row_ang is not None:
        meta["merged"] = "row_only"
        meta["final_angle"] = row_ang
        return row_ang, meta
    if hull_ang is not None:
        meta["merged"] = "hull_only"
        meta["final_angle"] = hull_ang
        return hull_ang, meta
    meta["merged"] = "none"
    return None, meta


def _angle_from_doc_detector(meta_results: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Tilt from DocDetector segmentation polygon (minAreaRect), in Angle90 image space."""
    meta: Dict[str, Any] = {"method": "doc_segm_minAreaRect"}
    if not isinstance(meta_results, dict):
        return None, {**meta, "reason": "bad_meta"}

    dd = meta_results.get("DocDetector")
    if not isinstance(dd, dict):
        return None, {**meta, "reason": "no_doc_detector"}

    segm = dd.get("segm")
    if segm is None:
        return None, {**meta, "reason": "no_segm"}

    best_area = 0.0
    best_ang: Optional[float] = None

    try:
        for seg in segm:
            cnt = np.asarray(seg, dtype=np.float32)
            if cnt.size < 8:
                continue
            cnt = cnt.reshape(-1, 2)
            if cnt.shape[0] < 4:
                continue
            area = float(cv2.contourArea(cnt))
            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), raw = rect
            ang = _normalize_min_area_rect_angle(rw, rh, raw)
            if ang is None:
                continue
            if area > best_area:
                best_area = area
                best_ang = float(ang)
    except Exception as exc:  # pragma: no cover
        return None, {**meta, "error": str(exc)}

    if best_ang is None:
        return None, {**meta, "reason": "no_valid_contour"}

    meta["doc_contour_area"] = best_area
    meta["doc_angle"] = best_ang
    return best_ang, meta


def _rotate_bgr(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rusdocs_resize_rgb(rgb: np.ndarray, img_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Same scaling as RussianDocsOCR.Pipeline._prepare_image (RGB ndarray)."""
    h, w = rgb.shape[:2]
    ratio = max(max(h, w) / float(img_size), 1.0)
    new_h, new_w = int(h // ratio), int(w // ratio)
    resized = cv2.resize(rgb, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, {"resize_ratio": ratio, "net_shape": [new_h, new_w]}


def _normalize_bbox_list(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, list):
        return raw
    return None


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _aabb_from_doc_detector_payload(dd: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """Axis-aligned bounds from DocDetector segm polygons or raw bbox array (coords in net image space)."""
    segm = dd.get("segm")
    xs: List[float] = []
    ys: List[float] = []
    if isinstance(segm, (list, tuple)):
        for seg in segm:
            try:
                cnt = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                if cnt.shape[0] < 3:
                    continue
                xs.extend(cnt[:, 0].tolist())
                ys.extend(cnt[:, 1].tolist())
            except Exception:
                continue
    if len(xs) >= 3 and len(ys) >= 3:
        return int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys)))

    bbox = dd.get("bbox")
    if bbox is None:
        return None
    b = np.asarray(bbox, dtype=np.float64)
    if b.size == 0:
        return None
    if b.ndim == 2 and b.shape[1] >= 4:
        return (
            int(np.floor(b[:, 0].min())),
            int(np.floor(b[:, 1].min())),
            int(np.ceil(b[:, 2].max())),
            int(np.ceil(b[:, 3].max())),
        )
    if b.ndim == 1 and b.size >= 4:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        return (
            int(np.floor(min(x0, x1))),
            int(np.floor(min(y0, y1))),
            int(np.ceil(max(x0, x1))),
            int(np.ceil(max(y0, y1))),
        )
    return None


def _crop_rect_from_russian_doc_detector(
    image_bgr: np.ndarray,
    img_size: int,
    margin_frac: float,
    min_doc_frac: float,
    max_doc_frac: float,
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    """Map DocDetector AABB from net-sized RGB to full-resolution BGR crop (x1,y1,x2,y2 exclusive end)."""
    meta: Dict[str, Any] = {}
    h_f, w_f = image_bgr.shape[:2]
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        small, rmeta = _rusdocs_resize_rgb(rgb, img_size)
        h_n, w_n = small.shape[:2]
        meta["resize"] = rmeta
        pipeline = _get_pipeline()
        with _PIPELINE_LOCK:
            doc_pack = pipeline.doc_detector.predict(small)
        dd = doc_pack.get("DocDetector", {}) if isinstance(doc_pack, dict) else {}
        segm = dd.get("segm")
        meta["segm_count"] = len(segm) if isinstance(segm, (list, tuple)) else 0
        rect_net = _aabb_from_doc_detector_payload(dd if isinstance(dd, dict) else {})
    except Exception as exc:  # pragma: no cover
        meta["error"] = str(exc)
        return None, meta

    if rect_net is None:
        meta["reason"] = "no_aabb"
        return None, meta

    x1n, y1n, x2n, y2n = rect_net
    net_area = float(h_n * w_n)
    doc_area = float(max(0, x2n - x1n) * max(0, y2n - y1n))
    frac = doc_area / net_area if net_area > 0 else 0.0
    meta["doc_area_frac_net"] = round(frac, 4)
    if frac < min_doc_frac or frac > max_doc_frac:
        meta["reason"] = "area_out_of_range"
        return None, meta

    sx = w_f / float(w_n)
    sy = h_f / float(h_n)
    xf1 = int(np.floor(x1n * sx))
    xf2 = int(np.ceil(x2n * sx))
    yf1 = int(np.floor(y1n * sy))
    yf2 = int(np.ceil(y2n * sy))

    mx = max(4, int(w_f * margin_frac))
    my = max(4, int(h_f * margin_frac))
    xf1 = max(0, xf1 - mx)
    yf1 = max(0, yf1 - my)
    xf2 = min(w_f, xf2 + mx)
    yf2 = min(h_f, yf2 + my)

    if xf2 - xf1 < 32 or yf2 - yf1 < 32:
        meta["reason"] = "too_small_roi"
        return None, meta

    crop_frac = (xf2 - xf1) * (yf2 - yf1) / float(w_f * h_f)
    meta["crop_frac_full"] = round(crop_frac, 4)
    if crop_frac > float(os.getenv("PREP_CROP_MAX_CROP_FRAC", "0.985")):
        meta["reason"] = "crop_almost_full"
        return None, meta

    return (xf1, yf1, xf2, yf2), meta


def _crop_rect_from_contours(
    image_bgr: np.ndarray,
    margin_frac: float,
    min_doc_frac: float,
    max_doc_frac: float,
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    """Largest plausible outer contour → bounding rect (full image coords)."""
    h_f, w_f = image_bgr.shape[:2]
    image_area = float(h_f * w_f)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    best: Optional[Tuple[int, int, int, int]] = None
    best_area = 0.0
    for edges in _build_edge_maps(gray):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:22]:
            area = float(cv2.contourArea(c))
            if area < image_area * min_doc_frac or area > image_area * max_doc_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 30 or bh < 30:
                continue
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(x + bw), int(y + bh))

    meta = {"best_contour_area_frac": round(best_area / image_area, 4) if image_area else 0.0}
    if best is None:
        meta["reason"] = "no_contour"
        return None, meta

    x1, y1, x2, y2 = best
    mx = max(4, int(w_f * margin_frac))
    my = max(4, int(h_f * margin_frac))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w_f, x2 + mx)
    y2 = min(h_f, y2 + my)
    crop_frac = (x2 - x1) * (y2 - y1) / image_area
    meta["crop_frac_full"] = round(crop_frac, 4)
    if crop_frac > float(os.getenv("PREP_CROP_MAX_CROP_FRAC", "0.985")):
        meta["reason"] = "crop_almost_full"
        return None, meta

    return (x1, y1, x2, y2), meta


def _auto_crop_document_bgr(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tight crop so the document fills more of the frame (similar to manual crop).
    Tries RussianDocs DocDetector first, then OpenCV contours if method is auto.
    """
    method = os.getenv("PREP_CROP_METHOD", "auto").strip().lower()
    img_size = int(os.getenv("RUS_DOCS_IMG_SIZE", "1500"))
    margin_frac = float(os.getenv("PREP_CROP_MARGIN_FRAC", "0.035"))
    min_doc_frac = float(os.getenv("PREP_CROP_MIN_DOC_FRAC", "0.08"))
    max_doc_frac = float(os.getenv("PREP_CROP_MAX_DOC_FRAC", "0.93"))

    meta: Dict[str, Any] = {"enabled": True, "method_requested": method}

    if method in ("auto", "russian", "russian_doc", "doc"):
        rect, rmeta = _crop_rect_from_russian_doc_detector(
            image_bgr, img_size, margin_frac, min_doc_frac, max_doc_frac
        )
        meta["russian_doc"] = rmeta
        if rect is not None:
            x1, y1, x2, y2 = rect
            cropped = image_bgr[y1:y2, x1:x2]
            meta.update(
                {
                    "method": "russian_doc",
                    "roi_xyxy": [x1, y1, x2, y2],
                    "original_shape": [image_bgr.shape[0], image_bgr.shape[1]],
                    "cropped_shape": [cropped.shape[0], cropped.shape[1]],
                }
            )
            return cropped, meta

    if method in ("auto", "contour", "opencv"):
        rect, cmeta = _crop_rect_from_contours(image_bgr, margin_frac, min_doc_frac, max_doc_frac)
        meta["contour"] = cmeta
        if rect is not None:
            x1, y1, x2, y2 = rect
            cropped = image_bgr[y1:y2, x1:x2]
            meta.update(
                {
                    "method": "contour",
                    "roi_xyxy": [x1, y1, x2, y2],
                    "original_shape": [image_bgr.shape[0], image_bgr.shape[1]],
                    "cropped_shape": [cropped.shape[0], cropped.shape[1]],
                }
            )
            return cropped, meta

    meta["skipped_reason"] = "no_detection"
    meta["method"] = "none"
    return image_bgr, meta


def _probe_skew_from_russian_docs_fields(rgb: np.ndarray) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Fine skew from RussianDocsOCR **without** Pipeline.__call__.

    RussianDocsOCR.Pipeline stops after DocType when doctype == 'NONE', so DocDetector /
    TextFieldsDetector never run — the old probe often returned nothing on difficult photos.

    We call Angle90 → DocDetector → TextFieldsDetector on the shared Pipeline instance.
    Geometry matches the unwarped branch: fields on the Angle90 image (no perspective warp),
    same as find_text_fields with get_doc_borders=False inside the library.
    """
    if os.getenv("DESKEW_USE_RUS_DOCS_FIELDS", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return None, {"russian_docs_fields": "disabled"}

    pipeline = _get_pipeline()
    img_size = int(os.getenv("RUS_DOCS_IMG_SIZE", "1500"))
    mode = os.getenv("DESKEW_RUSDOC_PROBE_MODE", "full").strip().lower()
    # full | doc_only | fields_only

    meta_out: Dict[str, Any] = {
        "probe_mode": mode,
        "probe_backend": "direct_modules",
        "note": "bypasses Pipeline NONE early-exit",
    }
    ang_doc: Optional[float] = None
    ang_f: Optional[float] = None

    try:
        small, rmeta = _rusdocs_resize_rgb(rgb, img_size)
        meta_out["resize"] = rmeta
        h_net, w_net = small.shape[:2]

        with _PIPELINE_LOCK:
            # Default skip: probe skew in the same frame as OpenCV-aligned image (see process_document_image).
            if os.getenv("DESKEW_RUS_SKIP_ANGLE90", "1").strip().lower() in {"1", "true", "yes", "on"}:
                after90 = small
                meta_out["angle90"] = {"skipped": True}
            else:
                angle_pack = pipeline.angle90.predict_transform(small)
                a90 = angle_pack.get("Angle90", {}) if isinstance(angle_pack, dict) else {}
                after90 = a90.get("warped_img", small)
                meta_out["angle90"] = {
                    "angle": a90.get("angle"),
                    "confidence": a90.get("confidence"),
                }

            if mode in ("full", "doc_only"):
                doc_pack = pipeline.doc_detector.predict_transform(after90)
                dd = doc_pack.get("DocDetector", {}) if isinstance(doc_pack, dict) else {}
                segm = dd.get("segm")
                meta_out["doc"] = {
                    "segm_count": len(segm) if isinstance(segm, (list, tuple)) else 0,
                    "had_perspective_fix": bool(segm),
                }
                ang_doc, doc_meta = _angle_from_doc_detector({"DocDetector": dd})
                meta_out["doc"].update(doc_meta)

            if mode in ("full", "fields_only"):
                fields_pack = pipeline.text_fields.predict_transform(after90)
                tf = fields_pack.get("TextFieldsDetector", {}) if isinstance(fields_pack, dict) else {}
                bboxes = _normalize_bbox_list(tf.get("bbox") if isinstance(tf, dict) else None)
                meta_out["fields"] = {"bbox_count": len(bboxes) if bboxes else 0}
                if bboxes and len(bboxes) >= 2:
                    min_boxes = int(os.getenv("DESKEW_FIELDS_FORCE_MIN_BOXES", "2"))
                    if len(bboxes) >= min_boxes:
                        ang_f, fm = _angle_from_field_bboxes_combined(bboxes, h_net, w_net)
                        meta_out["fields"].update(fm)
                    else:
                        meta_out["fields"]["reason"] = "too_few_boxes"
                else:
                    meta_out["fields"]["reason"] = "no_bboxes"

    except Exception as exc:  # pragma: no cover
        meta_out["error"] = str(exc)
        return None, meta_out

    merge_tol = float(os.getenv("DESKEW_DOC_FIELDS_MERGE_TOL", "22.0"))

    if ang_doc is not None and ang_f is not None:
        meta_out["angles"] = {"doc": ang_doc, "fields": ang_f}
        if abs(ang_doc - ang_f) <= merge_tol:
            ang_final = float(np.mean([ang_doc, ang_f]))
            meta_out["picked"] = "doc+fields_mean"
        else:
            use_on_disagree = os.getenv("DESKEW_DOC_FIELDS_ON_DISAGREE", "doc").strip().lower()
            if use_on_disagree in {"fields", "field"}:
                ang_final = ang_f
                meta_out["picked"] = "fields_only_disagree"
            else:
                ang_final = ang_doc
                meta_out["picked"] = "doc_only_disagree"
    elif ang_doc is not None:
        ang_final = ang_doc
        meta_out["picked"] = "doc_only"
    elif ang_f is not None:
        ang_final = ang_f
        meta_out["picked"] = "fields_only"
    else:
        meta_out["reason"] = "no_angle_from_russian_docs"
        return None, meta_out

    if os.getenv("DESKEW_RUS_INVERT_ANGLE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        ang_final = -float(ang_final)
        meta_out["inverted"] = True

    min_a = float(os.getenv("DESKEW_FIELDS_MIN_ANGLE", "0.4"))
    max_a = float(os.getenv("DESKEW_FIELDS_MAX_ANGLE", "50.0"))
    if abs(ang_final) < min_a or abs(ang_final) > max_a:
        meta_out["reason"] = "angle_out_of_range"
        meta_out["final_angle"] = ang_final
        meta_out["min"] = min_a
        meta_out["max"] = max_a
        return None, meta_out

    meta_out["final_angle"] = ang_final
    return float(ang_final), meta_out


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
    # Optional: zoom in on the document (like a manual crop) so it occupies more of the frame.
    if _truthy_env("PREP_AUTO_DOC_CROP", "0"):
        work_bgr, prep_meta = _auto_crop_document_bgr(original_bgr)
    else:
        work_bgr, prep_meta = original_bgr, {"enabled": False}

    # Coarse 90° can confuse skew on diagonal real-world photos; enable if scans are often upside-down.
    use_coarse = os.getenv("DESKEW_COARSE_90", "0").strip().lower() in {"1", "true", "yes", "on"}
    if use_coarse:
        coarse_bgr, coarse_angle = _coarse_rotate_90(work_bgr)
    else:
        coarse_bgr, coarse_angle = work_bgr, 0.0
    aligned_bgr, fine_angle, align_debug = _deskew_image(coarse_bgr)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    fields_angle = 0.0
    fields_align_meta: Dict[str, Any] = {}
    f_ang, f_meta = _probe_skew_from_russian_docs_fields(aligned_rgb)
    fields_align_meta = f_meta

    # If the probe ran RussianDocs Angle90, doc/field geometry is in that rotated frame — mirror
    # the same 0/90/180/270 CCW steps on our canvas before applying the fine fields_angle.
    angle90_applied_deg = 0.0
    a90_info = f_meta.get("angle90") or {}
    if not a90_info.get("skipped") and isinstance(a90_info.get("angle"), (int, float)):
        raw_a = float(a90_info["angle"])
        steps = max(0, min(int(raw_a) // 90, 4))
        for _ in range(steps):
            aligned_bgr = cv2.rotate(aligned_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        angle90_applied_deg = raw_a if steps else 0.0
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    if f_ang is not None:
        fields_angle = f_ang
        aligned_bgr = _rotate_bgr(aligned_bgr, fields_angle)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    total_angle = coarse_angle + fine_angle + angle90_applied_deg + fields_angle
    # Normalize to [-180, 180] for easier interpretation.
    if total_angle > 180:
        total_angle -= 360
    elif total_angle < -180:
        total_angle += 360

    pipeline = _get_pipeline()
    docconf = float(os.getenv("RUS_DOCS_DOCCONF", "0.5"))
    img_size = int(os.getenv("RUS_DOCS_IMG_SIZE", "1500"))

    with _PIPELINE_LOCK:
        results = pipeline(
            img_path=aligned_rgb,
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
        best_rgb = aligned_rgb
    best_bgr = cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR)

    annotated = _annotate_image(best_bgr, detections)
    annotated_b64 = _encode_image_to_base64(annotated)

    out: Dict[str, Any] = {
        "angle": float(total_angle),
        "structured_fields": structured,
        "raw_lines": raw_lines,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "document_type": results.doctype,
        "quality": results.quality,
    }
    out["alignment"] = {
        "coarse_angle": float(coarse_angle),
        "fine_angle": float(fine_angle),
        "russian_probe_angle90": float(angle90_applied_deg),
        "fields_angle": float(fields_angle),
        "fields_probe": fields_align_meta,
        "auto_crop": prep_meta,
        "coarse_enabled": use_coarse,
        "annotated_canvas": (
            "russian_docs_warped"
            if results.img_with_fixed_perspective is not None or results.rotated_image is not None
            else "aligned_input"
        ),
        **align_debug,
    }
    # Final annotated image may use RussianDocsOCR dewarp (different geometry than our deskew).
    if os.getenv("DESKEW_PREVIEW", "0").strip().lower() in {"1", "true", "yes", "on"}:
        out["aligned_preview_base64"] = _encode_image_to_base64(aligned_bgr)
    return out
