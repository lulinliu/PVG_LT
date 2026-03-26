from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .schema import BBox


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def save_binary_mask(mask: np.ndarray, path: str | Path) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def crop_array(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    return arr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]


def paste_mask(canvas_shape: tuple[int, int], bbox: BBox, crop_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] = crop_mask.astype(np.uint8)
    return canvas


def bbox_from_mask(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def dilate(mask: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)


def erode(mask: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)


def close_open(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


def find_components(binary_mask: np.ndarray) -> list[tuple[np.ndarray, BBox, int]]:
    mask_u8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components: list[tuple[np.ndarray, BBox, int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue
        component = (labels == label).astype(np.uint8)
        components.append((component, BBox(int(x), int(y), int(x + w), int(y + h)), int(area)))
    return components


def strict_json_load(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            payload, end_idx = decoder.raw_decode(stripped[start_idx:])
        except json.JSONDecodeError:
            continue
        trailing = stripped[start_idx + end_idx :].strip()
        if trailing.startswith("```"):
            trailing_lines = trailing.splitlines()
            if trailing_lines and trailing_lines[0].startswith("```"):
                trailing = "\n".join(trailing_lines[1:]).strip()
        if trailing:
            continue
        return payload
    raise json.JSONDecodeError("Unable to locate a valid JSON object in model output", stripped, 0)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def mask_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    inner_bool = inner > 0
    inner_area = float(inner_bool.sum())
    if inner_area <= 0:
        return 0.0
    overlap = float(np.logical_and(inner_bool, outer > 0).sum())
    return overlap / inner_area


def mask_compactness(mask: np.ndarray) -> float:
    bbox = bbox_from_mask(mask)
    if bbox is None or bbox.area() <= 0:
        return 0.0
    return float((mask > 0).sum()) / float(bbox.area())


def connected_component_count(mask: np.ndarray) -> int:
    count, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return max(0, count - 1)


def distance_to_mask(mask: np.ndarray, x: float, y: float) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("inf")
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    return float(np.sqrt(dx * dx + dy * dy).min())


def infer_scene_category(source_scene: Optional[str], scene_root: Path) -> str:
    if source_scene:
        parts = Path(source_scene).parts
        if len(parts) >= 2:
            return parts[-2]
    if scene_root.parent.name not in {"data_longtail", "data"}:
        return scene_root.parent.name
    return "unknown"
