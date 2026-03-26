from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import DINOv3Config, ROIConfig
from .schema import BBox, CoreMaskSelection, PromptCluster
from .utils import bbox_from_mask, clamp01, crop_array, find_components, normalize_map


@dataclass
class ROIResult:
    roi_box: BBox
    local_core_mask: np.ndarray


class DINOv3FeatureMiner:
    def __init__(self, dino_config: DINOv3Config, roi_config: ROIConfig) -> None:
        self.dino_config = dino_config
        self.roi_config = roi_config
        self._torch = None
        self._processor = None
        self._model = None
        self._patch_size = 16
        self._register_tokens = 4
        self._load_or_raise()

    def _load_or_raise(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from torchvision.transforms import v2
        except Exception as exc:
            raise RuntimeError(
                "DINOv3 requires torch, torchvision, and the DINOv3 torch.hub entrypoint."
            ) from exc

        device = self.dino_config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = self._resolve_weights_path()
        try:
            self._model = torch.hub.load(
                self.dino_config.repo_or_dir,
                self.dino_config.model_entrypoint,
                weights=str(weights_path),
                trust_repo=True,
            ).to(device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv3 from torch.hub with "
                f"repo={self.dino_config.repo_or_dir}, entrypoint={self.dino_config.model_entrypoint}, "
                f"weights={weights_path}."
            ) from exc
        self._processor = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._torch = torch
        patch_size = getattr(self._model, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self._patch_size = int(patch_size or 16)
        self._register_tokens = 0

    def _resolve_weights_path(self) -> Path:
        weights_path = Path(self.dino_config.weights_path).expanduser()
        if weights_path.is_file():
            return weights_path
        raise RuntimeError(
            f"DINOv3 weights file not found at {weights_path}. Download the Meta checkpoint there before running tail_miner."
        )

    def build_roi(self, image_shape: tuple[int, int, int], core_selection: CoreMaskSelection) -> ROIResult:
        height, width = image_shape[:2]
        core_box = core_selection.core_box
        cx, cy = core_box.center()
        half_w = max(core_box.width() * self.roi_config.expansion_scale / 2.0, float(self.roi_config.center_allowance_px))
        half_h = max(core_box.height() * self.roi_config.expansion_scale / 2.0, float(self.roi_config.center_allowance_px))
        roi_box = BBox(
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        ).clipped(width, height)
        return ROIResult(roi_box=roi_box, local_core_mask=crop_array(core_selection.mask, roi_box).astype(np.uint8))

    def mine_clusters(self, frame_rgb: np.ndarray, roi_result: ROIResult) -> list[PromptCluster]:
        roi_rgb = crop_array(frame_rgb, roi_result.roi_box)
        image_height, image_width = frame_rgb.shape[:2]
        resized_rgb, scale_x, scale_y = self._resize_roi(roi_rgb)
        patch_features = self._extract_patch_features(resized_rgb)
        patch_h, patch_w, _ = patch_features.shape
        if patch_h == 0 or patch_w == 0:
            return []
        core_patch_mask = self._mask_to_patch_grid(roi_result.local_core_mask, patch_w, patch_h)
        if core_patch_mask.sum() == 0:
            return []
        prototype = self._build_prototype(patch_features, core_patch_mask)
        similarity = np.tensordot(patch_features, prototype, axes=([2], [0]))
        retained = self._threshold_similarity(similarity)
        if retained.sum() == 0:
            return []
        clusters = self._components_to_clusters(
            retained,
            similarity,
            core_patch_mask,
            roi_result.roi_box,
            scale_x,
            scale_y,
            image_width,
            image_height,
        )
        clusters.sort(key=lambda item: (item.average_similarity, item.max_similarity), reverse=True)
        return clusters[: self.roi_config.max_clusters]

    def _resize_roi(self, roi_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = roi_rgb.shape[:2]
        max_side = max(height, width)
        min_side = min(height, width)
        scale = 1.0
        if max_side > self.dino_config.target_max_side:
            scale = self.dino_config.target_max_side / max_side
        elif min_side < self.dino_config.target_min_side:
            scale = self.dino_config.target_min_side / max(min_side, 1)

        resized_w = max(self._patch_size, int(round(width * scale)))
        resized_h = max(self._patch_size, int(round(height * scale)))
        resized_w = max(self._patch_size, (resized_w // self._patch_size) * self._patch_size)
        resized_h = max(self._patch_size, (resized_h // self._patch_size) * self._patch_size)
        resized = cv2.resize(roi_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        scale_x = width / resized_w
        scale_y = height / resized_h
        return resized, scale_x, scale_y

    def _extract_patch_features(self, resized_rgb: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(resized_rgb)
        model_device = next(self._model.parameters()).device
        pixel_values = self._processor(pil_image).unsqueeze(0).to(model_device)
        with self._torch.no_grad():
            outputs = self._model.forward_features(pixel_values)
        patch_tokens = outputs["x_norm_patchtokens"]
        patch_h = resized_rgb.shape[0] // self._patch_size
        patch_w = resized_rgb.shape[1] // self._patch_size
        patch_tokens = patch_tokens.reshape(1, patch_h, patch_w, patch_tokens.shape[-1])[0]
        features = patch_tokens.float().cpu().numpy()
        norm = np.linalg.norm(features, axis=2, keepdims=True)
        return features / np.clip(norm, 1e-6, None)

    def _mask_to_patch_grid(self, mask: np.ndarray, patch_w: int, patch_h: int) -> np.ndarray:
        resized = cv2.resize(mask.astype(np.uint8), (patch_w, patch_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def _build_prototype(self, patch_features: np.ndarray, core_patch_mask: np.ndarray) -> np.ndarray:
        selected = patch_features[core_patch_mask > 0]
        prototype = selected.mean(axis=0)
        return prototype / np.clip(np.linalg.norm(prototype), 1e-6, None)

    def _threshold_similarity(self, similarity: np.ndarray) -> np.ndarray:
        keep = similarity > self.roi_config.similarity_threshold
        if keep.sum() == 0:
            return np.zeros_like(similarity, dtype=np.uint8)
        max_keep = max(1, int(np.ceil(similarity.size * self.roi_config.similarity_top_percent)))
        values = similarity[keep]
        if values.size > max_keep:
            order = np.sort(values)
            cutoff = float(order[-max_keep])
            keep = np.logical_and(keep, similarity >= cutoff)
        return keep.astype(np.uint8)

    def _components_to_clusters(
        self,
        retained: np.ndarray,
        similarity: np.ndarray,
        core_patch_mask: np.ndarray,
        roi_box: BBox,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> list[PromptCluster]:
        components = []
        for component_mask, component_box, area in find_components(retained):
            if area < self.roi_config.min_cluster_patches:
                continue
            coverage = float((component_mask > 0)[core_patch_mask > 0].sum()) / max(float((component_mask > 0).sum()), 1.0)
            if coverage > self.roi_config.redundant_coverage_ratio:
                continue
            components.append((component_mask, component_box, area))
        merged = self._merge_components(components)
        clusters: list[PromptCluster] = []
        for component_mask, component_box in merged:
            ys, xs = np.where(component_mask > 0)
            if len(xs) == 0:
                continue
            avg_sim = float(similarity[component_mask > 0].mean())
            max_idx = int(np.argmax(similarity[component_mask > 0]))
            max_y = int(ys[max_idx])
            max_x = int(xs[max_idx])
            patch_box = component_box
            pixel_box = self._patch_box_to_image_box(
                patch_box,
                roi_box,
                scale_x,
                scale_y,
                image_width,
                image_height,
            )
            prompt_box = pixel_box.expand_by_ratio(
                self.roi_config.second_round_box_expand_ratio,
                image_width,
                image_height,
            ).clipped(image_width, image_height)
            prompt_point = self._patch_center_to_image_point(
                max_x,
                max_y,
                roi_box,
                scale_x,
                scale_y,
                image_width,
                image_height,
            )
            clusters.append(
                PromptCluster(
                    patch_mask=component_mask,
                    patch_box=patch_box,
                    prompt_box=prompt_box,
                    prompt_point=prompt_point,
                    average_similarity=clamp01(avg_sim),
                    max_similarity=clamp01(float(similarity[max_y, max_x])),
                )
            )
        return clusters

    def _merge_components(
        self,
        components: list[tuple[np.ndarray, BBox, int]],
    ) -> list[tuple[np.ndarray, BBox]]:
        merged: list[tuple[np.ndarray, BBox]] = []
        for component_mask, component_box, _ in components:
            merged_into_existing = False
            for idx, (existing_mask, existing_box) in enumerate(merged):
                iou = self._patch_mask_iou(existing_mask, component_mask)
                if iou > self.roi_config.cluster_merge_iou:
                    union_mask = np.logical_or(existing_mask > 0, component_mask > 0).astype(np.uint8)
                    union_box = bbox_from_mask(union_mask) or existing_box.union(component_box)
                    merged[idx] = (union_mask, union_box)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged.append((component_mask.astype(np.uint8), component_box))
        return merged

    def _patch_mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
        if inter <= 0:
            return 0.0
        union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
        return inter / max(union, 1.0)

    def _patch_box_to_image_box(
        self,
        patch_box: BBox,
        roi_box: BBox,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> BBox:
        x1 = roi_box.x1 + int(round(patch_box.x1 * self._patch_size * scale_x))
        y1 = roi_box.y1 + int(round(patch_box.y1 * self._patch_size * scale_y))
        x2 = roi_box.x1 + int(round(patch_box.x2 * self._patch_size * scale_x))
        y2 = roi_box.y1 + int(round(patch_box.y2 * self._patch_size * scale_y))
        return BBox(x1, y1, x2, y2).clipped(image_width, image_height)

    def _patch_center_to_image_point(
        self,
        patch_x: int,
        patch_y: int,
        roi_box: BBox,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        x = roi_box.x1 + int(round((patch_x + 0.5) * self._patch_size * scale_x))
        y = roi_box.y1 + int(round((patch_y + 0.5) * self._patch_size * scale_y))
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        return (x, y)
