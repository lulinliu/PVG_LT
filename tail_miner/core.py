from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .config import LocalSam3Config
from .schema import BBox, CandidateProposal, CoreMaskSelection, PromptCluster
from .utils import (
    bbox_from_mask,
    clamp01,
    close_open,
    connected_component_count,
    crop_array,
    mask_compactness,
    paste_mask,
)


class Sam3Segmenter:
    def __init__(self, config: LocalSam3Config) -> None:
        self.config = config
        self._torch = None
        self._predictor = None
        self._processor = None
        self._load_or_raise()

    def _load_or_raise(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch

            sam_root = Path(self.config.root)
            if str(sam_root) not in sys.path:
                sys.path.insert(0, str(sam_root))
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError("Local SAM3 could not be imported. Check the local checkout and its dependencies.") from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(
            checkpoint_path=self.config.checkpoint,
            load_from_HF=False,
            device=device,
            enable_inst_interactivity=True,
            eval_mode=True,
        )
        processor = Sam3Processor(model)
        predictor = model.inst_interactive_predictor
        if predictor is None:
            raise RuntimeError("SAM3 interactive predictor is unavailable.")
        self._torch = torch
        self._processor = processor
        self._predictor = predictor

    def select_primary_mask(self, frame_rgb: np.ndarray, candidate: CandidateProposal) -> Optional[CoreMaskSelection]:
        masks, scores = self._predict_text_prompt(
            frame_rgb,
            candidate.text_prompt,
        )
        return self._select_best_mask(
            masks=masks,
            sam_scores=scores,
            image_shape=frame_rgb.shape[:2],
            max_area_ratio=self.config.primary_mask_max_area_ratio,
            min_area_ratio=self.config.primary_mask_min_area_ratio,
        )

    def segment_cluster(self, frame_rgb: np.ndarray, cluster: PromptCluster) -> Optional[np.ndarray]:
        crop = crop_array(frame_rgb, cluster.prompt_box)
        local_point = self._clamp_point(
            crop.shape[:2],
            int(cluster.prompt_point[0] - cluster.prompt_box.x1),
            int(cluster.prompt_point[1] - cluster.prompt_box.y1),
        )
        local_box = BBox(0, 0, cluster.prompt_box.width(), cluster.prompt_box.height())
        masks = self._predict_local_crop(crop, local_box, local_point)
        selection = self._select_best_local_mask(
            masks=masks,
            center_point=local_point,
            local_box=local_box,
            max_area_ratio=self.config.secondary_mask_max_area_ratio,
        )
        if selection is None:
            return None
        return paste_mask(frame_rgb.shape[:2], cluster.prompt_box, selection.mask)

    def _predict_text_prompt(
        self,
        frame_rgb: np.ndarray,
        text_prompt: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        inference_state = self._processor.set_image(Image.fromarray(frame_rgb))
        output = self._processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None:
            return (
                np.zeros((0, frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
            )
        if hasattr(masks, "detach"):
            masks = masks.detach().float().cpu().numpy()
        if scores is None:
            scores = np.zeros((len(masks),), dtype=np.float32)
        elif hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return (masks > 0).astype(np.uint8), scores

    def _predict_full_frame(
        self,
        frame_rgb: np.ndarray,
        box: BBox,
        point: tuple[int, int],
    ) -> np.ndarray:
        self._predictor.set_image(frame_rgb)
        masks, _, _ = self._predictor.predict(
            point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=np.array(box.to_list(), dtype=np.float32),
            multimask_output=True,
            normalize_coords=False,
        )
        return (masks > 0).astype(np.uint8)

    def _predict_local_crop(
        self,
        crop_rgb: np.ndarray,
        local_box: BBox,
        local_point: tuple[int, int],
    ) -> np.ndarray:
        self._predictor.set_image(crop_rgb)
        masks, _, _ = self._predictor.predict(
            point_coords=np.array([[local_point[0], local_point[1]]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=np.array(local_box.to_list(), dtype=np.float32),
            multimask_output=True,
            normalize_coords=False,
        )
        return (masks > 0).astype(np.uint8)

    def _select_best_mask(
        self,
        masks: np.ndarray,
        sam_scores: np.ndarray,
        image_shape: tuple[int, int],
        max_area_ratio: float,
        min_area_ratio: float,
    ) -> Optional[CoreMaskSelection]:
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None
        image_area = max(image_shape[0] * image_shape[1], 1)
        best_selection: Optional[CoreMaskSelection] = None
        best_score = float("-inf")
        for mask_idx, mask in enumerate(masks):
            area = int((mask > 0).sum())
            if area <= 0:
                continue
            mask_box = bbox_from_mask(mask)
            if mask_box is None:
                continue
            box_area = max(mask_box.area(), 1)
            area_ratio = area / box_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            image_area_ratio = area / image_area
            compactness = mask_compactness(mask)
            component_penalty = max(0, connected_component_count(mask) - 1)
            sam_score = float(sam_scores[mask_idx]) if mask_idx < len(sam_scores) else 0.0
            small_object_bonus = 1.0 - min(1.0, np.sqrt(image_area_ratio / 0.15))
            shape_bonus = 1.0 - min(1.0, abs(mask_box.width() - mask_box.height()) / max(mask_box.width() + mask_box.height(), 1))
            score = (
                2.5 * sam_score
                + 1.5 * compactness
                + 1.0 * small_object_bonus
                + 0.5 * shape_bonus
                - 0.5 * component_penalty
            )
            if score > best_score:
                best_score = score
                best_selection = CoreMaskSelection(mask=(mask > 0).astype(np.uint8), core_box=mask_box, score=score)
        if best_selection is None:
            return None
        best_selection.mask = close_open(best_selection.mask, 3)
        best_selection.core_box = bbox_from_mask(best_selection.mask) or best_selection.core_box
        best_selection.score = clamp01(best_selection.score)
        return best_selection

    def _select_best_local_mask(
        self,
        masks: np.ndarray,
        center_point: tuple[int, int],
        local_box: BBox,
        max_area_ratio: float,
    ) -> Optional[CoreMaskSelection]:
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None
        local_area = max(local_box.area(), 1)
        best_selection: Optional[CoreMaskSelection] = None
        best_score = float("-inf")
        for mask in masks:
            area = int((mask > 0).sum())
            if area <= 0:
                continue
            area_ratio = area / local_area
            if area_ratio > max_area_ratio:
                continue
            compactness = mask_compactness(mask)
            component_penalty = max(0, connected_component_count(mask) - 1)
            px, py = self._clamp_point(mask.shape, int(center_point[0]), int(center_point[1]))
            center_inside = 1.0 if mask[py, px] > 0 else 0.0
            score = 3.0 * center_inside + 1.5 * compactness - 0.5 * component_penalty
            mask_box = bbox_from_mask(mask)
            if mask_box is None:
                continue
            if score > best_score:
                best_score = score
                best_selection = CoreMaskSelection(
                    mask=(mask > 0).astype(np.uint8),
                    core_box=mask_box,
                    score=score,
                )
        if best_selection is None:
            return None
        best_selection.mask = close_open(best_selection.mask, 3)
        best_selection.core_box = bbox_from_mask(best_selection.mask) or best_selection.core_box
        best_selection.score = clamp01(best_selection.score)
        return best_selection

    def _clamp_point(self, image_shape: tuple[int, int] | tuple[int, int, int], x: int, y: int) -> tuple[int, int]:
        height, width = image_shape[:2]
        return (
            max(0, min(x, width - 1)),
            max(0, min(y, height - 1)),
        )
