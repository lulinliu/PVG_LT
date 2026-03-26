from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import TailMinerConfig
from .core import Sam3Segmenter
from .expand import DINOv3FeatureMiner
from .proposal import Qwen3VLCandidateProposer
from .scene import PVGScene
from .schema import CandidateProposal, CoreMaskSelection
from .utils import close_open, ensure_dir, mask_containment, mask_iou, save_binary_mask


class TailMinerV1:
    def __init__(self, config: TailMinerConfig) -> None:
        self.config = config
        self.proposer = Qwen3VLCandidateProposer(config.vlm)
        self.segmenter = Sam3Segmenter(config.sam)
        self.feature_miner = DINOv3FeatureMiner(config.dino, config.roi)

    def _log(self, message: str) -> None:
        print(f"[tail_miner] {message}", flush=True)

    def run_scene(self, scene_root: str | Path, output_root: str | Path) -> dict:
        scene = PVGScene(
            scene_root,
            cameras=self.config.cameras,
            frame_cache_size=self.config.frame_cache_size,
        )
        output_root = ensure_dir(Path(output_root))
        original_root = ensure_dir(output_root / "original_longtail_masks" / scene.metadata.scene_id)
        dino_root = ensure_dir(output_root / "dino_helped_longtail_masks" / scene.metadata.scene_id)
        frames_processed = 0
        original_masks_written = 0
        final_masks_written = 0
        target_frames = scene.iter_target_frames(
            self.config.frame_start,
            self.config.frame_end,
            self.config.frame_stride,
        )

        self._log(
            "start "
            f"scene={scene.metadata.scene_id} cameras={','.join(self.config.cameras)} "
            f"frames={len(target_frames)} range={target_frames[0] if target_frames else 'n/a'}-{target_frames[-1] if target_frames else 'n/a'}"
        )

        for camera in self.config.cameras:
            if camera not in scene.frame_paths:
                self._log(f"skip camera={camera} reason=no_frames")
                continue
            self._log(f"camera_start camera={camera} frames={len(target_frames)}")
            for frame_index in target_frames:
                frames_processed += 1
                self._log(f"frame_start camera={camera} frame={frame_index:06d}")
                frame_rgb = scene.load_frame(camera, frame_index)
                original_candidate_masks: list[np.ndarray] = []
                candidate_masks: list[np.ndarray] = []
                candidates = self.proposer.propose(
                    scene=scene,
                    camera=camera,
                    frame_index=frame_index,
                    context_size=self.config.vlm_context_size,
                    max_candidates=self.config.max_candidates_per_frame,
                )
                self._log(
                    f"frame_candidates camera={camera} frame={frame_index:06d} count={len(candidates)} "
                    f"labels={[candidate.label for candidate in candidates]}"
                )
                selected_per_label = self._select_primary_per_label(frame_rgb, candidates)
                self._log(
                    f"frame_primary camera={camera} frame={frame_index:06d} selected={len(selected_per_label)}"
                )
                for candidate, primary in selected_per_label:
                    original_candidate_masks.append((primary.mask > 0).astype(np.uint8))
                    roi_result = self.feature_miner.build_roi(frame_rgb.shape, primary)
                    clusters = self.feature_miner.mine_clusters(frame_rgb, roi_result)
                    accepted_masks = []
                    for cluster in clusters:
                        mask = self.segmenter.segment_cluster(frame_rgb, cluster)
                        if mask is not None and mask.any():
                            accepted_masks.append(mask)
                    final_mask = self._merge_candidate_masks(primary.mask, accepted_masks)
                    self._log(
                        "candidate_result "
                        f"camera={camera} frame={frame_index:06d} label={candidate.label} "
                        f"clusters={len(clusters)} accepted={len(accepted_masks)} "
                        f"final_nonempty={int(final_mask.any())}"
                    )
                    if final_mask.any():
                        candidate_masks.append(final_mask)
                frame_original_written = self._save_frame_masks(
                    original_root,
                    camera,
                    frame_index,
                    original_candidate_masks,
                )
                original_masks_written += frame_original_written
                frame_masks = self._deduplicate_frame_masks(candidate_masks)
                frame_final_written = self._save_frame_masks(dino_root, camera, frame_index, frame_masks)
                final_masks_written += frame_final_written
                self._log(
                    "frame_done "
                    f"camera={camera} frame={frame_index:06d} "
                    f"original_candidates={len(original_candidate_masks)} "
                    f"final_candidates={len(candidate_masks)} "
                    f"original_written={frame_original_written} "
                    f"final_written={frame_final_written}"
                )

            self._log(f"camera_done camera={camera}")

        self._log(
            "scene_done "
            f"scene={scene.metadata.scene_id} frames_processed={frames_processed} "
            f"original_masks_written={original_masks_written} final_masks_written={final_masks_written}"
        )

        return {
            "scene_id": scene.metadata.scene_id,
            "frames_processed": frames_processed,
            "original_masks_written": original_masks_written,
            "final_masks_written": final_masks_written,
        }

    def _select_primary_per_label(
        self,
        frame_rgb: np.ndarray,
        candidates: list[CandidateProposal],
    ) -> list[tuple[CandidateProposal, CoreMaskSelection]]:
        grouped: dict[str, list[CandidateProposal]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.label].append(candidate)

        selected: list[tuple[CandidateProposal, CoreMaskSelection]] = []
        for label, label_candidates in grouped.items():
            representative = self._choose_representative_candidate(label_candidates)
            primary = self.segmenter.select_primary_mask(frame_rgb, representative)
            if primary is not None:
                selected.append((representative, primary))
        return selected

    def _choose_representative_candidate(
        self,
        label_candidates: list[CandidateProposal],
    ) -> CandidateProposal:
        label_candidates = sorted(label_candidates, key=lambda item: item.confidence, reverse=True)
        best = label_candidates[0]
        return CandidateProposal(
            camera=best.camera,
            frame_index=best.frame_index,
            label=best.label,
            text_prompt=best.text_prompt,
            confidence=best.confidence,
        )

    def _merge_candidate_masks(self, primary_mask: np.ndarray, accepted_masks: list[np.ndarray]) -> np.ndarray:
        merged = (primary_mask > 0).astype(np.uint8)
        for mask in accepted_masks:
            merged = np.logical_or(merged > 0, mask > 0).astype(np.uint8)
        return close_open(merged, self.config.merge.cleanup_kernel_size)

    def _deduplicate_frame_masks(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        deduped = [(mask > 0).astype(np.uint8) for mask in masks if mask.any()]
        changed = True
        while changed:
            changed = False
            next_masks: list[np.ndarray] = []
            used = set()
            for idx, current in enumerate(deduped):
                if idx in used:
                    continue
                base = current.copy()
                base_area = int(base.sum())
                for other_idx in range(idx + 1, len(deduped)):
                    if other_idx in used:
                        continue
                    other = deduped[other_idx]
                    iou = mask_iou(base, other)
                    if iou > self.config.merge.candidate_mask_merge_iou:
                        base = np.logical_or(base > 0, other > 0).astype(np.uint8)
                        base_area = int(base.sum())
                        used.add(other_idx)
                        changed = True
                        continue
                    contain_other = mask_containment(other, base)
                    contain_base = mask_containment(base, other)
                    if contain_other > self.config.merge.candidate_containment_ratio:
                        used.add(other_idx)
                        changed = True
                    elif contain_base > self.config.merge.candidate_containment_ratio:
                        base = other.copy() if int(other.sum()) >= base_area else base
                        base_area = int(base.sum())
                        used.add(other_idx)
                        changed = True
                used.add(idx)
                next_masks.append(close_open(base, self.config.merge.cleanup_kernel_size))
            deduped = next_masks
        return deduped

    def _save_frame_masks(
        self,
        scene_out: Path,
        camera: str,
        frame_index: int,
        masks: list[np.ndarray],
    ) -> int:
        frame_dir = scene_out / camera / f"{frame_index:06d}"
        ensure_dir(frame_dir)
        for existing in frame_dir.glob("mask_*.png"):
            existing.unlink()
        written = 0
        for instance_idx, mask in enumerate(masks):
            if not mask.any():
                continue
            mask_path = frame_dir / f"mask_{instance_idx:02d}.png"
            save_binary_mask(mask, mask_path)
            written += 1
        return written
