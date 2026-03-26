from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Qwen3VLConfig:
    model_path: str = "/ssd2/wenyan/CarTwin/longtail/Qwen3-VL-8B-Thinking"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 2048


@dataclass
class LocalSam3Config:
    root: str = "/ssd2/wenyan/CarTwin/longtail/sam3"
    checkpoint: str = "/ssd2/wenyan/CarTwin/longtail/sam3/ckpt/sam3.pt"
    device: str = "auto"
    primary_mask_min_area_ratio: float = 0.10
    primary_mask_max_area_ratio: float = 2.00
    secondary_mask_max_area_ratio: float = 2.50


@dataclass
class DINOv3Config:
    repo_or_dir: str = "facebookresearch/dinov3"
    model_entrypoint: str = "dinov3_vith16plus"
    weights_path: str = "/ssd2/wenyan/CarTwin/longtail/PVG/tail_miner/dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    device: str = "auto"
    target_min_side: int = 224
    target_max_side: int = 560


@dataclass
class ROIConfig:
    expansion_scale: float = 2.5
    center_allowance_px: int = 200
    similarity_threshold: float = 0.65
    similarity_top_percent: float = 0.10
    min_cluster_patches: int = 3
    max_clusters: int = 3
    cluster_merge_iou: float = 0.70
    redundant_coverage_ratio: float = 0.80
    second_round_box_expand_ratio: float = 0.15


@dataclass
class MergeConfig:
    accepted_mask_merge_iou: float = 0.80
    candidate_mask_merge_iou: float = 0.70
    candidate_containment_ratio: float = 0.85
    cleanup_kernel_size: int = 3


@dataclass
class TailMinerConfig:
    cameras: list[str] = field(default_factory=lambda: ["front", "left", "right"])
    frame_start: int = 0
    frame_end: int | None = None
    frame_stride: int = 1
    frame_cache_size: int = 96
    vlm_context_size: int = 5
    max_candidates_per_frame: int = 3
    vlm: Qwen3VLConfig = field(default_factory=Qwen3VLConfig)
    sam: LocalSam3Config = field(default_factory=LocalSam3Config)
    dino: DINOv3Config = field(default_factory=DINOv3Config)
    roi: ROIConfig = field(default_factory=ROIConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)

    def __post_init__(self) -> None:
        if self.vlm_context_size < 1 or self.vlm_context_size % 2 == 0:
            raise ValueError("vlm_context_size must be an odd integer >= 1")
        if self.frame_start < 0:
            raise ValueError("frame_start must be >= 0")
        if self.frame_end is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.frame_cache_size < 1:
            raise ValueError("frame_cache_size must be >= 1")
        if self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be >= 1")
