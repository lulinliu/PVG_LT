from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import infer_scene_category


CAMERA_TO_DIR = {
    "front": "image_0",
    "left": "image_1",
    "right": "image_2",
}


@dataclass(frozen=True)
class SceneMetadata:
    scene_id: str
    scene_category: str
    frames: int
    width: int
    height: int
    source_scene: str | None


class PVGScene:
    def __init__(
        self,
        scene_root: str | Path,
        cameras: list[str] | None = None,
        frame_cache_size: int = 96,
    ) -> None:
        self.scene_root = Path(scene_root)
        if not self.scene_root.is_dir():
            raise FileNotFoundError(f"Scene root not found: {self.scene_root}")
        self.frame_cache_size = frame_cache_size
        self._frame_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

        meta_path = self.scene_root / "pvg_prepare_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        source_scene = meta.get("source_scene")
        self.scene_category = infer_scene_category(source_scene, self.scene_root)

        chosen_cameras = cameras or list(CAMERA_TO_DIR.keys())
        self.frame_paths: dict[str, list[Path]] = {}
        for camera in chosen_cameras:
            image_dir = self.scene_root / CAMERA_TO_DIR[camera]
            if image_dir.is_dir():
                paths = sorted(image_dir.glob("*.png"))
                if paths:
                    self.frame_paths[camera] = paths
        if not self.frame_paths:
            raise RuntimeError(f"No camera frames found in {self.scene_root}")

        sample = np.array(Image.open(next(iter(self.frame_paths.values()))[0]).convert("RGB"))
        self.metadata = SceneMetadata(
            scene_id=self.scene_root.name,
            scene_category=self.scene_category,
            frames=min(len(paths) for paths in self.frame_paths.values()),
            width=int(sample.shape[1]),
            height=int(sample.shape[0]),
            source_scene=source_scene,
        )

    def load_frame(self, camera: str, frame_index: int) -> np.ndarray:
        key = (camera, frame_index)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        frame = np.array(Image.open(self.frame_paths[camera][frame_index]).convert("RGB"))
        self._frame_cache[key] = frame
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def frame_path(self, camera: str, frame_index: int) -> Path:
        return self.frame_paths[camera][frame_index]

    def context_frame_indices(self, frame_index: int, context_size: int) -> list[int]:
        half = context_size // 2
        indices = []
        for delta in range(-half, half + 1):
            idx = min(max(frame_index + delta, 0), self.metadata.frames - 1)
            indices.append(idx)
        return indices

    def context_frame_paths(self, camera: str, frame_index: int, context_size: int) -> list[Path]:
        return [self.frame_path(camera, idx) for idx in self.context_frame_indices(frame_index, context_size)]

    def context_frames(self, camera: str, frame_index: int, context_size: int) -> list[np.ndarray]:
        return [self.load_frame(camera, idx) for idx in self.context_frame_indices(frame_index, context_size)]

    def iter_target_frames(self, frame_start: int, frame_end: int | None, frame_stride: int) -> list[int]:
        last = self.metadata.frames - 1 if frame_end is None else min(frame_end, self.metadata.frames - 1)
        return list(range(frame_start, last + 1, frame_stride))
