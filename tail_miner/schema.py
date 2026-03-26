from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

    def clipped(self, width: int, height: int) -> "BBox":
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        x2 = max(x1 + 1, min(self.x2, width))
        y2 = max(y1 + 1, min(self.y2, height))
        return BBox(x1, y1, x2, y2)

    def expand(self, scale: float, image_width: int, image_height: int) -> "BBox":
        cx, cy = self.center()
        half_w = max(1.0, self.width() * scale / 2.0)
        half_h = max(1.0, self.height() * scale / 2.0)
        return BBox(
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        ).clipped(image_width, image_height)

    def expand_by_ratio(self, ratio: float, image_width: int, image_height: int) -> "BBox":
        return self.expand(1.0 + ratio, image_width, image_height)

    def intersection(self, other: "BBox") -> "BBox | None":
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox(x1, y1, x2, y2)

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def iou(self, other: "BBox") -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area()
        union_area = self.area() + other.area() - inter_area
        return inter_area / max(union_area, 1)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class CandidateProposal:
    camera: str
    frame_index: int
    label: str
    text_prompt: str
    confidence: float


@dataclass
class CoreMaskSelection:
    mask: object
    core_box: BBox
    score: float


@dataclass
class PromptCluster:
    patch_mask: object
    patch_box: BBox
    prompt_box: BBox
    prompt_point: tuple[int, int]
    average_similarity: float
    max_similarity: float
