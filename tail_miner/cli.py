from __future__ import annotations

import argparse
import json

from .config import TailMinerConfig
from .pipeline import TailMinerV1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V1 long-tail miner for single-frame per-camera candidate refinement.")
    parser.add_argument("--scene-root", required=True, help="PVG scene directory with image_0/image_1/image_2.")
    parser.add_argument("--output-root", required=True, help="Directory where final binary masks will be written.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["front", "left", "right"],
        choices=["front", "left", "right"],
        help="One or more cameras to process.",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-cache-size", type=int, default=96)
    parser.add_argument("--vlm-context-size", type=int, default=5)
    parser.add_argument("--max-candidates-per-frame", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TailMinerConfig(
        cameras=args.cameras,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        frame_cache_size=args.frame_cache_size,
        vlm_context_size=args.vlm_context_size,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    miner = TailMinerV1(config)
    summary = miner.run_scene(args.scene_root, args.output_root)
    print(json.dumps(summary, indent=2))
    return 0
