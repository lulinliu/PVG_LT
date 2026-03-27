#!/usr/bin/env python3
import argparse
import glob
import os
import re
from pathlib import Path

from omegaconf import OmegaConf


def infer_scale(iteration, resolution_scales, scale_increase_interval):
    scale_index = len(resolution_scales) - 1
    transitions = iteration // int(scale_increase_interval)
    scale_index = max(0, scale_index - transitions)
    return float(resolution_scales[scale_index])


def main():
    parser = argparse.ArgumentParser(description="Inspect hierarchical run progress")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    vis_files = sorted((model_path / "visualization").glob("*.png"))
    ckpts = sorted(glob.glob(str(model_path / "chkpnt*.pth")))

    latest_vis = vis_files[-1] if vis_files else None
    latest_iter = None
    if latest_vis is not None:
        m = re.match(r"(\d+)_", latest_vis.name)
        if m:
            latest_iter = int(m.group(1))
    if latest_iter is None and ckpts:
        latest_iter = max(int(re.search(r"chkpnt(\d+)\.pth", p).group(1)) for p in ckpts)

    print(f"model_path: {model_path}")
    print(f"latest_visualization: {latest_vis if latest_vis else 'N/A'}")
    print(f"latest_checkpoint: {ckpts[-1] if ckpts else 'N/A'}")
    print(f"latest_iteration: {latest_iter if latest_iter is not None else 'unknown'}")

    if args.config:
        conf = OmegaConf.load(args.config)
        if args.base_config:
            conf = OmegaConf.merge(OmegaConf.load(args.base_config), conf)
        resolution_scales = list(conf.resolution_scales)
        scale_increase_interval = int(conf.scale_increase_interval)
        lt_activate_max_scale = float(conf.lt_activate_max_scale)
        if latest_iter is not None:
            current_scale = infer_scale(latest_iter, resolution_scales, scale_increase_interval)
            lt_active = current_scale <= lt_activate_max_scale
            print(f"current_scale: {current_scale}")
            print(f"lt_active_now: {lt_active}")
            if not lt_active:
                print("note: LT branch is not active yet; current visualization mostly reflects BG branch.")
                first_lt_iter = max(0, (len(resolution_scales) - 1) * scale_increase_interval)
                print(f"expected_lt_activation_after_iter: ~{first_lt_iter}")
            else:
                print("note: LT branch should now be active; inspect bird-specific artifacts.")

    eval_dir = model_path / "eval"
    if eval_dir.exists():
        render_only = sorted(eval_dir.glob("*_render_only"))
        videos = sorted(eval_dir.glob("*_videos"))
        print(f"eval_render_dirs: {len(render_only)}")
        print(f"eval_video_dirs: {len(videos)}")
        if render_only:
            print(f"latest_render_dir: {render_only[-1]}")
        if videos:
            print(f"latest_video_dir: {videos[-1]}")


if __name__ == "__main__":
    main()
