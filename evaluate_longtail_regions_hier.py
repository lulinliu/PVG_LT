import glob
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from evaluate_longtail_regions import (
    _infer_camera_name,
    _write_camera_videos,
    _write_combined_camera_video,
    finalize_metric_store,
    init_metric_store,
    region_masks,
    update_metric_store,
    write_markdown_report,
)
from scene import EnvLight, GaussianModel, Scene
from utils.general_utils import seed_everything
from utils.hierarchical_utils import (
    load_checkpoint_bundle,
    make_branch_args,
    render_hierarchical,
    save_debug_overlay,
)


@torch.no_grad()
def evaluation(iteration, scene, bg_gaussians, lt_gaussians, args, env_map=None):
    scale = scene.resolution_scales[0]
    if "kitti" in args.model_path:
        num = len(scene.getTrainCameras()) // 2
        eval_train_frame = num // 5
        train_cameras = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(scale=scale)},
            {"name": "train", "cameras": train_cameras[:num][-eval_train_frame:] + train_cameras[num:][-eval_train_frame:]},
        )
    else:
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(scale=scale)},
            {"name": "train", "cameras": scene.getTrainCameras()},
        )

    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    results = {}
    for config in validation_configs:
        cameras = config["cameras"]
        if not cameras:
            continue

        split_name = config["name"]
        outdir = os.path.join(args.model_path, "eval", split_name + f"_{iteration}" + "_region_metrics")
        os.makedirs(outdir, exist_ok=True)
        render_only_outdir = os.path.join(args.model_path, "eval", split_name + f"_{iteration}" + "_render_only")
        video_outdir = os.path.join(args.model_path, "eval", split_name + f"_{iteration}" + "_videos")
        os.makedirs(render_only_outdir, exist_ok=True)
        os.makedirs(video_outdir, exist_ok=True)

        frame_paths = {
            "final": defaultdict(list),
            "bg_only": defaultdict(list),
            "lt_only": defaultdict(list),
        }
        diagnostics = []
        region_stores = {
            "overall": init_metric_store(),
            "longtail": init_metric_store(),
            "non_longtail": init_metric_store(),
        }

        for viewpoint in tqdm(cameras, desc=f"Evaluating {split_name}"):
            render_bundle = render_hierarchical(
                viewpoint,
                bg_gaussians,
                lt_gaussians,
                args,
                background,
                env_map=env_map,
                bg_other=[bg_gaussians.get_scaling_t.clamp_max(2), bg_gaussians.get_inst_velocity],
                lt_other=[lt_gaussians.get_scaling_t.clamp_max(2), lt_gaussians.get_inst_velocity],
                lt_enabled=bool(scale <= args.lt_activate_max_scale),
            )
            final_image = torch.clamp(render_bundle["final"]["render"], 0.0, 1.0)
            bg_image = torch.clamp(render_bundle["bg"]["render"], 0.0, 1.0)
            lt_image = torch.clamp(render_bundle["lt"]["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            masks = region_masks(viewpoint)

            cam_name = _infer_camera_name(viewpoint, args.cam_num)
            frame_name = viewpoint.image_name if viewpoint.image_name is not None else f"{viewpoint.colmap_id:06d}"
            for tag, image in [("final", final_image), ("bg_only", bg_image), ("lt_only", lt_image)]:
                cam_outdir = os.path.join(render_only_outdir, tag, cam_name)
                os.makedirs(cam_outdir, exist_ok=True)
                frame_path = os.path.join(cam_outdir, f"{frame_name}.png")
                save_image(image, frame_path)
                frame_paths[tag][cam_name].append(frame_path)

            overlay_path = os.path.join(render_only_outdir, "debug_overlay", cam_name, f"{frame_name}.png")
            save_debug_overlay(
                overlay_path,
                gt_image,
                render_bundle["lt_mask"],
                render_bundle["bg"]["alpha"],
                render_bundle["lt"]["alpha"],
            )

            diagnostics.append(
                {
                    "image_name": viewpoint.image_name,
                    "bg_alpha_in_lt": float((render_bundle["bg"]["alpha"] * render_bundle["lt_mask"]).mean().item()),
                    "lt_alpha_outside_lt": float((render_bundle["lt"]["alpha"] * (1.0 - render_bundle["lt_mask"])).mean().item()),
                }
            )

            for region_name, mask in masks.items():
                update_metric_store(
                    region_stores[region_name],
                    viewpoint.image_name,
                    region_name,
                    final_image,
                    gt_image,
                    mask,
                )

        split_metrics = {"regions": {}, "diagnostics": diagnostics}
        for region_name, store in region_stores.items():
            split_metrics["regions"][region_name] = finalize_metric_store(store)

        metrics_json_path = os.path.join(outdir, "metrics_regions.json")
        metrics_md_path = os.path.join(outdir, "metrics_regions.md")
        with open(metrics_json_path, "w") as f:
            json.dump(
                {
                    "split": split_name,
                    "iteration": iteration,
                    "regions": split_metrics["regions"],
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
            )
        write_markdown_report({split_name: split_metrics}, metrics_md_path, args.loaded_checkpoint, iteration)

        video_fps = getattr(args, "eval_render_video_fps", getattr(args, "train_render_video_fps", 10))
        for tag, paths in frame_paths.items():
            branch_video_outdir = os.path.join(video_outdir, tag)
            _write_camera_videos(paths, branch_video_outdir, video_fps)
            _write_combined_camera_video(paths, branch_video_outdir, video_fps, filename=f"{tag}_cam0_cam1_cam2.mp4")
        results[split_name] = split_metrics

    merged_md_path = os.path.join(args.model_path, "eval", f"region_metrics_{iteration}.md")
    write_markdown_report(results, merged_md_path, args.loaded_checkpoint, iteration)
    merged_json_path = os.path.join(args.model_path, "eval", f"region_metrics_{iteration}.json")
    with open(merged_json_path, "w") as f:
        json.dump({"iteration": iteration, "splits": results}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Hierarchical long-tail region evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)

    seed_everything(args.seed)
    os.makedirs(os.path.join(args.model_path, "eval"), exist_ok=True)

    bg_args = make_branch_args(args, enable_long_tail_branch=False)
    lt_args = make_branch_args(
        args,
        t_init=float(getattr(args, "lt_branch_t_init", args.t_init)),
        lt_gate_max_span_factor=float(getattr(args, "lt_branch_gate_span_factor", getattr(args, "lt_gate_max_span_factor", 1.0))),
    )
    bg_gaussians = GaussianModel(bg_args)
    scene = Scene(args, bg_gaussians, shuffle=False)
    lt_gaussians = GaussianModel(lt_args)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.model_path}")
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    args.loaded_checkpoint = checkpoint

    bundle = load_checkpoint_bundle(checkpoint)
    if not bundle["hierarchical_longtail"] or bundle["lt"] is None:
        raise ValueError(f"Checkpoint is not hierarchical: {checkpoint}")
    bg_gaussians.restore(bundle["bg"], bg_args)
    lt_gaussians.restore(bundle["lt"], lt_args)
    first_iter = int(bundle["iteration"])

    if env_map is not None:
        env_checkpoint = os.path.join(
            os.path.dirname(checkpoint),
            os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"),
        )
        light_params, _ = torch.load(env_checkpoint)
        env_map.restore(light_params)

    evaluation(first_iter, scene, bg_gaussians, lt_gaussians, args, env_map=env_map)
    print("Long-tail hierarchical region evaluation complete.")
