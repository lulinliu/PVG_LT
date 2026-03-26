import glob
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image

from gaussian_renderer import render
from lpipsPyTorch import lpips
from scene import EnvLight, GaussianModel, Scene
from utils.general_utils import seed_everything
from utils.loss_utils import create_window

EPS = 1e-8


def _infer_camera_name(viewpoint, default_cam_num):
    image_path = getattr(viewpoint, "image_path", None)
    if image_path:
        cam_dir = os.path.basename(os.path.dirname(image_path))
        if cam_dir.startswith("image_"):
            return cam_dir
    colmap_id = getattr(viewpoint, "colmap_id", None)
    if isinstance(colmap_id, int):
        return f"image_{int(colmap_id) % max(1, int(default_cam_num))}"
    return "image_unknown"


def _sorted_camera_names(frame_paths_by_cam):
    def camera_order_key(cam_name):
        suffix = cam_name.split("_")[-1]
        return int(suffix) if suffix.isdigit() else 1_000_000

    return sorted(frame_paths_by_cam.keys(), key=camera_order_key)


def _ordered_frame_paths(frame_paths):
    return sorted(frame_paths, key=lambda path: os.path.basename(path))


def _write_camera_videos(frame_paths_by_cam, outdir, fps):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed, skip video export.")
        return

    os.makedirs(outdir, exist_ok=True)
    for cam_name in _sorted_camera_names(frame_paths_by_cam):
        ordered_frame_paths = _ordered_frame_paths(frame_paths_by_cam[cam_name])
        if not ordered_frame_paths:
            continue
        video_path = os.path.join(outdir, f"{cam_name}.mp4")
        with imageio.get_writer(video_path, fps=float(fps), macro_block_size=1) as writer:
            for frame_path in ordered_frame_paths:
                writer.append_data(imageio.imread(frame_path))


def _write_combined_camera_video(frame_paths_by_cam, outdir, fps, filename="cam0_cam1_cam2.mp4"):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed, skip combined video export.")
        return

    ordered_all = []
    for cam_name in _sorted_camera_names(frame_paths_by_cam):
        ordered_all.extend(_ordered_frame_paths(frame_paths_by_cam[cam_name]))
    if not ordered_all:
        return

    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(outdir, filename)
    with imageio.get_writer(video_path, fps=float(fps), macro_block_size=1) as writer:
        for frame_path in ordered_all:
            writer.append_data(imageio.imread(frame_path))


def masked_psnr(pred, target, mask):
    mask3 = mask.expand_as(pred)
    denom = mask3.sum().clamp_min(EPS)
    mse = ((pred - target) ** 2 * mask3).sum() / denom
    return 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(EPS)))


def masked_ssim(pred, target, mask, window_size=11):
    channel = pred.size(0)
    window = create_window(window_size, channel).to(device=pred.device, dtype=pred.dtype)

    pred_b = pred.unsqueeze(0)
    target_b = target.unsqueeze(0)

    mu1 = F.conv2d(pred_b, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target_b, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_b * pred_b, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target_b * target_b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred_b * target_b, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + EPS
    )
    ssim_map = ssim_map.mean(dim=1, keepdim=True)
    weight = mask.unsqueeze(0)
    return (ssim_map * weight).sum() / weight.sum().clamp_min(EPS)


def _mask_bbox(mask):
    ys, xs = torch.where(mask[0] > 0)
    if ys.numel() == 0:
        return None
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    return y0, y1, x0, x1


def masked_lpips(pred, target, mask):
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    pred_crop = pred[:, y0:y1, x0:x1]
    target_crop = target[:, y0:y1, x0:x1]
    mask_crop = mask[:, y0:y1, x0:x1]
    pred_crop = pred_crop * mask_crop
    target_crop = target_crop * mask_crop
    min_side = 32
    height, width = pred_crop.shape[-2:]
    if height < min_side or width < min_side:
        out_h = max(height, min_side)
        out_w = max(width, min_side)
        pred_crop = F.interpolate(pred_crop.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)
        target_crop = F.interpolate(target_crop.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)
    return lpips(pred_crop, target_crop, net_type="vgg").double()


def region_masks(viewpoint):
    h, w = viewpoint.image_height, viewpoint.image_width
    device = viewpoint.original_image.device
    lt_mask = viewpoint.lt_mask_conf
    if lt_mask is None:
        lt_mask = torch.zeros((1, h, w), device=device, dtype=viewpoint.original_image.dtype)
    else:
        lt_mask = (lt_mask > 0).float()
    non_lt_mask = 1.0 - lt_mask
    overall_mask = torch.ones_like(lt_mask)
    return {
        "overall": overall_mask,
        "longtail": lt_mask,
        "non_longtail": non_lt_mask,
    }


def init_metric_store():
    return {
        "sum": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0},
        "count": {"psnr": 0, "ssim": 0, "lpips": 0},
        "images": [],
    }


def update_metric_store(store, image_name, region_name, pred, target, mask):
    area = float(mask.sum().item())
    image_metrics = {"image_name": image_name, "region": region_name, "pixels": int(area)}
    if area < 1.0:
        image_metrics.update({"psnr": None, "ssim": None, "lpips": None})
        store["images"].append(image_metrics)
        return

    psnr_value = masked_psnr(pred, target, mask).double()
    ssim_value = masked_ssim(pred, target, mask).double()
    lpips_value = masked_lpips(pred, target, mask)

    store["sum"]["psnr"] += float(psnr_value.item())
    store["sum"]["ssim"] += float(ssim_value.item())
    store["count"]["psnr"] += 1
    store["count"]["ssim"] += 1
    image_metrics["psnr"] = float(psnr_value.item())
    image_metrics["ssim"] = float(ssim_value.item())

    if lpips_value is not None:
        store["sum"]["lpips"] += float(lpips_value.item())
        store["count"]["lpips"] += 1
        image_metrics["lpips"] = float(lpips_value.item())
    else:
        image_metrics["lpips"] = None

    store["images"].append(image_metrics)


def finalize_metric_store(store):
    result = {"per_image": store["images"]}
    for metric_name in ("psnr", "ssim", "lpips"):
        count = store["count"][metric_name]
        result[metric_name] = store["sum"][metric_name] / count if count > 0 else None
        result[f"{metric_name}_count"] = count
    return result


def write_markdown_report(metrics_by_split, output_path, checkpoint_path, iteration):
    lines = [
        "# Long-Tail Region Evaluation",
        "",
        f"- Iteration: `{iteration}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "## Region Metrics",
        "",
        "| Split | Region | PSNR | SSIM | LPIPS | Valid Images |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for split_name, split_metrics in metrics_by_split.items():
        for region_name, region_metrics in split_metrics["regions"].items():
            psnr_value = "N/A" if region_metrics["psnr"] is None else f'{region_metrics["psnr"]:.4f}'
            ssim_value = "N/A" if region_metrics["ssim"] is None else f'{region_metrics["ssim"]:.4f}'
            lpips_value = "N/A" if region_metrics["lpips"] is None else f'{region_metrics["lpips"]:.4f}'
            valid_images = region_metrics["lpips_count"]
            lines.append(
                f"| {split_name} | {region_name} | {psnr_value} | {ssim_value} | {lpips_value} | {valid_images} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `overall` uses the full image.",
            "- `longtail` uses pixels where `lt_mask_conf > 0`.",
            "- `non_longtail` uses the complement of the long-tail mask.",
            "- Masked LPIPS is computed on the tight bounding box of the target region, with pixels outside the mask zeroed within that crop.",
            "",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


@torch.no_grad()
def evaluation(iteration, scene, render_func, render_args, args, env_map=None):
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
        frame_paths_by_cam = defaultdict(list)

        region_stores = {
            "overall": init_metric_store(),
            "longtail": init_metric_store(),
            "non_longtail": init_metric_store(),
        }

        for viewpoint in tqdm(cameras, desc=f"Evaluating {split_name}"):
            lt_point_mask = None
            if args.enable_long_tail_branch and scale <= args.lt_activate_max_scale and viewpoint.lt_mask_conf is not None:
                scene.gaussians.set_long_tail_active(True)
                lt_point_mask = scene.gaussians.get_current_lt_point_mask(
                    viewpoint,
                    mask_threshold=getattr(args, "lt_current_mask_threshold", 0.25),
                    enable_long_tail=True,
                )
            else:
                scene.gaussians.set_long_tail_active(False)
            render_pkg = render_func(viewpoint, scene.gaussians, *render_args, env_map=env_map, lt_point_mask=lt_point_mask)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            masks = region_masks(viewpoint)

            cam_name = _infer_camera_name(viewpoint, args.cam_num)
            cam_outdir = os.path.join(render_only_outdir, cam_name)
            os.makedirs(cam_outdir, exist_ok=True)
            frame_name = viewpoint.image_name if viewpoint.image_name is not None else f"{viewpoint.colmap_id:06d}"
            frame_path = os.path.join(cam_outdir, f"{frame_name}.png")
            save_image(image, frame_path)
            frame_paths_by_cam[cam_name].append(frame_path)

            for region_name, mask in masks.items():
                update_metric_store(
                    region_stores[region_name],
                    viewpoint.image_name,
                    region_name,
                    image,
                    gt_image,
                    mask,
                )

        split_metrics = {"regions": {}}
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
                },
                f,
                indent=2,
            )
        write_markdown_report({split_name: split_metrics}, metrics_md_path, args.loaded_checkpoint, iteration)
        video_fps = getattr(args, "eval_render_video_fps", getattr(args, "train_render_video_fps", 10))
        _write_camera_videos(frame_paths_by_cam, video_outdir, video_fps)
        _write_combined_camera_video(frame_paths_by_cam, video_outdir, video_fps)
        results[split_name] = split_metrics

        print(f"\n[ITER {iteration}] {split_name} region metrics:")
        for region_name, region_metrics in split_metrics["regions"].items():
            print(
                f"  - {region_name}: "
                f"PSNR={region_metrics['psnr']} "
                f"SSIM={region_metrics['ssim']} "
                f"LPIPS={region_metrics['lpips']}"
            )
        print(f"[ITER {iteration}] Saved render videos to {video_outdir}")

    merged_md_path = os.path.join(args.model_path, "eval", f"region_metrics_{iteration}.md")
    write_markdown_report(results, merged_md_path, args.loaded_checkpoint, iteration)
    merged_json_path = os.path.join(args.model_path, "eval", f"region_metrics_{iteration}.json")
    with open(merged_json_path, "w") as f:
        json.dump({"iteration": iteration, "splits": results}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Long-tail region evaluation")
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

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)

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

    model_params, first_iter = torch.load(checkpoint)
    gaussians.restore(model_params, args)

    if env_map is not None:
        env_checkpoint = os.path.join(
            os.path.dirname(checkpoint),
            os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"),
        )
        light_params, _ = torch.load(env_checkpoint)
        env_map.restore(light_params)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    evaluation(first_iter, scene, render, (args, background), args, env_map=env_map)

    print("Long-tail region evaluation complete.")
