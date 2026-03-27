import json
import os
from datetime import datetime
from collections import defaultdict
from random import randint

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from scene import EnvLight, GaussianModel, Scene
from train import (
    EPS,
    _build_viewpoint_stack,
    _dice_loss,
    _dilate_mask,
    _lt_masks_for_view,
    _masked_l1,
)
from utils.general_utils import seed_everything, visualize_depth
from utils.hierarchical_utils import (
    initialize_hierarchical_pools,
    load_checkpoint_bundle,
    make_branch_args,
    render_hierarchical,
    save_checkpoint_bundle,
)
from utils.loss_utils import psnr, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _safe_feature_maps(render_pkg):
    alpha = render_pkg["alpha"].clamp_min(EPS)
    feature = render_pkg["feature"] / alpha
    if feature.shape[0] >= 4:
        t_map = feature[0:1]
        v_map = feature[1:4]
    else:
        t_map = torch.zeros_like(alpha)
        v_map = torch.zeros((3,) + alpha.shape[1:], device=alpha.device, dtype=alpha.dtype)
    return t_map, v_map


def _erode_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    return 1.0 - _dilate_mask(1.0 - mask, radius)


def _lt_supervision_masks(lt_mask_conf, args):
    support_mask = (lt_mask_conf > float(getattr(args, "lt_support_threshold", 0.0))).float()
    support_dilate_px = int(getattr(args, "lt_support_dilate_px", 0))
    if support_dilate_px > 0:
        support_mask = _dilate_mask(support_mask, support_dilate_px).clamp(0.0, 1.0)
    core_mask = support_mask
    core_erode_px = int(getattr(args, "lt_core_erode_px", 0))
    if core_erode_px > 0:
        core_mask = _erode_mask(core_mask, core_erode_px).clamp(0.0, 1.0)
    if core_mask.sum() < 1:
        core_mask = support_mask
    outside_ring = (support_mask - core_mask).clamp_min(0.0)
    return core_mask, support_mask, outside_ring


def _mask_bbox(mask):
    ys, xs = torch.where(mask[0] > 0)
    if ys.numel() == 0:
        return None
    return int(ys.min().item()), int(ys.max().item()) + 1, int(xs.min().item()), int(xs.max().item()) + 1


def _cropped_l1(pred, target, mask):
    bbox = _mask_bbox(mask)
    if bbox is None:
        return torch.zeros((), device=pred.device)
    y0, y1, x0, x1 = bbox
    pred_crop = pred[:, y0:y1, x0:x1]
    target_crop = target[:, y0:y1, x0:x1]
    mask_crop = mask[:, y0:y1, x0:x1]
    return _masked_l1(pred_crop, target_crop, mask_crop)


def _save_scale2_final_checkpoint(args, scene, bg_gaussians, lt_gaussians, iteration, env_map=None, reason=""):
    if not bool(getattr(args, "save_scale2_final_checkpoint", False)):
        return

    run_name = os.path.basename(os.path.normpath(scene.model_path))
    local_root = os.path.join(scene.model_path, "managed_scale2_final")
    managed_root = getattr(args, "scale2_final_checkpoint_root", None)
    target_roots = [local_root]
    if managed_root:
        target_roots.append(str(managed_root))

    metadata = {
        "run_name": run_name,
        "model_path": scene.model_path,
        "iteration": int(iteration),
        "current_scale": 2,
        "reason": reason,
        "hierarchical_longtail": True,
        "source_path": args.source_path,
        "start_frame": int(getattr(args, "start_frame", 0)),
        "end_frame": int(getattr(args, "end_frame", 0)),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    for root in target_roots:
        dest_dir = os.path.join(root, run_name, f"scale2_final_iter{iteration}")
        os.makedirs(dest_dir, exist_ok=True)
        save_checkpoint_bundle(os.path.join(dest_dir, f"chkpnt{iteration}.pth"), bg_gaussians, iteration, lt_gaussians=lt_gaussians)
        if env_map is not None:
            torch.save((env_map.capture(), iteration), os.path.join(dest_dir, f"env_light_chkpnt{iteration}.pth"))
        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def training(args):
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")

    vis_path = os.path.join(args.model_path, "visualization")
    os.makedirs(vis_path, exist_ok=True)

    bg_args = make_branch_args(args, enable_long_tail_branch=False)
    lt_args = make_branch_args(
        args,
        t_init=float(getattr(args, "lt_branch_t_init", args.t_init)),
        lt_gate_max_span_factor=float(getattr(args, "lt_branch_gate_span_factor", getattr(args, "lt_gate_max_span_factor", 1.0))),
    )

    bg_gaussians = GaussianModel(bg_args)
    scene = Scene(args, bg_gaussians)
    lt_gaussians = GaussianModel(lt_args)
    initialize_hierarchical_pools(scene, bg_gaussians, lt_gaussians, args)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    first_iter = 0
    if args.start_checkpoint:
        bundle = load_checkpoint_bundle(args.start_checkpoint)
        first_iter = int(bundle["iteration"])
        bg_gaussians.restore(bundle["bg"], bg_args)
        if bundle["lt"] is None:
            raise ValueError(f"Expected hierarchical checkpoint for {args.start_checkpoint}")
        lt_gaussians.restore(bundle["lt"], lt_args)
    else:
        bg_gaussians.training_setup(bg_args)
        lt_gaussians.training_setup(lt_args)

    background = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    viewpoint_stack = None
    ema_dict_for_log = defaultdict(float)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        iter_start.record()
        bg_gaussians.update_learning_rate(iteration)
        lt_gaussians.update_learning_rate(iteration)

        if iteration % args.sh_increase_interval == 0:
            bg_gaussians.oneupSHdegree()
            lt_gaussians.oneupSHdegree()

        current_scale = scene.resolution_scales[scene.scale_index]
        lt_scale_active = bool(current_scale <= args.lt_activate_max_scale)

        if not viewpoint_stack:
            viewpoint_stack = _build_viewpoint_stack(scene, args)
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        if np.random.random() < float(getattr(args, "bg_lambda_self_supervision", args.lambda_self_supervision)):
            bg_time_shift = 3 * (np.random.random() - 0.5) * scene.time_interval
        else:
            bg_time_shift = None

        render_bundle = render_hierarchical(
            viewpoint_cam,
            bg_gaussians,
            lt_gaussians,
            args,
            background,
            env_map=env_map,
            bg_other=[bg_gaussians.get_scaling_t.clamp_max(2), bg_gaussians.get_inst_velocity],
            lt_other=[lt_gaussians.get_scaling_t.clamp_max(2), lt_gaussians.get_inst_velocity],
            bg_time_shift=bg_time_shift,
            lt_time_shift=None,
            lt_enabled=lt_scale_active,
            is_training=True,
        )

        final_pkg = render_bundle["final"]
        bg_pkg = render_bundle["bg"]
        lt_pkg = render_bundle["lt"]

        image = final_pkg["render"]
        alpha = final_pkg["alpha"]
        depth = final_pkg["depth"] / alpha.clamp_min(EPS)
        if env_map is not None:
            if args.depth_blend_mode == 0:
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / 900).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * 900

        gt_image = viewpoint_cam.original_image.cuda()
        sky_mask, lt_mask, lt_mask_conf, lt_mask_dilated, sky_only_mask, overlap_mask = _lt_masks_for_view(
            viewpoint_cam,
            alpha,
            args,
            lt_scale_active,
        )
        lt_core_mask, lt_support_mask, lt_outside_ring = _lt_supervision_masks(lt_mask_conf, args)
        bg_mask = (1.0 - lt_support_mask).clamp_min(0.0)

        bg_image = bg_pkg["render"]
        bg_alpha = bg_pkg["alpha"]
        lt_image = lt_pkg["render"]
        lt_alpha = lt_pkg["alpha"]

        loss_bg_recon = _masked_l1(bg_image, gt_image, bg_mask) if bg_mask.sum() > 0 else torch.zeros((), device=image.device)
        lt_target_mode = str(getattr(args, "lt_target_mode", "foreground"))
        if lt_target_mode == "residual":
            lt_target = torch.clamp(gt_image - (1.0 - lt_support_mask) * bg_image.detach(), 0.0, 1.0)
        elif lt_target_mode == "foreground":
            lt_target = gt_image * lt_support_mask
        else:
            lt_target = gt_image
        loss_lt_recon = _masked_l1(lt_image, lt_target, lt_core_mask) if lt_scale_active and lt_core_mask.sum() > 0 else torch.zeros((), device=image.device)
        loss_final_l1 = F.l1_loss(image, gt_image)
        loss_final_ssim = 1.0 - ssim(image, gt_image)
        loss_bg_suppress = (bg_alpha * lt_support_mask).sum() / lt_support_mask.sum().clamp_min(EPS) if lt_scale_active else torch.zeros((), device=image.device)
        outside_mask = sky_only_mask if sky_only_mask.sum() > 0 else bg_mask
        loss_lt_suppress = (lt_alpha * outside_mask).sum() / outside_mask.sum().clamp_min(EPS) if lt_scale_active else torch.zeros((), device=image.device)
        if lt_scale_active:
            alpha_target = lt_support_mask
            loss_lt_mask = F.binary_cross_entropy(lt_alpha.clamp(EPS, 1.0 - EPS), alpha_target) + _dice_loss(lt_alpha, alpha_target)
            loss_lt_sparse, loss_lt_gate, loss_lt_smooth = lt_gaussians.get_lt_regularization_losses(scene.time_interval)
            loss_lt_ring = (lt_alpha * lt_outside_ring).sum() / lt_outside_ring.sum().clamp_min(EPS) if lt_outside_ring.sum() > 0 else torch.zeros((), device=image.device)
            loss_lt_crop = _cropped_l1(lt_image, lt_target, lt_core_mask) if lt_core_mask.sum() > 0 else torch.zeros((), device=image.device)
            loss_lt_scale = lt_gaussians.get_scaling.max(dim=1).values.mean() if lt_gaussians.get_xyz.shape[0] > 0 else torch.zeros((), device=image.device)
        else:
            zero = torch.zeros((), device=image.device)
            loss_lt_mask = zero
            loss_lt_sparse = zero
            loss_lt_gate = zero
            loss_lt_smooth = zero
            loss_lt_ring = zero
            loss_lt_crop = zero
            loss_lt_scale = zero

        loss = (
            float(args.bg_loss_weight) * loss_bg_recon
            + float(args.lt_loss_weight) * loss_lt_recon
            + float(args.final_loss_weight) * ((1.0 - args.lambda_dssim) * loss_final_l1 + args.lambda_dssim * loss_final_ssim)
            + float(args.bg_suppress_in_lt_weight) * loss_bg_suppress
            + float(args.lt_suppress_outside_weight) * loss_lt_suppress
            + args.lt_mask_weight * loss_lt_mask
            + args.lt_sparse_weight * loss_lt_sparse
            + args.lt_gate_weight * loss_lt_gate
            + args.lt_motion_smooth_weight * loss_lt_smooth
            + float(getattr(args, "lt_ring_weight", 0.0)) * loss_lt_ring
            + float(getattr(args, "lt_crop_loss_weight", 0.0)) * loss_lt_crop
            + float(getattr(args, "lt_scale_penalty_weight", 0.0)) * loss_lt_scale
        )

        bg_t_map, bg_v_map = _safe_feature_maps(bg_pkg)
        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(bg_v_map).mean()
            loss = loss + args.lambda_v_reg * loss_v_reg
        else:
            loss_v_reg = torch.zeros((), device=image.device)

        if args.lambda_inv_depth > 0:
            inverse_depth = 1 / (depth + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
            loss = loss + args.lambda_inv_depth * loss_inv_depth
        else:
            loss_inv_depth = torch.zeros((), device=image.device)

        if args.lambda_lidar > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = pts_depth > 0
            loss_lidar = torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay) if args.lidar_decay > 0 else 1
            loss = loss + iter_decay * args.lambda_lidar * loss_lidar
        else:
            loss_lidar = torch.zeros((), device=image.device)

        if args.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_sky_opa = (-sky_only_mask * torch.log(1 - o)).sum() / sky_only_mask.sum().clamp_min(EPS)
            loss = loss + args.lambda_sky_opa * loss_sky_opa
        else:
            loss_sky_opa = torch.zeros((), device=image.device)

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy
        else:
            loss_opacity_entropy = torch.zeros((), device=image.device)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            bg_visibility_filter = bg_pkg["visibility_filter"]
            bg_radii = bg_pkg["radii"]
            bg_gaussians.max_radii2D[bg_visibility_filter] = torch.max(
                bg_gaussians.max_radii2D[bg_visibility_filter],
                bg_radii[bg_visibility_filter],
            )
            bg_gaussians.add_densification_stats(bg_pkg["viewspace_points"], bg_visibility_filter)

            lt_visibility_filter = lt_pkg["visibility_filter"]
            if lt_scale_active and lt_visibility_filter.numel() > 0:
                lt_gaussians.max_radii2D[lt_visibility_filter] = torch.max(
                    lt_gaussians.max_radii2D[lt_visibility_filter],
                    lt_pkg["radii"][lt_visibility_filter],
                )
                lt_gaussians.add_densification_stats(lt_pkg["viewspace_points"], lt_visibility_filter)

            if iteration > args.densify_until_iter * args.time_split_frac:
                bg_gaussians.no_time_split = False
                if lt_scale_active:
                    lt_gaussians.no_time_split = False

            if iteration < args.densify_until_iter:
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None
                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]
                    bg_gaussians.densify_and_prune(
                        args.densify_grad_threshold,
                        args.thresh_opa_prune,
                        scene.cameras_extent,
                        size_threshold,
                        args.densify_grad_t_threshold,
                    )
                    lt_point_cap = int(getattr(args, "lt_densify_until_num_points", 400000))
                    if lt_scale_active and (lt_point_cap < 0 or lt_gaussians.get_xyz.shape[0] < lt_point_cap):
                        lt_gaussians.densify_and_prune(
                            args.densify_grad_threshold,
                            args.thresh_opa_prune,
                            scene.cameras_extent,
                            size_threshold,
                            args.densify_grad_t_threshold,
                        )
                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    bg_gaussians.reset_opacity()
                    if lt_scale_active:
                        lt_gaussians.reset_opacity()

            bg_gaussians.optimizer.step()
            bg_gaussians.optimizer.zero_grad(set_to_none=True)
            lt_gaussians.optimizer.step()
            lt_gaussians.optimizer.zero_grad(set_to_none=True)
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none=True)

            psnr_for_log = psnr(image, gt_image).double()
            log_dict = {
                "loss": float(loss.item()),
                "loss_bg_recon": float(loss_bg_recon.item()),
                "loss_lt_recon": float(loss_lt_recon.item()),
                "loss_final_l1": float(loss_final_l1.item()),
                "loss_final_ssim": float(loss_final_ssim.item()),
                "loss_bg_suppress_in_lt": float(loss_bg_suppress.item()),
                "loss_lt_suppress_outside": float(loss_lt_suppress.item()),
                "loss_lt_mask": float(loss_lt_mask.item()),
                "loss_lt_sparse": float(loss_lt_sparse.item()),
                "loss_lt_gate": float(loss_lt_gate.item()),
                "loss_lt_smooth": float(loss_lt_smooth.item()),
                "loss_lt_ring": float(loss_lt_ring.item()),
                "loss_lt_crop": float(loss_lt_crop.item()),
                "loss_lt_scale": float(loss_lt_scale.item()),
                "loss_lidar": float(loss_lidar.item()),
                "loss_inv_depth": float(loss_inv_depth.item()),
                "loss_sky_opa": float(loss_sky_opa.item()),
                "loss_opacity_entropy": float(loss_opacity_entropy.item()),
                "loss_v_reg": float(loss_v_reg.item()),
                "psnr": float(psnr_for_log.item()),
                "bg_points": int(bg_gaussians.get_xyz.shape[0]),
                "lt_points": int(lt_gaussians.get_xyz.shape[0]),
                "iter_time": float(iter_start.elapsed_time(iter_end)),
                "scale": float(current_scale),
            }
            for key in ["loss", "loss_final_l1", "psnr"]:
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{ema_dict_for_log['loss']:.5f}",
                        "l1": f"{ema_dict_for_log['loss_final_l1']:.5f}",
                        "psnr": f"{ema_dict_for_log['psnr']:.5f}",
                        "bg": log_dict["bg_points"],
                        "lt": log_dict["lt_points"],
                    }
                )

            if tb_writer:
                for key, value in log_dict.items():
                    tb_writer.add_scalar(f"train/{key}", value, iteration)

            if iteration % args.vis_step == 0 or iteration == 1:
                grid = make_grid(
                    [
                        image,
                        gt_image,
                        bg_image,
                        lt_image,
                        alpha.repeat(3, 1, 1),
                        bg_alpha.repeat(3, 1, 1),
                        lt_alpha.repeat(3, 1, 1),
                        lt_mask.repeat(3, 1, 1),
                        lt_support_mask.repeat(3, 1, 1),
                        lt_core_mask.repeat(3, 1, 1),
                        sky_only_mask.repeat(3, 1, 1),
                        overlap_mask.repeat(3, 1, 1),
                        visualize_depth(depth),
                        visualize_depth(bg_t_map, near=0.01, far=1),
                    ],
                    nrow=4,
                )
                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))

            if iteration % args.scale_increase_interval == 0:
                current_scale = scene.resolution_scales[scene.scale_index]
                next_scale = scene.resolution_scales[max(0, scene.scale_index - 1)]
                if float(current_scale) == 2.0 and float(next_scale) == 1.0:
                    _save_scale2_final_checkpoint(
                        args,
                        scene,
                        bg_gaussians,
                        lt_gaussians,
                        iteration,
                        env_map=env_map,
                        reason="before_upscale_to_scale1",
                    )
                scene.upScale()
                viewpoint_stack = None

            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_checkpoint_bundle(os.path.join(scene.model_path, f"chkpnt{iteration}.pth"), bg_gaussians, iteration, lt_gaussians=lt_gaussians)
                if env_map is not None:
                    torch.save((env_map.capture(), iteration), os.path.join(scene.model_path, f"env_light_chkpnt{iteration}.pth"))

            if iteration == args.iterations and float(scene.resolution_scales[scene.scale_index]) == 2.0:
                _save_scale2_final_checkpoint(
                    args,
                    scene,
                    bg_gaussians,
                    lt_gaussians,
                    iteration,
                    env_map=env_map,
                    reason="final_iteration_at_scale2",
                )

            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Hierarchical long-tail training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    print(args)

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    seed_everything(args.seed)
    os.makedirs(args.model_path, exist_ok=True)
    training(args)
    print("\nTraining complete.")
