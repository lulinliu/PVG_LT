#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import os
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
import kornia
from omegaconf import OmegaConf
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

EPS = 1e-5


def _infer_camera_name(viewpoint, default_cam_num=1):
    if viewpoint.image_path is not None:
        cam_name = os.path.basename(os.path.dirname(viewpoint.image_path))
        if cam_name:
            return cam_name
    colmap_id = getattr(viewpoint, "colmap_id", None)
    if isinstance(colmap_id, (int, np.integer)):
        return f"image_{int(colmap_id) % max(1, int(default_cam_num))}"
    return "image_unknown"


def _write_camera_videos(frame_paths_by_cam, outdir, fps):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed, skip video export.")
        return

    os.makedirs(outdir, exist_ok=True)
    for cam_name, frame_paths in sorted(frame_paths_by_cam.items()):
        ordered_frame_paths = sorted(frame_paths)
        if not ordered_frame_paths:
            continue
        video_path = os.path.join(outdir, f"{cam_name}.mp4")
        with imageio.get_writer(video_path, fps=float(fps), macro_block_size=1) as writer:
            for frame_path in ordered_frame_paths:
                writer.append_data(imageio.imread(frame_path))


def _camera_has_lt(viewpoint):
    if viewpoint.lt_mask_conf is None:
        return False
    return bool(viewpoint.lt_mask_conf.max().item() > 0)


def _build_viewpoint_stack(scene, args):
    cameras = scene.getTrainCameras()
    if not getattr(args, "enable_long_tail_branch", True):
        return list(range(len(cameras)))
    boost = max(1, int(round(float(getattr(args, "lt_sampling_boost", 1.0)))))
    stack = []
    for idx, cam in enumerate(cameras):
        repeat = boost if _camera_has_lt(cam) else 1
        stack.extend([idx] * repeat)
    return stack


def _dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)


def _masked_l1(pred, target, mask):
    mask3 = mask.expand_as(pred)
    return (torch.abs(pred - target) * mask3).sum() / mask3.sum().clamp_min(EPS)


def _dice_loss(pred, target):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + EPS) / (pred_flat.sum() + target_flat.sum() + EPS)


def _lt_masks_for_view(viewpoint_cam, alpha, args, lt_scale_active):
    zero_mask = torch.zeros_like(alpha)
    sky_mask = viewpoint_cam.sky_mask.cuda().float() if viewpoint_cam.sky_mask is not None else zero_mask
    if viewpoint_cam.lt_mask_conf is None:
        lt_mask_conf = zero_mask
    else:
        lt_mask_conf = viewpoint_cam.lt_mask_conf.cuda().clamp(0.0, 1.0)
    lt_mask = (lt_mask_conf > 0).float()
    if lt_scale_active and viewpoint_cam.lt_mask_conf is not None:
        lt_mask_dilated = _dilate_mask(lt_mask, getattr(args, "lt_mask_dilate_px", 0))
    else:
        lt_mask_dilated = zero_mask
    sky_only_mask = sky_mask * (1.0 - lt_mask_dilated)
    overlap_mask = sky_mask * lt_mask_dilated
    return sky_mask, lt_mask, lt_mask_conf, lt_mask_dilated, sky_only_mask, overlap_mask


def _save_scale2_final_checkpoint(args, scene, gaussians, iteration, env_map=None, reason=""):
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
        "source_path": args.source_path,
        "start_frame": int(getattr(args, "start_frame", 0)),
        "end_frame": int(getattr(args, "end_frame", 0)),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    for root in target_roots:
        dest_dir = os.path.join(root, run_name, f"scale2_final_iter{iteration}")
        os.makedirs(dest_dir, exist_ok=True)
        torch.save((gaussians.capture(), iteration), os.path.join(dest_dir, f"chkpnt{iteration}.pth"))
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
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians)
    gaussians.training_setup(args)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        current_scale = scene.resolution_scales[scene.scale_index]
        lt_scale_active = bool(args.enable_long_tail_branch and current_scale <= args.lt_activate_max_scale)
        gaussians.set_long_tail_active(lt_scale_active)

        if not viewpoint_stack:
            viewpoint_stack = _build_viewpoint_stack(scene, args)
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        other = [t_scale, v]

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3 * (np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None

        current_lt_point_mask = None
        if lt_scale_active and viewpoint_cam.lt_mask_conf is not None:
            current_lt_point_mask = gaussians.get_current_lt_point_mask(
                viewpoint_cam,
                mask_threshold=getattr(args, "lt_current_mask_threshold", 0.25),
                enable_long_tail=lt_scale_active,
            )

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            args,
            background,
            env_map=env_map,
            other=other,
            time_shift=time_shift,
            lt_point_mask=current_lt_point_mask,
            is_training=True,
        )

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        log_dict = {}

        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:]

        if args.enable_long_tail_branch and viewpoint_cam.lt_mask_conf is not None:
            gaussians.update_long_tail_stats(viewpoint_cam, visibility_filter, args.lt_ema_decay, args.lt_min_obs, args.lt_prob_threshold)

        sky_mask, lt_mask, lt_mask_conf, lt_mask_dilated, sky_only_mask, overlap_mask = _lt_masks_for_view(viewpoint_cam, alpha, args, lt_scale_active)

        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)
        if env_map is not None:
            if args.depth_blend_mode == 0:
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        gt_image = viewpoint_cam.original_image.cuda()

        alpha_lt = torch.zeros_like(alpha)
        loss_l1_lt = torch.zeros((), device=image.device)
        if lt_scale_active and viewpoint_cam.lt_mask_conf is not None:
            bg_mask = (1.0 - lt_mask).clamp_min(0.0)
            loss_l1_bg = _masked_l1(image, gt_image, bg_mask)
            if lt_mask.sum() > 0:
                loss_l1_lt = _masked_l1(image, gt_image, lt_mask)
                loss_l1 = loss_l1_bg + args.lt_rgb_weight * loss_l1_lt
            else:
                loss_l1 = loss_l1_bg

            lt_point_mask = current_lt_point_mask
            if lt_point_mask is None:
                lt_point_mask = gaussians.get_current_lt_point_mask(
                    viewpoint_cam,
                    visibility_filter,
                    getattr(args, "lt_current_mask_threshold", 0.25),
                    enable_long_tail=lt_scale_active,
                )
            if lt_point_mask.any():
                alpha_lt = render(
                    viewpoint_cam,
                    gaussians,
                    args,
                    background,
                    env_map=env_map,
                    mask=lt_point_mask,
                    lt_point_mask=lt_point_mask,
                    is_training=True,
                )["alpha"]
        else:
            loss_l1 = F.l1_loss(image, gt_image)
            loss_l1_bg = loss_l1

        log_dict['loss_l1'] = loss_l1.item()
        log_dict['loss_l1_lt'] = loss_l1_lt.item() if isinstance(loss_l1_lt, torch.Tensor) else float(loss_l1_lt)
        loss_ssim = 1.0 - ssim(image, gt_image)
        log_dict['loss_ssim'] = loss_ssim.item()
        loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if lt_scale_active and viewpoint_cam.lt_mask_conf is not None:
            a_pos = lt_mask_conf.sum()
            a_neg = float(lt_mask_conf.numel()) - a_pos
            w_pos = torch.clamp(a_neg / (a_pos + EPS), min=1.0, max=float(args.lt_positive_weight_cap))
            weight = 1.0 + (w_pos - 1.0) * lt_mask_conf
            loss_lt_mask = F.binary_cross_entropy(alpha_lt.clamp(EPS, 1.0 - EPS), lt_mask_conf, weight=weight) + _dice_loss(alpha_lt, lt_mask_conf)
            log_dict['loss_lt_mask'] = loss_lt_mask.item()
            loss = loss + args.lt_mask_weight * loss_lt_mask

            loss_lt_sparse, loss_lt_gate, loss_lt_smooth = gaussians.get_lt_regularization_losses(scene.time_interval)
            log_dict['loss_lt_sparse'] = loss_lt_sparse.item()
            log_dict['loss_lt_gate'] = loss_lt_gate.item()
            log_dict['loss_lt_smooth'] = loss_lt_smooth.item()
            loss = loss + args.lt_sparse_weight * loss_lt_sparse
            loss = loss + args.lt_gate_weight * loss_lt_gate
            loss = loss + args.lt_motion_smooth_weight * loss_lt_smooth
        else:
            log_dict['loss_lt_mask'] = 0.0
            log_dict['loss_lt_sparse'] = 0.0
            log_dict['loss_lt_gate'] = 0.0
            log_dict['loss_lt_smooth'] = 0.0

        if args.lambda_lidar > 0:
            assert viewpoint_cam.pts_depth is not None
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = pts_depth > 0
            loss_lidar = torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay) if args.lidar_decay > 0 else 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_t_reg > 0:
            loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(v_map).mean()
            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += args.lambda_v_reg * loss_v_reg

        if args.lambda_inv_depth > 0:
            inverse_depth = 1 / (depth + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
            log_dict['loss_inv_depth'] = loss_inv_depth.item()
            loss = loss + args.lambda_inv_depth * loss_inv_depth

        if args.lambda_v_smooth > 0:
            loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
            log_dict['loss_v_smooth'] = loss_v_smooth.item()
            loss = loss + args.lambda_v_smooth * loss_v_smooth

        if args.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            sky = sky_only_mask if lt_scale_active else sky_mask
            loss_sky_opa = (-sky * torch.log(1 - o)).sum() / sky.sum().clamp_min(EPS)
            log_dict['loss_sky_opa'] = loss_sky_opa.item()
            loss = loss + args.lambda_sky_opa * loss_sky_opa

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

        loss.backward()
        log_dict['loss'] = loss.item()
        log_dict['lt_flag_count'] = int(gaussians.get_lt_flag_mask.sum().item())
        log_dict['lt_prob_mean'] = gaussians._lt_prob.mean().item() if gaussians._lt_prob.numel() > 0 else 0.0
        log_dict['lt_active_scale'] = float(lt_scale_active)

        iter_end.record()

        with torch.no_grad():
            psnr_for_log = psnr(image, gt_image).double()
            log_dict["psnr"] = psnr_for_log
            for key in ['loss', 'loss_l1', 'psnr']:
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.{5}f}" for k in ['loss', 'loss_l1', 'psnr']}
                postfix["scale"] = current_scale
                postfix["lt"] = int(log_dict['lt_flag_count'])
                progress_bar.set_postfix(postfix)

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            complete_eval(tb_writer, iteration, args.test_iterations, scene, render, (args, background), log_dict, env_map=env_map)

            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            if iteration < args.densify_until_iter and (args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None
                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]
                    gaussians.densify_and_prune(args.densify_grad_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold, args.densify_grad_t_threshold)

                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if iteration % args.vis_step == 0 or iteration == 1:
                other_img = []
                feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                t_map = feature[0:1]
                v_map = feature[1:]
                v_norm_map = v_map.norm(dim=0, keepdim=True)

                other_img.append(visualize_depth(t_map, near=0.01, far=1))
                other_img.append(visualize_depth(v_norm_map, near=0.01, far=1))
                other_img.append(alpha_lt.repeat(3, 1, 1))
                other_img.append(lt_mask.repeat(3, 1, 1))
                other_img.append(sky_only_mask.repeat(3, 1, 1))
                other_img.append(overlap_mask.repeat(3, 1, 1))

                if viewpoint_cam.pts_depth is not None:
                    other_img.append(visualize_depth(viewpoint_cam.pts_depth))

                grid = make_grid([
                    image,
                    gt_image,
                    alpha.repeat(3, 1, 1),
                    torch.logical_not(sky_mask.bool()[:1]).float().repeat(3, 1, 1),
                    visualize_depth(depth),
                ] + other_img, nrow=4)
                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))

            if iteration % args.scale_increase_interval == 0:
                current_scale = scene.resolution_scales[scene.scale_index]
                next_scale = scene.resolution_scales[max(0, scene.scale_index - 1)]
                if float(current_scale) == 2.0 and float(next_scale) == 1.0:
                    _save_scale2_final_checkpoint(
                        args,
                        scene,
                        gaussians,
                        iteration,
                        env_map=env_map,
                        reason="before_upscale_to_scale1",
                    )
                scene.upScale()
                viewpoint_stack = None

            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if env_map is not None:
                    torch.save((env_map.capture(), iteration), scene.model_path + "/env_light_chkpnt" + str(iteration) + ".pth")

            if iteration == args.iterations:
                current_scale = scene.resolution_scales[scene.scale_index]
                if float(current_scale) == 2.0:
                    _save_scale2_final_checkpoint(
                        args,
                        scene,
                        gaussians,
                        iteration,
                        env_map=env_map,
                        reason="final_iteration_at_scale2",
                    )


def complete_eval(tb_writer, iteration, test_iterations, scene: Scene, renderFunc, renderArgs, log_dict, env_map=None):
    from lpipsPyTorch import lpips

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(f'train/{key}', value, iteration)

    if iteration in test_iterations:
        scale = scene.resolution_scales[scene.scale_index]
        scene.gaussians.set_long_tail_active(args.enable_long_tail_branch and scale <= args.lt_activate_max_scale)
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if "kitti" in args.model_path:
                num = len(scene.getTrainCameras()) // 2
                eval_train_frame = num // 5
                traincamera = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
                validation_configs = (
                    {'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                    {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:] + traincamera[num:][-eval_train_frame:]},
                )
            else:
                validation_configs = (
                    {'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                    {'name': 'train', 'cameras': scene.getTrainCameras()},
                )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir, exist_ok=True)
                export_train_video = (
                    config['name'] == 'train'
                    and iteration == args.iterations
                    and getattr(args, "save_train_render_video", False)
                )
                if export_train_video:
                    render_only_outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render_only")
                    video_outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_videos")
                    os.makedirs(render_only_outdir, exist_ok=True)
                    frame_paths_by_cam = defaultdict(list)
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    lt_point_mask = None
                    if args.enable_long_tail_branch and scale <= args.lt_activate_max_scale and viewpoint.lt_mask_conf is not None:
                        lt_point_mask = scene.gaussians.get_current_lt_point_mask(
                            viewpoint,
                            mask_threshold=getattr(args, "lt_current_mask_threshold", 0.25),
                            enable_long_tail=True,
                        )
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, lt_point_mask=lt_point_mask)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = render_pkg['depth']
                    alpha = render_pkg['alpha']
                    sky_depth = 900
                    depth = depth / alpha.clamp_min(EPS)
                    if env_map is not None:
                        if args.depth_blend_mode == 0:
                            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                        elif args.depth_blend_mode == 1:
                            depth = alpha * depth + (1 - alpha) * sky_depth

                    depth = visualize_depth(depth)
                    alpha = alpha.repeat(3, 1, 1)

                    grid = [gt_image, image, alpha, depth]
                    grid = make_grid(grid, nrow=2)
                    save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))
                    if export_train_video:
                        cam_name = _infer_camera_name(viewpoint, args.cam_num)
                        cam_outdir = os.path.join(render_only_outdir, cam_name)
                        os.makedirs(cam_outdir, exist_ok=True)
                        frame_name = viewpoint.image_name if viewpoint.image_name is not None else f"{idx:06d}"
                        frame_path = os.path.join(cam_outdir, f"{frame_name}.png")
                        save_image(image, frame_path)
                        frame_paths_by_cam[cam_name].append(frame_path)

                    l1_test += F.l1_loss(image, gt_image).double()
                    psnr_test += psnr(image, gt_image).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item(), "ssim": ssim_test.item(), "lpips": lpips_test.item()}, f)
                if export_train_video:
                    _write_camera_videos(frame_paths_by_cam, video_outdir, getattr(args, "train_render_video_fps", 10))
                    print(f"[ITER {iteration}] Saved per-camera train videos to {video_outdir}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
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

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0, args.iterations, args.test_interval)]

    print("Optimizing " + args.model_path)
    seed_everything(args.seed)
    training(args)
    print("\nTraining complete.")
