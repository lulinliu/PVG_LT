import os
import math

import torch
from torch import nn
from omegaconf import OmegaConf

from gaussian_renderer import render
from utils.general_utils import inverse_sigmoid


def make_branch_args(args, **updates):
    branch_args = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    for key, value in updates.items():
        branch_args[key] = value
    return branch_args


def initialize_hierarchical_pools(scene, bg_gaussians, lt_gaussians, args):
    lt_gaussians.clone_from(bg_gaussians)
    if lt_gaussians.get_xyz.shape[0] == 0:
        bg_gaussians.long_tail_enabled = False
        return

    support = torch.zeros(bg_gaussians.get_xyz.shape[0], dtype=torch.bool, device=bg_gaussians.get_xyz.device)
    support_count = torch.zeros(bg_gaussians.get_xyz.shape[0], dtype=torch.int32, device=bg_gaussians.get_xyz.device)
    best_score = torch.zeros(bg_gaussians.get_xyz.shape[0], dtype=bg_gaussians.get_xyz.dtype, device=bg_gaussians.get_xyz.device)

    for viewpoint in scene.getTrainCameras():
        if viewpoint.lt_mask_conf is None or viewpoint.lt_mask_conf.max().item() <= 0:
            continue
        sampled, valid = lt_gaussians._sample_current_lt_mask(
            viewpoint,
            enable_long_tail=False,
        )
        if sampled is None:
            continue
        sampled = sampled.clamp_min(0.0)
        best_score = torch.maximum(best_score, sampled)
        positive = valid & (sampled > float(getattr(args, "lt_pool_init_threshold", 0.5)))
        support |= positive
        support_count[positive] += 1

    min_views = max(1, int(getattr(args, "lt_init_min_views", 1)))
    support = support & (support_count >= min_views)

    scale_quantile = float(getattr(args, "lt_init_scale_quantile", 1.0))
    if support.any() and scale_quantile < 0.999:
        candidate_scales = bg_gaussians.get_scaling.max(dim=1).values
        threshold = torch.quantile(candidate_scales[support], torch.tensor(scale_quantile, device=candidate_scales.device))
        support = support & (candidate_scales <= threshold)

    if not support.any():
        candidate_count = min(int(getattr(args, "lt_pool_fallback_topk", 4096)), best_score.shape[0])
        if candidate_count > 0:
            topk = torch.topk(best_score, k=candidate_count, largest=True).indices
            support[topk] = True

    bg_keep = ~support
    if not bg_keep.any():
        bg_keep[torch.argmin(best_score)] = True
    if not support.any():
        support[torch.argmax(best_score)] = True

    bg_gaussians.hard_keep_points(bg_keep)
    bg_gaussians.long_tail_enabled = False
    bg_gaussians.long_tail_active = False
    bg_gaussians.mark_all_long_tail(False)

    lt_gaussians.hard_keep_points(support)
    lt_gaussians.long_tail_enabled = True
    lt_gaussians.long_tail_active = True
    lt_gaussians.lt_frame_local_only = False
    lt_gaussians.reset_temporal_support(
        t_init=getattr(args, "lt_branch_t_init", getattr(args, "t_init", 0.02)),
        gate_span_factor=getattr(args, "lt_branch_gate_span_factor", 1.0),
    )
    if bool(getattr(args, "lt_reinit_features", False)):
        lt_gaussians._features_dc = nn.Parameter(
            torch.full_like(lt_gaussians._features_dc, float(getattr(args, "lt_reinit_dc_value", 0.0))).detach().clone().requires_grad_(True)
        )
        lt_gaussians._features_rest = nn.Parameter(
            torch.zeros_like(lt_gaussians._features_rest).detach().clone().requires_grad_(True)
        )
        init_opacity = float(getattr(args, "lt_init_opacity", 0.01))
        lt_gaussians._opacity = nn.Parameter(
            inverse_sigmoid(
                torch.full_like(lt_gaussians.get_opacity, init_opacity)
            ).detach().clone().requires_grad_(True)
        )
        shrink_factor = float(getattr(args, "lt_init_scale_shrink", 1.0))
        if shrink_factor < 0.999:
            lt_gaussians._scaling = nn.Parameter(
                (lt_gaussians._scaling.detach() + math.log(max(shrink_factor, 1e-4))).clone().requires_grad_(True)
            )
    lt_gaussians.mark_all_long_tail(True)


def save_checkpoint_bundle(path, bg_gaussians, iteration, lt_gaussians=None):
    payload = (bg_gaussians.capture(), iteration)
    if lt_gaussians is not None:
        payload = {
            "hierarchical_longtail": True,
            "bg": bg_gaussians.capture(),
            "lt": lt_gaussians.capture(),
            "iteration": int(iteration),
        }
    torch.save(payload, path)


def load_checkpoint_bundle(path):
    payload = torch.load(path)
    if isinstance(payload, dict) and payload.get("hierarchical_longtail"):
        return payload
    model_params, iteration = payload
    return {
        "hierarchical_longtail": False,
        "bg": model_params,
        "lt": None,
        "iteration": iteration,
    }


def _zero_render_pkg(viewpoint_camera, bg_color, other_channels=0):
    h = int(viewpoint_camera.image_height)
    w = int(viewpoint_camera.image_width)
    device = bg_color.device
    dtype = bg_color.dtype
    return {
        "render": torch.zeros((3, h, w), device=device, dtype=dtype),
        "render_nobg": torch.zeros((3, h, w), device=device, dtype=dtype),
        "alpha": torch.zeros((1, h, w), device=device, dtype=dtype),
        "depth": torch.zeros((1, h, w), device=device, dtype=dtype),
        "feature": torch.zeros((other_channels, h, w), device=device, dtype=dtype),
        "viewspace_points": torch.zeros((0, 3), device=device, dtype=dtype, requires_grad=True),
        "visibility_filter": torch.zeros((0,), device=device, dtype=torch.bool),
        "radii": torch.zeros((0,), device=device, dtype=dtype),
        "contrib": None,
    }


def render_hierarchical(
    viewpoint_camera,
    bg_gaussians,
    lt_gaussians,
    pipe,
    bg_color,
    env_map=None,
    bg_other=None,
    lt_other=None,
    bg_time_shift=None,
    lt_time_shift=None,
    lt_enabled=True,
    is_training=False,
):
    bg_other = [] if bg_other is None else bg_other
    lt_other = [] if lt_other is None else lt_other

    bg_gaussians.set_long_tail_active(False)
    bg_pkg = render(
        viewpoint_camera,
        bg_gaussians,
        pipe,
        bg_color,
        env_map=env_map,
        other=bg_other,
        time_shift=bg_time_shift,
        is_training=is_training,
    )

    if lt_enabled and lt_gaussians is not None and lt_gaussians.get_xyz.shape[0] > 0:
        lt_gaussians.set_long_tail_active(True)
        lt_point_mask = torch.ones((lt_gaussians.get_xyz.shape[0],), dtype=torch.bool, device=lt_gaussians.get_xyz.device)
        lt_pkg = render(
            viewpoint_camera,
            lt_gaussians,
            pipe,
            torch.zeros_like(bg_color),
            env_map=None,
            other=lt_other,
            time_shift=lt_time_shift,
            lt_point_mask=lt_point_mask,
            is_training=is_training,
        )
    else:
        lt_pkg = _zero_render_pkg(viewpoint_camera, bg_color, sum(t.shape[0] for t in lt_other))

    if viewpoint_camera.lt_mask_conf is None:
        lt_mask = torch.zeros_like(bg_pkg["alpha"])
    else:
        lt_mask = viewpoint_camera.lt_mask_conf.cuda().float().clamp(0.0, 1.0)

    lt_composite = lt_pkg["render"] + (1.0 - lt_pkg["alpha"]) * bg_pkg["render"]
    final_image = (1.0 - lt_mask) * bg_pkg["render"] + lt_mask * lt_composite
    final_alpha = (1.0 - lt_mask) * bg_pkg["alpha"] + lt_mask * torch.clamp(
        lt_pkg["alpha"] + (1.0 - lt_pkg["alpha"]) * bg_pkg["alpha"],
        0.0,
        1.0,
    )
    final_depth = (1.0 - lt_mask) * bg_pkg["depth"] + lt_mask * (
        torch.where(lt_pkg["alpha"] > 0.0, lt_pkg["depth"], bg_pkg["depth"])
    )
    final_feature = bg_pkg["feature"]

    return {
        "final": {
            "render": final_image,
            "alpha": final_alpha,
            "depth": final_depth,
            "feature": final_feature,
            "render_nobg": final_image,
        },
        "bg": bg_pkg,
        "lt": lt_pkg,
        "lt_mask": lt_mask,
    }


def save_debug_overlay(path, gt_image, lt_mask, bg_alpha, lt_alpha):
    overlay = torch.cat(
        [
            gt_image,
            lt_mask.repeat(3, 1, 1),
            bg_alpha.repeat(3, 1, 1),
            lt_alpha.repeat(3, 1, 1),
        ],
        dim=2,
    )
    from torchvision.utils import save_image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(overlay, path)
