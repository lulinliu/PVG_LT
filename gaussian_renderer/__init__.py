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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=None, mask=None, lt_point_mask=None, is_training=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if other is None:
        other = []

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    if pipe.neg_fov:
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D, opacity, visibility = pc.get_render_state(
        viewpoint_camera.timestamp,
        time_shift=time_shift,
        enable_long_tail=pc.long_tail_active,
        lt_point_mask=lt_point_mask,
    )
    means2D = screenspace_points
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if len(other) > 0:
        features = torch.cat(other, dim=1)
        s_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        s_other = 0

    if mask is None:
        point_mask = visibility[:, 0] > 0.05
    else:
        point_mask = mask & (visibility[:, 0] > 0.05)

    masked_means3D = means3D[point_mask]
    masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
    masked_depth = masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3]
    depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
    depth_alpha[point_mask] = torch.cat([masked_depth, torch.ones_like(masked_depth)], dim=1)
    features = torch.cat([features, depth_alpha], dim=1)

    contrib, rendered_image, rendered_feature, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=point_mask,
    )

    rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([s_other, 1, 1], dim=0)
    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map(viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

    return {
        "render": rendered_image,
        "render_nobg": rendered_image_before,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "contrib": contrib,
        "depth": rendered_depth,
        "alpha": rendered_opacity,
        "feature": rendered_other,
    }
