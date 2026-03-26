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
import math
import torch
import numpy as np
import torch.nn.functional as F
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self._velocity = torch.empty(0)
        self._lt_basis = torch.empty(0)
        self._lt_on = torch.empty(0)
        self._lt_off = torch.empty(0)
        self._lt_beta_on = torch.empty(0)
        self._lt_beta_off = torch.empty(0)
        self._lt_prob = torch.empty(0)
        self._lt_obs_count = torch.empty(0)
        self._lt_flag = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.t_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.time_duration = args.time_duration
        self.no_time_split = args.no_time_split
        self.t_grad = args.t_grad
        self.contract = args.contract
        self.t_init = args.t_init
        self.big_point_threshold = args.big_point_threshold

        self.T = args.cycle
        self.velocity_decay = args.velocity_decay
        self.random_init_point = args.random_init_point

        self.long_tail_enabled = bool(getattr(args, "enable_long_tail_branch", True))
        self.long_tail_active = False
        self.lt_frame_local_only = bool(getattr(args, "lt_frame_local_only", False))
        self.lt_prob_threshold = float(getattr(args, "lt_prob_threshold", 0.6))
        self.lt_ema_decay = float(getattr(args, "lt_ema_decay", 0.9))
        self.lt_min_obs = int(getattr(args, "lt_min_obs", 5))
        self.lt_current_mask_threshold = float(getattr(args, "lt_current_mask_threshold", 0.25))
        self.lt_min_obs_before_prune = int(getattr(args, "lt_min_obs_before_prune", 10))
        self.lt_prune_opacity_factor = float(getattr(args, "lt_prune_opacity_factor", 0.5))
        self.lt_densify_grad_factor = float(getattr(args, "lt_densify_grad_factor", 0.5))
        self.lt_gate_max_span_factor = float(getattr(args, "lt_gate_max_span_factor", 4.0))

        self.setup_functions()

    def _beta_inverse(self, value):
        return torch.log(value.clamp_min(1e-6))

    def _beta_activation(self, value):
        return torch.exp(value).clamp_min(1e-6)

    def _lt_default_window(self):
        if self._t.numel() == 0:
            return torch.empty(0, device="cuda")
        base_window = self.get_scaling_t.detach().clamp_min(1e-3)
        min_window = max((self.time_duration[1] - self.time_duration[0]) * 0.05, 1e-3)
        return base_window.clamp_min(min_window)

    def _ensure_lt_state_from_existing(self):
        if self._xyz.numel() == 0:
            return
        num_points = self._xyz.shape[0]
        device = self._xyz.device
        dtype = self._xyz.dtype
        window = self._lt_default_window()
        if window.numel() == 0:
            window = torch.full((num_points, 1), 0.05, device=device, dtype=dtype)
        beta_init = (0.25 * window).clamp_min(1e-3)

        if self._lt_basis.numel() != num_points * 18:
            self._lt_basis = nn.Parameter(torch.zeros((num_points, 3, 6), device=device, dtype=dtype).requires_grad_(True))
        if self._lt_on.numel() != num_points:
            self._lt_on = nn.Parameter((self._t.detach() - window).clone().requires_grad_(True))
        if self._lt_off.numel() != num_points:
            self._lt_off = nn.Parameter((self._t.detach() + window).clone().requires_grad_(True))
        if self._lt_beta_on.numel() != num_points:
            self._lt_beta_on = nn.Parameter(self._beta_inverse(beta_init).clone().requires_grad_(True))
        if self._lt_beta_off.numel() != num_points:
            self._lt_beta_off = nn.Parameter(self._beta_inverse(beta_init).clone().requires_grad_(True))
        if self._lt_prob.numel() != num_points:
            self._lt_prob = torch.zeros((num_points, 1), device=device, dtype=dtype)
        if self._lt_obs_count.numel() != num_points:
            self._lt_obs_count = torch.zeros((num_points, 1), device=device, dtype=dtype)
        if self._lt_flag.numel() != num_points:
            self._lt_flag = torch.zeros((num_points, 1), device=device, dtype=dtype)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t,
            self._scaling_t,
            self._velocity,
            self._lt_basis,
            self._lt_on,
            self._lt_off,
            self._lt_beta_on,
            self._lt_beta_off,
            self._lt_prob,
            self._lt_obs_count,
            self._lt_flag,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.t_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.T,
            self.velocity_decay,
        )

    def restore(self, model_args, training_args=None):
        if len(model_args) >= 26:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._t,
                self._scaling_t,
                self._velocity,
                self._lt_basis,
                self._lt_on,
                self._lt_off,
                self._lt_beta_on,
                self._lt_beta_off,
                self._lt_prob,
                self._lt_obs_count,
                self._lt_flag,
                self.max_radii2D,
                xyz_gradient_accum,
                t_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.T,
                self.velocity_decay,
            ) = model_args
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._t,
                self._scaling_t,
                self._velocity,
                self.max_radii2D,
                xyz_gradient_accum,
                t_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.T,
                self.velocity_decay,
            ) = model_args
            self._lt_basis = torch.empty(0, device=self._xyz.device)
            self._lt_on = torch.empty(0, device=self._xyz.device)
            self._lt_off = torch.empty(0, device=self._xyz.device)
            self._lt_beta_on = torch.empty(0, device=self._xyz.device)
            self._lt_beta_off = torch.empty(0, device=self._xyz.device)
            self._lt_prob = torch.empty(0, device=self._xyz.device)
            self._lt_obs_count = torch.empty(0, device=self._xyz.device)
            self._lt_flag = torch.empty(0, device=self._xyz.device)
        self.setup_functions()
        self._ensure_lt_state_from_existing()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_t_activation(self._scaling_t)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz_SHM(self, t):
        a = 1 / self.T * np.pi * 2
        return self._xyz + self._velocity * torch.sin((t - self._t) * a) / a

    @property
    def get_inst_velocity(self):
        return self._velocity * torch.exp(-self.get_scaling_t / self.T / 2 * self.velocity_decay)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    @property
    def get_lt_flag_mask(self):
        if self._lt_flag.numel() == 0:
            return torch.zeros((self.get_xyz.shape[0],), device=self.get_xyz.device, dtype=torch.bool)
        return self._lt_flag[:, 0] > 0.5

    @property
    def get_lt_on(self):
        return torch.minimum(self._lt_on, self._lt_off)

    @property
    def get_lt_off(self):
        return torch.maximum(self._lt_on, self._lt_off)

    @property
    def get_lt_beta_on(self):
        return self._beta_activation(self._lt_beta_on)

    @property
    def get_lt_beta_off(self):
        return self._beta_activation(self._lt_beta_off)

    def set_long_tail_active(self, active):
        self.long_tail_active = bool(self.long_tail_enabled and active)

    def get_marginal_t(self, timestamp):
        return torch.exp(-0.5 * (self.get_t - timestamp) ** 2 / self.get_scaling_t.clamp_min(1e-6) ** 2)

    def _expand_timestamp(self, timestamp):
        ts = torch.as_tensor(timestamp, device=self._t.device, dtype=self._t.dtype)
        if ts.ndim == 0:
            ts = torch.full_like(self._t, float(ts.item()))
        return ts.reshape_as(self._t)

    def _lt_basis_values(self, dt):
        omega1 = 2 * math.pi / self.T
        omega2 = 4 * math.pi / self.T
        return torch.cat([
            dt,
            dt ** 2,
            torch.sin(omega1 * dt),
            torch.cos(omega1 * dt),
            torch.sin(omega2 * dt),
            torch.cos(omega2 * dt),
        ], dim=1)

    def _lt_offset(self, dt):
        basis = self._lt_basis_values(dt)
        return torch.einsum('ndk,nk->nd', self._lt_basis, basis)

    def get_lt_xyz(self, timestamp):
        dt = self._expand_timestamp(timestamp) - self._t
        return self._xyz + self._lt_offset(dt)

    def get_lt_gate(self, timestamp):
        ts = self._expand_timestamp(timestamp)
        gate_on = torch.sigmoid((ts - self.get_lt_on) / self.get_lt_beta_on)
        gate_off = torch.sigmoid((self.get_lt_off - ts) / self.get_lt_beta_off)
        return gate_on * gate_off

    def get_render_state(self, timestamp, time_shift=None, enable_long_tail=None, lt_point_mask=None):
        if enable_long_tail is None:
            enable_long_tail = self.long_tail_active
        query_time = timestamp if time_shift is None else timestamp - time_shift
        base_means = self.get_xyz_SHM(query_time)
        if time_shift is not None:
            base_means = base_means + self.get_inst_velocity * time_shift
        base_visibility = self.get_marginal_t(query_time)
        opacity = self.get_opacity

        if enable_long_tail and self.long_tail_enabled and self._lt_flag.numel() > 0:
            if lt_point_mask is None:
                if self.lt_frame_local_only:
                    lt_mask = None
                else:
                    lt_mask = self.get_lt_flag_mask[:, None]
            else:
                lt_mask = lt_point_mask[:, None].bool()

            if lt_mask is not None:
                lt_means = self.get_lt_xyz(query_time)
                lt_visibility = self.get_lt_gate(query_time)
                means = torch.where(lt_mask.expand(-1, 3), lt_means, base_means)
                visibility = torch.where(lt_mask, lt_visibility, base_visibility)
            else:
                means = base_means
                visibility = base_visibility
        else:
            means = base_means
            visibility = base_visibility

        return means, opacity * visibility, visibility

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        r_max = 100000
        r_min = 2
        num_sph = self.random_init_point

        theta = 2 * torch.pi * torch.rand(num_sph)
        phi = (torch.pi / 2 * 0.99 * torch.rand(num_sph)) ** 1.5
        s = torch.rand(num_sph)
        r_1 = s * 1 / r_min + (1 - s) * 1 / r_max
        r = 1 / r_1
        pts_sph = torch.stack([r * torch.cos(theta) * torch.cos(phi), r * torch.sin(theta) * torch.cos(phi), r * torch.sin(phi)], dim=-1).cuda()

        r_rec = r_min
        num_rec = self.random_init_point
        pts_rec = torch.stack([
            r_rec * (torch.rand(num_rec) - 0.5),
            r_rec * (torch.rand(num_rec) - 0.5),
            r_rec * (torch.rand(num_rec)),
        ], dim=-1).cuda()

        pts_sph = torch.cat([pts_rec, pts_sph], dim=0)
        pts_sph[:, 2] = -pts_sph[:, 2] + 1

        fused_point_cloud = torch.cat([fused_point_cloud, pts_sph], dim=0)
        features = torch.cat([
            features,
            torch.zeros([pts_sph.size(0), features.size(1), features.size(2)]).float().cuda(),
        ], dim=0)

        if pcd.time is None or pcd.time.shape[0] != fused_point_cloud.shape[0]:
            if pcd.time is None:
                time = (np.random.rand(pcd.points.shape[0], 1) * 1.2 - 0.1) * (
                    self.time_duration[1] - self.time_duration[0]
                ) + self.time_duration[0]
            else:
                time = pcd.time

            if self.t_init < 1:
                random_times = (torch.rand(fused_point_cloud.shape[0] - pcd.points.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                    self.time_duration[1] - self.time_duration[0]
                ) + self.time_duration[0]
                pts_times = torch.from_numpy(time.copy()).float().cuda()
                fused_times = torch.cat([pts_times, random_times], dim=0)
            else:
                fused_times = torch.full_like(
                    fused_point_cloud[..., :1],
                    0.5 * (self.time_duration[1] + self.time_duration[0]),
                )
        else:
            fused_times = torch.from_numpy(np.asarray(pcd.time.copy())).cuda().float()
            fused_times_sh = torch.full_like(pts_sph[..., :1], 0.5 * (self.time_duration[1] + self.time_duration[0]))
            fused_times = torch.cat([fused_times, fused_times_sh], dim=0)

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        dist_t = torch.full_like(fused_times, (self.time_duration[1] - self.time_duration[0]) * self.t_init)
        scales_t = self.scaling_t_inverse_activation(torch.sqrt(dist_t))
        velocity = torch.full((fused_point_cloud.shape[0], 3), 0.0, device="cuda")
        opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), device="cuda")
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._velocity = nn.Parameter(velocity.requires_grad_(True))
        self._ensure_lt_state_from_existing()

    def training_setup(self, training_args):
        self._ensure_lt_state_from_existing()
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        lr_scale = training_args.velocity_lr * self.spatial_lr_scale
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._t], 'lr': training_args.t_lr_init, "name": "t"},
            {'params': [self._scaling_t], 'lr': training_args.scaling_t_lr, "name": "scaling_t"},
            {'params': [self._velocity], 'lr': lr_scale, "name": "velocity"},
            {'params': [self._lt_basis], 'lr': lr_scale, "name": "lt_basis"},
            {'params': [self._lt_on], 'lr': training_args.t_lr_init, "name": "lt_on"},
            {'params': [self._lt_off], 'lr': training_args.t_lr_init, "name": "lt_off"},
            {'params': [self._lt_beta_on], 'lr': training_args.scaling_t_lr, "name": "lt_beta_on"},
            {'params': [self._lt_beta_off], 'lr': training_args.scaling_t_lr, "name": "lt_beta_off"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        final_decay = training_args.position_lr_final / training_args.position_lr_init
        self.t_scheduler_args = get_expon_lr_func(
            lr_init=training_args.t_lr_init,
            lr_final=training_args.t_lr_init * final_decay,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = self.xyz_scheduler_args(iteration)
            if param_group["name"] in {"t", "lt_on", "lt_off"}:
                param_group['lr'] = self.t_scheduler_args(iteration)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        self._velocity = optimizable_tensors['velocity']
        self._lt_basis = optimizable_tensors['lt_basis']
        self._lt_on = optimizable_tensors['lt_on']
        self._lt_off = optimizable_tensors['lt_off']
        self._lt_beta_on = optimizable_tensors['lt_beta_on']
        self._lt_beta_off = optimizable_tensors['lt_beta_off']

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._lt_prob = self._lt_prob[valid_points_mask]
        self._lt_obs_count = self._lt_obs_count[valid_points_mask]
        self._lt_flag = self._lt_flag[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_t,
        new_scaling_t,
        new_velocity,
        new_lt_basis,
        new_lt_on,
        new_lt_off,
        new_lt_beta_on,
        new_lt_beta_off,
        new_lt_prob,
        new_lt_obs_count,
        new_lt_flag,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "t": new_t,
            "scaling_t": new_scaling_t,
            "velocity": new_velocity,
            "lt_basis": new_lt_basis,
            "lt_on": new_lt_on,
            "lt_off": new_lt_off,
            "lt_beta_on": new_lt_beta_on,
            "lt_beta_off": new_lt_beta_off,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        self._velocity = optimizable_tensors['velocity']
        self._lt_basis = optimizable_tensors['lt_basis']
        self._lt_on = optimizable_tensors['lt_on']
        self._lt_off = optimizable_tensors['lt_off']
        self._lt_beta_on = optimizable_tensors['lt_beta_on']
        self._lt_beta_off = optimizable_tensors['lt_beta_off']

        self._lt_prob = torch.cat([self._lt_prob, new_lt_prob], dim=0)
        self._lt_obs_count = torch.cat([self._lt_obs_count, new_lt_obs_count], dim=0)
        self._lt_flag = torch.cat([self._lt_flag, new_lt_flag], dim=0)

        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _point_grad_threshold(self, base_threshold, n_points):
        thresholds = torch.full((n_points,), float(base_threshold), device="cuda")
        if self.long_tail_enabled and (not self.lt_frame_local_only) and self._lt_flag.numel() == n_points:
            thresholds[self.get_lt_flag_mask] *= self.lt_densify_grad_factor
        return thresholds

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2, time_split=False,
                          joint_sample=True):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        grad_thresholds = self._point_grad_threshold(grad_threshold, n_init_points)
        selected_pts_mask = padded_grad >= grad_thresholds

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1) * scene_extent - 1
            scale_factor = torch.where(scale_factor <= 1, 1, scale_factor) / scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:, 0] / scene_extent

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent * scale_factor,
        )
        decay_factor = N * 0.8
        if not self.no_time_split:
            N = N + 1

        if time_split:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_time_mask = torch.where(padded_grad_t >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense
            selected_time_mask = torch.logical_and(selected_time_mask, torch.max(self.get_scaling_t, dim=1).values > extend_thresh)
            if joint_sample:
                selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / decay_factor)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        xyz = self.get_xyz[selected_pts_mask]
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
        means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
        samples_t = torch.normal(mean=means_t, std=stds_t)
        new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
        new_scaling_t = self.scaling_t_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N, 1) / decay_factor)
        new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyz + self.get_inst_velocity[selected_pts_mask].repeat(N, 1) * samples_t

        not_split_xyz_mask = torch.max(self.get_scaling[selected_pts_mask], dim=1).values < self.percent_dense * scene_extent * scale_factor[selected_pts_mask]
        new_scaling[not_split_xyz_mask.repeat(N)] = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1))[not_split_xyz_mask.repeat(N)]

        if time_split:
            not_split_t_mask = self.get_scaling_t[selected_pts_mask].squeeze() < extend_thresh
            new_scaling_t[not_split_t_mask.repeat(N)] = self.scaling_t_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N, 1))[not_split_t_mask.repeat(N)]

        if self.no_time_split:
            new_scaling_t = self.scaling_t_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N, 1))

        new_lt_basis = self._lt_basis[selected_pts_mask].repeat(N, 1, 1)
        new_lt_on = self._lt_on[selected_pts_mask].repeat(N, 1) + samples_t
        new_lt_off = self._lt_off[selected_pts_mask].repeat(N, 1) + samples_t
        new_lt_beta_on = self._lt_beta_on[selected_pts_mask].repeat(N, 1)
        new_lt_beta_off = self._lt_beta_off[selected_pts_mask].repeat(N, 1)
        new_lt_prob = self._lt_prob[selected_pts_mask].repeat(N, 1)
        new_lt_obs_count = self._lt_obs_count[selected_pts_mask].repeat(N, 1)
        new_lt_flag = self._lt_flag[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_velocity,
            new_lt_basis,
            new_lt_on,
            new_lt_off,
            new_lt_beta_on,
            new_lt_beta_off,
            new_lt_prob,
            new_lt_obs_count,
            new_lt_flag,
        )
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, time_clone=False):
        if self.contract:
            scale_factor = self._xyz.norm(dim=-1) * scene_extent - 1
            scale_factor = torch.where(scale_factor <= 1, 1, scale_factor) / scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:, 0] / scene_extent

        grad_thresholds = self._point_grad_threshold(grad_threshold, self.get_xyz.shape[0])
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_thresholds
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent * scale_factor)
        if time_clone:
            selected_time_mask = torch.where(torch.norm(grads_t, dim=-1) >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense
            selected_time_mask = torch.logical_and(selected_time_mask, torch.max(self.get_scaling_t, dim=1).values <= extend_thresh)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)

        self.densification_postfix(
            self._xyz[selected_pts_mask],
            self._features_dc[selected_pts_mask],
            self._features_rest[selected_pts_mask],
            self._opacity[selected_pts_mask],
            self._scaling[selected_pts_mask],
            self._rotation[selected_pts_mask],
            self._t[selected_pts_mask],
            self._scaling_t[selected_pts_mask],
            self._velocity[selected_pts_mask],
            self._lt_basis[selected_pts_mask],
            self._lt_on[selected_pts_mask],
            self._lt_off[selected_pts_mask],
            self._lt_beta_on[selected_pts_mask],
            self._lt_beta_off[selected_pts_mask],
            self._lt_prob[selected_pts_mask],
            self._lt_obs_count[selected_pts_mask],
            self._lt_flag[selected_pts_mask],
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            grads_t = self.t_gradient_accum / self.denom
            grads_t[grads_t.isnan()] = 0.0

            if self.t_grad:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=True)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=True)
            else:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=False)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=False)

        min_opacity_tensor = torch.full_like(self.get_opacity[:, 0], float(min_opacity))
        if self.long_tail_enabled and (not self.lt_frame_local_only) and self._lt_flag.numel() == self.get_xyz.shape[0]:
            min_opacity_tensor[self.get_lt_flag_mask] *= self.lt_prune_opacity_factor
        prune_mask = self.get_opacity[:, 0] < min_opacity_tensor

        tentative_lt = torch.zeros_like(prune_mask)
        if not self.lt_frame_local_only:
            tentative_lt = (self._lt_obs_count[:, 0] < self.lt_min_obs_before_prune) & (self._lt_prob[:, 0] > 0)
        prune_mask = prune_mask & (~tentative_lt)

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1) * extent - 1
            scale_factor = torch.where(scale_factor <= 1, 1, scale_factor) / extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:, 0] / extent

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.t_gradient_accum[update_filter] += self._t.grad.clone()[update_filter]

    @torch.no_grad()
    def _sample_current_lt_mask(self, viewpoint_camera, visibility_filter=None, enable_long_tail=None):
        mask_conf = viewpoint_camera.lt_mask_conf if viewpoint_camera.lt_mask_conf is not None else viewpoint_camera.lt_mask
        if mask_conf is None or self.get_xyz.shape[0] == 0:
            return None, torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)

        if visibility_filter is None:
            visibility_filter = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)

        means3D, _, _ = self.get_render_state(viewpoint_camera.timestamp, enable_long_tail=enable_long_tail)
        xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
        cam_x = xyz_homo @ viewpoint_camera.world_view_transform[:, 0:1]
        cam_y = xyz_homo @ viewpoint_camera.world_view_transform[:, 1:2]
        cam_z = xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3]

        u = cam_x / cam_z.clamp_min(1e-6) * viewpoint_camera.fx + viewpoint_camera.cx
        v = cam_y / cam_z.clamp_min(1e-6) * viewpoint_camera.fy + viewpoint_camera.cy
        h = viewpoint_camera.image_height
        w = viewpoint_camera.image_width
        valid = visibility_filter & (cam_z[:, 0] > 1e-6) & (u[:, 0] >= 0) & (u[:, 0] <= w - 1) & (v[:, 0] >= 0) & (v[:, 0] <= h - 1)
        if not valid.any():
            return torch.zeros(means3D.shape[0], device=means3D.device, dtype=means3D.dtype), valid

        grid = torch.zeros((1, means3D.shape[0], 1, 2), device=means3D.device, dtype=means3D.dtype)
        grid[0, :, 0, 0] = (u[:, 0] / max(w - 1, 1)) * 2 - 1
        grid[0, :, 0, 1] = (v[:, 0] / max(h - 1, 1)) * 2 - 1
        sampled = F.grid_sample(mask_conf[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True).view(-1)
        return sampled, valid

    @torch.no_grad()
    def get_current_lt_point_mask(self, viewpoint_camera, visibility_filter=None, mask_threshold=None, enable_long_tail=None):
        threshold = self.lt_current_mask_threshold if mask_threshold is None else float(mask_threshold)
        sampled, valid = self._sample_current_lt_mask(
            viewpoint_camera,
            visibility_filter,
            enable_long_tail=self.long_tail_active if enable_long_tail is None else enable_long_tail,
        )
        if sampled is None:
            return torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        return valid & (sampled > threshold)

    @torch.no_grad()
    def update_long_tail_stats(self, viewpoint_camera, visibility_filter, ema_decay=None, min_obs=None, prob_threshold=None):
        if not self.long_tail_enabled:
            return
        if self.get_xyz.shape[0] == 0:
            return
        if self.lt_frame_local_only:
            if self._lt_obs_count.numel() == self.get_xyz.shape[0]:
                self._lt_obs_count.zero_()
            if self._lt_prob.numel() == self.get_xyz.shape[0]:
                self._lt_prob.zero_()
            if self._lt_flag.numel() == self.get_xyz.shape[0]:
                self._lt_flag.zero_()
            return

        ema_decay = self.lt_ema_decay if ema_decay is None else ema_decay
        min_obs = self.lt_min_obs if min_obs is None else min_obs
        prob_threshold = self.lt_prob_threshold if prob_threshold is None else prob_threshold
        self._lt_obs_count.mul_(ema_decay)
        self._lt_prob.mul_(ema_decay)

        sampled, valid = self._sample_current_lt_mask(
            viewpoint_camera,
            visibility_filter,
            enable_long_tail=self.long_tail_active,
        )
        if sampled is not None and valid.any():
            positive = valid & (sampled > self.lt_current_mask_threshold)
            self._lt_obs_count[positive] += 1.0
            self._lt_prob[valid] += (1.0 - ema_decay) * sampled[valid, None]
        self._lt_flag = ((self._lt_obs_count >= float(min_obs)) & (self._lt_prob >= float(prob_threshold))).float()

    def get_lt_regularization_losses(self, time_interval):
        mask = self.get_lt_flag_mask
        zero = torch.zeros((), device=self.get_xyz.device)
        if mask.sum() == 0:
            return zero, zero, zero

        sparse = self._lt_basis[mask].abs().mean()
        gate_span = self.get_lt_off[mask] - self.get_lt_on[mask]
        min_span = max(float(time_interval), 1e-3)
        max_span = max(min_span, float(self.lt_gate_max_span_factor) * min_span)
        gate_short = torch.relu(min_span - gate_span).mean()
        gate_long = torch.relu(gate_span - max_span).mean()
        gate = gate_short + gate_long + 0.1 * (self.get_lt_beta_on[mask] + self.get_lt_beta_off[mask]).mean()

        dt = max(float(time_interval), 1e-3)
        prev_xyz = self._xyz[mask] + self._lt_offset(torch.full_like(self._t, -dt))[mask]
        curr_xyz = self._xyz[mask] + self._lt_offset(torch.zeros_like(self._t))[mask]
        next_xyz = self._xyz[mask] + self._lt_offset(torch.full_like(self._t, dt))[mask]
        smooth = (prev_xyz - 2 * curr_xyz + next_xyz).abs().mean()
        return sparse, gate, smooth
