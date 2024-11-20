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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
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

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 6), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
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
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 6), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    # def densify_and_split(self, grads, grad_threshold, scene_extent):
    #     n_init_points = self.get_xyz.shape[0]
        
    #     # 生成每个维度的 selected_pts_mask 并计算总的 selected_pts_mask
    #     masks = []
    #     for i in range(6):
    #         padded_grad = torch.zeros((n_init_points), device="cuda")
    #         padded_grad[:grads[:, i:i+1].shape[0]] = grads[:, i:i+1].squeeze()
    #         mask = torch.where(padded_grad >= grad_threshold, True, False)
    #         mask = torch.logical_and(mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
    #         masks.append(mask)
        
    #     # 按位统计每个点被 mask 覆盖的次数
    #     selected_pts_mask = torch.stack(masks, dim=1).any(dim=1)
    #     mask_count = torch.stack(masks, dim=1).sum(dim=1)  # 记录每个点被几个 mask 覆盖
        
    #     # 初始化列表来保存所有的 new_xyz 和其他特性
    #     all_new_xyz, all_new_scaling, all_new_rotation = [], [], []
    #     all_new_features_dc, all_new_features_rest, all_new_opacity = [], [], []

    #     direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

    #     # 对 selected_pts_mask 中每个被选中的点，根据 mask_count 复制 n+1 次
    #     for i in range(6):
    #         # 针对第 i 个方向的 mask 选择相应点
    #         direction_mask = direction_masks[i]
    #         specific_mask = masks[i]  # 当前维度的掩码
            
    #         # 针对被该方向 mask 覆盖的点，按 mask_count 决定复制次数
    #         selected_points = self.get_xyz[specific_mask]
    #         num_copies = mask_count[specific_mask] + 1
            
    #         # 设置均值与标准差，用于生成随机扰动
    #         stds = self.get_scaling[specific_mask].repeat_interleave(num_copies, dim=0)
    #         means = torch.zeros((stds.size(0), 3), device="cuda")
    #         samples = torch.normal(mean=means, std=stds)
            
    #         # 旋转扰动
    #         rots = build_rotation(self._rotation[specific_mask]).repeat_interleave(num_copies, dim=0)
    #         random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            
    #         # 根据 direction_mask 控制方向和符号
    #         for j in range(3):
    #             if direction_mask[j] == 0:
    #                 random_samples[:, j] *= 0.5  # 对应维度置为0
    #             elif direction_mask[j] == 1:
    #                 random_samples[:, j] = torch.abs(random_samples[:, j] * 1.5)  # 保证正扰动
    #             elif direction_mask[j] == -1:
    #                 random_samples[:, j] = -torch.abs(random_samples[:, j] * 1.5)  # 保证负扰动

    #         # 生成新坐标
    #         new_xyz = selected_points.repeat_interleave(num_copies, dim=0) + random_samples
    #         all_new_xyz.append(new_xyz)
            
    #         # 根据 num_copies 复制其他属性
    #         all_new_scaling.append(self.scaling_inverse_activation(self.get_scaling[specific_mask].repeat_interleave(num_copies, dim=0) / (0.8 * num_copies)))
    #         all_new_rotation.append(self._rotation[specific_mask].repeat_interleave(num_copies, dim=0))
    #         all_new_features_dc.append(self._features_dc[specific_mask].repeat_interleave(num_copies, dim=0))
    #         all_new_features_rest.append(self._features_rest[specific_mask].repeat_interleave(num_copies, dim=0))
    #         all_new_opacity.append(self._opacity[specific_mask].repeat_interleave(num_copies, dim=0))

    #     # 将所有新生成的点和属性拼接
    #     final_new_xyz = torch.cat(all_new_xyz, dim=0)
    #     final_new_scaling = torch.cat(all_new_scaling, dim=0)
    #     final_new_rotation = torch.cat(all_new_rotation, dim=0)
    #     final_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    #     final_new_features_rest = torch.cat(all_new_features_rest, dim=0)
    #     final_new_opacity = torch.cat(all_new_opacity, dim=0)

    #     # 整合生成的新点到原数据中
    #     self.densification_postfix(final_new_xyz, final_new_features_dc, final_new_features_rest, final_new_opacity, final_new_scaling, final_new_rotation)

    #     # 调整 prune_filter 以保留原始点
    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(final_new_xyz.shape[0], device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)

    def densify_and_split_v(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        all_new_xyz, all_new_scaling, all_new_rotation = [], [], []
        all_new_features_dc, all_new_features_rest, all_new_opacity = [], [], []
        num=0
        for i in range(6):
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            direction_mask = direction_masks[i]

            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] *= 0  # 对应维度置为0
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*1.5)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*1.0)  # 保证负扰动

            # 将扰动与原始坐标结合
            all_new_xyz.append(self.get_xyz[selected_pts_mask] + random_samples)

            # stds = self.get_scaling[selected_pts_mask]  # 尺度
            # means = torch.zeros((stds.size(0), 3), device="cuda")  # 均值（新分布的中心点）
            # samples = torch.normal(mean=means, std=stds)  # 随机采样新的位置
            # rots = build_rotation(self._rotation[selected_pts_mask]) # 旋转
        
            # # 计算新的位置
            # all_new_xyz.append(torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask])

            all_new_scaling.append(self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (1.6)))
            all_new_rotation.append(self._rotation[selected_pts_mask])
            all_new_features_dc.append(self._features_dc[selected_pts_mask])
            all_new_features_rest.append(self._features_rest[selected_pts_mask])
            all_new_opacity.append(self._opacity[selected_pts_mask])
            mask = selected_pts_mask if i == 0 else torch.logical_or(mask, selected_pts_mask)

            num += selected_pts_mask.sum()
            
        all_new_xyz.append(self.get_xyz[mask])
        all_new_scaling.append(self.scaling_inverse_activation(self.get_scaling[mask] / (1.6)))
        all_new_rotation.append(self._rotation[mask])
        all_new_features_dc.append(self._features_dc[mask])
        all_new_features_rest.append(self._features_rest[mask])
        all_new_opacity.append(self._opacity[mask])
        num += mask.sum()

        # Convert all lists of tensors into a single tensor by concatenating
        new_xyz = torch.cat(all_new_xyz, dim=0)
        new_scaling = torch.cat(all_new_scaling, dim=0)
        new_rotation = torch.cat(all_new_rotation, dim=0)
        new_features_dc = torch.cat(all_new_features_dc, dim=0)
        new_features_rest = torch.cat(all_new_features_rest, dim=0)
        new_opacity = torch.cat(all_new_opacity, dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((mask, torch.zeros(num, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_v0(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        
        for i in range(6):
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
            mask = selected_pts_mask if i == 0 else torch.logical_or(mask, selected_pts_mask)
            
        stds = self.get_scaling[mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[mask].repeat(N,1)
        new_features_dc = self._features_dc[mask].repeat(N,1,1)
        new_features_rest = self._features_rest[mask].repeat(N,1,1)
        new_opacity = self._opacity[mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((mask, torch.zeros(N * mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_v1(self, grads, grad_threshold, scene_extent, small):
        # Extract points that satisfy the gradient condition
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            stds = self.get_scaling[selected_pts_mask]  # 尺度
            means = torch.zeros((stds.size(0), 3), device="cuda")  # 均值（新分布的中心点）
            samples = torch.normal(mean=means, std=stds)  # 随机采样新的位置
            rots = build_rotation(self._rotation[selected_pts_mask]) # 旋转
        
            # 计算新的位置
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (small))
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            
            if i == 0:
                mask = selected_pts_mask
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
            else:
                mask = torch.logical_or(mask, selected_pts_mask)
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))

        new_xyz = self.get_xyz[mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask] / (small))
        new_rotation = self._rotation[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacity = self._opacity[mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((mask, torch.zeros(mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_v2(self, grads, grad_threshold, scene_extent, x, y, small):
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            direction_mask = direction_masks[i]
            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] = random_samples[:, j]*y
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*x)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*x)  # 保证负扰动

            # 将扰动与原始坐标结合
            new_xyz = self.get_xyz[selected_pts_mask] + random_samples
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (small))
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            
            if i == 0:
                mask = selected_pts_mask
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
            else:
                mask = torch.logical_or(mask, selected_pts_mask)
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))

        new_xyz = self.get_xyz[mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask] / (small))
        new_rotation = self._rotation[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacity = self._opacity[mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((mask, torch.zeros(mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_v3(self, grads, grad_threshold, scene_extent, x, y, small):
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        import pdb; pdb.set_trace()
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            if (i%2 == 1):
                selected_pts_mask = torch.logical_and(pre_mask, selected_pts_mask)

            pre_mask = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))

            direction_mask = direction_masks[i]
            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] = random_samples[:, j]*y
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*x)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*x)  # 保证负扰动

            # 将扰动与原始坐标结合
            new_xyz = self.get_xyz[selected_pts_mask] + random_samples
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (small))
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            
            if i == 0:
                mask = selected_pts_mask
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
            else:
                mask = torch.logical_or(mask, selected_pts_mask)
                mask = torch.cat((mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))

        new_xyz = self.get_xyz[mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask] / (small))
        new_rotation = self._rotation[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacity = self._opacity[mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((mask, torch.zeros(mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_v4(self, grads, grad_threshold, scene_extent, x, y, small):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        all_new_xyz, all_new_scaling, all_new_rotation = [], [], []
        all_new_features_dc, all_new_features_rest, all_new_opacity = [], [], []
        num=0
        for i in range(6):
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            direction_mask = direction_masks[i]

            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] = random_samples[:, j]*y
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*x)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*x)  # 保证负扰动

            # 将扰动与原始坐标结合
            all_new_xyz.append(self.get_xyz[selected_pts_mask] + random_samples)
            all_new_scaling.append(self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (small)))
            all_new_rotation.append(self._rotation[selected_pts_mask])
            all_new_features_dc.append(self._features_dc[selected_pts_mask])
            all_new_features_rest.append(self._features_rest[selected_pts_mask])
            all_new_opacity.append(self._opacity[selected_pts_mask])
            mask = selected_pts_mask if i == 0 else torch.logical_or(mask, selected_pts_mask)

            num += selected_pts_mask.sum()
            
        all_new_xyz.append(self.get_xyz[mask])
        all_new_scaling.append(self.scaling_inverse_activation(self.get_scaling[mask] / (small)))
        all_new_rotation.append(self._rotation[mask])
        all_new_features_dc.append(self._features_dc[mask])
        all_new_features_rest.append(self._features_rest[mask])
        all_new_opacity.append(self._opacity[mask])
        num += mask.sum()

        # Convert all lists of tensors into a single tensor by concatenating
        new_xyz = torch.cat(all_new_xyz, dim=0)
        new_scaling = torch.cat(all_new_scaling, dim=0)
        new_rotation = torch.cat(all_new_rotation, dim=0)
        new_features_dc = torch.cat(all_new_features_dc, dim=0)
        new_features_rest = torch.cat(all_new_features_rest, dim=0)
        new_opacity = torch.cat(all_new_opacity, dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((mask, torch.zeros(num, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # import pdb; pdb.set_trace()
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_clone_v0(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        for i in range(6):
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            mask = selected_pts_mask if i == 0 else torch.logical_or(mask, selected_pts_mask)
        
        new_xyz = self._xyz[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacities = self._opacity[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify_and_clone_v(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        all_new_xyz, all_new_scaling, all_new_rotation = [], [], []
        all_new_features_dc, all_new_features_rest, all_new_opacity = [], [], []
        num=0
        for i in range(6):
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

            direction_mask = direction_masks[i]

            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] *= 0  # 对应维度置为0
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*1.5)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*1.5)  # 保证负扰动

            # 将扰动与原始坐标结合
            all_new_xyz.append(self.get_xyz[selected_pts_mask] + random_samples)
            all_new_scaling.append(self._scaling[selected_pts_mask])
            all_new_rotation.append(self._rotation[selected_pts_mask])
            all_new_features_dc.append(self._features_dc[selected_pts_mask])
            all_new_features_rest.append(self._features_rest[selected_pts_mask])
            all_new_opacity.append(self._opacity[selected_pts_mask])

        self.densification_postfix(all_new_xyz, all_new_features_dc, all_new_features_rest, all_new_opacity, all_new_scaling, all_new_rotation)

    def densify_and_clone_v1(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

            stds = self.get_scaling[selected_pts_mask]  # 尺度
            means = torch.zeros((stds.size(0), 3), device="cuda")  # 均值（新分布的中心点）
            samples = torch.normal(mean=means, std=stds)  # 随机采样新的位置
            rots = build_rotation(self._rotation[selected_pts_mask]) # 旋转
        
            # 计算新的位置
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_clone_v2(self, grads, grad_threshold, scene_extent, x, y):
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

            direction_mask = direction_masks[i]
            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] = random_samples[:, j]*y  
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*x)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*x)  # 保证负扰动

            # 将扰动与原始坐标结合
            new_xyz = self.get_xyz[selected_pts_mask] + random_samples
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_clone_v3(self, grads, grad_threshold, scene_extent, x, y):
        # Extract points that satisfy the gradient condition
        direction_masks = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in range(6):
            n_init_points = self.get_xyz.shape[0]
            grad = grads[:, i:i+1]
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grad.shape[0]] = grad.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            if (i%2 == 1):
                selected_pts_mask = torch.logical_and(pre_mask, selected_pts_mask)

            pre_mask = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))

            direction_mask = direction_masks[i]
            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            # 计算扰动
            random_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 根据direction_mask控制保留的方向和符号
            for j in range(3):
                if direction_mask[j] == 0:
                    random_samples[:, j] = random_samples[:, j]*y  
                elif direction_mask[j] == 1:
                    random_samples[:, j] = torch.abs(random_samples[:, j]*x)  # 保证正扰动
                elif direction_mask[j] == -1:
                    random_samples[:, j] = -torch.abs(random_samples[:, j]*x)  # 保证负扰动

            # 将扰动与原始坐标结合
            new_xyz = self.get_xyz[selected_pts_mask] + random_samples
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, dataset, opt):
        # grads = self.xyz_gradient_accum / self.denom
        # grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        # grads[grads.isnan()] = 0.0

        grads = self.xyz_gradient_accum / self.denom  # 直接进行六个维度的除法运算
        grads[grads.isnan()] = 0.0  # 将 NaN 替换为 0

        # import pdb; pdb.set_trace()
        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)
        if (dataset.clone == 'v0'):
            self.densify_and_clone_v0(grads, max_grad, extent)
            # self.densify_and_clone(grads[:, 0:1], max_grad, extent)

        if (dataset.clone == 'v1'):
            self.densify_and_clone_v1(grads, max_grad, extent)

        if (dataset.clone == 'v2'):
            self.densify_and_clone_v2(grads, max_grad, extent, opt.x, opt.y)
        
        if (dataset.clone == 'v3'):
            self.densify_and_clone_v3(grads, max_grad, extent, opt.x, opt.y)

        if (dataset.split == 'v0'):
            self.densify_and_split_v0(grads, max_grad, extent)

        if (dataset.split == 'v1'):
            self.densify_and_split_v1(grads, max_grad, extent, opt.small)
            # self.densify_and_split(grads[:, 0:1], max_grad, extent)

        if (dataset.split == 'v2'):
            self.densify_and_split_v2(grads, max_grad, extent, opt.x, opt.y, opt.small)

        if (dataset.split == 'v3'):
            self.densify_and_split_v3(grads, max_grad, extent, opt.x, opt.y, opt.small)

        if (dataset.split == 'v4'):
            self.densify_and_split_v4(grads, max_grad, extent, opt.x, opt.y, opt.small)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_x, viewspace_point_tensor_y, viewspace_point_tensor_z, update_filter):
        # 对 xyz_gradient_accum 的第一个维度进行更新，确保仅在该维度上计算
        # import pdb; pdb.set_trace()
        # self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor_x.grad[update_filter, 0:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter, 1:2] += torch.norm(viewspace_point_tensor_x.grad[update_filter, 2:4], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter, 2:3] += torch.norm(viewspace_point_tensor_y.grad[update_filter, 0:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter, 3:4] += torch.norm(viewspace_point_tensor_y.grad[update_filter, 2:4], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter, 4:5] += torch.norm(viewspace_point_tensor_z.grad[update_filter, 0:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter, 5:6] += torch.norm(viewspace_point_tensor_z.grad[update_filter, 2:4], dim=-1, keepdim=True)
        
        # 更新 denom 对应的索引
        self.denom[update_filter] += 1
