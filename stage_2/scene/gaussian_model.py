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
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, matrix_to_quaternion, quaternion_multiply
from pathlib import Path
from glob import glob
from tqdm import tqdm
from scene.gaussian_skinning import blend_weights_points


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


    def __init__(self, sh_degree : int, mode: None, npg_args: None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self._lrf = torch.empty(0)
        self._gaussian_indices = torch.empty(0)
        self._weights_blend = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.mode_req = mode
        self.npg_args = npg_args
        self.use_bgkd = bool(npg_args['use_bgkd'])


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
            self._lrf,
            self._gaussian_indices,
            self._weights_blend,

            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale
            
        )
    
    def restore(self, model_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self._lrf,
        self._gaussian_indices,
        self._weights_blend,

        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        # self.training_setup(training_args)
        # self.xyz_gradient_accum = xyz_gradient_accum
        # self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

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
    def get_bgkd_pts(self):

        #####
        a_min,b_min,c_min = float(self.npg_args['a_min']),float(self.npg_args['b_min']),float(self.npg_args['c_min'])
        a_max,b_max,c_max = float(self.npg_args['a_max']),float(self.npg_args['b_max']),float(self.npg_args['c_max'])
        
        c_x,c_y,c_z = (a_max+a_min)*0.5 , (b_max+b_min)*0.5, (c_max+c_min)*0.5

        return torch.tensor([[a_min,b_min,c_min],[a_min,b_max,c_min],[a_max,b_max,c_min],[a_max,b_min,c_min],\
                                    [a_max,b_max,c_max],[a_min,b_max,c_max],[a_min,b_min,c_max],[a_max,b_min,c_max],\
                                    [c_x,b_max,c_min],[a_min,c_y,c_min],[c_x,b_min,c_min],[a_max,c_y,c_min],\
                                    [c_x,b_max,c_max],[a_min,c_y,c_max],[c_x,b_min,c_max],[a_max,c_y,c_max],\
                                    [a_max,b_max,c_z],[a_min,b_max,c_z],[a_max,b_min,c_z],[a_min,b_min,c_z]]).cuda()[None,...]
    # # @property
    # def get_xyz_copy(self):
    #     return self._xyz_copy

    @property
    def get_weights(self):
        return self._weights_blend
        
    def get_xg(self,xyz,ids):
        if self.use_bgkd:
            cage1 = self.get_bgkd_pts
        else:
            cage1 = None
        
        self._weighted_gaussians = blend_weights_points(xyz, self._weights_blend,self._gaussian_indices, id_s = ids,cage=cage1)
        
        return self._weighted_gaussians
        # if warm:
        #     return self._weighted_gaussians, 0.0
        # else:
        #     off_position,_,off_scale = self.get_offsets(self._weighted_gaussians,ids)
        #     return self._weighted_gaussians, off_scale  #[ids] #torch.sum(torch.nn.functional.softmax(self._weights_blend,dim=1)[...,None]*self.all_points_stack[ids],dim=1)/3  #self.weighted_gaussians[ids]#blend_weights_points(self.all_points_stack,self.ref_fr_index, self._weights_blend, id_s = ids)

    @property
    def get_lrf(self):
        return self._lrf
         
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
        
    def blend_weights_points(self,kps):
        # # weighted point positions

        points_tensor = kps#torch.tensor(np.asarray(kps)).squeeze().float().cuda() # (150x1500x3)
        dis_mat_kpt = torch.cdist(points_tensor,points_tensor)**2  # (150, 1500, 1500)

        self.knn = 20

        _,nn_dist_ind_fixed= torch.topk(dis_mat_kpt, k = self.knn,largest=False,dim=2)  # (150, 1500, knn)

        # Fixed the triangles based on first Frame
        ind_first_frame_ref = nn_dist_ind_fixed[0][None,...].repeat(points_tensor.shape[0],1,1) #(150, 1500, knn)



        num_frames, num_kps,_ = points_tensor.shape
        self.weights_blend = torch.rand((num_kps,self.knn)).cuda()
        # self.all_points_stack = torch.zeros((num_frames, num_kps,self.knn,3)).cuda()
        position_gaussian = torch.zeros((num_frames,num_kps,3)).cuda()  # size- 150 x 1500 x 3

        for i in tqdm(range(num_frames),desc='Building Gaussians from PCD'):
            for j in range(num_kps):
                mask_full = torch.arange(start=0, end=num_kps, step=1).cuda()
                frame_indices = ind_first_frame_ref[i,j]
                mask_full[frame_indices] = -1
                mask_new = mask_full<0
                # self.triangle_dict[j] = mask_new
                first_neighbours = torch.masked_select(points_tensor[i], mask_new[...,None]).view(-1, 3) # knnx3
                # self.all_points_stack[i,j] = first_neighbours
                weighted = torch.nn.functional.softmax(self.weights_blend[j][...,None],dim=1) * first_neighbours
                position_gaussian[i,j] = torch.sum(weighted,dim=0)/self.knn        

        new_positions = position_gaussian#torch.cat((points_tensor,position_gaussian),dim=1)
        

        local_reference_frame = torch.zeros((points_tensor.shape[0],points_tensor.shape[1], 4))
        local_reference_frame[:,:, 0] = 1
        for idx in tqdm(range(0,num_frames),desc='Loading LRF'):
            for jdx in range(num_kps):
                mask_full_1 = torch.arange(start=0, end=num_kps, step=1)
                frame_indices_1 = ind_first_frame_ref[idx,jdx]
                mask_full_1[frame_indices_1] = -1
                mask_new_1 = mask_full_1<0
                
                # These are the two traingles(frame_idx is the current time triangle. Frame_0 is the reference frame triangle)
                frame_idx = torch.masked_select(points_tensor[idx].cpu(), mask_new_1[...,None]).view(-1, 3) # size- knnx3
                frame_0 = torch.masked_select(points_tensor[0].cpu(), mask_new_1[...,None]).view(-1, 3)  # size- knnx3
                
                # Take 3 points from KNN
                # centroid_knn = torch.mean(frame_idx,dim=0)
                frame_idx = frame_idx[:3] # size- 3x3
                frame_0 = frame_0[:3] # size- 3x3                

                # # Calculating the Normals
                # Step-1
                first_frame_normal = torch.cross(frame_0[1] - frame_0[0],frame_0[2] - frame_0[0])
                current_frame_normal = torch.cross(frame_idx[1] - frame_idx[0],frame_idx[2] - frame_idx[0])

                # make the normals unit norms and they should point in the same direction
                normal_vector_1 = first_frame_normal/torch.linalg.vector_norm(first_frame_normal)
                normal_vector_c = current_frame_normal/torch.linalg.vector_norm(current_frame_normal)

                if torch.dot(normal_vector_c,normal_vector_1)<0:
                    normal_vector_1 = -normal_vector_1
                # Step-2
                ceter_idx = torch.mean(frame_idx,dim=0)
                second_vector = ceter_idx-frame_idx[0]

                # Step-3
                third_vector = torch.cross(second_vector,normal_vector_1)

                rotation_matrix = torch.stack((normal_vector_1,second_vector,third_vector), dim = 0).T  # 3x3
                
                quat = matrix_to_quaternion(rotation_matrix[None,...])   # Function imported from pytorch3d 
                local_reference_frame[idx,jdx] = quat.squeeze(0)        

        

        # ###### New background Local Volume
        if self.use_bgkd:
            pts_backgnd = self.get_bgkd_pts # 1 x 20 x 3

            ##bkgd Gaussians weights
            weights_bgkd = torch.rand((self.knn,self.knn)).cuda()
            bgkd_gaussians_weighted = torch.nn.functional.softmax(weights_bgkd,dim=1)[...,None]*pts_backgnd        
            bgkd_gaussians = torch.sum(bgkd_gaussians_weighted,dim=0)/self.knn  # size- 20 x 3
            bgkd_gaussians = bgkd_gaussians.expand(num_frames,20,3)
            self.weights_blend = torch.cat((self.weights_blend,weights_bgkd),dim=0)
            new_positions = torch.cat((new_positions,bgkd_gaussians),dim=1) # 150 x 1520 x 3
            # ####### # ######### bkgd Gaussians LRF- just unit quaternion
            bgkd_lrf = torch.zeros((points_tensor.shape[0],20, 4))
            bgkd_lrf[:,:, 0] = 1 # size= 150 x 20 x 4
            local_reference_frame = torch.cat((local_reference_frame,bgkd_lrf),dim=1) # 150 x 1520 x 3
            #########
            ######### Arrange indices
            ind_strt,ind_end = num_kps,num_kps+20
            ind = torch.arange(start=ind_strt, end=ind_end, step=1).long()[None,...].expand(20,20).cuda()
            tot_ind = torch.cat((ind_first_frame_ref[0],ind),dim=0) # size- 1520X20
            return_ind = tot_ind
            #########
        if not self.use_bgkd:
            return_ind = ind_first_frame_ref[0]
        return new_positions, local_reference_frame, return_ind #tot_ind

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).squeeze().float().cuda()
        if fused_point_cloud.dim()==2:
            fused_point_cloud = fused_point_cloud[None,...]
        else:
             pass       
        self.weighted_gaussians, lrf, gaussian_indices = self.blend_weights_points(fused_point_cloud)#.cuda()

        # fused_point_cloud = blended_gaussians.cuda()

        # x_feat = torch.tensor(np.asarray(pcd.colors)[0])
        x_feat = torch.tensor(SH2RGB(np.random.random((self.weighted_gaussians.shape[1], 3)) / 255.0))
        feat_up = x_feat#torch.cat((x_feat,x_feat),dim=0)
        fused_color = RGB2SH(feat_up.squeeze().float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[1])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).squeeze().float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scale_list = []

        for i in range(len(fused_point_cloud)):
            dist2 = torch.clamp_min(distCUDA2(self.weighted_gaussians[i]), 0.0000001)
            scale_list.append(torch.log(torch.sqrt(dist2)))
        scale_tensor,_ = torch.min(torch.stack(scale_list),dim=0)

        scales = scale_tensor[...,None].repeat(1, 3)
        rots = torch.zeros((self.weighted_gaussians.shape[1], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((self.weighted_gaussians.shape[1], 1), dtype=torch.float, device="cuda"))


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        # self._xyz_copy = fused_point_cloud.detach().clone()

        self._gaussian_indices = gaussian_indices.cuda()
        self._lrf = lrf #nn.Parameter(lrf.requires_grad_(False)).float()

        self._weights_blend =  nn.Parameter(self.weights_blend ,requires_grad = True)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_weights.shape[0]), device="cuda")

        # Fused Pt Cld: torch.Size([182686, 3]) [02/09 00:03:02]
        # features : torch.Size([182686, 3, 16]) [02/09 00:03:02]
        # features DC: torch.Size([182686, 1, 3]) [02/09 00:03:02]
        # features Rest: torch.Size([182686, 15, 3]) [02/09 00:03:02]
        # Fused Color: torch.Size([182686, 3]) [02/09 00:03:02]
        # scales : torch.Size([182686, 3]) [02/09 00:03:02]
        # rotation : torch.Size([182686, 4]) [02/09 00:03:02]
        # opacity : torch.Size([182686, 1]) [02/09 00:03:02]
        # max_radii2D : 182686 [02/09 00:03:02]

        # our case
        # Fused Pt Cld: torch.Size([1500, 3]) [21/09 03:21:29]
        # features : torch.Size([1500, 3, 4]) [21/09 03:21:29]
        # features DC: torch.Size([1500, 1, 3]) [21/09 03:21:29]
        # features Rest: torch.Size([1500, 3, 3]) [21/09 03:21:29]
        # Fused Color: torch.Size([1500, 3]) [21/09 03:21:29]
        # scales : torch.Size([1500, 3]) [21/09 03:21:29]
        # rotation : torch.Size([1500, 4]) [21/09 03:21:29]
        # opacity : torch.Size([1500, 1]) [21/09 03:21:29]
        # max_radii2D : torch.Size([1500]) [21/09 03:21:29]
        print('Fused Pt Cld: {}'.format(fused_point_cloud.shape))
        print('features : {}'.format(features.shape))
        print('features DC: {}'.format(features[:,:,0:1].transpose(1, 2).shape))
        print('features Rest: {}'.format(features[:,:,1:].transpose(1, 2).shape))
        print('Fused Color: {}'.format(fused_color.shape))
        print('scales : {}'.format(scales.shape))
        print('rotation : {}'.format(rots.shape))
        print('opacity : {}'.format(opacities.shape))
        print('max_radii2D : {}'.format(self.max_radii2D.shape))
        print('lrf: {}'.format(lrf.shape))
        print('Weights: {}'.format(self.weights_blend.shape))
        print('indices: {}'.format(gaussian_indices.shape))
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_weights.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_weights.shape[0], 1), device="cuda")
        print(self.spatial_lr_scale)
        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': list(self._offsets_net.parameters()),'lr': training_args.position_lr_init * self.spatial_lr_scale,"name": "offsets"},
            {'params': [self._weights_blend], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "weights"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.weights_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.scale_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                    lr_final=training_args.scaling_lr_final,
                                                    lr_delay_mult=training_args.scaling_lr_delay_mult,
                                                    max_steps=training_args.scaling_lr_max_steps)        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "weights":
                lr = self.weights_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def scale_update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scale_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr    

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        # for i in range(self._scaling.shape[1]):
        #     l.append('scale_{}'.format(i))
        # for i in range(self._rotation.shape[1]):
        #     l.append('rot_{}'.format(i))

        # for i in range(self._lrf.shape[2]):
        #     l.append('lrf_rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        path_1 = path.split('/')[:-1]
        path_1 = '/'.join(path_1)

        tot_frames = len(self._xyz)
        for idx in range(tot_frames):
            xyz = self.get_xyz[idx].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # opacities = self._opacity.detach().cpu().numpy()
            # scale = self._scaling.detach().cpu().numpy()
            # rotation = self._rotation.detach().cpu().numpy()
            # lrf = self._lrf[idx].cpu().numpy()

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, lrf), axis=1)
            attributes = np.concatenate((xyz, normals), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')

            file_name = os.path.join(path_1, "{:0>8d}_pointCloud.ply".format(idx))
            PlyData([el]).write(file_name)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        
        path_1 = path.split('/')[:-1]
        path_1 = '/'.join(path_1)
        all_ply_path = sorted(glob(os.path.join(path_1, '*.ply')))
        xyz_all = []
        rots_all = []
        lrf_all = []
        for path_idx in all_ply_path:
            # print(path_idx)
            plydata = PlyData.read(path_idx)
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
            # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

            # features_dc = np.zeros((xyz.shape[0], 3, 1))
            # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            # for idx, attr_name in enumerate(extra_f_names):
            #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

            # scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            # scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
            # scales = np.zeros((xyz.shape[0], len(scale_names)))
            # for idx, attr_name in enumerate(scale_names):
            #     scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
            # rots = np.zeros((xyz.shape[0], len(rot_names)))
            # for idx, attr_name in enumerate(rot_names):
            #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            # lrf_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("lrf_rot")]
            # lrf_names = sorted(lrf_names, key = lambda x: int(x.split('_')[-1]))
            # lrf_rots = np.zeros((xyz.shape[0], len(lrf_names)))
            # for idx, attr_name in enumerate(lrf_names):
            #     lrf_rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

            xyz_all.append(xyz)
            # lrf_all.append(lrf_rots)

        xyz_stack = np.stack(xyz_all,axis=0)
        # lrf_stack = np.stack(lrf_all,axis=0)
        # rots_stack = np.stack(rots_all,axis=0)
        # print(xyz_stack.shape)
        # print(rots_stack.shape)
        self._xyz = nn.Parameter(torch.tensor(xyz_stack, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._lrf = nn.Parameter(torch.tensor(lrf_stack, dtype=torch.float, device="cuda").requires_grad_(False))
        self.active_sh_degree = self.max_sh_degree
        # print(self._lrf[1])
    
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
            if group["name"] =='xyz':
                continue
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

    def prune_points(self, mask, flag=True):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._weights_blend = optimizable_tensors["weights"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # ####### code added here
        # print('Prune Weights : {} '.format(self._weights_blend.shape))
        if not flag:
            tot_frames = self._lrf.shape[0]
            new_mask = valid_points_mask[None,...].repeat(tot_frames,1)#torch.nonzero(valid_points_mask).squeeze()[None,...].repeat(tot_frames,1)
            # select_traingles = torch.masked_select(self.all_points_stack,new_mask[...,None,None]).view(tot_frames,-1,3,3)
            # self.all_points_stack = select_traingles
            select_rotations = torch.masked_select(self._lrf,new_mask[...,None].detach().cpu()).view(tot_frames,-1,4)
            self._lrf =  select_rotations#torch.cat((self._lrf,select_rotations),dim=1)  #150 x 1500 x 4             
            
            select_gaussian_indices = torch.masked_select(self._gaussian_indices, valid_points_mask[...,None]).view(-1,self.knn)
            self._gaussian_indices = select_gaussian_indices#torch.cat((self._gaussian_indices,select_gaussian_indices),dim=0)            
            # print('All stack: {}'.format(self.all_points_stack.shape))
            # print('New Mask: {}'.format(new_mask[...,None,None].shape))
            # print('Weights: {}'.format(self._weights_blend.shape))
        ## Currently not using the below comments but don't remove it
        # select_traingles = torch.masked_select(self.all_points_stack,new_mask[...,None,None]).view(tot_frames,-1,3,3)

        # self.all_points_stack = select_traingles#torch.cat((self.all_points_stack,select_traingles),dim=1)
        
        # select_rotations = torch.masked_select(self._lrf,new_mask[...,None].detach().cpu()).view(tot_frames,-1,4)
        # self._lrf =  select_rotations#torch.cat((self._lrf,select_rotations),dim=1)  #150 x 1500 x 4   

        #################
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"]=='xyz':
                continue
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

    def densification_postfix(self, new_weights, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"weights": new_weights,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._weights_blend = optimizable_tensors["weights"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_weights.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_weights.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_weights.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_weights.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)


        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        # new_weights = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_weights[selected_pts_mask].repeat(N, 1)

        new_weights = torch.rand(self.get_weights[selected_pts_mask].repeat(N, 1).shape,device="cuda") + self.get_weights[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        #### our case for finding which triangle index needs updates
        tot_frames = self._lrf.shape[0]
        new_mask = selected_pts_mask[None,...].repeat(tot_frames,1)
        # select_traingles = torch.masked_select(self.all_points_stack,new_mask[...,None,None]).view(tot_frames,-1,self.knn,3)
        # self.all_points_stack = torch.cat((self.all_points_stack,select_traingles),dim=1)
        # For rotations
        select_rotations = torch.masked_select(self._lrf,new_mask[...,None].detach().cpu()).view(tot_frames,-1,4)
        self._lrf =  torch.cat((self._lrf,select_rotations),dim=1)  #150 x 1500 x 4
        # For gaussian indices
        select_gaussian_indices = torch.masked_select(self._gaussian_indices, selected_pts_mask[...,None]).view(-1,self.knn)
        self._gaussian_indices = torch.cat((self._gaussian_indices,select_gaussian_indices),dim=0)
        ######### 

        self.densification_postfix(new_weights, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        

        new_weights = self._weights_blend[selected_pts_mask] + torch.rand(self._weights_blend[selected_pts_mask].shape,device="cuda")
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        #### our case for finding which triangle index needs updates. Select the indices and concat them
        # indexs_req = torch.nonzero(selected_pts_mask).squeeze()[None,...]
        
        tot_frames = self._lrf.shape[0]
        new_mask = selected_pts_mask[None,...].repeat(tot_frames,1)

        # # select_traingles = self.all_points_stack[new_mask][:,None,:,:]
        # select_traingles = torch.masked_select(self.all_points_stack,new_mask[...,None,None]).view(tot_frames,-1,self.knn,3)
  
        # self.all_points_stack = torch.cat((self.all_points_stack,select_traingles),dim=1)
        # For rotations
        select_rotations = torch.masked_select(self._lrf,new_mask[...,None].detach().cpu()).view(tot_frames,-1,4)
        self._lrf =  torch.cat((self._lrf,select_rotations),dim=1)  #150 x 1500 x 4
        # For gaussian indices
        select_gaussian_indices = torch.masked_select(self._gaussian_indices, selected_pts_mask[...,None]).view(-1,self.knn)
        self._gaussian_indices = torch.cat((self._gaussian_indices,select_gaussian_indices),dim=0)
        ######### 

        self.densification_postfix(new_weights, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask,flag=False)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1