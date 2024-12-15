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
from utils.general_utils import strip_symmetric, build_scaling_rotation, matrix_to_quaternion, quaternion_multiply
from pathlib import Path
from glob import glob
from tqdm import tqdm

# input is original point position, and weights        
def blend_weights_points(points_tensor, weights, gaussian_indices, id_s = None,cage=None):
    # # weighted point positions
   # points_tensor: (150x1500x3)
   # weights : (num_gaussians, 8)
   # gaussian_indices = (num_gaussians, 8)

    num_gs, knn = weights.shape
    # num_kps,_ = points_tensor.shape

    # index selection: First expand the dimensions of point along a dimesion based on the number of gaussians and 
    # then select using the indices

    ##### Add the bagkd cage (20 points)
    if cage is not None:
        points_tensor = torch.cat((points_tensor,cage.squeeze()),dim=0)  # size=1520x3
    num_kps,_ = points_tensor.shape
    ############

    expanded_pts = points_tensor[None,...].expand(num_gs,num_kps,3) # num_gs x num_kps x 3
    index_select = torch.gather(expanded_pts,1,gaussian_indices[...,None].expand(num_gs,knn,3)) # num_gs x knn x 3


    position_gaussian = torch.sum(torch.nn.functional.softmax(weights,dim=1)[...,None]*index_select,dim=1)#/knn  

    new_positions = position_gaussian#torch.cat((points_tensor,position_gaussian),dim=1)
    # print(new_positions[0])
    return position_gaussian

