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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# from scene.offsets_field import OffsetModel
from scene.npg_model import NPGModel
import imageio
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
from glob import glob

def render_set(model_path, name, iteration, views, gaussians, pipeline, data_back,npg_model,total_frames):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    renderings = []
    depth_max = []
    depth_min = []
    all_depth = []
    print(total_frames)
    bg_mode = 'black' # white
    if bg_mode == 'black':
        bg_color = [1,1,1] if data_back.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    else:
        bg_color = [1,1,1] #if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    
        view_id = view.tid

        hg,wg = view.original_image.shape[1],view.original_image.shape[2]        

        # xyz_learn = gaussians.get_xyz[view_id]

        get_xyz = npg_model.step(torch.tensor(float(view_id)).cuda(),img_size=tuple([hg,wg]), total_frames= total_frames)["keypoint_3D"]

        get_xg = gaussians.get_xg(get_xyz, view_id)
        num_gaussians = get_xg.shape[0]

        render_pkg = render(view, gaussians, pipeline, background,id_p = view_id,x_g= get_xg)

        rendering, depth, alpha = render_pkg["render"], render_pkg["depth"],render_pkg["alpha"]
        masked = (view.is_masked).cuda()
        if bg_mode=='black':
            gt = view.original_image[0:3, :, :]*(~masked) #+ masked * background[:,None,None]
        else:
            gt = view.original_image[0:3, :, :]*(~masked) + masked * background[:,None,None]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        depth_norm1 = depth.cpu().permute(1,2,0).squeeze().numpy()
        # print(depth_norm.max())
        depth_norm = depth_norm1/(np.max(depth_norm1)*1.1-np.min(depth_norm1))
        depth_te =  ((depth_norm)*255).astype(np.uint8)
        depth_max.append(depth_norm.max())    
        depth_min.append(depth_norm.min())

        alpha_te =  ((alpha.cpu().permute(1,2,0).squeeze().numpy())*255).astype(np.uint8)
        alpha_te = np.stack((alpha_te,) * 3, axis=-1)

        # # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        cv.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"),cv.cvtColor(depth_te,cv.COLOR_BGR2RGB))
        cv.imwrite(os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"),cv.cvtColor(alpha_te,cv.COLOR_BGR2RGB))
        renderings.append(to8b(rendering.cpu().numpy()))
        
        if name=='train':
            pt_cld_path = os.path.join(model_path, "point_cloud","npg")
            npg_model.save_ply(pt_cld_path,get_xyz,view_id)
            gaussian_path = os.path.join(model_path, "point_cloud","gaussians")
            npg_model.save_ply(gaussian_path,get_xg,view_id)
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    print('Max values:{}'.format(max(depth_max)))
    print('Min values:{}'.format(min(depth_min)))    
def render_interpolate(model_path, name, iteration, views, gaussians, pipeline, data_back,npg_model,total_frames):
    render_path = os.path.join(model_path, name, "ours_interpolate_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_interpolate_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_interpolate_{}".format(iteration), "alpha")
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    renderings = []
    depth_max = []
    depth_min = []
    all_depth = []
    total_frames = 150
    print(total_frames)
    bg_mode = 'white' # white
    if bg_mode == 'black':
        bg_color = [1,1,1] if data_back.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    else:
        bg_color = [1,1,1] #if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ids_12 = []
    for idx, view in enumerate(tqdm(views, desc="Rendering interpolation progress")):
    
        view_id = view.tid
        ids_12.append(idx)
        hg,wg = view.original_image.shape[1],view.original_image.shape[2]        

        # xyz_learn = gaussians.get_xyz[view_id]

        get_xyz = npg_model.step(torch.tensor(float(idx)).cuda(),img_size=tuple([hg,wg]), total_frames= total_frames)["keypoint_3D"]

        get_xg = gaussians.get_xg(get_xyz, view_id)
        num_gaussians = get_xg.shape[0]

        render_pkg = render(view, gaussians, pipeline, background,id_p = view_id,x_g= get_xg)

        rendering, depth, alpha = render_pkg["render"], render_pkg["depth"],render_pkg["alpha"]
        masked = (view.is_masked).cuda()

        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        depth_norm1 = depth.cpu().permute(1,2,0).squeeze().numpy()
        # print(depth_norm.max())
        depth_norm = depth_norm1/(np.max(depth_norm1)*1.1-np.min(depth_norm1))
        depth_te =  ((depth_norm)*255).astype(np.uint8)
        depth_max.append(depth_norm.max())    
        depth_min.append(depth_norm.min())

        alpha_te =  ((alpha.cpu().permute(1,2,0).squeeze().numpy())*255).astype(np.uint8)
        alpha_te = np.stack((alpha_te,) * 3, axis=-1)

        # # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        cv.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"),cv.cvtColor(depth_te,cv.COLOR_BGR2RGB))
        cv.imwrite(os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"),cv.cvtColor(alpha_te,cv.COLOR_BGR2RGB))
        renderings.append(to8b(rendering.cpu().numpy()))
        
        if name=='train':
            pt_cld_path = os.path.join(model_path, "point_cloud","npg")
            npg_model.save_ply(pt_cld_path,get_xyz,view_id)
            gaussian_path = os.path.join(model_path, "point_cloud","gaussians")
            npg_model.save_ply(gaussian_path,get_xg,view_id)
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    print(ids_12) 
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, npg_conf_args,npg_ckps_1):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 'test')
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        chkpt_name = sorted(glob(os.path.join(dataset.model_path,'*.pth')))
        (model_params, first_iter) = torch.load(chkpt_name[-1])
        gaussians.restore(model_params)

        npg_model = NPGModel(npg_conf_args,dataset.source_path,npg_ckps_1)
        npg_model.load_weights(dataset.model_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        tot_frames = len(scene.getTrainCameras())
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, dataset,npg_model,tot_frames)

        if not skip_test:
            mode = 'test_view'  # test_view , interpolate_view
            if mode == 'test_view':
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, dataset,npg_model,tot_frames)
            elif mode == 'interpolate_view':
                render_interpolate(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, dataset,npg_model,tot_frames)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    paraser.add_argument("--npg_config",type=str,required=True,help='npg model configs from stage 1')
    paraser.add_argument("--npg_ckp",type=str,required=True,help='npg model checkpoint from stage 1')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    npg_config_path = os.path.join('config',args.npg_config+'.yaml') 

    with open(npg_config_path, 'r') as f:
        npg_args = yaml.safe_load(f)
        
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, npg_args,args.npg_ckp)