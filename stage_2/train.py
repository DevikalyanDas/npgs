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
import cv2 as cv
import numpy as np
import pathlib
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
# from scene.offsets_field import OffsetModel
from scene.npg_model import NPGModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import yaml
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, npg_conf_args, npg_ckps_1):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,'train',npg_conf_args)

    npg_model = NPGModel(npg_conf_args,dataset.source_path,npg_ckps_1)
    npg_model.train_setting(opt)


    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Lpips loss
    # loss_fn_vgg = lpips.LPIPS(net='alex').cuda()

    #check training images
    check_tr_img = os.path.join(scene.model_path,'test_op')
    pathlib.Path(check_tr_img).mkdir(parents=True, exist_ok=True)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            test_viewpoint_stack = scene.getTestCameras().copy()
            total_frames = len(viewpoint_stack)
        
        id_sel = randint(0, len(viewpoint_stack)-1)
        
        viewpoint_cam = viewpoint_stack.pop(id_sel)
        # test_viewpoint_cam = test_viewpoint_stack.pop(id_sel)

        view_id = viewpoint_cam.tid #view_id_stack.pop(id_sel)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        warm_cond = True if iteration < opt.warm_up else False

        ### obtain corresponding time step gaussians
        # if warm_cond:
        #     get_xyz = gaussians.get_xyz[view_id]
        # else:
            # img_size(h,w)
        ## npg model gives pts
        hg,wg = viewpoint_cam.original_image.shape[1],viewpoint_cam.original_image.shape[2]
        get_xyz = npg_model.step(torch.tensor(float(view_id)).cuda(),img_size=tuple([hg,wg]), total_frames= total_frames)["keypoint_3D"]
        
        ## Original point : no gradient
        xyz_learn = gaussians.get_xyz[view_id]

        # Obtain gaussians
        get_xg = gaussians.get_xg(get_xyz, view_id)
        
        num_gaussians = get_xg.shape[0]


        if iteration%2 == 0: 
            bg_color = [1, 1, 1]  
        else:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,id_p = view_id,x_g= get_xg)
        
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],render_pkg["depth"]
        
        ## Loss for point positions

        xyz_l2 = l1_loss(get_xyz, xyz_learn)

        ##Loss
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.is_masked  # fore-True, back- False

        if mask is not None:
            gt_image *= (~mask).cuda() 
            # mask = (~mask)
            if iteration%2 == 0:
                gt_image += (mask).cuda().float()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) \
                + float(npg_conf_args['w_reg']) * xyz_l2
        ## 100-real cactus,synthetic_human| 0.1-dnerf, real_human|1.0-synthetic_cactus, trex
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration%1000==0:    
                image_te = ((image.detach().clone().cpu().permute(1,2,0).numpy())*255).astype(np.uint8)
                image_ori = ((gt_image.cpu().permute(1,2,0).numpy())*255).astype(np.uint8)

                # depth_te =  ((depth.cpu().permute(1,2,0).squeeze().numpy())*255).astype(np.uint8)
                # depth_st = np.stack((depth_te,) * 3, axis=-1)

                novel_views = render(test_viewpoint_stack[0], gaussians, pipe, background,id_p = view_id,x_g= get_xg)["render"]
                novel_view_np = ((novel_views.detach().cpu().permute(1,2,0).numpy())*255).astype(np.uint8)
                # print(image_ori.shape, image_te.shape,depth_st.shape)
                vis = np.concatenate((image_ori, image_te,novel_view_np), axis=1)
                
                cv.imwrite(os.path.join(check_tr_img,'iteration_gt_{}_{}.png'.format(iteration,view_id)),cv.cvtColor(vis,cv.COLOR_BGR2RGB))                
                
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),id_n = view_id)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # break
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.scale_update_learning_rate(iteration)
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

                npg_model.optimizer.step()
                npg_model.update_npg_learning_rate(iteration)
                npg_model.optimizer.zero_grad()
                
            if (iteration in checkpoint_iterations):
                print('Final number of Gaussians: {}'.format(num_gaussians))
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                npg_model.save_weights(args.model_path, iteration)
        torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,id_n = None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/point_update', xyz_l2.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    paraser.add_argument("--npg_config",type=str,required=True,help='npg model configs from stage 1')
    paraser.add_argument("--npg_ckp",type=str,required=True,help='npg model checkpoint from stage 1')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[70_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[70_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    npg_config_path = os.path.join('config',args.npg_config+'.yaml') 

    with open(npg_config_path, 'r') as f:
        npg_args = yaml.safe_load(f)
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,npg_args,args.npg_ckp)

    # All done
    print("\nTraining complete.")